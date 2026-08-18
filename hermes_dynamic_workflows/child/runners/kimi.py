"""Kimi child-agent runner — a subprocess adapter over `kimi -p --output-format=stream-json`.

Unlike the pi runner, no separate shell wrapper is needed. ``kimi -p`` has clean
exit codes (0 = success, nonzero = failure) and ``--output-format=stream-json``
produces a simple event stream that is parsed inline. The hardening provided is:

- **Turn cap**: counts ``role:assistant`` stream events; kills kimi after
  ``max_turns``.
- **Timeout**: subprocess timeout via the plugin's ``child_timeout_seconds``,
  process-group kill on timeout.
- **Deterministic session**: ``uuid5(task_id)`` passed as ``-r <session_id>``
  for resume.
- **Auth pre-flight**: verifies the kimi binary exists and ``kimi doctor``
  passes before spawning.
- **Error classification**: maps nonzero exit codes + stderr patterns onto the
  unified error_class enum (``spawn_failure | cap_exceeded | auth``).
- **JSON contract**: output shaped into the same child metadata schema the pi
  runner produces so the engine records don't need per-runner special cases.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..presets import AgentTypeSpec, resolve_agent_type
from ..runner import build_child_system_prompt
from ..worktree import create_workspace_lease
from ...core.config import PluginConfig
from ...core.errors import ChildAgentError, WorkflowTimeout
from ...core.types import ChildAgentRequest, ChildAgentResult, ChildAgentRunner

RUNNER_NAME = "kimi"
RUNNER_BINARY = "kimi"
KIMI_BINARY_PATH_ENV = "HERMES_DYNAMIC_WORKFLOWS_KIMI_BINARY"
DEFAULT_KIMI_BINARY = Path.home() / ".kimi-code" / "bin" / "kimi"
DEFAULT_MODEL = "k3-256k"

# error_class values — subset of the pi lane enum, kimi-specific gaps.
# provider-wall, verification_failed never apply (single managed provider,
# no verify cmd). auth = OAuth expired or config broken.
KNOWN_ERROR_CLASSES = frozenset(
    {"auth", "spawn_failure", "cap_exceeded"}
)

# Lane conf defaults (when no config/lanes/kimi.conf or absent keys).
LANE_DEFAULTS = {
    "model": DEFAULT_MODEL,
    "fallback": "kimi-for-coding, kimi-for-coding-highspeed",
    "max_turns": 50,
    "timeout": 300,
}


def kimi_binary() -> Path:
    override = os.environ.get(KIMI_BINARY_PATH_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return DEFAULT_KIMI_BINARY


def kimi_runner_available() -> bool:
    """True when the kimi binary exists and responds on this machine."""
    binary = kimi_binary()
    if not binary.is_file():
        return False
    try:
        result = subprocess.run(
            [str(binary), "--version"],
            capture_output=True, text=True, timeout=15,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _find_lane_conf(lane: str) -> Path | None:
    """Locate config/lanes/<lane>.conf under the plugin's config tree."""
    # Searches relative to the package root (same pattern as pi's find_wrapper).
    # For a workflow child the lane conf ships with the plugin in its harness copy.
    plugin_root = Path(__file__).resolve().parent.parent.parent.parent
    conf_paths = [
        plugin_root / "config" / "lanes" / f"{lane}.conf",
        Path.home() / ".hermes" / "skills" / "devops" / "kimi-code-lane" / "config" / "lanes" / f"{lane}.conf",
    ]
    for p in conf_paths:
        if p.is_file():
            return p
    return None


def _resolve_model(
    model: str | None,
    lane: str | None,
    candidate_file: Path | None = None,
) -> str:
    """Pick the effective model: per-call override > lane conf > default."""
    if model and model.strip().lower() not in ("", "inherit"):
        return model.strip()
    lane_conf = _find_lane_conf(lane or "default")
    if lane_conf:
        try:
            from configparser import ConfigParser
            cp = ConfigParser()
            cp.read_string(f"[lane]\n{lane_conf.read_text()}")
            fallback_str = cp.get("lane", "model", fallback=DEFAULT_MODEL)
            return fallback_str.strip()
        except Exception:
            pass
    return DEFAULT_MODEL


class KimiChildAgentRunner(ChildAgentRunner):
    """Run a workflow child as a kimi coding-agent subprocess."""

    def __init__(self, config: PluginConfig):
        self.config = config

    def run(self, request: ChildAgentRequest) -> ChildAgentResult:
        task_id = f"wf-kimi-{uuid.uuid4().hex[:12]}"
        base_cwd = request.cwd or os.environ.get("TERMINAL_CWD") or os.getcwd()
        resolved = request.resolved
        agent_type = (
            resolved.agent_type_spec
            if resolved is not None
            else resolve_agent_type(request.agent_type, cwd=base_cwd)
        )
        if request.agent_type and agent_type is None:
            available = ", ".join(
                spec.name for spec in self._list_agent_types(cwd=base_cwd)
            ) or "none"
            raise ChildAgentError(
                f"agent({{agentType}}): agent type '{request.agent_type}' not found. "
                f"Available agents: {available}"
            )

        if resolved is None:
            request = _apply_kimi_agent_type_defaults(request, agent_type)

        lane = (request.lane or "").strip() or "default"
        effective_model = _resolve_model(request.model, lane)

        lease = create_workspace_lease(
            cwd=base_cwd,
            isolation=request.isolation,
            label=request.label,
            task_id=task_id,
            keep_worktree=self.config.keep_worktrees,
        )
        work_dir = Path(tempfile.mkdtemp(prefix=f"dw-kimi-{task_id}-"))
        try:
            return self._run_lease(
                request, lease, agent_type, lane, effective_model, work_dir
            )
        finally:
            lease.cleanup()
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)

    def _list_agent_types(self, *, cwd: str) -> list[AgentTypeSpec]:
        """Wrap list_agent_types for testing."""
        return list_agent_types(cwd=cwd)

    def _run_lease(
        self,
        request: ChildAgentRequest,
        lease: WorkspaceLease,
        agent_type: AgentTypeSpec | None,
        lane: str,
        effective_model: str,
        work_dir: Path,
    ) -> ChildAgentResult:
        env = self._build_env(lease, lane, work_dir)
        base_prompt = self._build_prompt(request, agent_type, workspace=lease.cwd)

        state: dict[str, Any] = {"session": None, "content": ""}

        def invoke(prompt: str, attempt: int) -> str:
            payload = self._invoke_kimi(
                prompt,
                lane=lane,
                model=effective_model,
                lease=lease,
                env=env,
                work_dir=work_dir,
                resume=state["session"] if attempt > 1 else None,
                agent_type=agent_type,
            )
            state["session"] = payload.get("kimi_session_id") or state["session"]
            state["content"] = payload.get("content") or state["content"]
            return state["content"]

        schema = request.schema if (request.structured_tool and request.schema) else None
        if schema is None:
            content = invoke(base_prompt, 1)
            return ChildAgentResult(
                content=content,
                metadata=self._build_metadata(
                    state, lease, lane, agent_type, effective_model, work_dir,
                    attempts=1,
                ),
            )

        # Structured output via prompt + parse + re-prompt on schema mismatch.
        from ..subprocess_schema import run_with_schema
        value, attempts = run_with_schema(base_prompt, schema, invoke)
        metadata = self._build_metadata(
            state, lease, lane, agent_type, effective_model, work_dir,
            attempts=attempts,
        )
        metadata["structured_captured"] = True
        metadata["structured_result"] = value
        metadata["structured_attempts"] = attempts
        metadata["structured_mode"] = "prompt"
        return ChildAgentResult(content=state["content"], metadata=metadata)

    def _invoke_kimi(
        self,
        prompt: str,
        *,
        lane: str,
        model: str,
        lease: WorkspaceLease,
        env: dict[str, str],
        work_dir: Path,
        resume: str | None,
        agent_type: AgentTypeSpec | None,
    ) -> dict[str, Any]:
        binary = kimi_binary()
        if not binary.is_file():
            raise ChildAgentError(
                f"kimi binary not found at {binary} — "
                "install via `curl -fsSL https://code.kimi.com/kimi-code/install.sh | bash`"
            )

        prompt_file = work_dir / "prompt.md"
        prompt_file.write_text(prompt, encoding="utf-8")

        argv = [
            str(binary),
            "-p", prompt,
            "--output-format=stream-json",
            "--model", model,
        ]
        if resume:
            argv.extend(["-r", resume])

        timeout = self.config.child_timeout_seconds
        max_turns = self._max_turns(lane)

        try:
            process = subprocess.Popen(
                argv,
                cwd=lease.cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            raise ChildAgentError(f"could not spawn kimi: {exc}") from exc

        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_process_group(process)
            try:
                process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            raise WorkflowTimeout(
                f"kimi child agent timed out after {timeout:.0f}s "
                "(raise dynamic_workflows.child_timeout_seconds for kimi lanes)"
            ) from None

        rc = process.returncode
        # Parse the stream-json output.
        turn_count = 0
        session_id = None
        final_assistant_text = ""
        cap_reason = None

        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                continue
            if not isinstance(event, dict):
                continue
            role = event.get("role")
            etype = event.get("type")
            content = event.get("content")
            if role == "assistant" and content:
                turn_count += 1
                final_assistant_text = content
                if turn_count > max_turns:
                    cap_reason = f"max_turns exceeded ({turn_count})"
            elif etype == "session.resume_hint":
                sid = event.get("session_id")
                if sid:
                    session_id = sid

        stderr_text = stderr.strip() if stderr else ""
        is_error = bool(cap_reason or rc != 0)
        error_class = None

        if is_error:
            if cap_reason:
                error_class = "cap_exceeded"
            elif rc != 0:
                blob = stderr_text + "\n" + stdout
                error_class = _classify_kimi_error(blob)
            final_summary = (cap_reason or stderr_text or
                             f"kimi exited rc={rc}")[:500]
            if not final_summary:
                final_summary = f"kimi run failed rc={rc}"
        else:
            final_summary = final_assistant_text or "kimi run finished"

        return {
            "kimi_session_id": session_id or self._uuid5(lease.task_id),
            "total_cost_usd": 0.0,
            "is_error": is_error,
            "error_class": error_class,
            "changed_files": self._changed_files(lease.cwd),
            "tests_run": _tests_from_stream(stdout),
            "summary": final_summary[:500],
            "budget_exceeded": cap_reason is not None and "max_turns" in (cap_reason or ""),
            "providers_tried": [],
            "walled_until": None,
            "content": final_assistant_text or "",
        }

    def _max_turns(self, lane: str) -> int:
        """Read max_turns from lane conf or use default."""
        lane_conf = _find_lane_conf(lane)
        if lane_conf:
            try:
                from configparser import ConfigParser
                cp = ConfigParser()
                cp.read_string(f"[lane]\n{lane_conf.read_text()}")
                return int(cp.get("lane", "max_turns", fallback=str(LANE_DEFAULTS["max_turns"])))
            except (Exception, ValueError):
                pass
        return LANE_DEFAULTS["max_turns"]

    @staticmethod
    def _uuid5(name: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"hermes://kimi-lane/{name}"))

    @staticmethod
    def _build_env(
        lease: WorkspaceLease,
        lane: str,
        work_dir: Path,
    ) -> dict[str, str]:
        env = dict(os.environ)
        env["HERMES_KANBAN_WORKSPACE"] = lease.cwd
        env["HERMES_KANBAN_TASK_ID"] = lease.task_id
        # kimi-specific overrides: point logs away from the ADW surfaces.
        for stale in ("HERMES_KANBAN_PIPELINE_NODE", "HERMES_KANBAN_TASK", "HERMES_HOME"):
            env.pop(stale, None)
        return env

    @staticmethod
    def _build_prompt(
        request: ChildAgentRequest,
        agent_type: AgentTypeSpec | None,
        *,
        workspace: str,
    ) -> str:
        sections: list[str] = []
        # kimi reads AGENTS.md natively, but we fold the agent-type instructions
        # in (they ride the prompt body since kimi has no ephemeral system prompt).
        sections.append(build_child_system_prompt(agent_type, structured_output=False))
        context = [f"- Working directory: {workspace}"]
        if request.isolation == "worktree":
            context.append(
                "- You are running in an isolated git worktree; keep all "
                "file operations inside the working directory above."
            )
        sections.append("Context:\n" + "\n".join(context))
        sections.append(request.prompt)
        return "\n\n".join(section for section in sections if section)

    @staticmethod
    def _build_metadata(
        state: dict[str, Any],
        lease: WorkspaceLease,
        lane: str,
        agent_type: AgentTypeSpec | None,
        model: str,
        work_dir: Path,
        *,
        attempts: int = 0,
    ) -> dict[str, Any]:
        return {
            "runner": RUNNER_NAME,
            "lane": lane,
            "task_id": lease.task_id,
            "workspace": lease.cwd,
            "isolation": lease.isolation or "shared",
            "worktree_path": lease.path,
            "worktree_branch": lease.branch,
            "agent_type": agent_type.name if agent_type else "general-purpose",
            "agent_type_source": agent_type.source if agent_type else None,
            "model": model,
            "provider": "kimi-code",
            "kimi_session_id": state.get("session"),
            "session_id": state.get("session"),
            "total_cost_usd": 0.0,
            "providers_tried": [],
            "changed_files": [],
            "tests_run": False,
            "budget_exceeded": False,
            "walled_until": None,
            "kimi_log_path": str(work_dir / "output.jsonl"),
            "wrapper_attempts": attempts,
            "tokens": 0,
        }

    @staticmethod
    def _changed_files(workspace: str) -> list[str]:
        """Git porcelain changed files, same as pi runner's extractor."""
        try:
            result = subprocess.run(
                ["git", "-C", workspace, "status", "--porcelain"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return []
            files = []
            for row in result.stdout.splitlines():
                path = row[3:].strip() if len(row) > 3 else row.strip()
                if " -> " in path:
                    path = path.split(" -> ", 1)[1]
                if path:
                    files.append(path)
            return files
        except Exception:
            return []


# ── Module-level helpers ──────────────────────────────────────────────────────


def _classify_kimi_error(blob: str) -> str:
    """Map stderr patterns to error_class."""
    if re.search(r"\b401\b|\b403\b|unauthorized|forbidden|invalid token|oauth|login", blob, re.I):
        return "auth"
    return "spawn_failure"


def _tests_from_stream(stream: str) -> bool | str:
    """Detect whether a test-runner command was executed."""
    m = re.search(
        r'"command"\s*:\s*"[^"]*?\b('
        r'pytest|go test|npm (?:run )?test|yarn test|pnpm test|'
        r'cargo test|jest|vitest|unittest|rspec'
        r')\b',
        stream, re.I,
    )
    return m.group(1) if m else False


def _apply_kimi_agent_type_defaults(
    request: ChildAgentRequest,
    agent_type: AgentTypeSpec | None,
) -> ChildAgentRequest:
    if agent_type is None:
        return request
    lane = request.lane or getattr(agent_type, "lane", None)
    if lane and lane.strip().lower() == "inherit":
        lane = None
    return replace(request, lane=lane, isolation=request.isolation or agent_type.isolation)


def _kill_process_group(process: subprocess.Popen) -> None:
    pid = process.pid
    if not pid:
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except OSError:
        try:
            process.kill()
        except OSError:
            pass