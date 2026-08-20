"""Acpx child-agent runner — a subprocess adapter over the acpx headless ACP client.

``acpx`` (github.com/openclaw/acpx, npm ``acpx``) is a headless ACP client with
a built-in agent registry. Its ``kimi`` profile spawns the bare ``kimi acp``
agent server and exposes persistent named sessions keyed ``(agentCommand, cwd,
name)`` plus one-shot ``exec``. This runner replaces kimi.py's bespoke argv
glue — ``[kimi, -p, <prompt>, --output-format=stream-json]`` with ps/ARG_MAX
exposure — with acpx's stdin-prompt contract:

- **Arity**: the prompt rides stdin (``-f -``), so a long workflow prompt never
  lands in argv (kimi.py:244-252's ps/ARG_MAX exposure is not replicated).
- **Persistent named session**: ``sessions ensure --name dw-<task_id>`` before
  the first prompt; schema retries re-prompt the SAME named session (acpx queues
  follow-ups), replacing kimi.py's ``-r <session>`` resume. ``sessions close``
  soft-closes on completion (keeps history).
- **Cost honesty (spec 029 null-not-zero)**: A3 observed that kimi acp emits
  token usage via ``session/update`` ``usage_update`` but NO cost/price shape
  anywhere in the NDJSON (no ``result._meta``). So ``total_cost_usd`` carries
  ``null`` plus a ``cost_unavailable`` disclosure — never a fabricated 0.0.
- **Stable exit codes** (0/1/2/3/4/5/130) map onto the unified error_class enum.
- **Auth pre-flight**: the runner must prepend ``~/.kimi-code/bin`` to the child
  env (A2: bare ``kimi acp`` fails to spawn without it) and verify the binary
  resolves before spawning.

``--format json --json-strict`` (BOTH global, before the ``kimi`` token) emits
raw NDJSON ACP traffic with non-JSON stderr suppressed; the terminal
``session/prompt`` result line carries ``result.stopReason``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..presets import AgentTypeSpec, list_agent_types, resolve_agent_type
from ..runner import build_child_system_prompt
from ..worktree import WorkspaceLease, create_workspace_lease
from ...core.config import PluginConfig
from ...core.errors import ChildAgentError, WorkflowTimeout
from ...core.types import ChildAgentRequest, ChildAgentResult, ChildAgentRunner

RUNNER_NAME = "acpx"
RUNNER_BINARY = "acpx"
ACPX_BINARY_PATH_ENV = "HERMES_DYNAMIC_WORKFLOWS_ACPX_BINARY"
KIMI_BINARY_PATH_ENV = "HERMES_DYNAMIC_WORKFLOWS_KIMI_BINARY"
DEFAULT_KIMI_BINARY = Path.home() / ".kimi-code" / "bin" / "kimi"
# acpx `--model` takes the full `kimi-code/<id>` form (A4 probe: bare aliases
# are rejected by the kimi acp adapter). kimi.conf uses this exact id.
DEFAULT_MODEL = "kimi-code/k3-256k"

# error_class values — subset of the pi lane enum, acpx-specific gaps.
# provider-wall, verification_failed never apply (single managed provider,
# no verify cmd). auth = OAuth expired or config broken.
KNOWN_ERROR_CLASSES = frozenset(
    {"auth", "spawn_failure", "cap_exceeded"}
)

# Lane conf defaults (when no config/lanes/<lane>.conf or absent keys).
LANE_DEFAULTS = {
    "model": DEFAULT_MODEL,
    "fallback": "kimi-code/kimi-for-coding, kimi-code/kimi-for-coding-highspeed",
    "max_turns": 50,
    "timeout": 300,
}

# A3 verdict: kimi acp emits token usage (`usage_update`) but no cost shape.
COST_UNAVAILABLE_DISCLOSURE = "kimi-acp-emits-no-cost"

# ACP `session/update` variants we consume.
_UPDATE_MESSAGE_CHUNK = "agent_message_chunk"
_UPDATE_THOUGHT_CHUNK = "agent_thought_chunk"
_UPDATE_USAGE = "usage_update"


def acpx_binary() -> Path:
    """Return the acpx binary, honoring the env override (mirrors kimi.py:43)."""
    override = os.environ.get(ACPX_BINARY_PATH_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    resolved = shutil.which("acpx")
    if resolved:
        return Path(resolved)
    return Path("/usr/local/bin/acpx")  # fallback; availability gate checks is_file()


def acpx_runner_available() -> bool:
    """True when acpx resolves AND `kimi acp` resolves under the child PATH.

    The built-in ``kimi`` profile spawns bare ``kimi acp``; ~/.kimi-code/bin is
    not on every PATH, so the child env must prepend it. Availability therefore
    requires both the acpx binary AND the kimi binary to exist.
    """
    if not acpx_binary().is_file():
        return False
    kimi_bin = Path(
        os.environ.get(KIMI_BINARY_PATH_ENV, "").strip() or str(DEFAULT_KIMI_BINARY)
    ).expanduser()
    if not kimi_bin.is_file():
        return False
    try:
        result = subprocess.run(
            [str(acpx_binary()), "--version"],
            capture_output=True, text=True, timeout=15,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _find_lane_conf(lane: str) -> Path | None:
    """Locate config/lanes/<lane>.conf under the plugin's config tree."""
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
) -> str:
    """Pick the effective model: per-call override > lane conf > default.

    Returns the full ``kimi-code/<id>`` form that acpx ``--model`` accepts.
    """
    raw: str | None = model or None
    if raw is None or raw.strip().lower() in ("", "inherit"):
        lane_conf = _find_lane_conf(lane or "default")
        if lane_conf:
            try:
                from configparser import ConfigParser
                cp = ConfigParser()
                cp.read_string(f"[lane]\n{lane_conf.read_text()}")
                raw = cp.get("lane", "model", fallback=DEFAULT_MODEL).strip()
            except Exception:
                raw = None
        raw = raw or DEFAULT_MODEL
    raw = raw.strip()
    if not raw.startswith("kimi-code/"):
        raw = f"kimi-code/{raw}"
    return raw


class AcpxChildAgentRunner(ChildAgentRunner):
    """Run a workflow child as an acpx (kimi acp) coding-agent subprocess."""

    def __init__(self, config: PluginConfig):
        self.config = config

    def run(self, request: ChildAgentRequest) -> ChildAgentResult:
        task_id = f"wf-acpx-{uuid.uuid4().hex[:12]}"
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
                f"agent({request.agent_type}): agent type '{request.agent_type}' not found. "
                f"Available agents: {available}"
            )

        if resolved is None:
            request = _apply_acpx_agent_type_defaults(request, agent_type)

        lane = (request.lane or "").strip() or "default"
        effective_model = _resolve_model(request.model, lane)

        lease = create_workspace_lease(
            cwd=base_cwd,
            isolation=request.isolation,
            label=request.label,
            task_id=task_id,
            keep_worktree=self.config.keep_worktrees,
        )
        work_dir = Path(tempfile.mkdtemp(prefix=f"dw-acpx-{task_id}-"))
        try:
            return self._run_lease(
                request, lease, agent_type, lane, effective_model, work_dir, task_id
            )
        finally:
            lease.cleanup()
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
        task_id: str,
    ) -> ChildAgentResult:
        env = self._build_env(lease, lane, work_dir)
        base_prompt = self._build_prompt(request, agent_type, workspace=lease.cwd)

        # Named session keyed (agentCommand, cwd, name). Schema retries reuse it.
        session_name = f"dw-{task_id}"
        ensure = self._sessions_ensure(session_name, lease, env, work_dir)
        if ensure is None:
            raise ChildAgentError(
                f"acpx: could not ensure named session '{session_name}' — "
                "acpx kimi sessions ensure returned no identity record. "
                "Check the acpx binary and kimi login."
            )

        session_id = (
            ensure.get("agentSessionId")
            or ensure.get("acpxRecordId")
            or ensure.get("acpxSessionId")
        )

        state: dict[str, Any] = {"session": session_id, "content": ""}

        def invoke(prompt: str, attempt: int) -> str:
            payload = self._invoke_acpx(
                prompt,
                lane=lane,
                model=effective_model,
                lease=lease,
                env=env,
                work_dir=work_dir,
                session_name=session_name,
                task_id=task_id,
                agent_type=agent_type,
            )
            state["session"] = (
                payload.get("acpx_session_id") or state["session"]
            )
            state["content"] = payload.get("content") or state["content"]
            if payload.get("usage_tokens") is not None:
                state["tokens"] = payload["usage_tokens"]
            state["cost_unavailable"] = payload.get("cost_unavailable")
            if payload.get("is_error"):
                state["error_class"] = payload.get("error_class")
                state["summary"] = payload.get("summary") or ""
                raise ChildAgentError(
                    f"acpx child failed ({payload.get('error_class') or 'unknown'}): "
                    f"{(payload.get('summary') or '')[:300]}"
                )
            return state["content"]

        try:
            schema = (
                request.schema if (request.structured_tool and request.schema) else None
            )
            if schema is None:
                content = invoke(base_prompt, 1)
                return ChildAgentResult(
                    content=content,
                    metadata=self._build_metadata(
                        state, lease, lane, agent_type, effective_model, work_dir,
                        attempts=1, session_name=session_name,
                    ),
                )

            from ..subprocess_schema import run_with_schema
            value, attempts = run_with_schema(base_prompt, schema, invoke)
            metadata = self._build_metadata(
                state, lease, lane, agent_type, effective_model, work_dir,
                attempts=attempts, session_name=session_name,
            )
            metadata["structured_captured"] = True
            metadata["structured_result"] = value
            metadata["structured_attempts"] = attempts
            metadata["structured_mode"] = "prompt"
            return ChildAgentResult(content=state["content"], metadata=metadata)
        finally:
            self._sessions_close(session_name, lease, env, work_dir)

    def _sessions_ensure(
        self, session_name: str, lease: WorkspaceLease, env: dict[str, str],
        work_dir: Path,
    ) -> dict[str, Any] | None:
        """acpx kimi sessions ensure --name dw-<task_id>, return identity fields.

        Returns the ``session_ensured`` record with acpxRecordId /
        acpxSessionId / agentSessionId, or None when no record can be parsed.
        """
        binary = acpx_binary()
        argv = [
            str(binary),
            "--format", "json",
            "--json-strict",
            "kimi",
            "sessions", "ensure",
            "--name", session_name,
        ]
        try:
            result = subprocess.run(
                argv,
                cwd=lease.cwd,
                env=env,
                capture_output=True, text=True, timeout=self.config.child_timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            try:
                record = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                continue
            if isinstance(record, dict) and record.get("action") == "session_ensured":
                return record
        return None

    def _sessions_close(
        self, session_name: str, lease: WorkspaceLease, env: dict[str, str],
        work_dir: Path,
    ) -> None:
        """Soft-close the named session (keeps history). Best-effort."""
        binary = acpx_binary()
        argv = [str(binary), "--format", "json", "--json-strict", "kimi", "sessions", "close", session_name]
        try:
            subprocess.run(
                argv, cwd=lease.cwd, env=env,
                capture_output=True, text=True, timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _invoke_acpx(
        self,
        prompt: str,
        *,
        lane: str,
        model: str,
        lease: WorkspaceLease,
        env: dict[str, str],
        work_dir: Path,
        session_name: str,
        task_id: str,
        agent_type: AgentTypeSpec | None,
    ) -> dict[str, Any]:
        binary = acpx_binary()
        if not binary.is_file():
            raise ChildAgentError(
                f"acpx binary not found at {binary} — install via `npm install -g acpx`"
            )

        # Global flags MUST precede the `kimi` token; --model takes kimi-code/<id>.
        argv = [
            str(binary),
            "--format", "json",
            "--json-strict",
            "--approve-all",
            "--cwd", lease.cwd,
            "--timeout", str(int(self.config.child_timeout_seconds)),
            "--model", model,
            "kimi",
            "-s", session_name,
            "-f", "-",
        ]

        timeout = self.config.child_timeout_seconds
        try:
            process = subprocess.Popen(
                argv,
                cwd=lease.cwd,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            raise ChildAgentError(f"could not spawn acpx: {exc}") from exc

        try:
            stdout, stderr = process.communicate(input=prompt, timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_process_group(process)
            try:
                process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            raise WorkflowTimeout(
                f"acpx child agent timed out after {timeout:.0f}s "
                "(raise dynamic_workflows.child_timeout_seconds for acpx lanes)"
            ) from None

        rc = process.returncode
        stderr_text = stderr.strip() if stderr else ""
        return self._parse_output(
            stdout, stderr_text, rc, lease, session_name, task_id,
        )

    def _parse_output(
        self, stdout: str, stderr_text: str, rc: int,
        lease: WorkspaceLease, session_name: str, task_id: str,
    ) -> dict[str, Any]:
        """Parse the NDJSON ACP stream and map exit codes onto error_class.

        Final assistant text = concatenated ``session/update``
        ``agent_message_chunk`` content. The terminal ``session/prompt`` result
        line carries ``stopReason``. Usage rides ``usage_update`` → ``used``.
        """
        message_chunks: list[str] = []
        final_text = ""
        stop_reason = None
        usage_tokens: int | None = None
        acpx_session_id = None

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
            method = event.get("method")
            if method == "session/update":
                params = event.get("params") or {}
                update = params.get("update") or {}
                if update.get("sessionUpdate") == _UPDATE_MESSAGE_CHUNK:
                    content = (update.get("content") or {})
                    text = content.get("text") or ""
                    if text:
                        message_chunks.append(text)
                        final_text = "".join(message_chunks)
                elif update.get("sessionUpdate") == _UPDATE_USAGE:
                    used = update.get("used")
                    if isinstance(used, (int, float)):
                        usage_tokens = int(used)
            elif method == "session/prompt":
                params = event.get("params") or {}
                sid = params.get("sessionId")
                if sid:
                    acpx_session_id = sid
            # Terminal result line: {"id":2,"result":{"stopReason":...}}
            elif "result" in event and isinstance(event.get("result"), dict):
                if event.get("id") is not None and "stopReason" in event["result"]:
                    stop_reason = event["result"].get("stopReason")

        # Context for the agent's resumed session id (from agent-response lines).
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                continue
            if not isinstance(event, dict):
                continue
            if event.get("id") == 1 and event.get("result"):
                sid = (event["result"] or {}).get("sessionId")
                if sid:
                    acpx_session_id = sid

        error_class, is_error, summary = self._classify(
            rc, stderr_text, stdout, stop_reason, bool(message_chunks),
        )

        return {
            "acpx_session_id": acpx_session_id or session_name,
            "session_name": session_name,
            "usage_tokens": usage_tokens,
            "cost_unavailable": COST_UNAVAILABLE_DISCLOSURE,
            "is_error": is_error,
            "error_class": error_class,
            "changed_files": self._changed_files(lease.cwd),
            "tests_run": _tests_from_stream(stdout),
            "summary": summary[:500],
            "budget_exceeded": False,
            "providers_tried": [],
            "walled_until": None,
            "content": final_text or "",
            "stop_reason": stop_reason,
        }

    @staticmethod
    def _classify(
        rc: int, stderr_text: str, stdout: str, stop_reason: str | None,
        had_message_chunks: bool,
    ) -> tuple[str | None, bool, str]:
        """Map acpx exit codes onto the unified error_class / is_error / summary.

        Stable exit codes (docs/exit-codes.md):
          0 → success
          3 → WorkflowTimeout                 (acpx's own timeout)
          1 → stderr classification (auth/spawn_failure)
          2 → ChildAgentError (our argv bug, never the agent's)
          4 → spawn_failure                   (B2 ensures the session, so 4 = ordering broke)
          5 → spawn_failure with stderr preserved (unreachable under --approve-all)
        """
        if rc == 0:
            if stop_reason == "max_tokens":
                return "cap_exceeded", True, stderr_text or "acpx stopped: max_tokens"
            summary = "acpx run finished"
            if stop_reason:
                summary = f"acpx run finished (stopReason={stop_reason})"
            return None, False, summary

        if rc == 3:
            raise WorkflowTimeout(
                "acpx child agent timed out at the acpx --timeout boundary "
                "(raise dynamic_workflows.child_timeout_seconds)"
            )
        if rc == 2:
            raise ChildAgentError(
                f"acpx argv error (rc=2): {stderr_text or 'malformed argv'} — "
                "this is a runner bug, never the agent's"
            )
        if rc == 4:
            return "spawn_failure", True, (
                stderr_text or f"acpx rc=4: session ordering broke (expected after ensure)"
            )
        if rc == 5:
            return "spawn_failure", True, (
                stderr_text or "acpx rc=5: session closed unexpectedly"
            )
        # rc == 1 or anything else: classify stderr with the auth regex.
        blob = stderr_text + "\n" + stdout
        error_class = _classify_acpx_error(blob)
        final_summary = (stderr_text or f"acpx exited rc={rc}")[:500]
        if not final_summary:
            final_summary = f"acpx run failed rc={rc}"
        return error_class, True, final_summary

    @staticmethod
    def _build_env(
        lease: WorkspaceLease,
        lane: str,
        work_dir: Path,
    ) -> dict[str, str]:
        env = dict(os.environ)
        env["HERMES_KANBAN_WORKSPACE"] = lease.cwd
        env["HERMES_KANBAN_TASK_ID"] = lease.task_id
        # acpx-specific: kimi-code/bin must be on the child PATH for `kimi acp`.
        kimi_dir = str(Path.home() / ".kimi-code" / "bin")
        old_path = env.get("PATH", "")
        env["PATH"] = kimi_dir + (os.pathsep + old_path if old_path else "")
        # Point logs away from the ADW surfaces.
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
        session_name: str = "",
    ) -> dict[str, Any]:
        session_id = state.get("session")
        cost = state.get("total_cost_usd")
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
            "acpx_session_id": session_id,
            "session_id": session_id,          # agentSessionId or acpxRecordId
            "session_name": session_name,
            "total_cost_usd": cost,            # null when unavailable — never 0.0
            "cost_unavailable": state.get("cost_unavailable"),
            "tokens": state.get("tokens"),
            "providers_tried": [],
            "changed_files": [],
            "tests_run": False,
            "budget_exceeded": False,
            "walled_until": None,
            "acpx_log_path": str(work_dir / "ndjson.txt"),
            "wrapper_attempts": attempts,
        }

    @staticmethod
    def _changed_files(workspace: str) -> list[str]:
        """Git porcelain changed files, same as kimi runner's extractor."""
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


def _classify_acpx_error(blob: str) -> str:
    """Map stderr patterns to error_class (reuses kimi.py's auth-regex)."""
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


def _apply_acpx_agent_type_defaults(
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