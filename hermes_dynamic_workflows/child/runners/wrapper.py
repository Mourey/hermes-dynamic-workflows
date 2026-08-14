"""Shared adapter for the coding-lane wrappers.

`run-pi-task.sh` and `run-claude-task.sh` expose the SAME process contract —

    argv:   <wrapper> --prompt-file <tmp> --task-id <synthesized> [--resume <id>]
    env:    HERMES_KANBAN_WORKSPACE=<lease.cwd>
            HERMES_PROFILE=<lane>          -> config/lanes/<lane>.conf
            HERMES_KANBAN_TASK_ID=<id>
    stdout: exactly one JSON object
    exit:   0 clean, nonzero on failure (is_error + error_class set)

— so the subprocess mechanics, the workspace lease, the schema-retry bracket
and the dispatch fence belong here once rather than in each lane's module. A
lane subclass supplies only what genuinely differs between the two: which
wrapper file and binary to look for, which env redirections keep a workflow
child off the board's shared surfaces, how that lane's event log names the
final assistant message, and what its metadata reports.

Neither lane is driven directly. Both wrappers carry the turn cap, budget
accounting, provider degradation and process-group kill that a raw adapter
would otherwise have to reimplement; see the note at the top of ``pi.py``.
"""

from __future__ import annotations

import json
import os
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
from ..subprocess_schema import run_with_schema
from ..worktree import WorkspaceLease, create_workspace_lease
from ...core.config import PluginConfig
from ...core.errors import ChildAgentError, WorkflowTimeout
from ...core.types import ChildAgentRequest, ChildAgentResult, ChildAgentRunner

# error_class values the wrappers emit (unified enum, shared by both lanes).
# Surfaced verbatim in the ChildAgentError message so a workflow author can tell
# a provider wall from a bad model slug without reading logs.
KNOWN_ERROR_CLASSES = frozenset(
    {"provider-wall", "auth", "verification_failed", "spawn_failure", "cap_exceeded"}
)


# ── lane discovery ───────────────────────────────────────────────────────────


def profiles_root() -> Path:
    override = os.getenv("HERMES_DYNAMIC_WORKFLOWS_PI_PROFILES_ROOT", "").strip()
    if override:
        return Path(override).expanduser()
    return Path(os.getenv("HERMES_ROOT", "").strip() or Path.home() / ".hermes") / "profiles"


def find_wrapper(lane: str, wrapper_name: str, *, script_override: str = "") -> Path | None:
    """Locate the wrapper whose config dir carries ``lanes/<lane>.conf``.

    A wrapper is installed per lane profile
    (``~/.hermes/profiles/<lane>/skills/devops/<x>-code-lane/scripts/``) and
    resolves its lane conf relative to its own location, so the copy we pick
    must be one that actually ships the requested lane.
    """
    override = (script_override or "").strip()
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_file() else None

    root = profiles_root()
    ordered: list[Path] = []
    ordered.extend(sorted((root / lane).glob(f"skills/*/*/scripts/{wrapper_name}")))
    ordered.extend(sorted(root.glob(f"*/skills/*/*/scripts/{wrapper_name}")))
    seen: set[Path] = set()
    for candidate in ordered:
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        if (candidate.parent.parent / "config" / "lanes" / f"{lane}.conf").is_file():
            return candidate
    return None


def available_lanes(wrapper_name: str) -> list[str]:
    root = profiles_root()
    lanes: set[str] = set()
    for wrapper in root.glob(f"*/skills/*/*/scripts/{wrapper_name}"):
        conf_dir = wrapper.parent.parent / "config" / "lanes"
        for conf in conf_dir.glob("*.conf"):
            lanes.add(conf.stem)
    return sorted(lanes)


# ── payload helpers ──────────────────────────────────────────────────────────


def parse_wrapper_json(stdout: str) -> dict[str, Any] | None:
    """Return the wrapper's one JSON object, scanning from the last line back."""
    lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def message_text(message: dict[str, Any]) -> str:
    blocks = message.get("content") or []
    if not isinstance(blocks, list):
        return ""
    texts = [
        str(block.get("text") or "")
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(text for text in texts if text)


def real_changed_files(values: Any) -> list[str]:
    """Drop the wrapper's own `_project/` scaffold from the changed-file list."""
    if not isinstance(values, list):
        return []
    return [
        str(path)
        for path in values
        if str(path).strip() and not str(path).lstrip("./").startswith("_project")
    ]


def dispatch_block(request: ChildAgentRequest) -> str:
    """Per-call overrides expressed as a dispatch-spec v2 block.

    Only `model` today. Both lanes' resolvers catalog-verify it and fail fast
    (exit 65 -> error_class spawn_failure) before any API dollars are spent, so
    a bad slug surfaces as a clean error rather than a mystery run.
    """
    model = (request.model or "").strip()
    if not model or model.lower() == "inherit":
        return ""
    return f"```dispatch\nmodel: {model}\n```"


def build_lane_prompt(
    request: ChildAgentRequest,
    agent_type: AgentTypeSpec | None,
    *,
    workspace: str,
) -> str:
    """Fold the child system prompt into the task text.

    Neither wrapper exposes a system-prompt flag for the CHILD's agent type, so
    the instructions that the Hermes runner passes as ``ephemeral_system_prompt``
    have to ride in the prompt body instead.
    """
    sections: list[str] = []
    dispatch = dispatch_block(request)
    if dispatch:
        sections.append(dispatch)
    sections.append(build_child_system_prompt(agent_type, structured_output=False))
    context = [f"- Workspace: {workspace}"]
    if request.isolation == "worktree":
        context.append(
            "- You are running in an isolated git worktree; keep all file "
            "operations inside the workspace above."
        )
    sections.append("Task context:\n" + "\n".join(context))
    sections.append(request.prompt)
    return "\n\n".join(section for section in sections if section)


def apply_agent_type_defaults(
    request: ChildAgentRequest,
    agent_type: AgentTypeSpec | None,
) -> ChildAgentRequest:
    if agent_type is None:
        return request
    lane = request.lane or getattr(agent_type, "lane", None)
    if lane and lane.strip().lower() == "inherit":
        lane = None
    return replace(request, lane=lane, isolation=request.isolation or agent_type.isolation)


def kill_process_group(process: subprocess.Popen) -> None:
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


# ── the base runner ──────────────────────────────────────────────────────────


class LaneWrapperRunner(ChildAgentRunner):
    """Run a workflow child as a coding-lane subprocess.

    Subclasses set the four class attributes and override the lane hooks at the
    bottom. Everything above them is contract-shared and must stay that way —
    the two lanes diverging here is exactly the drift this base exists to stop.
    """

    runner_name: str = ""
    wrapper_name: str = ""
    task_id_prefix: str = "wf"
    #: JSON key the wrapper uses for its resumable session id.
    session_key: str = ""

    def __init__(self, config: PluginConfig):
        self.config = config

    # ── one child, one lease ─────────────────────────────────────────────

    def run(self, request: ChildAgentRequest) -> ChildAgentResult:
        task_id = f"{self.task_id_prefix}-{uuid.uuid4().hex[:12]}"
        base_cwd = request.cwd or os.environ.get("TERMINAL_CWD") or os.getcwd()
        resolved = request.resolved
        agent_type = (
            resolved.agent_type_spec
            if resolved is not None
            else resolve_agent_type(request.agent_type, cwd=base_cwd)
        )
        if request.agent_type and agent_type is None:
            available = ", ".join(spec.name for spec in list_agent_types(cwd=base_cwd)) or "none"
            raise ChildAgentError(
                f"agent({{agentType}}): agent type '{request.agent_type}' not found. "
                f"Available agents: {available}"
            )
        if resolved is None:
            request = apply_agent_type_defaults(request, agent_type)

        lane = (request.lane or "").strip() or self.default_lane()
        wrapper = self.find_wrapper(lane)
        if wrapper is None:
            known = ", ".join(self.available_lanes()) or "none"
            raise ChildAgentError(
                f"runner {self.runner_name!r}: no {self.wrapper_name} found for lane "
                f"{lane!r} under {profiles_root()}. Lanes available here: {known}"
            )

        lease = create_workspace_lease(
            cwd=base_cwd,
            isolation=request.isolation,
            label=request.label,
            task_id=task_id,
            keep_worktree=self.config.keep_worktrees,
        )
        work_dir = Path(tempfile.mkdtemp(prefix=f"dw-{self.runner_name}-{task_id}-"))
        # A wrapper may write a `_project/README.md` navigation aid into the
        # workspace. Board cards run in throwaway worktrees so it never matters
        # there; a workflow child with isolation="shared" runs in the user's own
        # checkout, so remove the scaffold again — but only if we created it.
        project_scaffold = Path(lease.cwd) / "_project"
        scaffold_pre_existed = project_scaffold.exists()
        try:
            return self._run_lease(request, lease, agent_type, lane, wrapper, work_dir)
        finally:
            if not scaffold_pre_existed:
                shutil.rmtree(project_scaffold, ignore_errors=True)
            lease.cleanup()
            shutil.rmtree(work_dir, ignore_errors=True)

    # ── one child, possibly several wrapper invocations (schema retries) ──

    def _run_lease(
        self,
        request: ChildAgentRequest,
        lease: WorkspaceLease,
        agent_type: AgentTypeSpec | None,
        lane: str,
        wrapper: Path,
        work_dir: Path,
    ) -> ChildAgentResult:
        log_dir = work_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{lease.task_id}.jsonl"
        env = self.build_env(lease, lane, work_dir, log_dir)
        base_prompt = build_lane_prompt(request, agent_type, workspace=lease.cwd)

        state: dict[str, Any] = {"payload": {}, "session": None, "content": ""}

        if request.on_start is not None:
            try:
                request.on_start(
                    self.metadata(
                        {},
                        lease=lease,
                        lane=lane,
                        agent_type=agent_type,
                        log_path=log_path,
                    )
                )
            except Exception:
                pass

        def invoke(prompt: str, attempt: int) -> str:
            payload = self._invoke_wrapper(
                prompt,
                wrapper=wrapper,
                lease=lease,
                env=env,
                work_dir=work_dir,
                resume=state["session"] if attempt > 1 else None,
            )
            state["payload"] = payload
            state["session"] = payload.get(self.session_key) or state["session"]
            content = self.final_text(payload, log_path)
            state["content"] = content
            metadata = self.metadata(
                payload,
                lease=lease,
                lane=lane,
                agent_type=agent_type,
                log_path=log_path,
                attempts=attempt,
            )
            if request.on_update is not None:
                try:
                    request.on_update(metadata)
                except Exception:
                    pass
            return content

        schema = request.schema if (request.structured_tool and request.schema) else None
        if schema is None:
            content = invoke(base_prompt, 1)
            metadata = self.metadata(
                state["payload"],
                lease=lease,
                lane=lane,
                agent_type=agent_type,
                log_path=log_path,
                attempts=1,
            )
            return ChildAgentResult(content=content, metadata=metadata)

        # Neither lane's wrapper exposes a schema-constrained output mode to the
        # workflow child, so the contract is enforced by prompt + parse +
        # validate + re-prompt, on the same retry budget as the Hermes tool-call
        # path.
        value, attempts = run_with_schema(base_prompt, schema, invoke)
        metadata = self.metadata(
            state["payload"],
            lease=lease,
            lane=lane,
            agent_type=agent_type,
            log_path=log_path,
            attempts=attempts,
        )
        metadata["structured_captured"] = True
        metadata["structured_result"] = value
        metadata["structured_attempts"] = attempts
        metadata["structured_mode"] = "prompt"
        return ChildAgentResult(content=state["content"], metadata=metadata)

    def _invoke_wrapper(
        self,
        prompt: str,
        *,
        wrapper: Path,
        lease: WorkspaceLease,
        env: dict[str, str],
        work_dir: Path,
        resume: str | None,
    ) -> dict[str, Any]:
        prompt_file = work_dir / "prompt.md"
        prompt_file.write_text(prompt, encoding="utf-8")
        argv = [
            "bash",
            str(wrapper),
            "--prompt-file",
            str(prompt_file),
            "--task-id",
            lease.task_id,
        ]
        if resume:
            argv.extend(["--resume", str(resume)])

        timeout = self.config.child_timeout_seconds
        try:
            # Popen rather than subprocess.run: on a timeout we need the pid to
            # kill the whole process group. run() only kills the wrapper shell,
            # leaving the agent tree it spawned alive.
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
            raise ChildAgentError(f"could not spawn {self.wrapper_name}: {exc}") from exc

        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_process_group(process)
            try:
                process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            raise WorkflowTimeout(
                f"{self.runner_name} child agent timed out after {timeout:.0f}s "
                f"(raise dynamic_workflows.child_timeout_seconds for {self.runner_name} lanes)"
            ) from None

        payload = parse_wrapper_json(stdout)
        if payload is None:
            detail = (stderr or stdout or "").strip()[-500:]
            raise ChildAgentError(
                f"{self.wrapper_name} produced no JSON result (exit {process.returncode})"
                + (f": {detail}" if detail else "")
            )
        if payload.get("is_error") or process.returncode != 0:
            error_class = str(payload.get("error_class") or "spawn_failure")
            if error_class not in KNOWN_ERROR_CLASSES:
                error_class = f"{error_class} (unrecognized)"
            summary = str(payload.get("summary") or f"{self.runner_name} run failed")
            raise ChildAgentError(f"{self.runner_name} lane failed [{error_class}]: {summary}")
        return payload

    # ── lane hooks ───────────────────────────────────────────────────────

    def default_lane(self) -> str:
        raise NotImplementedError

    def find_wrapper(self, lane: str) -> Path | None:
        raise NotImplementedError

    def available_lanes(self) -> list[str]:
        raise NotImplementedError

    def build_env(
        self,
        lease: WorkspaceLease,
        lane: str,
        work_dir: Path,
        log_dir: Path,
    ) -> dict[str, str]:
        raise NotImplementedError

    def final_text(self, payload: dict[str, Any], log_path: Path) -> str:
        raise NotImplementedError

    def metadata(
        self,
        payload: dict[str, Any],
        *,
        lease: WorkspaceLease,
        lane: str,
        agent_type: AgentTypeSpec | None,
        log_path: Path,
        attempts: int = 0,
    ) -> dict[str, Any]:
        raise NotImplementedError


def base_metadata(
    payload: dict[str, Any],
    *,
    runner: str,
    lease: WorkspaceLease,
    lane: str,
    agent_type: AgentTypeSpec | None,
    attempts: int,
) -> dict[str, Any]:
    """The metadata keys both lanes report identically."""
    dispatch = payload.get("dispatch") if isinstance(payload.get("dispatch"), dict) else {}
    return {
        "runner": runner,
        "lane": lane,
        "task_id": lease.task_id,
        "workspace": lease.cwd,
        "isolation": lease.isolation or "shared",
        "worktree_path": lease.path,
        "worktree_branch": lease.branch,
        "agent_type": agent_type.name if agent_type else "general-purpose",
        "agent_type_source": agent_type.source if agent_type else None,
        "model": dispatch.get("model"),
        "provider": dispatch.get("provider"),
        "total_cost_usd": payload.get("total_cost_usd"),
        "changed_files": real_changed_files(payload.get("changed_files")),
        "tests_run": payload.get("tests_run"),
        "wrapper_attempts": attempts,
        # Both lanes report spend, not token counts; the engine's token budget is
        # therefore not charged for their children — cost lands in metadata.
        "tokens": 0,
    }


__all__ = [
    "KNOWN_ERROR_CLASSES",
    "LaneWrapperRunner",
    "apply_agent_type_defaults",
    "available_lanes",
    "base_metadata",
    "build_lane_prompt",
    "dispatch_block",
    "find_wrapper",
    "kill_process_group",
    "message_text",
    "parse_wrapper_json",
    "profiles_root",
    "real_changed_files",
]
