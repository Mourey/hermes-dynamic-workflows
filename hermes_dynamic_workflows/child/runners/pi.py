"""Pi child-agent runner — a subprocess adapter over the pi coding lane wrapper.

This does NOT drive `pi` directly. pi has no turn cap, no budget cap, and under
``--mode json`` it exits 0 even on a failed turn, so a raw adapter would have to
reimplement the turn counter, cost accumulator, ``agent_end``/stopReason check,
provider degradation and process-group kill that
``skills/devops/*-code-lane/scripts/run-pi-task.sh`` (~1100 lines, load-bearing
for the ADW factory) already implements. We invoke that wrapper instead and map
its one-JSON-object contract onto :class:`ChildAgentResult`.

Wrapper contract we depend on:

    argv:   run-pi-task.sh --prompt-file <tmp> --task-id <synthesized>
    env:    HERMES_KANBAN_WORKSPACE=<lease.cwd>
            HERMES_PROFILE=<lane>          -> config/lanes/<lane>.conf
            HERMES_KANBAN_TASK_ID=<id>
    stdout: exactly one JSON object
    exit:   0 clean, nonzero on failure (is_error + error_class set)
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

RUNNER_NAME = "pi"
DEFAULT_LANE = "builder"
WRAPPER_NAME = "run-pi-task.sh"

# error_class values the wrapper emits (unified enum, shared with the claude
# lane). Surfaced verbatim in the ChildAgentError message so a workflow author
# can tell a provider wall from a bad model slug without reading logs.
KNOWN_ERROR_CLASSES = frozenset(
    {"provider-wall", "auth", "verification_failed", "spawn_failure", "cap_exceeded"}
)


def default_lane() -> str:
    return os.getenv("HERMES_DYNAMIC_WORKFLOWS_PI_LANE", "").strip() or DEFAULT_LANE


def profiles_root() -> Path:
    override = os.getenv("HERMES_DYNAMIC_WORKFLOWS_PI_PROFILES_ROOT", "").strip()
    if override:
        return Path(override).expanduser()
    return Path(os.getenv("HERMES_ROOT", "").strip() or Path.home() / ".hermes") / "profiles"


def find_wrapper(lane: str) -> Path | None:
    """Locate the run-pi-task.sh whose config dir carries ``lanes/<lane>.conf``.

    The wrapper is installed per lane profile
    (``~/.hermes/profiles/<lane>/skills/devops/<x>-code-lane/scripts/``) and
    resolves its lane conf relative to its own location, so the copy we pick
    must be one that actually ships the requested lane.
    """
    override = os.getenv("HERMES_DYNAMIC_WORKFLOWS_PI_TASK_SCRIPT", "").strip()
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_file() else None

    root = profiles_root()
    ordered: list[Path] = []
    ordered.extend(sorted((root / lane).glob(f"skills/*/*/scripts/{WRAPPER_NAME}")))
    ordered.extend(sorted(root.glob(f"*/skills/*/*/scripts/{WRAPPER_NAME}")))
    seen: set[Path] = set()
    for candidate in ordered:
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        if (candidate.parent.parent / "config" / "lanes" / f"{lane}.conf").is_file():
            return candidate
    return None


def available_lanes() -> list[str]:
    root = profiles_root()
    lanes: set[str] = set()
    for wrapper in root.glob(f"*/skills/*/*/scripts/{WRAPPER_NAME}"):
        conf_dir = wrapper.parent.parent / "config" / "lanes"
        for conf in conf_dir.glob("*.conf"):
            lanes.add(conf.stem)
    return sorted(lanes)


def pi_runner_available() -> bool:
    """True when a pi binary and at least one lane wrapper exist on this box."""
    if not shutil.which("pi"):
        return False
    if os.getenv("HERMES_DYNAMIC_WORKFLOWS_PI_TASK_SCRIPT", "").strip():
        return find_wrapper(default_lane()) is not None
    return bool(available_lanes())


class PiChildAgentRunner(ChildAgentRunner):
    """Run a workflow child as a pi coding-lane subprocess."""

    def __init__(self, config: PluginConfig):
        self.config = config

    def run(self, request: ChildAgentRequest) -> ChildAgentResult:
        task_id = f"wf-pi-{uuid.uuid4().hex[:12]}"
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
            request = _apply_pi_agent_type_defaults(request, agent_type)

        lane = (request.lane or "").strip() or default_lane()
        wrapper = find_wrapper(lane)
        if wrapper is None:
            known = ", ".join(available_lanes()) or "none"
            raise ChildAgentError(
                f"runner 'pi': no {WRAPPER_NAME} found for lane {lane!r} under "
                f"{profiles_root()}. Lanes available here: {known}"
            )

        lease = create_workspace_lease(
            cwd=base_cwd,
            isolation=request.isolation,
            label=request.label,
            task_id=task_id,
            keep_worktree=self.config.keep_worktrees,
        )
        work_dir = Path(tempfile.mkdtemp(prefix=f"dw-pi-{task_id}-"))
        # The wrapper writes a `_project/README.md` navigation aid into the
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
        env = _build_env(lease, lane, work_dir, log_dir)
        base_prompt = _build_pi_prompt(request, agent_type, workspace=lease.cwd)

        state: dict[str, Any] = {"payload": {}, "session": None, "content": ""}

        if request.on_start is not None:
            try:
                request.on_start(
                    _pi_metadata(
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
            state["session"] = payload.get("pi_session_id") or state["session"]
            content = _final_text(payload, log_path)
            state["content"] = content
            metadata = _pi_metadata(
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
            metadata = _pi_metadata(
                state["payload"],
                lease=lease,
                lane=lane,
                agent_type=agent_type,
                log_path=log_path,
                attempts=1,
            )
            return ChildAgentResult(content=content, metadata=metadata)

        # pi has no --json-schema and no schema-constrained output mode, so the
        # contract is enforced by prompt + parse + validate + re-prompt, on the
        # same retry budget as the Hermes tool-call path.
        value, attempts = run_with_schema(base_prompt, schema, invoke)
        metadata = _pi_metadata(
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
            # leaving the pi tree it spawned alive.
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
            raise ChildAgentError(f"could not spawn {WRAPPER_NAME}: {exc}") from exc

        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_process_group(process)
            try:
                process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            raise WorkflowTimeout(
                f"pi child agent timed out after {timeout:.0f}s "
                "(raise dynamic_workflows.child_timeout_seconds for pi lanes)"
            ) from None

        payload = _parse_wrapper_json(stdout)
        if payload is None:
            detail = (stderr or stdout or "").strip()[-500:]
            raise ChildAgentError(
                f"{WRAPPER_NAME} produced no JSON result (exit {process.returncode})"
                + (f": {detail}" if detail else "")
            )
        if payload.get("is_error") or process.returncode != 0:
            error_class = str(payload.get("error_class") or "spawn_failure")
            if error_class not in KNOWN_ERROR_CLASSES:
                error_class = f"{error_class} (unrecognized)"
            summary = str(payload.get("summary") or "pi run failed")
            raise ChildAgentError(f"pi lane failed [{error_class}]: {summary}")
        return payload


def _apply_pi_agent_type_defaults(
    request: ChildAgentRequest,
    agent_type: AgentTypeSpec | None,
) -> ChildAgentRequest:
    if agent_type is None:
        return request
    lane = request.lane or getattr(agent_type, "lane", None)
    if lane and lane.strip().lower() == "inherit":
        lane = None
    return replace(request, lane=lane, isolation=request.isolation or agent_type.isolation)


def _build_env(
    lease: WorkspaceLease,
    lane: str,
    work_dir: Path,
    log_dir: Path,
) -> dict[str, str]:
    env = dict(os.environ)
    env["HERMES_KANBAN_WORKSPACE"] = lease.cwd
    env["HERMES_PROFILE"] = lane
    env["HERMES_KANBAN_TASK_ID"] = lease.task_id
    # Keep the run out of the ADW factory's shared surfaces: a workflow child is
    # not a board card, so it must not write kanban telemetry, refresh the
    # board-ticker's stuck-reclaim heartbeats, or have its dispatch keys
    # re-sourced from a card id that does not exist.
    env["PI_LANE_LOG_DIR"] = str(log_dir)
    env["PI_LANE_KANBAN_DB"] = str(work_dir / "no-board.db")
    env["HERMES_HEARTBEAT_DIR"] = str(work_dir / "heartbeats")
    env["CODE_NODES"] = str(work_dir / "no-telemetry")
    # provider-status.json (PI_LANE_STATE_DIR) is deliberately NOT redirected:
    # provider walls are a machine-wide fact and workflow children should both
    # honour and stamp them.
    for stale in ("HERMES_KANBAN_PIPELINE_NODE", "HERMES_KANBAN_TASK", "HERMES_HOME"):
        env.pop(stale, None)
    return env


def _build_pi_prompt(
    request: ChildAgentRequest,
    agent_type: AgentTypeSpec | None,
    *,
    workspace: str,
) -> str:
    """Fold the child system prompt into the task text.

    The wrapper exposes no system-prompt flag, so the agent-type instructions
    that the Hermes runner passes as ``ephemeral_system_prompt`` have to ride in
    the prompt body instead.
    """
    sections: list[str] = []
    dispatch = _dispatch_block(request)
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


def _dispatch_block(request: ChildAgentRequest) -> str:
    """Per-call overrides expressed as a dispatch-spec v2 block.

    Only `model` today. The resolver catalog-verifies it and fails fast (exit
    65 -> error_class spawn_failure) before any API dollars are spent, so a bad
    slug surfaces as a clean error rather than a mystery run.
    """
    model = (request.model or "").strip()
    if not model or model.lower() == "inherit":
        return ""
    return f"```dispatch\nmodel: {model}\n```"


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


def _parse_wrapper_json(stdout: str) -> dict[str, Any] | None:
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


def _final_text(payload: dict[str, Any], log_path: Path) -> str:
    """The child's full final message.

    The wrapper's ``summary`` is capped at 500 chars (an operator-facing board
    line), so it is not usable as an agent return value. The complete final
    assistant message is recovered from the tee'd event stream; ``summary`` is
    only the fallback when the log is unreadable.
    """
    text = _final_assistant_text(log_path)
    if text:
        return text
    return str(payload.get("summary") or "")


def _final_assistant_text(log_path: Path) -> str:
    agent_end_messages: Any = None
    last_streamed = ""
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(event, dict):
                    continue
                kind = event.get("type")
                if kind == "agent_end":
                    agent_end_messages = event.get("messages")
                elif kind == "message_end":
                    message = event.get("message")
                    if isinstance(message, dict) and message.get("role") == "assistant":
                        text = _message_text(message)
                        if text:
                            last_streamed = text
    except OSError:
        return ""

    if isinstance(agent_end_messages, list):
        for message in reversed(agent_end_messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                text = _message_text(message)
                if text:
                    return text
                break
    return last_streamed


def _message_text(message: dict[str, Any]) -> str:
    blocks = message.get("content") or []
    if not isinstance(blocks, list):
        return ""
    texts = [
        str(block.get("text") or "")
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(text for text in texts if text)


def _real_changed_files(values: Any) -> list[str]:
    """Drop the wrapper's own `_project/` scaffold from the changed-file list."""
    if not isinstance(values, list):
        return []
    return [
        str(path)
        for path in values
        if str(path).strip() and not str(path).lstrip("./").startswith("_project")
    ]


def _pi_metadata(
    payload: dict[str, Any],
    *,
    lease: WorkspaceLease,
    lane: str,
    agent_type: AgentTypeSpec | None,
    log_path: Path,
    attempts: int = 0,
) -> dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}
    dispatch = payload.get("dispatch") if isinstance(payload.get("dispatch"), dict) else {}
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
        "model": dispatch.get("model"),
        "provider": dispatch.get("provider"),
        "pi_session_id": payload.get("pi_session_id"),
        "total_cost_usd": payload.get("total_cost_usd"),
        "providers_tried": payload.get("providers_tried") or [],
        "changed_files": _real_changed_files(payload.get("changed_files")),
        "tests_run": payload.get("tests_run"),
        "budget_exceeded": bool(payload.get("budget_exceeded")),
        "walled_until": payload.get("walled_until"),
        "pi_log_path": str(log_path),
        "wrapper_attempts": attempts,
        # pi reports spend, not token counts; the engine's token budget is
        # therefore not charged for pi children — cost lands in metadata instead.
        "tokens": 0,
    }
