"""Pi child-agent runner — a subprocess adapter over the pi coding lane wrapper.

This does NOT drive `pi` directly. pi has no turn cap, no budget cap, and under
``--mode json`` it exits 0 even on a failed turn, so a raw adapter would have to
reimplement the turn counter, cost accumulator, ``agent_end``/stopReason check,
provider degradation and process-group kill that
``skills/devops/*-code-lane/scripts/run-pi-task.sh`` (~1100 lines, load-bearing
for the ADW factory) already implements. We invoke that wrapper instead and map
its one-JSON-object contract onto :class:`ChildAgentResult`.

The wrapper contract itself, and every mechanic built on it, lives in
``wrapper.py`` — ``run-claude-task.sh`` speaks the same protocol. What stays
here is only what is pi-shaped: the wrapper filename, the env redirections that
keep a pi child off the board's shared surfaces, how pi's event log names the
final assistant message, and pi's own metadata keys.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from .wrapper import (
    KNOWN_ERROR_CLASSES,
    LaneWrapperRunner,
    apply_agent_type_defaults,
    available_lanes as _available_lanes,
    base_metadata,
    build_lane_prompt,
    dispatch_block,
    find_wrapper as _find_wrapper,
    kill_process_group,
    message_text,
    parse_wrapper_json,
    profiles_root,
    real_changed_files,
)
from ..presets import AgentTypeSpec
from ..worktree import WorkspaceLease

RUNNER_NAME = "pi"
DEFAULT_LANE = "builder"
WRAPPER_NAME = "run-pi-task.sh"


def default_lane() -> str:
    return os.getenv("HERMES_DYNAMIC_WORKFLOWS_PI_LANE", "").strip() or DEFAULT_LANE


def find_wrapper(lane: str) -> Path | None:
    return _find_wrapper(
        lane,
        WRAPPER_NAME,
        script_override=os.getenv("HERMES_DYNAMIC_WORKFLOWS_PI_TASK_SCRIPT", ""),
    )


def available_lanes() -> list[str]:
    return _available_lanes(WRAPPER_NAME)


def pi_runner_available() -> bool:
    """True when a pi binary and at least one lane wrapper exist on this box."""
    if not shutil.which("pi"):
        return False
    if os.getenv("HERMES_DYNAMIC_WORKFLOWS_PI_TASK_SCRIPT", "").strip():
        return find_wrapper(default_lane()) is not None
    return bool(available_lanes())


class PiChildAgentRunner(LaneWrapperRunner):
    """Run a workflow child as a pi coding-lane subprocess."""

    runner_name = RUNNER_NAME
    wrapper_name = WRAPPER_NAME
    task_id_prefix = "wf-pi"
    session_key = "pi_session_id"

    def default_lane(self) -> str:
        return default_lane()

    def find_wrapper(self, lane: str) -> Path | None:
        return find_wrapper(lane)

    def available_lanes(self) -> list[str]:
        return available_lanes()

    def build_env(
        self,
        lease: WorkspaceLease,
        lane: str,
        work_dir: Path,
        log_dir: Path,
    ) -> dict[str, str]:
        return _build_env(lease, lane, work_dir, log_dir)

    def final_text(self, payload: dict[str, Any], log_path: Path) -> str:
        return _final_text(payload, log_path)

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
        return _pi_metadata(
            payload,
            lease=lease,
            lane=lane,
            agent_type=agent_type,
            log_path=log_path,
            attempts=attempts,
        )


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
    metadata = base_metadata(
        payload,
        runner=RUNNER_NAME,
        lease=lease,
        lane=lane,
        agent_type=agent_type,
        attempts=attempts,
    )
    metadata.update(
        {
            "pi_session_id": payload.get("pi_session_id"),
            # Also under the generic key: _apply_child_metadata reads
            # `session_id` to populate the agent record, so without this the
            # resumable session never reaches the run record for a lane child.
            "session_id": payload.get("pi_session_id"),
            "providers_tried": payload.get("providers_tried") or [],
            "budget_exceeded": bool(payload.get("budget_exceeded")),
            "walled_until": payload.get("walled_until"),
            "pi_log_path": str(log_path),
        }
    )
    return metadata


# Names kept at module scope because they are this module's tested surface and
# because the generic implementations read better under their pi-flavoured
# aliases at the call sites above.
_apply_pi_agent_type_defaults = apply_agent_type_defaults
_build_pi_prompt = build_lane_prompt
_dispatch_block = dispatch_block
_kill_process_group = kill_process_group
_message_text = message_text
_parse_wrapper_json = parse_wrapper_json
_real_changed_files = real_changed_files

__all__ = [
    "DEFAULT_LANE",
    "KNOWN_ERROR_CLASSES",
    "RUNNER_NAME",
    "WRAPPER_NAME",
    "PiChildAgentRunner",
    "available_lanes",
    "default_lane",
    "find_wrapper",
    "pi_runner_available",
    "profiles_root",
]
