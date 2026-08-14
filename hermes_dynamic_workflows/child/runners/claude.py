"""Claude child-agent runner — a subprocess adapter over the claude coding lane.

The sibling of ``pi.py``, and for the same reason: ``claude -p`` is not driven
directly. ``skills/devops/claude-code-lane/scripts/run-claude-task.sh`` already
carries the dispatch-spec validation that fails before a bad card burns
subscription messages, the workspace jail, the goal Stop hook, the post-run
verify gate, and the deterministic result extraction. We invoke that wrapper and
map its one-JSON-object contract — the SAME contract ``run-pi-task.sh`` speaks —
onto :class:`ChildAgentResult`.

Registering this is what makes "Claude writes, pi runs" expressible inside a
single workflow (spec 019 §12.1 PHASE A step 2): per-node runner selection
already existed, and the only barrier was that the registry knew one wrapper
filename.

Two honest differences from the pi lane, both in the wrapper's favour:

* ``summary`` is the run's COMPLETE final message here (the stream's ``result``
  event), not a 500-char board line, so the child's return value does not have
  to be recovered from the event log.
* the wrapper can emit ``structured_output`` natively (``--json-schema``). This
  adapter does NOT use it yet — it takes the same prompt-and-validate retry path
  as pi so both runners behave identically under ``agent(schema=...)``. Wiring
  the native path is a follow-up, not a silent divergence.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from .wrapper import (
    LaneWrapperRunner,
    available_lanes as _available_lanes,
    base_metadata,
    find_wrapper as _find_wrapper,
    message_text,
)
from ..presets import AgentTypeSpec
from ..worktree import WorkspaceLease

RUNNER_NAME = "claude"
DEFAULT_LANE = "claude-worker"
WRAPPER_NAME = "run-claude-task.sh"


def default_lane() -> str:
    return os.getenv("HERMES_DYNAMIC_WORKFLOWS_CLAUDE_LANE", "").strip() or DEFAULT_LANE


def find_wrapper(lane: str) -> Path | None:
    return _find_wrapper(
        lane,
        WRAPPER_NAME,
        script_override=os.getenv("HERMES_DYNAMIC_WORKFLOWS_CLAUDE_TASK_SCRIPT", ""),
    )


def available_lanes() -> list[str]:
    return _available_lanes(WRAPPER_NAME)


def claude_runner_available() -> bool:
    """True when a claude binary and at least one lane wrapper exist on this box."""
    if not shutil.which("claude"):
        return False
    if os.getenv("HERMES_DYNAMIC_WORKFLOWS_CLAUDE_TASK_SCRIPT", "").strip():
        return find_wrapper(default_lane()) is not None
    return bool(available_lanes())


class ClaudeChildAgentRunner(LaneWrapperRunner):
    """Run a workflow child as a claude coding-lane subprocess."""

    runner_name = RUNNER_NAME
    wrapper_name = WRAPPER_NAME
    task_id_prefix = "wf-claude"
    session_key = "claude_session_id"

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
        return _claude_metadata(
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
    # board-ticker's stuck-reclaim heartbeats, or have its dispatch block
    # re-sourced from a card id that does not exist. (board_dispatch_block in
    # dispatch-resolve.py treats an unreadable board as "no block", so pointing
    # at a file that will never exist is the supported way to say "no card".)
    env["CLAUDE_LANE_LOG_DIR"] = str(log_dir)
    env["CLAUDE_LANE_KANBAN_DB"] = str(work_dir / "no-board.db")
    env["HERMES_HEARTBEAT_DIR"] = str(work_dir / "heartbeats")
    env["CODE_NODES"] = str(work_dir / "no-telemetry")
    # budget-breaker.json (HERMES_BUDGET_BREAKER_FILE) is deliberately NOT
    # redirected: a tripped breaker is a machine-wide fact and workflow children
    # should honour it, exactly as the pi lane honours a provider wall.
    for stale in ("HERMES_KANBAN_PIPELINE_NODE", "HERMES_KANBAN_TASK", "HERMES_HOME"):
        env.pop(stale, None)
    return env


def _final_text(payload: dict[str, Any], log_path: Path) -> str:
    """The child's full final message.

    Unlike the pi lane, ``summary`` here is not truncated — the wrapper lifts it
    straight off the stream's ``result`` event, which IS the final assistant
    message. So it is the primary source, and the event-log scan below is only
    the fallback for a stream that ended without one.
    """
    summary = str(payload.get("summary") or "")
    if summary:
        return summary
    return _final_assistant_text(log_path)


def _final_assistant_text(log_path: Path) -> str:
    """Last assistant text in a Claude Code stream-json log.

    Event shape: ``{"type": "assistant", "message": {"role": ..., "content": [
    {"type": "text", "text": ...}]}}`` — the same content-block layout the pi
    lane uses, under a different envelope.
    """
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
                if event.get("type") == "result":
                    text = str(event.get("result") or "")
                    if text:
                        last_streamed = text
                    continue
                if event.get("type") != "assistant":
                    continue
                message = event.get("message")
                if isinstance(message, dict) and message.get("role") in (None, "assistant"):
                    text = message_text(message)
                    if text:
                        last_streamed = text
    except OSError:
        return ""
    return last_streamed


def _claude_metadata(
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
            "claude_session_id": payload.get("claude_session_id"),
            # Also under the generic key: _apply_child_metadata reads
            # `session_id` to populate the agent record, so without this the
            # resumable session never reaches the run record for a lane child.
            "session_id": payload.get("claude_session_id"),
            "claude_log_path": str(log_path),
        }
    )
    # The wrapper copies the stream's structured_output through when the card
    # carried a result_schema. Nothing in this adapter asks for one yet, so its
    # presence means an agent type or lane conf did — surface it rather than
    # dropping it on the floor.
    if "structured_output" in payload:
        metadata["structured_output"] = payload["structured_output"]
    return metadata


__all__ = [
    "DEFAULT_LANE",
    "RUNNER_NAME",
    "WRAPPER_NAME",
    "ClaudeChildAgentRunner",
    "available_lanes",
    "claude_runner_available",
    "default_lane",
    "find_wrapper",
]
