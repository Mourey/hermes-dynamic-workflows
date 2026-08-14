"""Child-agent runner registry.

`agent()` picks its executor by name: "hermes" runs an in-process Hermes AIAgent
(the default, unchanged); "pi" and "claude" shell out to their coding lane's
hardened wrapper. A runner is only registered when it is actually usable on this
machine — an unavailable one is left out so the error names what *is* available
instead of failing deep inside a subprocess.

Both subprocess runners are thin lane hooks over `wrapper.LaneWrapperRunner`,
because the two wrappers speak one process contract. With "claude" registered,
a single workflow can hand one node to Claude and the next to pi.

`HermesChildAgentRunner` deliberately stays in `child/runner.py`: it predates
this package and moving it would churn every import site for no gain.
"""

from __future__ import annotations

from typing import Any

from ...core.config import PluginConfig
from ...core.types import ChildAgentRunner

HERMES_RUNNER = "hermes"
PI_RUNNER = "pi"
CLAUDE_RUNNER = "claude"


def build_runner_registry(
    config: PluginConfig,
    hermes_runner: ChildAgentRunner,
    **_: Any,
) -> dict[str, ChildAgentRunner]:
    """Return {name: runner} for every runner usable in this process."""
    registry: dict[str, ChildAgentRunner] = {HERMES_RUNNER: hermes_runner}
    pi_runner = _build_pi_runner(config)
    if pi_runner is not None:
        registry[PI_RUNNER] = pi_runner
    claude_runner = _build_claude_runner(config)
    if claude_runner is not None:
        registry[CLAUDE_RUNNER] = claude_runner
    return registry


def _build_pi_runner(config: PluginConfig) -> ChildAgentRunner | None:
    try:
        from .pi import PiChildAgentRunner, pi_runner_available
    except Exception:
        return None
    if not pi_runner_available():
        return None
    return PiChildAgentRunner(config)


def _build_claude_runner(config: PluginConfig) -> ChildAgentRunner | None:
    try:
        from .claude import ClaudeChildAgentRunner, claude_runner_available
    except Exception:
        return None
    if not claude_runner_available():
        return None
    return ClaudeChildAgentRunner(config)


__all__ = [
    "CLAUDE_RUNNER",
    "HERMES_RUNNER",
    "PI_RUNNER",
    "build_runner_registry",
]
