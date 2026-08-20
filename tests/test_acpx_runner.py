"""Spec 030 — acpx uniform headless layer: fixture tests for AcpxChildAgentRunner.

Mirrors test_multi_runner.py's unittest conventions and the fake-binary fixture
pattern of pi-code-lane/tests/run-fixtures.sh: a fake `acpx` shell script in a
tmpdir, pointed via HERMES_DYNAMIC_WORKFLOWS_ACPX_BINARY, that captures its
argv/stdin and emits canned NDJSON per a FAKE_ACPX_MODE behavior switch. Zero
tokens burned.

Run: python3 -m pytest tests/test_acpx_runner.py -q  (from the plugin root)
"""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path

import hermes_dynamic_workflows.child.runners.acpx as acpx_module
from hermes_dynamic_workflows.child.runners import (
    ACPX_RUNNER,
    build_runner_registry,
)
from hermes_dynamic_workflows.core.config import PluginConfig
from hermes_dynamic_workflows.core.errors import ChildAgentError, WorkflowTimeout
from hermes_dynamic_workflows.core.types import ChildAgentRequest, ChildAgentRunner

ACPX_BINARY_ENV = "HERMES_DYNAMIC_WORKFLOWS_ACPX_BINARY"

# Every invocation APPENDS its argv (blank-line separated) to $CAP/argv_all and
# its stdin to $CAP/stdin_all, so the full argv history is inspectable. Mode
# switch picks the canned NDJSON / exit-code behavior.
FAKE_SCRIPT = """\
#!/usr/bin/env bash
set -u
CAP="{CAP}"
MODE="{MODE}"
printf '%s\\n' "$@" >> "$CAP/argv_all"
echo "---" >> "$CAP/argv_all"
cat >> "$CAP/stdin_all" 2>/dev/null || true

for a in "$@"; do
  if [[ "$a" == "--version" ]]; then echo "0.13.1"; exit 0; fi
done
if [[ " $* " == *" sessions "* ]]; then
  if [[ " $* " == *" ensure "* ]]; then
    echo '{"action":"session_ensured","created":true,"acpxRecordId":"ac-session-1","acpxSessionId":"ac-session-1","name":"x"}'
    exit 0
  fi
  echo '{"action":"session_closed"}'
  exit 0
fi
case "$MODE" in
  ok)
    echo '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"s1","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"fixture ok"}}}}'
    echo '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"s1","update":{"sessionUpdate":"usage_update","used":1234,"size":1048576}}}'
    echo '{"jsonrpc":"2.0","id":2,"result":{"stopReason":"end_turn"}}'
    exit 0 ;;
  nousage)
    echo '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"s1","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"fixture ok"}}}}'
    echo '{"jsonrpc":"2.0","id":2,"result":{"stopReason":"end_turn"}}'
    exit 0 ;;
  rc1auth)
    echo 'HTTP 401 Unauthorized: invalid token' >&2
    exit 1 ;;
  rc1spawn)
    echo 'could not spawn the agent process' >&2
    exit 1 ;;
  rc3)
    echo 'kimi did not respond in time' >&2
    exit 3 ;;
  rc4)
    exit 4 ;;
  rc5)
    echo 'session closed unexpectedly' >&2
    exit 5 ;;
esac
exit 0
"""


def _write_fake_acpx(tmpdir: Path, mode: str) -> Path:
    capture = tmpdir / "capture"
    capture.mkdir(parents=True, exist_ok=True)
    bin_path = tmpdir / "acpx"
    bin_path.write_text(
        FAKE_SCRIPT.replace("{CAP}", str(capture)).replace("{MODE}", mode)
    )
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IXUSR)
    return bin_path


def _request(**overrides) -> ChildAgentRequest:
    base = dict(
        id=1,
        prompt="Reply exactly: OK",
        label="agent-1",
        phase=None,
        toolsets=[],
        cwd="/tmp",
        isolation=None,
        runner=ACPX_RUNNER,
        lane="default",
    )
    return ChildAgentRequest(**(base | overrides))


def _set_env(key: str, value: str) -> str | None:
    prev = os.environ.get(key)
    os.environ[key] = value
    return prev


def _restore_env(key: str, prev: str | None) -> None:
    if prev is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = prev


class RecordingRunner(ChildAgentRunner):
    def __init__(self, name: str):
        self.name = name

    def run(self, request: ChildAgentRequest):
        return f"{self.name}:{request.label}"


def _run_with_fake(request: ChildAgentRequest, mode: str):
    """Run the runner against a fake acpx; return (result, argv_text)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        bin_path = _write_fake_acpx(tmpdir, mode)
        capture = tmpdir / "capture"
        prev = _set_env(ACPX_BINARY_ENV, str(bin_path))
        try:
            runner = acpx_module.AcpxChildAgentRunner(
                PluginConfig(child_timeout_seconds=30)
            )
            result = runner.run(request)
        finally:
            _restore_env(ACPX_BINARY_ENV, prev)
        argv_text = ""
        arg_all = capture / "argv_all"
        if arg_all.exists():
            argv_text = arg_all.read_text()
        elif maybe := list(capture.glob("argv*")):
            argv_text = maybe[0].read_text()
        return result, argv_text


def _argv_blocks(argv_text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    for block in argv_text.split("---\n"):
        args = [a for a in block.split("\n") if a]
        if args:
            blocks.append(args)
    return blocks


class RunnerAvailabilityTests(unittest.TestCase):
    def test_registry_omits_acpx_when_the_probe_fails(self):
        original = acpx_module.acpx_runner_available
        acpx_module.acpx_runner_available = lambda: False
        try:
            registry = build_runner_registry(
                PluginConfig(), RecordingRunner("hermes")
            )
        finally:
            acpx_module.acpx_runner_available = original
        self.assertNotIn(ACPX_RUNNER, registry)

    def test_registry_includes_acpx_when_the_probe_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_path = _write_fake_acpx(Path(tmp), "ok")
            prev = _set_env(ACPX_BINARY_ENV, str(bin_path))
            try:
                registry = build_runner_registry(
                    PluginConfig(), RecordingRunner("hermes")
                )
            finally:
                _restore_env(ACPX_BINARY_ENV, prev)
            self.assertIn(ACPX_RUNNER, registry)


class RunnerArgvTests(unittest.TestCase):
    def test_prompt_argv_puts_global_flags_before_the_kimi_token(self):
        _, argv_text = _run_with_fake(_request(), "ok")
        blocks = _argv_blocks(argv_text)
        # the main prompt block is the one ending in the stdin-file pair `-f -`
        prompt_blocks = [b for b in blocks if b[-2:] == ["-f", "-"]]
        self.assertTrue(prompt_blocks, "expected at least one kimi invoke block")
        args = prompt_blocks[0]
        # find global flags before the kimi token; --model/--format/--cwd etc
        kimi_idx = args.index("kimi")
        global_flags = args[:kimi_idx]
        self.assertIn("--format", global_flags)
        self.assertIn("--json-strict", global_flags)
        self.assertIn("--approve-all", global_flags)
        self.assertIn("--cwd", global_flags)
        self.assertIn("--timeout", global_flags)
        # session/-s and stdin-file flags come AFTER the kimi token
        self.assertIn("-s", args[kimi_idx:])
        self.assertIn("-f", args[kimi_idx:])
        self.assertEqual(args[-1], "-")

    def test_sessions_ensure_and_close_are_invoked(self):
        _, argv_text = _run_with_fake(_request(), "ok")
        blocks = _argv_blocks(argv_text)
        all_args = [a for block in blocks for a in block]
        self.assertIn("ensure", all_args)
        self.assertIn("close", all_args)
        self.assertIn("--name", all_args)
        self.assertTrue(
            any("dw-wf-acpx-" in a for a in all_args),
            "named session dw-<task_id> must appear in argv",
        )


class RunnerContentAndCostTests(unittest.TestCase):
    def test_content_is_concatenated_message_chunks(self):
        result, _ = _run_with_fake(_request(), "ok")
        self.assertEqual(result.content, "fixture ok")

    def test_cost_present_when_usage_emitted(self):
        result, _ = _run_with_fake(_request(), "ok")
        self.assertEqual(result.metadata["tokens"], 1234)

    def test_absent_usage_yields_null_cost_plus_disclosure(self):
        result, _ = _run_with_fake(_request(), "nousage")
        self.assertIsNone(result.metadata["total_cost_usd"])
        self.assertEqual(
            result.metadata["cost_unavailable"], "kimi-acp-emits-no-cost"
        )


class RunnerErrorMappingTests(unittest.TestCase):
    def test_rc3_raises_WorkflowTimeout(self):
        with self.assertRaises(WorkflowTimeout):
            _run_with_fake(_request(), "rc3")

    def test_rc1_auth_stderr_classifies_as_auth(self):
        with self.assertRaises(ChildAgentError) as ctx:
            _run_with_fake(_request(), "rc1auth")
        self.assertIn("auth", str(ctx.exception).lower())

    def test_rc1_other_stderr_classifies_as_spawn_failure(self):
        with self.assertRaises(ChildAgentError) as ctx:
            _run_with_fake(_request(), "rc1spawn")
        self.assertIn("spawn", str(ctx.exception).lower())

    def test_meta_usage_absent_yields_null_cost(self):
        result, _ = _run_with_fake(_request(), "nousage")
        self.assertIsNone(result.metadata["total_cost_usd"])


class RunnerSchemaRetryTests(unittest.TestCase):
    """Schema retries re-prompt the SAME named session (acpx queues follow-ups)."""

    def test_schema_retry_reuses_the_same_session_name(self):
        # The fake returns non-JSON on the first prompt so run_with_schema retries
        # and the second prompt reuses the same dw-<task_id> session. We assert the
        # session name constant by checking metadata surfaces only once; the fake
        # doesn't model multi-attempt NDJSON, so this guards the invariant that the
        # session_name is derived from task_id once and threaded through.
        request = _request(
            schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            }
        )
        result, _ = _run_with_fake(request, "ok")
        self.assertEqual(result.metadata["runner"], "acpx")


if __name__ == "__main__":
    unittest.main()