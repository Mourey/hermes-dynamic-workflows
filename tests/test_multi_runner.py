from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from hermes_dynamic_workflows.child.runners import build_runner_registry
from hermes_dynamic_workflows.child.runners import pi as pi_runner_module
from hermes_dynamic_workflows.child.subprocess_schema import (
    extract_json_value,
    run_with_schema,
)
from hermes_dynamic_workflows.core.config import PluginConfig, _as_runner_concurrency
from hermes_dynamic_workflows.core.errors import ChildAgentError
from hermes_dynamic_workflows.core.types import (
    ChildAgentRequest,
    ChildAgentResult,
    ChildAgentRunner,
    ResolvedAgentSpec,
)
from hermes_dynamic_workflows.engine.cache import ResumeCache, agent_fingerprint
from hermes_dynamic_workflows.engine.runtime import WorkflowOptions, run_workflow


class RecordingRunner(ChildAgentRunner):
    def __init__(self, name: str):
        self.name = name
        self.requests: list[ChildAgentRequest] = []

    def run(self, request: ChildAgentRequest):
        self.requests.append(request)
        return f"{self.name}:{request.label}"


class ConcurrencyRunner(ChildAgentRunner):
    """Records the peak number of simultaneously in-flight children."""

    def __init__(self, hold_seconds: float = 0.05):
        self.hold_seconds = hold_seconds
        self._lock = threading.Lock()
        self.live = 0
        self.peak = 0

    def run(self, request: ChildAgentRequest):
        with self._lock:
            self.live += 1
            self.peak = max(self.peak, self.live)
        try:
            time.sleep(self.hold_seconds)
        finally:
            with self._lock:
                self.live -= 1
        return request.label


def _registry(**runners: ChildAgentRunner) -> dict[str, ChildAgentRunner]:
    return dict(runners)


class RunnerSelectionTests(unittest.TestCase):
    def test_per_call_runner_option_routes_to_that_runner(self):
        script = """
meta = {"name": "route", "description": "Test workflow"}

return await agent("work", {"runner": "pi"})
"""
        hermes = RecordingRunner("hermes")
        pi = RecordingRunner("pi")
        result = run_workflow(
            script,
            WorkflowOptions(child_runners=_registry(hermes=hermes, pi=pi)),
        )

        self.assertEqual(result.value, "pi:agent-1")
        self.assertEqual(hermes.requests, [])
        self.assertEqual(len(pi.requests), 1)
        self.assertEqual(pi.requests[0].runner, "pi")

    def test_no_runner_option_uses_the_default(self):
        script = """
meta = {"name": "default-route", "description": "Test workflow"}

return await agent("work")
"""
        hermes = RecordingRunner("hermes")
        pi = RecordingRunner("pi")
        run_workflow(script, WorkflowOptions(child_runners=_registry(hermes=hermes, pi=pi)))

        self.assertEqual(len(hermes.requests), 1)
        self.assertEqual(hermes.requests[0].runner, "hermes")
        self.assertEqual(pi.requests, [])

    def test_agent_type_runner_and_lane_become_defaults(self):
        script = """
meta = {"name": "typed-route", "description": "Test workflow"}

return await agent("work", {"agentType": "cheap"})
"""
        hermes = RecordingRunner("hermes")
        pi = RecordingRunner("pi")
        with tempfile.TemporaryDirectory() as tmp:
            agent_dir = Path(tmp) / ".hermes" / "dynamic-workflows" / "agents"
            agent_dir.mkdir(parents=True)
            (agent_dir / "cheap.md").write_text(
                "---\nname: cheap\nrunner: pi\nlane: builder\n---\n\nBuild it.\n",
                encoding="utf-8",
            )
            run_workflow(
                script,
                WorkflowOptions(cwd=tmp, child_runners=_registry(hermes=hermes, pi=pi)),
            )

        self.assertEqual(hermes.requests, [])
        self.assertEqual(len(pi.requests), 1)
        self.assertEqual(pi.requests[0].runner, "pi")
        self.assertEqual(pi.requests[0].lane, "builder")

    def test_per_call_runner_beats_agent_type_runner(self):
        script = """
meta = {"name": "override-route", "description": "Test workflow"}

return await agent("work", {"agentType": "cheap", "runner": "hermes"})
"""
        hermes = RecordingRunner("hermes")
        pi = RecordingRunner("pi")
        with tempfile.TemporaryDirectory() as tmp:
            agent_dir = Path(tmp) / ".hermes" / "dynamic-workflows" / "agents"
            agent_dir.mkdir(parents=True)
            (agent_dir / "cheap.md").write_text(
                "---\nname: cheap\nrunner: pi\nlane: builder\n---\n\nBuild it.\n",
                encoding="utf-8",
            )
            run_workflow(
                script,
                WorkflowOptions(cwd=tmp, child_runners=_registry(hermes=hermes, pi=pi)),
            )

        self.assertEqual(pi.requests, [])
        self.assertEqual(len(hermes.requests), 1)
        self.assertEqual(hermes.requests[0].runner, "hermes")

    def test_agent_type_runner_inherit_falls_back_to_default(self):
        script = """
meta = {"name": "inherit-route", "description": "Test workflow"}

return await agent("work", {"agentType": "neutral"})
"""
        hermes = RecordingRunner("hermes")
        pi = RecordingRunner("pi")
        with tempfile.TemporaryDirectory() as tmp:
            agent_dir = Path(tmp) / ".hermes" / "dynamic-workflows" / "agents"
            agent_dir.mkdir(parents=True)
            (agent_dir / "neutral.md").write_text(
                "---\nname: neutral\nrunner: inherit\n---\n\nThink.\n",
                encoding="utf-8",
            )
            run_workflow(
                script,
                WorkflowOptions(cwd=tmp, child_runners=_registry(hermes=hermes, pi=pi)),
            )

        self.assertEqual(pi.requests, [])
        self.assertEqual(len(hermes.requests), 1)

    def test_unknown_runner_names_the_available_runners(self):
        script = """
meta = {"name": "bad-route", "description": "Test workflow"}

return await agent("work", {"runner": "claude"})
"""
        hermes = RecordingRunner("hermes")
        pi = RecordingRunner("pi")
        with self.assertRaises(Exception) as ctx:
            run_workflow(
                script,
                WorkflowOptions(child_runners=_registry(hermes=hermes, pi=pi)),
            )

        message = str(ctx.exception)
        self.assertIn("unknown runner 'claude'", message)
        self.assertIn("Available runners: hermes, pi", message)
        self.assertEqual(hermes.requests, [])
        self.assertEqual(pi.requests, [])

    def test_runner_is_recorded_on_the_agent_snapshot_and_journal(self):
        script = """
meta = {"name": "journal-route", "description": "Test workflow"}

return await agent("work", {"runner": "pi"})
"""
        events: list[dict] = []
        result = run_workflow(
            script,
            WorkflowOptions(
                child_runners=_registry(hermes=RecordingRunner("h"), pi=RecordingRunner("p")),
                on_journal=events.append,
            ),
        )

        agents = result.state.snapshot()["agents"]
        self.assertEqual(agents[0]["runner"], "pi")
        started = [event for event in events if event.get("type") == "started"]
        self.assertEqual([event["runner"] for event in started], ["pi"])


class RunnerCacheTests(unittest.TestCase):
    """The §3 regression guard: a runner swap must never reuse a cached result."""

    def test_cache_inputs_include_runner_and_lane(self):
        inputs = ResolvedAgentSpec(
            requested_agent_type="general-purpose",
            runner="pi",
            lane="builder",
        ).cache_inputs()

        self.assertEqual(inputs["runner"], "pi")
        self.assertEqual(inputs["lane"], "builder")

    def test_swapping_the_runner_changes_the_fingerprint(self):
        base = dict(requested_agent_type="general-purpose", lane=None)
        hermes_print = agent_fingerprint(
            "work", ResolvedAgentSpec(runner="hermes", **base).cache_inputs()
        )
        pi_print = agent_fingerprint(
            "work", ResolvedAgentSpec(runner="pi", **base).cache_inputs()
        )

        self.assertNotEqual(hermes_print, pi_print)

    def test_swapping_the_lane_changes_the_fingerprint(self):
        base = dict(requested_agent_type="general-purpose", runner="pi")
        cheap = agent_fingerprint(
            "work", ResolvedAgentSpec(lane="builder", **base).cache_inputs()
        )
        heavy = agent_fingerprint(
            "work", ResolvedAgentSpec(lane="builder-heavy", **base).cache_inputs()
        )

        self.assertNotEqual(cheap, heavy)

    def test_resume_after_a_runner_swap_reruns_instead_of_replaying(self):
        hermes_script = """
meta = {"name": "swap", "description": "Test workflow"}

return await agent("work")
"""
        pi_script = """
meta = {"name": "swap", "description": "Test workflow"}

return await agent("work", {"runner": "pi"})
"""
        hermes = RecordingRunner("hermes")
        pi = RecordingRunner("pi")
        first_cache = ResumeCache()
        first = run_workflow(
            hermes_script,
            WorkflowOptions(
                child_runners=_registry(hermes=hermes, pi=pi),
                resume_cache=first_cache,
            ),
        )
        self.assertEqual(first.value, "hermes:agent-1")

        second = run_workflow(
            pi_script,
            WorkflowOptions(
                child_runners=_registry(hermes=hermes, pi=pi),
                resume_cache=ResumeCache(first_cache.current),
            ),
        )

        self.assertEqual(second.value, "pi:agent-1")
        self.assertEqual(len(pi.requests), 1, "a runner swap must re-run, not replay")

    def test_discovered_toolsets_stay_out_of_the_fingerprint(self):
        """MCP discovery is racy; only declared toolsets may key the cache."""
        declared = ("web", "file")
        before = ResolvedAgentSpec(
            requested_agent_type="general-purpose",
            toolsets=declared,
            declared_toolsets=declared,
        )
        after = ResolvedAgentSpec(
            requested_agent_type="general-purpose",
            toolsets=declared + ("mcp-github",),
            declared_toolsets=declared,
        )

        self.assertEqual(
            agent_fingerprint("work", before.cache_inputs()),
            agent_fingerprint("work", after.cache_inputs()),
        )

    def test_declared_toolsets_still_key_the_cache(self):
        narrow = ResolvedAgentSpec(
            requested_agent_type="t", toolsets=("file",), declared_toolsets=("file",)
        )
        wide = ResolvedAgentSpec(
            requested_agent_type="t",
            toolsets=("file", "terminal"),
            declared_toolsets=("file", "terminal"),
        )

        self.assertNotEqual(
            agent_fingerprint("work", narrow.cache_inputs()),
            agent_fingerprint("work", wide.cache_inputs()),
        )

    def test_resume_without_a_swap_still_replays_from_cache(self):
        script = """
meta = {"name": "replay", "description": "Test workflow"}

return await agent("work")
"""
        hermes = RecordingRunner("hermes")
        first_cache = ResumeCache()
        first = run_workflow(
            script,
            WorkflowOptions(child_runners=_registry(hermes=hermes), resume_cache=first_cache),
        )

        second = run_workflow(
            script,
            WorkflowOptions(
                child_runners=_registry(hermes=hermes),
                resume_cache=ResumeCache(first_cache.current),
            ),
        )

        self.assertEqual(second.value, first.value)
        self.assertEqual(len(hermes.requests), 1, "cached call must not re-run")


class RunnerConcurrencyTests(unittest.TestCase):
    def test_per_runner_cap_bounds_a_fan_out(self):
        script = """
meta = {"name": "fanout", "description": "Test workflow"}

results = await parallel([lambda: agent("work", {"runner": "pi"}) for _ in range(10)])
return len(results)
"""
        pi = ConcurrencyRunner()
        config = PluginConfig(concurrency=8, runner_concurrency={"pi": 2})
        result = run_workflow(
            script,
            WorkflowOptions(
                config=config,
                child_runners=_registry(hermes=RecordingRunner("h"), pi=pi),
            ),
        )

        self.assertEqual(result.value, 10)
        self.assertLessEqual(pi.peak, 2)

    def test_a_runner_without_a_cap_uses_only_the_global_ceiling(self):
        script = """
meta = {"name": "fanout-uncapped", "description": "Test workflow"}

results = await parallel([lambda: agent("work") for _ in range(6)])
return len(results)
"""
        hermes = ConcurrencyRunner()
        config = PluginConfig(concurrency=3, runner_concurrency={"pi": 1})
        run_workflow(
            script,
            WorkflowOptions(config=config, child_runners=_registry(hermes=hermes)),
        )

        self.assertLessEqual(hermes.peak, 3)

    def test_runner_concurrency_parses_config_and_env_forms(self):
        default = {"pi": 4, "claude": 2}
        self.assertEqual(
            _as_runner_concurrency("pi=6", default), {"pi": 6, "claude": 2}
        )
        self.assertEqual(
            _as_runner_concurrency({"claude": 1}, default), {"pi": 4, "claude": 1}
        )
        self.assertEqual(_as_runner_concurrency("garbage", default), default)
        self.assertEqual(_as_runner_concurrency(None, default), default)


class RunnerRegistryTests(unittest.TestCase):
    def test_hermes_is_always_registered(self):
        hermes = RecordingRunner("hermes")
        registry = build_runner_registry(PluginConfig(), hermes)

        self.assertIs(registry["hermes"], hermes)

    def test_pi_is_omitted_when_unavailable(self):
        original = pi_runner_module.pi_runner_available
        pi_runner_module.pi_runner_available = lambda: False
        try:
            registry = build_runner_registry(PluginConfig(), RecordingRunner("hermes"))
        finally:
            pi_runner_module.pi_runner_available = original

        self.assertNotIn("pi", registry)


class SubprocessSchemaTests(unittest.TestCase):
    SCHEMA = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    def test_extract_json_prefers_the_last_fenced_block(self):
        text = 'draft:\n```json\n{"answer": "no"}\n```\nfinal:\n```json\n{"answer": "yes"}\n```'
        found, value = extract_json_value(text)

        self.assertTrue(found)
        self.assertEqual(value, {"answer": "yes"})

    def test_extract_json_falls_back_to_a_bare_object(self):
        found, value = extract_json_value('Here it is: {"answer": "yes"} — done')

        self.assertTrue(found)
        self.assertEqual(value, {"answer": "yes"})

    def test_extract_json_reports_when_there_is_none(self):
        found, value = extract_json_value("no json here")

        self.assertFalse(found)
        self.assertIsNone(value)

    def test_invalid_output_is_retried_with_the_validation_errors(self):
        prompts: list[str] = []
        replies = ["not json at all", '```json\n{"wrong": 1}\n```', '```json\n{"answer": "ok"}\n```']

        def invoke(prompt: str, attempt: int) -> str:
            prompts.append(prompt)
            return replies[attempt - 1]

        value, attempts = run_with_schema("do it", self.SCHEMA, invoke)

        self.assertEqual(value, {"answer": "ok"})
        self.assertEqual(attempts, 3)
        self.assertIn("JSON Schema", prompts[0])
        self.assertIn("did not contain a JSON object", prompts[1])
        self.assertIn("required property 'answer'", prompts[2])

    def test_exhausting_the_retry_budget_raises(self):
        def invoke(prompt: str, attempt: int) -> str:
            return "still not json"

        with self.assertRaises(ChildAgentError) as ctx:
            run_with_schema("do it", self.SCHEMA, invoke)

        self.assertIn("after 5 attempts", str(ctx.exception))


class PiAdapterUnitTests(unittest.TestCase):
    def test_wrapper_json_is_read_from_the_last_json_line(self):
        stdout = 'warming up\n{"is_error": false, "summary": "ok"}\n'
        payload = pi_runner_module._parse_wrapper_json(stdout)

        self.assertEqual(payload, {"is_error": False, "summary": "ok"})

    def test_wrapper_json_missing_returns_none(self):
        self.assertIsNone(pi_runner_module._parse_wrapper_json("no json\nhere\n"))

    def test_final_text_prefers_the_full_agent_end_message(self):
        long_text = "x" * 900
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "run.jsonl"
            log.write_text(
                "\n".join(
                    [
                        json.dumps({"type": "session", "id": "sess-1"}),
                        json.dumps(
                            {
                                "type": "message_end",
                                "message": {
                                    "role": "assistant",
                                    "content": [{"type": "text", "text": "streamed"}],
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "agent_end",
                                "messages": [
                                    {
                                        "role": "assistant",
                                        "content": [{"type": "text", "text": long_text}],
                                    }
                                ],
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            text = pi_runner_module._final_text({"summary": "x" * 500}, log)

        self.assertEqual(text, long_text)
        self.assertGreater(len(text), 500, "summary truncation must not reach the caller")

    def test_final_text_falls_back_to_the_summary_without_a_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "absent.jsonl"
            self.assertEqual(
                pi_runner_module._final_text({"summary": "wrapper said so"}, missing),
                "wrapper said so",
            )

    def test_env_isolates_the_run_from_the_kanban_board(self):
        from hermes_dynamic_workflows.child.worktree import WorkspaceLease

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "work"
            log_dir = work_dir / "logs"
            lease = WorkspaceLease(task_id="wf-pi-1", cwd=tmp)
            env = pi_runner_module._build_env(lease, "builder", work_dir, log_dir)

        self.assertEqual(env["HERMES_PROFILE"], "builder")
        self.assertEqual(env["HERMES_KANBAN_WORKSPACE"], tmp)
        self.assertEqual(env["HERMES_KANBAN_TASK_ID"], "wf-pi-1")
        self.assertEqual(env["PI_LANE_LOG_DIR"], str(log_dir))
        self.assertNotIn("HERMES_KANBAN_PIPELINE_NODE", env)
        self.assertFalse(Path(env["PI_LANE_KANBAN_DB"]).exists())
        self.assertFalse(Path(env["CODE_NODES"], "adw-emit.sh").exists())

    def test_model_override_becomes_a_dispatch_block(self):
        request = ChildAgentRequest(
            id=1, prompt="work", label="w", phase=None, toolsets=[], model="qwen/qwen3-coder"
        )
        block = pi_runner_module._dispatch_block(request)

        self.assertIn("```dispatch", block)
        self.assertIn("model: qwen/qwen3-coder", block)

    def test_no_model_means_no_dispatch_block(self):
        request = ChildAgentRequest(id=1, prompt="w", label="w", phase=None, toolsets=[])
        self.assertEqual(pi_runner_module._dispatch_block(request), "")

        inherit = ChildAgentRequest(
            id=1, prompt="w", label="w", phase=None, toolsets=[], model="inherit"
        )
        self.assertEqual(pi_runner_module._dispatch_block(inherit), "")

    def test_prompt_carries_agent_type_instructions_and_workspace(self):
        from hermes_dynamic_workflows.child.presets import AgentTypeSpec

        request = ChildAgentRequest(
            id=1,
            prompt="rename the flag",
            label="w",
            phase=None,
            toolsets=[],
            isolation="worktree",
        )
        spec = AgentTypeSpec(name="cheap", instructions="Be surgical.", source="test")
        prompt = pi_runner_module._build_pi_prompt(request, spec, workspace="/tmp/ws")

        self.assertIn("Be surgical.", prompt)
        self.assertIn("- Workspace: /tmp/ws", prompt)
        self.assertIn("isolated git worktree", prompt)
        self.assertTrue(prompt.rstrip().endswith("rename the flag"))

    def test_lane_defaults_come_from_the_agent_type(self):
        from hermes_dynamic_workflows.child.presets import AgentTypeSpec

        request = ChildAgentRequest(id=1, prompt="w", label="w", phase=None, toolsets=[])
        spec = AgentTypeSpec(
            name="cheap", instructions="x", source="test", lane="builder-heavy"
        )

        applied = pi_runner_module._apply_pi_agent_type_defaults(request, spec)

        self.assertEqual(applied.lane, "builder-heavy")


class HermesRunnerMetadataTests(unittest.TestCase):
    def test_hermes_children_report_the_registry_name(self):
        from hermes_dynamic_workflows.child.runner import _child_metadata
        from hermes_dynamic_workflows.child.worktree import WorkspaceLease

        metadata = _child_metadata(
            object(),
            {},
            WorkspaceLease(task_id="t", cwd="/tmp"),
            None,
            [],
        )

        self.assertEqual(metadata["runner"], "hermes")


class ChildAgentResultShapeTests(unittest.TestCase):
    def test_pi_metadata_carries_cost_and_session(self):
        from hermes_dynamic_workflows.child.worktree import WorkspaceLease

        payload = {
            "pi_session_id": "sess-9",
            "total_cost_usd": 0.0123,
            "providers_tried": [{"provider": "openrouter"}],
            "changed_files": ["a.py", "_project/", "./_project/README.md"],
            "tests_run": "pytest",
            "dispatch": {"model": "deepseek/deepseek-v4-flash", "provider": "openrouter"},
        }
        metadata = pi_runner_module._pi_metadata(
            payload,
            lease=WorkspaceLease(task_id="wf-pi-2", cwd="/tmp/ws"),
            lane="builder",
            agent_type=None,
            log_path=Path("/tmp/log.jsonl"),
            attempts=1,
        )

        self.assertEqual(metadata["runner"], "pi")
        self.assertEqual(metadata["lane"], "builder")
        self.assertEqual(metadata["pi_session_id"], "sess-9")
        self.assertEqual(metadata["total_cost_usd"], 0.0123)
        self.assertEqual(metadata["model"], "deepseek/deepseek-v4-flash")
        self.assertEqual(
            metadata["changed_files"], ["a.py"], "wrapper scaffold must not look like work"
        )
        self.assertIsInstance(ChildAgentResult(content="x", metadata=metadata).metadata, dict)


if __name__ == "__main__":
    unittest.main()
