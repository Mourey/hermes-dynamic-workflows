from __future__ import annotations

import os
import sys
import threading
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from hermes_dynamic_workflows.core.config import PluginConfig
from hermes_dynamic_workflows.run.manager import _approve_launch, _sanctioned_cron_launch

META = {"name": "demo", "description": "a workflow"}


@contextmanager
def fake_approval(
    *,
    gateway=False,
    gateway_choice="once",
    notify_present=True,
    cli_choice="once",
    legacy_wait=False,
    gateway_timeout=1,
    install_touch=False,
    cron=False,
):
    """Inject fake tools.approval / tools.terminal_tool so _approve_launch's
    channel logic can be exercised without the real Hermes engine."""
    appr = types.ModuleType("tools.approval")
    appr._is_gateway_approval_context = lambda: gateway
    appr._is_cron_approval_context = lambda: cron
    appr.get_current_session_key = lambda default="default": "sess"
    appr._lock = threading.RLock()
    appr._gateway_queues = {}
    appr._get_approval_config = lambda: {"gateway_timeout": gateway_timeout}
    appr._fire_approval_hook = lambda *a, **k: None

    class ApprovalEntry:
        def __init__(self, data):
            self.event = threading.Event()
            self.data = data
            self.result = None

    appr._ApprovalEntry = ApprovalEntry

    def notify(*a, **k):
        queue = appr._gateway_queues.get("sess", [])
        if queue and gateway_choice != "timeout":
            queue[-1].result = gateway_choice
            queue[-1].event.set()

    appr._gateway_notify_cbs = {"sess": notify} if notify_present else {}
    if legacy_wait:
        appr._await_gateway_decision = lambda sk, cb, data, surface=None: {
            "resolved": True,
            "choice": gateway_choice,
        }
    appr.prompt_dangerous_approval = lambda command, description, approval_callback=None: cli_choice

    term = types.ModuleType("tools.terminal_tool")
    term._get_approval_callback = lambda: None

    pkg = types.ModuleType("tools")
    pkg.approval = appr
    pkg.terminal_tool = term
    modules = {"tools": pkg, "tools.approval": appr, "tools.terminal_tool": term}
    if install_touch:
        env_pkg = types.ModuleType("tools.environments")
        base = types.ModuleType("tools.environments.base")
        base.touch_activity_if_due = lambda state, label: (state["start"], state["last_touch"])
        pkg.environments = env_pkg
        env_pkg.base = base
        modules["tools.environments"] = env_pkg
        modules["tools.environments.base"] = base

    with patch.dict(sys.modules, modules):
        yield


class LaunchApprovalConfigTests(unittest.TestCase):
    def test_default_is_on(self):
        self.assertTrue(PluginConfig().require_launch_approval)


class LaunchApprovalDecisionTests(unittest.TestCase):
    def test_off_always_approves(self):
        approved, _ = _approve_launch(META, PluginConfig(require_launch_approval=False), None)
        self.assertTrue(approved)

    def test_gateway_approve(self):
        with fake_approval(gateway=True, gateway_choice="once"):
            approved, _ = _approve_launch(META, PluginConfig(), None)
        self.assertTrue(approved)

    def test_gateway_deny(self):
        with fake_approval(gateway=True, gateway_choice="deny"):
            approved, reason = _approve_launch(META, PluginConfig(), None)
        self.assertFalse(approved)
        self.assertIn("denied", reason)

    def test_gateway_legacy_wait_compat(self):
        with fake_approval(gateway=True, gateway_choice="once", legacy_wait=True):
            approved, _ = _approve_launch(META, PluginConfig(), None)
        self.assertTrue(approved)

    def test_gateway_no_channel_denies(self):
        with fake_approval(gateway=True, notify_present=False):
            approved, reason = _approve_launch(META, PluginConfig(), None)
        self.assertFalse(approved)
        self.assertIn("no gateway approval channel", reason)

    def test_gateway_timeout_activity_state_is_initialized(self):
        with fake_approval(gateway=True, gateway_choice="timeout", gateway_timeout=0.01, install_touch=True):
            approved, reason = _approve_launch(META, PluginConfig(), None)
        self.assertFalse(approved)
        self.assertIn("timed out", reason)

    def test_cli_approve(self):
        with fake_approval(gateway=False, cli_choice="once"), \
                patch.dict(os.environ, {"HERMES_INTERACTIVE": "1"}):
            approved, _ = _approve_launch(META, PluginConfig(), None)
        self.assertTrue(approved)

    def test_cli_deny(self):
        with fake_approval(gateway=False, cli_choice="deny"), \
                patch.dict(os.environ, {"HERMES_INTERACTIVE": "1"}):
            approved, _ = _approve_launch(META, PluginConfig(), None)
        self.assertFalse(approved)

    def test_headless_no_channel_denies(self):
        env = {k: v for k, v in os.environ.items() if k != "HERMES_INTERACTIVE"}
        with fake_approval(gateway=False), patch.dict(os.environ, env, clear=True):
            approved, reason = _approve_launch(META, PluginConfig(), None)
        self.assertFalse(approved)
        self.assertIn("no interactive channel", reason)

    def test_headless_denial_names_the_lead_carve_out(self):
        """The remedy an operator should reach for is the narrow one, not the global one."""
        env = {k: v for k, v in os.environ.items() if k != "HERMES_INTERACTIVE"}
        with fake_approval(gateway=False), patch.dict(os.environ, env, clear=True):
            _, reason = _approve_launch(META, PluginConfig(), None)
        self.assertIn("lead_profiles", reason)


def _dispatched(profile="claude-worker", task="t_lead_1", **extra):
    """The env a kanban-dispatched worker actually runs under (kanban_db.py)."""
    env = {"HERMES_PROFILE": profile, "HERMES_KANBAN_TASK": task}
    env.update(extra)
    return env


class SanctionedLeadLaunchTests(unittest.TestCase):
    """Spec 019 §12.1 PHASE A: a dispatched Lead can launch; nothing else gains."""

    def test_lead_profiles_is_empty_by_default(self):
        self.assertEqual(PluginConfig().lead_profiles, ())

    def test_dispatched_lead_launches_unattended(self):
        cfg = PluginConfig(lead_profiles=("claude-worker",))
        with patch.dict(os.environ, _dispatched(), clear=True):
            approved, detail = _approve_launch(META, cfg, None)
        self.assertTrue(approved)
        self.assertEqual(detail, "sanctioned-lead")

    def test_profile_match_is_case_insensitive(self):
        cfg = PluginConfig(lead_profiles=("Claude-Worker",))
        with patch.dict(os.environ, _dispatched(profile="claude-worker"), clear=True):
            approved, _ = _approve_launch(META, cfg, None)
        self.assertTrue(approved)

    def test_interactive_session_on_the_same_profile_is_not_sanctioned(self):
        """No kanban task -> a human activated the profile by hand -> still gated."""
        cfg = PluginConfig(lead_profiles=("claude-worker",))
        env = _dispatched()
        del env["HERMES_KANBAN_TASK"]
        with fake_approval(gateway=False), patch.dict(os.environ, env, clear=True):
            approved, reason = _approve_launch(META, cfg, None)
        self.assertFalse(approved)
        self.assertIn("no interactive channel", reason)

    def test_dispatched_worker_on_an_unlisted_profile_is_not_sanctioned(self):
        cfg = PluginConfig(lead_profiles=("claude-worker",))
        with fake_approval(gateway=False), patch.dict(
            os.environ, _dispatched(profile="builder-heavy"), clear=True
        ):
            approved, reason = _approve_launch(META, cfg, None)
        self.assertFalse(approved)
        self.assertIn("no interactive channel", reason)

    def test_empty_allowlist_sanctions_nobody(self):
        with fake_approval(gateway=False), patch.dict(os.environ, _dispatched(), clear=True):
            approved, _ = _approve_launch(META, PluginConfig(), None)
        self.assertFalse(approved)

    def test_blank_entries_do_not_become_a_wildcard(self):
        cfg = PluginConfig(lead_profiles=("", "   "))
        with fake_approval(gateway=False), patch.dict(
            os.environ, _dispatched(profile=""), clear=True
        ):
            approved, _ = _approve_launch(META, cfg, None)
        self.assertFalse(approved)

    def test_sanction_is_checked_before_any_approval_channel(self):
        """A dispatched Lead must not block on a gateway prompt no one can tap."""
        cfg = PluginConfig(lead_profiles=("claude-worker",))
        with fake_approval(gateway=True, gateway_choice="deny"), patch.dict(
            os.environ, _dispatched(), clear=True
        ):
            approved, detail = _approve_launch(META, cfg, None)
        self.assertTrue(approved)
        self.assertEqual(detail, "sanctioned-lead")


class SanctionedCronLaunchTests(unittest.TestCase):
    """A scheduler-fired cron run can launch when cron_launch is on; nothing else gains."""

    def test_cron_launch_is_off_by_default(self):
        self.assertFalse(PluginConfig().cron_launch)

    def test_cron_session_launches_unattended_when_enabled(self):
        cfg = PluginConfig(cron_launch=True)
        with fake_approval(cron=True), patch.dict(os.environ, {}, clear=True):
            approved, detail = _approve_launch(META, cfg, None)
        self.assertTrue(approved)
        self.assertEqual(detail, "sanctioned-cron")

    def test_cron_session_denied_when_flag_off(self):
        with fake_approval(cron=True), patch.dict(os.environ, {}, clear=True):
            approved, reason = _approve_launch(META, PluginConfig(), None)
        self.assertFalse(approved)
        self.assertIn("no interactive channel", reason)

    def test_non_cron_session_stays_gated_when_flag_on(self):
        """The flag opens the gate for cron runs only, not for every headless session."""
        cfg = PluginConfig(cron_launch=True)
        with fake_approval(gateway=False, cron=False), patch.dict(os.environ, {}, clear=True):
            approved, reason = _approve_launch(META, cfg, None)
        self.assertFalse(approved)
        self.assertIn("no interactive channel", reason)

    def test_cron_sanction_never_reaches_an_approval_channel(self):
        """A sanctioned cron run must not block on a gateway prompt no one can tap."""
        cfg = PluginConfig(cron_launch=True)
        with fake_approval(gateway=True, gateway_choice="deny", cron=True), patch.dict(
            os.environ, {}, clear=True
        ):
            approved, detail = _approve_launch(META, cfg, None)
        self.assertTrue(approved)
        self.assertEqual(detail, "sanctioned-cron")

    def test_cron_classifier_import_failure_fails_closed(self):
        """No engine approval layer importable -> deny, never approve."""
        cfg = PluginConfig(cron_launch=True)
        with patch.dict(sys.modules, {"tools": None, "tools.approval": None}):
            self.assertFalse(_sanctioned_cron_launch(cfg))

    def test_headless_denial_names_the_cron_carve_out(self):
        """The remedy an operator should reach for is the narrow one, not the global one."""
        env = {k: v for k, v in os.environ.items() if k != "HERMES_INTERACTIVE"}
        with fake_approval(gateway=False, cron=True), patch.dict(os.environ, env, clear=True):
            _, reason = _approve_launch(META, PluginConfig(), None)
        self.assertIn("cron_launch", reason)


class CronLaunchConfigTests(unittest.TestCase):
    def test_env_var_enables(self):
        from hermes_dynamic_workflows.core.config import load_config

        with patch.dict(
            os.environ,
            {"HERMES_DYNAMIC_WORKFLOWS_CRON_LAUNCH": "1"},
            clear=False,
        ):
            self.assertTrue(load_config().cron_launch)


class LeadProfilesConfigTests(unittest.TestCase):
    def test_env_var_parses_the_comma_form(self):
        from hermes_dynamic_workflows.core.config import load_config

        with patch.dict(
            os.environ,
            {"HERMES_DYNAMIC_WORKFLOWS_LEAD_PROFILES": "claude-worker, lead"},
            clear=False,
        ):
            self.assertEqual(load_config().lead_profiles, ("claude-worker", "lead"))


if __name__ == "__main__":
    unittest.main()
