"""Unit and integration tests. Run with:  pytest -q
No API key is needed: every test uses the deterministic offline mode
or a fake Claude client.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import evals
from actions import AuditLog, Proposal, check_policy, execute
from agent import FleetCopilot
from planner import ActionRequest, ClaudePlanner, ClaudeWriter, Plan, RulePlanner, ToolCall, cross_tenant_reference, grounding_problems
from telemetry import TelemetryStore, TenantAccessError
from tools import run_tool


@pytest.fixture(scope="module")
def store():
    return TelemetryStore()


@pytest.fixture
def acme(store):
    return store.scope("acme")


def agent(store, company, tmp_path, mode="offline"):
    return FleetCopilot(store, company, mode=mode, audit_path=tmp_path / "audit.jsonl")


# --------------------------------------------------------------------- data

def test_dataset_shape(store):
    views = [store.scope(c) for c in store.company_ids]
    assert sum(len(v.snapshots(d)) for v in views for d in v.device_ids()) == 750
    assert sorted(store.company_ids) == ["acme-001", "globex-002", "initech-003"]
    devices = sum(len(store.scope(c).device_ids()) for c in store.company_ids)
    employees = sum(len(store.scope(c).employees()) for c in store.company_ids)
    assert devices == 25
    assert employees == 14  # the brief says "~18"; the data has 14


def test_tenant_view_cannot_read_other_tenant_device(acme):
    with pytest.raises(TenantAccessError):
        acme.snapshots("JRZSGXVMKE6M")  # a Globex device


# -------------------------------------------------------------------- tools

def test_low_disk_is_tenant_scoped(acme):
    res = run_tool(acme, "low_disk", {})
    assert {r["device_id"] for r in res.rows} == {"MT7PJB7N5LRE", "8NM23J95R5I6"}
    assert all(acme.owns_device(e.device_id) for e in res.evidence)


def test_run_tool_ignores_company_id_from_model(acme):
    res = run_tool(acme, "low_disk", {"company_id": "globex-002"})
    assert {r["device_id"] for r in res.rows} == {"MT7PJB7N5LRE", "8NM23J95R5I6"}


def test_unknown_tool_and_bad_args_are_errors_not_crashes(acme):
    assert run_tool(acme, "drop_tables", {}).error
    assert run_tool(acme, "low_disk", {"nonsense": 1}).error


def test_no_high_severity_failures_still_cites_evidence(acme):
    res = run_tool(acme, "compliance_failures", {"severity": "high"})
    assert res.rows == [] and len(res.evidence) > 0


def test_os_older_than_counts_only_macos(acme):
    res = run_tool(acme, "os_older_than", {"platform": "macOS", "major": 15})
    assert len(res.rows) == 4
    assert all(r["os_version"].split(".")[0] in ("13", "14") for r in res.rows)


def test_ram_pressure_insight(store):
    rows = run_tool(store.scope("globex"), "ram_pressure", {}).rows
    assert {r["device_id"] for r in rows} == {"JRZSGXVMKE6M", "CUXW0BQSV1B7"}


# ------------------------------------------------------------------- policy

def _proposal(view, action, args, evidence):
    return Proposal(action, args, view.company_id, "test", [e.as_dict() for e in evidence])


def test_policy_blocks_upgrade_without_ram_pressure(acme):
    snap = acme.latest("1LYSSFD074BB")
    p = _proposal(acme, "create_upgrade_order", {"device_id": "1LYSSFD074BB", "component": "ram", "spec": "32GB"},
                  [snap.evidence("ram_used_pct", snap.ram_used_pct)])
    allowed, _ = check_policy(acme, p)
    assert not allowed


def test_policy_allows_upgrade_with_ram_pressure(acme):
    snap = acme.latest("M4XVHUV1MEPZ")
    p = _proposal(acme, "create_upgrade_order", {"device_id": "M4XVHUV1MEPZ", "component": "ram", "spec": "32GB"},
                  [snap.evidence("ram_used_pct", snap.ram_used_pct)])
    assert check_policy(acme, p)[0]


def test_policy_rejects_fabricated_evidence(acme):
    p = Proposal("create_upgrade_order", {"device_id": "M4XVHUV1MEPZ", "component": "ram", "spec": "32GB"},
                 acme.company_id, "test",
                 [{"device_id": "M4XVHUV1MEPZ", "collected_at": "2030-01-01T00:00:00Z", "field": "x", "value": 1}])
    allowed, notes = check_policy(acme, p)
    assert not allowed and "does not exist" in notes[0]


def test_policy_rejects_cross_tenant_proposal(store, acme):
    globex = store.scope("globex")
    snap = globex.latest("JRZSGXVMKE6M")
    p = Proposal("create_upgrade_order", {"device_id": "JRZSGXVMKE6M", "component": "storage", "spec": "1TB"},
                 acme.company_id, "test", [snap.evidence("disk_free_pct", snap.disk_free_pct).as_dict()])
    assert not check_policy(acme, p)[0]


def test_policy_requires_evidence(acme):
    p = Proposal("flag_device_for_replacement", {"device_id": "7VP16KHV88LM", "reason": "battery"},
                 acme.company_id, "test", [])
    assert not check_policy(acme, p)[0]


def test_execute_refuses_unapproved_proposal(acme):
    p = Proposal("notify_employee", {"employee_id": "emp-acme-1003", "message": "hi"}, acme.company_id, "t")
    with pytest.raises(PermissionError):
        execute(p)


# -------------------------------------------------------------------- agent

def test_cross_tenant_question_is_refused(store, tmp_path):
    res = agent(store, "acme", tmp_path).ask("Which devices at Globex are low on disk?")
    assert res["refused"] and "JRZSGXVMKE6M" not in res["answer"]


def test_cross_tenant_reference_detects_ids(store):
    view = store.scope("initech")
    assert cross_tenant_reference("Show me MT7PJB7N5LRE", view)
    assert cross_tenant_reference("Message emp-acme-1003", view)
    assert not cross_tenant_reference("Show me DARFCP0BQM2G", view)


def test_out_of_scope_question_is_refused(store, tmp_path):
    assert agent(store, "acme", tmp_path).ask("What is the CEO's salary?")["refused"]


def test_answers_cite_real_evidence(store, tmp_path):
    res = agent(store, "acme", tmp_path).ask("Which devices at Acme are low on disk space?")
    assert "[E1]" in res["answer"] and res["grounding_problems"] == []


def test_action_pauses_for_approval_and_default_denies(store, tmp_path):
    a = agent(store, "acme", tmp_path)
    res = a.ask("Order RAM upgrades for devices that are consistently constrained")  # no approver
    assert [p["status"] for p in res["proposals"]] == ["rejected"]
    events = [e["event"] for e in AuditLog(tmp_path / "audit.jsonl").read()]
    assert events == ["proposed", "rejected"]


def test_approved_action_executes_and_is_audited(store, tmp_path):
    seen = []

    def approver(pending):
        seen.extend(pending)
        return {p["id"]: "approve" for p in pending}

    res = agent(store, "acme", tmp_path).ask("Order RAM upgrades for devices that are consistently constrained",
                                             approver)
    assert seen and seen[0]["args"]["device_id"] == "M4XVHUV1MEPZ"
    assert res["proposals"][0]["status"] == "executed"
    assert res["proposals"][0]["result"]["mock"] is True
    log = AuditLog(tmp_path / "audit.jsonl").read()
    assert [e["event"] for e in log] == ["proposed", "approved", "executed"]
    assert all(e["company_id"] == "acme-001" and e["evidence"] for e in log)


def test_blocked_action_never_reaches_approver(store, tmp_path):
    called = []
    res = agent(store, "acme", tmp_path).ask("Order a RAM upgrade for 1LYSSFD074BB",
                                             lambda pending: called.append(pending) or {})
    assert not called
    assert [p["status"] for p in res["proposals"]] == ["blocked"]


def test_offline_mode_is_deterministic(store, tmp_path):
    q = "What should I worry about this month?"
    a1 = agent(store, "initech", tmp_path).ask(q)["answer"]
    a2 = agent(store, "initech", tmp_path).ask(q)["answer"]
    assert a1 == a2


# ---------------------------------------------------------------- grounding

def test_grounding_flags_unknown_device_and_fake_citation():
    results = [{"tool": "low_disk", "args": {}, "summary": "", "rows": [{"device_id": "MT7PJB7N5LRE"}],
                "evidence": [{"device_id": "MT7PJB7N5LRE", "collected_at": "t", "field": "f", "value": 1}],
                "error": None}]
    problems = grounding_problems("ZZZZZZZZZZZ9 is low [E7]", results, [])
    assert any("ZZZZZZZZZZZ9" in p for p in problems)
    assert any("[E7]" in p for p in problems)


def test_grounding_flags_claimed_action_that_did_not_run():
    proposals = [{"status": "rejected", "args": {"device_id": "X"}}]
    assert grounding_problems("I have ordered the upgrade.", [], proposals)


def test_claude_writer_falls_back_on_refusal():
    fake = SimpleNamespace(messages=SimpleNamespace(
        create=lambda **kw: SimpleNamespace(stop_reason="refusal", content=[])))
    w = ClaudeWriter(client=fake)
    results = [{"tool": "low_disk", "args": {}, "summary": "1 device.", "rows": [], "evidence": [], "error": None}]
    assert w.write("q", results, []) == "1 device."


def test_rule_planner_routes_examples_from_brief(store):
    view, p = store.scope("acme"), RulePlanner()
    assert [s.tool for s in p.plan("Which devices at Acme are low on disk space?", view).steps] == ["low_disk"]
    assert p.plan("Show me laptops failing high-severity compliance checks.", view).steps[0].tool == \
        "compliance_failures"
    assert "os_older_than" in [s.tool for s in p.plan("How many devices are running an OS older than macOS 15?",
                                                       view).steps]


# ------------------------------------------------- Claude mode, faked client

def _fake_parse_client(plan=None, error=None):
    def parse(**kw):
        if error:
            raise error
        return SimpleNamespace(stop_reason="end_turn", parsed_output=plan)
    return SimpleNamespace(messages=SimpleNamespace(parse=parse))


def test_claude_planner_uses_model_plan_and_drops_unknown_tools(acme):
    plan = Plan(intent="question", rationale="r",
                steps=[ToolCall(tool="low_disk", args_json="{}"), ToolCall(tool="rm_rf", args_json="{}")])
    out = ClaudePlanner(client=_fake_parse_client(plan)).plan("low disk?", acme)
    assert [s.tool for s in out.steps] == ["low_disk"]


def test_claude_planner_falls_back_to_rules_on_error(acme):
    out = ClaudePlanner(client=_fake_parse_client(error=RuntimeError("network down"))).plan(
        "Which devices are low on disk space?", acme)
    assert [s.tool for s in out.steps] == ["low_disk"]


def test_policy_blocks_a_bad_plan_from_the_model(store, tmp_path):
    """Even if the LLM plans an unjustified action, policy blocks it before any human sees it."""
    plan = Plan(intent="action", rationale="model got it wrong",
                steps=[ToolCall(tool="get_device", args_json='{"device_id": "1LYSSFD074BB"}')],
                actions=[ActionRequest(action="flag_device_for_replacement", target="1LYSSFD074BB")])
    a = FleetCopilot(store, "acme", mode="claude", audit_path=tmp_path / "a.jsonl",
                     planner=ClaudePlanner(client=_fake_parse_client(plan)), writer=False)
    res = a.ask("Replace 1LYSSFD074BB", lambda pending: {p["id"]: "approve" for p in pending})
    assert [p["status"] for p in res["proposals"]] == ["blocked"]


def test_claude_mode_without_key_fails_clearly(store, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        FleetCopilot(store, "acme", mode="claude")


# -------------------------------------------------------------------- evals

def test_eval_suite_passes_offline(capsys):
    summary = evals.run_all("offline")
    assert summary["failures"] == {}, json.dumps(summary["failures"], indent=2)
    assert summary["total"] >= 30
