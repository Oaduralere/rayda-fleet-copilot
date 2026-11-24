"""The Fleet Copilot agent: an explicit LangGraph loop.

    guard -> plan -> act -> reflect --(problems, retry)--> plan
                              |
                              +--> propose -> policy -> approval (human, interrupt) -> execute
                              |                                                          |
                              +-------------------------------------------------------> answer -> ground -> END

The LLM (or the rule planner) only decides WHICH tools/actions to use.
Data access, tenant scoping, policy checks, approval and audit are all code.
"""
from __future__ import annotations

import os
import uuid
from typing import Any, Callable, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from actions import AuditLog, Proposal, check_policy, execute
from planner import (ClaudePlanner, ClaudeWriter, Plan, RulePlanner, cross_tenant_reference,
                     grounding_problems, template_answer)
from telemetry import TelemetryStore, TenantView
from tools import run_tool

MAX_PLAN_ATTEMPTS = 2


class AgentState(TypedDict, total=False):
    question: str
    plan: dict
    attempts: int
    feedback: str
    results: list[dict]
    proposals: list[dict]
    refusal: str
    answer: str
    grounding: list[str]
    trace: list[str]


def _trace(state: AgentState, line: str) -> list[str]:
    return list(state.get("trace", [])) + [line]


def _rows_by_device(results: list[dict], tool: str) -> dict[str, dict]:
    out = {}
    for r in results:
        if r["tool"] == tool:
            for row in r["rows"]:
                out.setdefault(row["device_id"], row)
    return out


def _evidence_for(results: list[dict], device_ids: set[str]) -> list[dict]:
    seen, out = set(), []
    for r in results:
        for e in r["evidence"]:
            key = (e["device_id"], e["collected_at"], e["field"])
            if e["device_id"] in device_ids and key not in seen:
                seen.add(key)
                out.append(e)
    return out


def build_proposals(plan: Plan, results: list[dict], view: TenantView) -> list[Proposal]:
    """Turn the planner's action requests into concrete, evidence-backed proposals."""
    proposals: list[Proposal] = []
    cid = view.company_id
    for req in plan.actions:
        target = req.target.strip()
        if req.action == "create_upgrade_order":
            component = "ram" if req.check_id.lower() in ("ram", "memory") else "storage"
            rows = _rows_by_device(results, "ram_pressure" if component == "ram" else "storage_trend")
            devices = [target] if target else sorted(rows)
            for d in devices:
                row = rows.get(d, {})
                if component == "ram":
                    spec = f"Next RAM tier above {row.get('total_memory', 'current')}"
                    reason = (f"RAM over 85% in {row['snapshots_over_threshold']}/{row['snapshots_total']} snapshots"
                              if row else "Requested by administrator")
                else:
                    spec = "Larger SSD (at least 2x current capacity)"
                    reason = (f"Free disk fell to {row['free_pct_last']}%" if row else "Requested by administrator")
                proposals.append(Proposal(req.action, {"device_id": d, "component": component, "spec": spec},
                                          cid, reason, _evidence_for(results, {d})))
        elif req.action == "flag_device_for_replacement":
            rows = _rows_by_device(results, "battery_health")
            devices = [target] if target else sorted(d for d, r in rows.items() if r["condition"] != "Normal")
            for d in devices:
                row = rows.get(d)
                reason = (f"Battery {row['condition']}, {row['cycle_count']} cycles, capacity down "
                          f"{row['capacity_drop_pct']}% in 30 days" if row else "Requested by administrator")
                proposals.append(Proposal(req.action, {"device_id": d, "reason": reason}, cid, reason,
                                          _evidence_for(results, {d})))
        elif req.action == "open_remediation_ticket":
            pairs = sorted({(row["device_id"], row["check_id"]) for r in results
                            if r["tool"] == "compliance_failures" for row in r["rows"]})
            if req.check_id:
                pairs = [p for p in pairs if p[1] == req.check_id]
            if target:
                pairs = [p for p in pairs if p[0] == target] or [(target, req.check_id or "unknown")]
            for d, check in pairs:
                note = f"{check} is failing on {d}. Please remediate and confirm on the next check-in."
                proposals.append(Proposal(req.action, {"device_id": d, "check_id": check, "note": note}, cid,
                                          f"{check} failing", _evidence_for(results, {d})))
        elif req.action == "notify_employee":
            affected: dict[str, list[str]] = {}
            emp_of = {d: e for e, ds in view.employees().items() for d in ds}
            for r in results:
                for row in r["rows"]:
                    d = row.get("device_id")
                    if d and d in emp_of:
                        affected.setdefault(emp_of[d], [])
                        if d not in affected[emp_of[d]]:
                            affected[emp_of[d]].append(d)
            if target:
                affected = {target: affected.get(target, [])} if target.startswith("emp-") else {
                    emp_of[target]: [target]} if target in emp_of else {}
            for emp, devs in sorted(affected.items()):
                msg = (f"Hi, IT noticed an issue on your device(s) {', '.join(devs)} that needs attention. "
                       f"We'll be in touch with next steps.") if devs else "Hi, IT would like to check in with you."
                proposals.append(Proposal(req.action, {"employee_id": emp, "message": msg}, cid,
                                          "Device issue found", _evidence_for(results, set(devs))))
    # one proposal per (action, args)
    uniq: dict[str, Proposal] = {}
    for p in proposals:
        uniq.setdefault(p.id, p)
    return list(uniq.values())


class FleetCopilot:
    """Session bound to ONE company. Build one per tenant."""

    def __init__(self, store: TelemetryStore, company: str, mode: str = "auto",
                 audit_path: str = "audit_log.jsonl", actor: str = "it-admin",
                 planner=None, writer=None):
        self.view = store.scope(company)
        if mode == "auto":
            mode = "claude" if os.environ.get("ANTHROPIC_API_KEY") else "offline"
        if mode == "claude" and planner is None and not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("Claude mode needs ANTHROPIC_API_KEY (see .env.example). "
                             "Use --mode offline to run without a key.")
        self.mode = mode
        # planner/writer can be injected (tests use a fake Claude client).
        self.planner = planner or (ClaudePlanner() if mode == "claude" else RulePlanner())
        self.writer = writer if writer is not None else (ClaudeWriter() if mode == "claude" else None)
        self.audit = AuditLog(audit_path)
        self.actor = actor
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ nodes
    def _guard(self, state: AgentState) -> AgentState:
        why = cross_tenant_reference(state["question"], self.view)
        if why:
            return {"refusal": why, "trace": _trace(state, f"guard: refused ({why})")}
        return {"attempts": 0, "trace": _trace(state, f"guard: ok, tenant={self.view.company_id}")}

    def _plan(self, state: AgentState) -> AgentState:
        plan = self.planner.plan(state["question"], self.view, state.get("feedback", ""))
        tools = ", ".join(s.tool for s in plan.steps) or "none"
        upd: AgentState = {"plan": plan.model_dump(), "attempts": state.get("attempts", 0) + 1,
                           "trace": _trace(state, f"plan[{self.planner.name}]: intent={plan.intent}, tools={tools}, "
                                                  f"actions={[a.action for a in plan.actions]}")}
        if plan.intent == "refuse":
            upd["refusal"] = plan.refusal_reason or "That is outside what the telemetry can answer."
        return upd

    def _act(self, state: AgentState) -> AgentState:
        plan = Plan(**state["plan"])
        results = []
        for step in plan.steps:
            res = run_tool(self.view, step.tool, step.args).as_dict()
            results.append(res)
        lines = []
        for r in results:
            line = f"act: {r['tool']}({r['args']}) -> {len(r['rows'])} rows, {len(r['evidence'])} evidence"
            if r["error"]:
                line += f" (error: {r['error']})"
            lines.append(line)
        return {"results": results, "trace": list(state.get("trace", [])) + lines}

    def _reflect(self, state: AgentState) -> AgentState:
        errors = [f"{r['tool']}: {r['summary']}" for r in state["results"] if r.get("error")]
        if errors and state.get("attempts", 0) < MAX_PLAN_ATTEMPTS:
            return {"feedback": "\n".join(errors), "trace": _trace(state, f"reflect: retry ({len(errors)} errors)")}
        ok = [r for r in state["results"] if not r.get("error")]
        return {"results": ok, "feedback": "", "trace": _trace(state, "reflect: results usable")}

    def _route_after_reflect(self, state: AgentState) -> str:
        if state.get("feedback"):
            return "plan"
        if not state.get("results"):
            return "answer"
        return "propose" if state["plan"]["actions"] else "answer"

    def _propose(self, state: AgentState) -> AgentState:
        plan = Plan(**state["plan"])
        proposals = build_proposals(plan, state["results"], self.view)
        out = []
        for p in proposals:
            allowed, notes = check_policy(self.view, p)
            p.policy_notes = notes
            p.status = "proposed" if allowed else "blocked"
            self.audit.write("proposed" if allowed else "policy_blocked", p, actor="agent", detail=notes)
            out.append(p.as_dict())
        summary = f"propose: {sum(p['status'] == 'proposed' for p in out)} allowed, " \
                  f"{sum(p['status'] == 'blocked' for p in out)} blocked by policy"
        return {"proposals": out, "trace": _trace(state, summary)}

    def _route_after_propose(self, state: AgentState) -> str:
        return "approval" if any(p["status"] == "proposed" for p in state["proposals"]) else "answer"

    def _approval(self, state: AgentState) -> AgentState:
        pending = [p for p in state["proposals"] if p["status"] == "proposed"]
        # Pauses the graph. The caller resumes with {"decisions": {action_id: "approve" | "reject"}}.
        # (Wrapped so that "nobody answered" is still a non-empty resume value.)
        resumed = interrupt({"type": "approval_required", "proposals": pending})
        decisions: dict = (resumed or {}).get("decisions", {}) if isinstance(resumed, dict) else {}
        updated = []
        for p in state["proposals"]:
            if p["status"] == "proposed":
                approved = str(decisions.get(p["id"], "reject")).lower() == "approve"  # default deny
                p = {**p, "status": "approved" if approved else "rejected"}
                prop = Proposal(p["action"], p["args"], p["company_id"], p["reason"], p["evidence"])
                self.audit.write(p["status"], prop, actor=self.actor)
            updated.append(p)
        n = sum(p["status"] == "approved" for p in updated)
        return {"proposals": updated, "trace": _trace(state, f"approval: {n} approved by {self.actor}")}

    def _execute(self, state: AgentState) -> AgentState:
        updated = []
        for p in state["proposals"]:
            if p["status"] == "approved":
                prop = Proposal(p["action"], p["args"], p["company_id"], p["reason"], p["evidence"],
                                status="approved")
                # Re-check policy right before acting (data could have changed since proposal).
                allowed, notes = check_policy(self.view, prop)
                if not allowed:
                    p = {**p, "status": "blocked", "policy_notes": notes}
                    self.audit.write("policy_blocked", prop, actor="agent", detail=notes)
                else:
                    result = execute(prop)
                    p = {**p, "status": "executed", "result": result}
                    self.audit.write("executed", prop, actor="agent", detail=result)
            updated.append(p)
        return {"proposals": updated, "trace": _trace(state, "execute: done")}

    def _answer(self, state: AgentState) -> AgentState:
        results, proposals = state.get("results", []), state.get("proposals", [])
        refusal = state.get("refusal", "")
        if self.writer:
            text = self.writer.write(state["question"], results, proposals, refusal)
        else:
            text = template_answer(state["question"], results, proposals, refusal)
        return {"answer": text, "trace": _trace(state, "answer: written")}

    def _ground(self, state: AgentState) -> AgentState:
        problems = grounding_problems(state["answer"], state.get("results", []), state.get("proposals", []),
                                      state.get("refusal", ""))
        if problems and self.writer:
            # The LLM answer failed the check: fall back to the deterministic, cited answer.
            text = template_answer(state["question"], state.get("results", []), state.get("proposals", []),
                                   state.get("refusal", ""))
            return {"answer": text, "grounding": problems,
                    "trace": _trace(state, f"ground: {len(problems)} problems, used template answer")}
        return {"grounding": problems, "trace": _trace(state, f"ground: {len(problems)} problems")}

    # ------------------------------------------------------------------ graph
    def _build_graph(self):
        g = StateGraph(AgentState)
        for name, fn in [("guard", self._guard), ("plan", self._plan), ("act", self._act),
                         ("reflect", self._reflect), ("propose", self._propose), ("approval", self._approval),
                         ("execute", self._execute), ("answer", self._answer), ("ground", self._ground)]:
            g.add_node(name, fn)
        g.add_edge(START, "guard")
        g.add_conditional_edges("guard", lambda s: "answer" if s.get("refusal") else "plan",
                                {"answer": "answer", "plan": "plan"})
        g.add_conditional_edges("plan", lambda s: "answer" if s.get("refusal") else "act",
                                {"answer": "answer", "act": "act"})
        g.add_edge("act", "reflect")
        g.add_conditional_edges("reflect", self._route_after_reflect,
                                {"plan": "plan", "propose": "propose", "answer": "answer"})
        g.add_conditional_edges("propose", self._route_after_propose,
                                {"approval": "approval", "answer": "answer"})
        g.add_edge("approval", "execute")
        g.add_edge("execute", "answer")
        g.add_edge("answer", "ground")
        g.add_edge("ground", END)
        return g.compile(checkpointer=InMemorySaver())

    # ------------------------------------------------------------------ API
    def ask(self, question: str,
            approver: Callable[[list[dict]], dict[str, str]] | None = None) -> dict[str, Any]:
        """Run one question. `approver` gets pending proposals and returns decisions.
        With no approver, every proposal is rejected (default deny)."""
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        state = self.graph.invoke({"question": question, "trace": []}, config)
        for _ in range(3):  # one approval round per question; the cap is a safety net
            if "__interrupt__" not in state:
                break
            pending = state["__interrupt__"][0].value["proposals"]
            decisions = approver(pending) if approver else {}
            state = self.graph.invoke(Command(resume={"decisions": decisions or {}}), config)
        return {
            "company_id": self.view.company_id,
            "question": question,
            "mode": self.mode,
            "answer": state.get("answer", ""),
            "refused": bool(state.get("refusal")),
            "plan": state.get("plan"),
            "results": state.get("results", []),
            "proposals": state.get("proposals", []),
            "grounding_problems": state.get("grounding", []),
            "trace": state.get("trace", []),
        }
