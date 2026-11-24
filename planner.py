"""Planning and answer writing.

Two interchangeable planners produce the same `Plan` object:
  * RulePlanner   - deterministic keyword routing. Used for offline runs, CI
                    and the eval suite, and as the fallback if Claude fails.
  * ClaudePlanner - asks Claude for a structured Plan (validated with Pydantic).

Two interchangeable writers turn tool results into an answer:
  * template_answer - deterministic, always cites evidence.
  * ClaudeWriter    - natural-language answer; must pass the grounding check
                      or we fall back to the template answer.
"""
from __future__ import annotations

import json
import os
import re
from typing import Literal

from pydantic import BaseModel, Field

from actions import ACTION_SPECS
from telemetry import TenantView
from tools import READ_TOOLS, tool_catalog_text

DEVICE_ID_RE = re.compile(r"\b[A-Z0-9]{12}\b")
EMPLOYEE_ID_RE = re.compile(r"\bemp-[a-z]+-\d+\b")
DEFAULT_MODEL = os.environ.get("FLEET_COPILOT_MODEL", "claude-sonnet-5-5")


class ToolCall(BaseModel):
    tool: str
    args_json: str = Field(default="{}", description="Tool arguments as a JSON object string.")

    @property
    def args(self) -> dict:
        try:
            val = json.loads(self.args_json or "{}")
            return val if isinstance(val, dict) else {}
        except json.JSONDecodeError:
            return {}


class ActionRequest(BaseModel):
    action: str
    target: str = Field(default="", description="device_id or employee_id the user named, or '' for all affected.")
    check_id: str = Field(default="", description="For remediation tickets: which check, or '' for any failing.")


class Plan(BaseModel):
    intent: Literal["question", "insight", "action", "refuse"]
    rationale: str
    steps: list[ToolCall] = Field(default_factory=list)
    actions: list[ActionRequest] = Field(default_factory=list)
    refusal_reason: str = ""


# ---------------------------------------------------------------------------
# Deterministic planner
# ---------------------------------------------------------------------------

def _tc(tool: str, **args) -> ToolCall:
    return ToolCall(tool=tool, args_json=json.dumps(args))


def cross_tenant_reference(question: str, view: TenantView) -> str | None:
    """Return why the question reaches outside this tenant, or None."""
    q = question.lower()
    for alias in view.other_company_aliases:
        if re.search(rf"\b{re.escape(alias)}\b", q):
            return f"That asks about another company ({alias.title()}). This session can only see {view.alias.title()} data."
    for dev in DEVICE_ID_RE.findall(question):
        if not view.owns_device(dev) and any(c.isdigit() for c in dev):
            return f"Device {dev} is not in {view.alias.title()}'s fleet, so I can't look it up or act on it."
    for emp in EMPLOYEE_ID_RE.findall(question):
        if not view.owns_employee(emp):
            return f"Employee {emp} is not in {view.alias.title()}, so I can't look them up or contact them."
    return None


class RulePlanner:
    name = "rules"

    def plan(self, question: str, view: TenantView, feedback: str = "") -> Plan:
        q = question.lower()
        why = cross_tenant_reference(question, view)
        if why:
            return Plan(intent="refuse", rationale="cross-tenant", refusal_reason=why)

        steps: list[ToolCall] = []
        actions: list[ActionRequest] = []
        devices = [d for d in DEVICE_ID_RE.findall(question) if view.owns_device(d)]
        employees = [e for e in EMPLOYEE_ID_RE.findall(question) if view.owns_employee(e)]
        target = (devices or employees or [""])[0]
        trend = any(w in q for w in ("consistent", "trend", "over time", "running out", "persist", "drift",
                                     "started failing", "getting worse", "pattern"))

        # --- actions the admin asked for --------------------------------
        wants_action = any(w in q for w in ("order", "upgrade", "ticket", "remediat", "replace", "flag ",
                                            "notify", "message", "tell ", "email", "fix "))
        if wants_action:
            if "upgrade" in q or "order" in q:
                if "ram" in q or "memory" in q:
                    steps.append(_tc("ram_pressure"))
                    actions.append(ActionRequest(action="create_upgrade_order", target=target, check_id="ram"))
                else:
                    steps.append(_tc("storage_trend"))
                    actions.append(ActionRequest(action="create_upgrade_order", target=target, check_id="storage"))
            if "replace" in q or "flag" in q:
                steps.append(_tc("battery_health"))
                actions.append(ActionRequest(action="flag_device_for_replacement", target=target))
            if "ticket" in q or "remediat" in q or "fix " in q:
                check = "screen_lock" if "screen" in q else ("os_up_to_date" if "os" in q.split() or "update" in q
                                                            else ("disk_encryption" if "encrypt" in q else ""))
                steps.append(_tc("compliance_failures", check_id=check or None))
                actions.append(ActionRequest(action="open_remediation_ticket", target=target, check_id=check))
            if any(w in q for w in ("notify", "message", "tell ", "email")):
                if not steps:
                    steps.append(_tc("storage_trend"))
                actions.append(ActionRequest(action="notify_employee", target=target))
            if devices:
                steps.insert(0, _tc("get_device", device_id=devices[0]))
            return Plan(intent="action", rationale="admin asked for an action", steps=steps, actions=actions)

        # --- questions and insights --------------------------------------
        if devices:
            steps.append(_tc("get_device", device_id=devices[0]))
        if any(w in q for w in ("disk", "storage", "space")):
            steps.append(_tc("storage_trend") if trend else _tc("low_disk"))
        if re.search(r"high[- ]severity|critical", q):
            steps.append(_tc("compliance_failures", severity="high", laptops_only="laptop" in q))
        elif any(w in q for w in ("compliance", "complian", "failing", "check", "screen lock", "encrypt")):
            steps.append(_tc("compliance_drift") if trend else _tc("compliance_failures"))
        m = re.search(r"(?:older than|before|below|earlier than)\s+(macos|windows)?\s*(\d+)", q)
        if m:
            platform = "Windows" if (m.group(1) == "windows") else "macOS"
            steps.append(_tc("os_older_than", platform=platform, major=int(m.group(2))))
        if "battery" in q or "batteries" in q:
            steps.append(_tc("battery_health"))
        if re.search(r"\bram\b|memory", q):
            steps.append(_tc("ram_pressure"))
        for name in ("utorrent", "teamviewer", "cleanmymac", "zoom", "slack", "chrome", "1password"):
            if name in q:
                steps.append(_tc("software_search", name=name))
        if any(w in q for w in ("new software", "installed", "new app", "risky software", "unapproved")):
            steps.append(_tc("new_software"))
        if any(w in q for w in ("insight", "worry", "attention", "risks", "trends", "health check",
                                "what should i", "anything wrong", "summary of issues")):
            steps += [_tc("battery_health"), _tc("ram_pressure"), _tc("storage_trend"),
                      _tc("compliance_drift"), _tc("new_software")]
        if any(w in q for w in ("how many devices", "how many employees", "overview", "fleet size",
                                "list devices", "list all devices", "what devices")):
            steps.insert(0, _tc("fleet_overview" if "list" not in q else "list_devices"))

        # de-duplicate, keep order
        seen, uniq = set(), []
        for s in steps:
            key = (s.tool, s.args_json)
            if key not in seen:
                seen.add(key)
                uniq.append(s)
        if not uniq:
            return Plan(intent="refuse", rationale="no tool covers this",
                        refusal_reason="I can only answer from device telemetry (hardware, OS, memory, disk, "
                                       "battery, installed software and compliance checks), and none of it "
                                       "covers that question.")
        intent = "insight" if trend or len(uniq) > 2 else "question"
        return Plan(intent=intent, rationale="keyword routing", steps=uniq)


# ---------------------------------------------------------------------------
# Claude planner
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = """You are the planning step of a Fleet Copilot for IT administrators.
You never answer the question yourself. You choose which read tools to call, and
which actions to propose, and return a Plan.

Rules:
- Only use the tools listed below. Pass arguments as a JSON object string in args_json.
- The session is bound to one company. Never try to access another company's data.
  If the question asks about another company, set intent="refuse" with a reason.
- If no tool can answer the question from telemetry, set intent="refuse" and explain.
- Only add actions when the administrator asks for an action. Allowed actions:
{actions}
- For an action, also add the read tools whose results justify it.

Read tools:
{tools}
"""


class ClaudePlanner:
    name = "claude"

    def __init__(self, model: str = DEFAULT_MODEL, client=None):
        import anthropic  # imported lazily so offline mode needs no SDK setup

        self.client = client or anthropic.Anthropic()
        self.model = model
        self.fallback = RulePlanner()

    def plan(self, question: str, view: TenantView, feedback: str = "") -> Plan:
        why = cross_tenant_reference(question, view)  # hard rule, not left to the model
        if why:
            return Plan(intent="refuse", rationale="cross-tenant", refusal_reason=why)
        system = PLANNER_SYSTEM.format(
            actions="\n".join(f"  - {k}: {v['description']}" for k, v in ACTION_SPECS.items()),
            tools=tool_catalog_text(),
        )
        user = f"Company: {view.alias.title()}\nQuestion: {question}"
        if feedback:
            user += f"\n\nYour previous plan had problems, fix them:\n{feedback}"
        try:
            resp = self.client.messages.parse(
                model=self.model,
                max_tokens=4000,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=Plan,
            )
            if resp.stop_reason == "refusal" or resp.parsed_output is None:
                return self.fallback.plan(question, view, feedback)
            plan: Plan = resp.parsed_output
            plan.steps = [s for s in plan.steps if s.tool in READ_TOOLS]
            plan.actions = [a for a in plan.actions if a.action in ACTION_SPECS]
            return plan
        except Exception:  # network, auth or schema errors: stay useful, stay deterministic
            return self.fallback.plan(question, view, feedback)


# ---------------------------------------------------------------------------
# Answer writing + grounding check
# ---------------------------------------------------------------------------

def number_evidence(results: list[dict]) -> list[dict]:
    """Flatten evidence across tool results and give each item a stable [E#] id."""
    out, seen = [], set()
    for r in results:
        for e in r["evidence"]:
            key = (e["device_id"], e["collected_at"], e["field"], str(e["value"]))
            if key in seen:
                continue
            seen.add(key)
            out.append({"ref": f"E{len(out) + 1}", **e})
    return out


def _refs_for(device_id: str, evidence: list[dict], limit: int = 3) -> str:
    refs = [e["ref"] for e in evidence if e["device_id"] == device_id][:limit]
    return "".join(f"[{r}]" for r in refs)


def _refs_for_result(result: dict, evidence: list[dict]) -> str:
    """Citation for a row that summarises many devices (e.g. fleet_overview)."""
    keys = {(e["device_id"], e["collected_at"], e["field"], str(e["value"])) for e in result["evidence"]}
    refs = [e["ref"] for e in evidence if (e["device_id"], e["collected_at"], e["field"], str(e["value"])) in keys]
    if not refs:
        return ""
    return f"[{refs[0]}]" if len(refs) == 1 else f"[{refs[0]}] to [{refs[-1]}]"


def _row_text(tool: str, row: dict) -> str:
    d = row.get("device_id", "")
    if tool in ("low_disk",):
        return f"{d} ({row['model']}): {row['disk_free_pct']}% free ({row['disk_free_gb']} GB)"
    if tool == "storage_trend":
        return f"{d}: free disk went from {row['free_pct_first']}% to {row['free_pct_last']}%"
    if tool == "compliance_failures":
        return f"{d} ({row['model']}): {row['check_id']} failing ({row['severity']})"
    if tool == "os_older_than":
        return f"{d} ({row['model']}): {row['os_version']}"
    if tool == "battery_health":
        return (f"{d}: {row['condition']}, {row['cycle_count']} cycles, capacity down "
                f"{row['capacity_drop_pct']}% ({row['capacity_first']} to {row['capacity_last']} mAh)")
    if tool == "ram_pressure":
        return (f"{d} ({row['model']}, {row['total_memory']}): over 85% RAM in {row['snapshots_over_threshold']} "
                f"of {row['snapshots_total']} snapshots, average {row['avg_ram_used_pct']}%")
    if tool == "compliance_drift":
        return f"{d}: {row['check_id']} passed on {row['passed_until']}, failing since {row['first_failed']}"
    if tool == "new_software":
        risk = f" ({row['risk']})" if row.get("risk") else ""
        return f"{d}: {row['software']} installed {row['first_seen']}{risk}"
    if tool == "software_search":
        return f"{d}: {row['software']}"
    if tool == "get_device":
        return (f"{d} ({row['model']}, {row['platform']} {row['os_version']}): {row['disk_free_pct']}% disk free, "
                f"{row['ram_used_pct']}% RAM used, battery {row['battery_condition'] or 'n/a'}, "
                f"failing checks: {', '.join(row['failing_checks']) or 'none'}")
    if tool == "list_devices":
        return f"{d} ({row['model']}, {row['platform']} {row['os_version']}), owner {row['employee_id']}"
    if tool == "fleet_overview":
        plats = ", ".join(f"{k} {v}" for k, v in row["platforms"].items())
        return (f"{row['devices']} devices ({plats}), {row['employees']} employees, "
                f"snapshots {row['first_snapshot'][:10]} to {row['last_snapshot'][:10]}")
    return json.dumps(row)


def template_answer(question: str, results: list[dict], proposals: list[dict], refusal: str = "") -> str:
    if refusal:
        return f"I can't help with that: {refusal}"
    evidence = number_evidence(results)
    lines: list[str] = []
    for r in results:
        if r.get("error"):
            lines.append(f"- {r['tool']}: {r['summary']}")
            continue
        lines.append(r["summary"])
        for row in r["rows"]:
            d = row.get("device_id", "")
            cite = _refs_for(d, evidence) if d else _refs_for_result(r, evidence)
            lines.append(f"  - {_row_text(r['tool'], row)} {cite}".rstrip())
        if not r["rows"] and r["evidence"]:
            lines.append(f"  (based on {len(r['evidence'])} cited snapshot values)")
    if proposals:
        lines.append("")
        lines.append("Proposed actions:")
        for p in proposals:
            target = p["args"].get("device_id") or p["args"].get("employee_id")
            lines.append(f"  - [{p['status']}] {p['action']} for {target}: {'; '.join(p['policy_notes'])}")
    return "\n".join(lines).strip()


WRITER_SYSTEM = """You write the final answer for a Fleet Copilot used by IT administrators.
Use ONLY the tool results provided. Every claim about a device must cite evidence
with its [E#] reference. Never mention a device that is not in the results.
If the results are empty, say so plainly. Do not invent numbers. Be brief:
a one-line answer, then one bullet per finding. If actions were proposed,
list them with their status; never say an action was done unless status is 'executed'."""


class ClaudeWriter:
    def __init__(self, model: str = DEFAULT_MODEL, client=None):
        import anthropic

        self.client = client or anthropic.Anthropic()
        self.model = model

    def write(self, question: str, results: list[dict], proposals: list[dict], refusal: str = "") -> str:
        if refusal:
            return template_answer(question, results, proposals, refusal)
        payload = {
            "question": question,
            "tool_results": [{k: r[k] for k in ("tool", "summary", "rows", "error")} for r in results],
            "evidence": number_evidence(results),
            "proposed_actions": proposals,
        }
        try:
            resp = self.client.messages.create(
                model=self.model, max_tokens=4000, system=WRITER_SYSTEM,
                messages=[{"role": "user", "content": json.dumps(payload)}],
            )
            if resp.stop_reason == "refusal":
                return template_answer(question, results, proposals)
            return "".join(b.text for b in resp.content if b.type == "text").strip()
        except Exception:
            return template_answer(question, results, proposals)


def grounding_problems(answer: str, results: list[dict], proposals: list[dict], refusal: str = "") -> list[str]:
    """Checks the answer only uses known devices and real evidence references."""
    if refusal:
        return []  # a refusal makes no claims about the fleet
    problems = []
    evidence = number_evidence(results)
    refs = {e["ref"] for e in evidence}
    known_devices = {e["device_id"] for e in evidence} | {
        p["args"].get("device_id") for p in proposals if p["args"].get("device_id")}
    for r in results:
        known_devices |= {row.get("device_id") for row in r["rows"] if row.get("device_id")}
    for dev in set(DEVICE_ID_RE.findall(answer)):
        if dev not in known_devices and any(c.isdigit() for c in dev):
            problems.append(f"Mentions device {dev} that no tool returned.")
    for ref in set(re.findall(r"\[(E\d+)\]", answer)):
        if ref not in refs:
            problems.append(f"Cites [{ref}] which does not exist.")
    has_rows = any(r["rows"] for r in results)
    if has_rows and not re.search(r"\[E\d+\]", answer):
        problems.append("Answer has findings but no evidence citations.")
    if re.search(r"\b(done|completed|executed|ordered|sent)\b", answer.lower()) and proposals and not any(
            p["status"] == "executed" for p in proposals):
        problems.append("Claims an action happened but none was executed.")
    return problems
