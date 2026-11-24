"""State-changing actions: proposals, policy checks, approval and audit log.

The agent can only *propose* an action. A proposal is executed only if:
  1. the policy engine accepts it (tenant ownership + evidence that is
     re-checked against the raw telemetry, not taken on the LLM's word), and
  2. a human explicitly approves it.
Every step (proposed, blocked, approved, rejected, executed) is appended to a
JSONL audit log.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from telemetry import TenantView
from tools import BATTERY_BAD_CONDITIONS, LOW_DISK_PCT, RAM_PRESSURE_PCT, RAM_CONSISTENT_SHARE

ACTION_SPECS: dict[str, dict] = {
    "create_upgrade_order": {
        "args": ["device_id", "component", "spec"],
        "description": "Order a RAM or storage upgrade for a device that is consistently constrained.",
    },
    "open_remediation_ticket": {
        "args": ["device_id", "check_id", "note"],
        "description": "Open an IT ticket to fix a compliance check that is failing right now.",
    },
    "flag_device_for_replacement": {
        "args": ["device_id", "reason"],
        "description": "Flag a device for replacement when its battery is at end of life.",
    },
    "notify_employee": {
        "args": ["employee_id", "message"],
        "description": "Send a message to the employee who owns a device with an open issue.",
    },
}


@dataclass
class Proposal:
    action: str
    args: dict
    company_id: str
    reason: str
    evidence: list[dict] = field(default_factory=list)
    status: str = "proposed"          # proposed | blocked | approved | rejected | executed
    policy_notes: list[str] = field(default_factory=list)
    result: dict | None = None

    @property
    def id(self) -> str:
        key = json.dumps([self.company_id, self.action, self.args], sort_keys=True)
        return "act_" + hashlib.sha256(key.encode()).hexdigest()[:10]

    def as_dict(self) -> dict:
        return {"id": self.id, "action": self.action, "args": self.args, "company_id": self.company_id,
                "reason": self.reason, "evidence": self.evidence, "status": self.status,
                "policy_notes": self.policy_notes, "result": self.result}


# ---------------------------------------------------------------------------
# Policy engine
# ---------------------------------------------------------------------------

def _evidence_is_real(view: TenantView, ev: dict) -> bool:
    """Evidence must point at a snapshot that exists inside THIS tenant."""
    return view.find_snapshot(ev.get("device_id", ""), ev.get("collected_at", "")) is not None


def check_policy(view: TenantView, p: Proposal) -> tuple[bool, list[str]]:
    """Return (allowed, reasons). Recomputes the justification from raw data."""
    notes: list[str] = []
    if p.action not in ACTION_SPECS:
        return False, [f"Unknown action {p.action!r}."]
    if p.company_id != view.company_id:
        return False, ["Proposal belongs to a different company than this session."]
    missing = [a for a in ACTION_SPECS[p.action]["args"] if not str(p.args.get(a, "")).strip()]
    if missing:
        return False, [f"Missing arguments: {', '.join(missing)}."]
    if not p.evidence:
        return False, ["No evidence attached. Actions need cited telemetry."]
    if not all(_evidence_is_real(view, e) for e in p.evidence):
        return False, ["Some cited evidence does not exist in this company's telemetry."]

    if p.action == "notify_employee":
        emp = p.args["employee_id"]
        if not view.owns_employee(emp):
            return False, [f"Employee {emp} is not in this company."]
        if len(p.args["message"]) > 500:
            return False, ["Message is longer than 500 characters."]
        own_devices = set(view.employees()[emp])
        if not any(e["device_id"] in own_devices for e in p.evidence):
            return False, ["Evidence is not about a device this employee owns."]
        return True, ["Employee owns a device with a cited issue."]

    device_id = p.args["device_id"]
    if not view.owns_device(device_id):
        return False, [f"Device {device_id} is not in this company."]
    snaps = view.snapshots(device_id)
    last = snaps[-1]

    if p.action == "create_upgrade_order":
        comp = p.args["component"].lower()
        if comp == "ram":
            share = sum(s.ram_used_pct >= RAM_PRESSURE_PCT for s in snaps) / len(snaps)
            if share < RAM_CONSISTENT_SHARE:
                return False, [f"RAM was over {RAM_PRESSURE_PCT}% in only {share:.0%} of snapshots; "
                               f"policy needs {RAM_CONSISTENT_SHARE:.0%}."]
            notes.append(f"RAM over {RAM_PRESSURE_PCT}% in {share:.0%} of snapshots.")
        elif comp in ("storage", "disk", "ssd"):
            if last.disk_free_pct >= LOW_DISK_PCT:
                return False, [f"Latest free disk is {last.disk_free_pct}%, not under {LOW_DISK_PCT}%."]
            notes.append(f"Latest free disk {last.disk_free_pct}%.")
        else:
            return False, [f"Component {comp!r} is not upgradeable here (ram or storage only)."]
        return True, notes

    if p.action == "open_remediation_ticket":
        check = p.args["check_id"]
        failing = {c["check_id"] for c in last.failing_checks()}
        if check not in failing:
            return False, [f"Check {check} is not failing on the latest snapshot."]
        return True, [f"{check} is failing on the latest snapshot ({last.collected_at})."]

    if p.action == "flag_device_for_replacement":
        batt = next((s.battery for s in reversed(snaps) if s.battery), None)
        if not batt or batt["condition"] not in BATTERY_BAD_CONDITIONS:
            return False, ["Battery condition does not justify replacement."]
        return True, [f"Battery condition is {batt['condition']} at {batt['cycle_count']} cycles."]

    return False, ["No policy rule matched."]


# ---------------------------------------------------------------------------
# Audit log + mock execution
# ---------------------------------------------------------------------------

class AuditLog:
    """Append-only JSONL file. One line per event."""

    def __init__(self, path: str | Path = "audit_log.jsonl"):
        self.path = Path(path)

    def write(self, event: str, proposal: Proposal, actor: str = "agent", detail: Any = None) -> dict:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": event,
            "actor": actor,
            "company_id": proposal.company_id,
            "action_id": proposal.id,
            "action": proposal.action,
            "args": proposal.args,
            "evidence": proposal.evidence,
            "detail": detail,
        }
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        return entry

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]


def execute(p: Proposal) -> dict:
    """Mock side effect. A real system would call the ticketing/procurement API here."""
    if p.status != "approved":
        raise PermissionError("Only approved proposals can be executed.")
    prefix = {"create_upgrade_order": "PO", "open_remediation_ticket": "TICKET",
              "flag_device_for_replacement": "REPL", "notify_employee": "MSG"}[p.action]
    return {"reference": f"{prefix}-{p.id[-6:].upper()}", "mock": True}
