"""Evaluation suite.

Runs every case in eval_cases.json through the full agent graph and checks
the answer against expectations written from the raw data:
  - correctness: devices that must / must not appear, phrases that must appear
  - grounding: citations present, no grounding-check problems
  - guardrails: refusals, tenant isolation, default-deny approval
  - action quality: proposal status counts, targets and audit log events

Usage:
  python evals.py                  # offline (deterministic), no API key needed
  python evals.py --mode claude    # same cases with Claude planning and writing
  python evals.py --only acme-low-disk --verbose
Exit code is 0 only when every case passes.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from actions import AuditLog
from agent import FleetCopilot
from telemetry import DEFAULT_DATA_PATH, TelemetryStore

CASES_PATH = Path(__file__).with_name("eval_cases.json")


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def approver_for(policy: str):
    if policy == "all":
        return lambda pending: {p["id"]: "approve" for p in pending}
    if policy == "none":
        return lambda pending: {p["id"]: "reject" for p in pending}
    return None  # "default": nobody answers, so the agent must default to deny


def check_case(case: dict, res: dict, audit: list[dict]) -> list[str]:
    exp, answer, fails = case["expect"], res["answer"], []
    if res["refused"] != exp.get("refused", False):
        fails.append(f"refused={res['refused']}, expected {exp.get('refused', False)}")
    for dev in exp.get("include", []):
        if dev not in answer:
            fails.append(f"missing device {dev}")
    for dev in exp.get("exclude", []):
        if dev in answer:
            fails.append(f"leaked device {dev}")
    for phrase in exp.get("contains", []):
        if phrase.lower() not in answer.lower():
            fails.append(f"missing phrase {phrase!r}")
    if exp.get("cited") and not re.search(r"\[E\d+\]|cited snapshot values", answer):
        fails.append("no evidence citation")
    if res["grounding_problems"]:
        fails.append("grounding: " + "; ".join(res["grounding_problems"]))
    if "proposals" in exp:
        got = dict(Counter(p["status"] for p in res["proposals"]))
        if got != exp["proposals"]:
            fails.append(f"proposal statuses {got}, expected {exp['proposals']}")
    for target in exp.get("proposal_targets", []):
        if not any(target in json.dumps(p["args"]) for p in res["proposals"]):
            fails.append(f"no proposal for {target}")
    if "audit_events" in exp:
        events = sorted({e["event"] for e in audit})
        if events != sorted(exp["audit_events"]):
            fails.append(f"audit events {events}, expected {sorted(exp['audit_events'])}")
    # Every evidence item must come from the session's own tenant.
    for r in res["results"]:
        for e in r["evidence"]:
            if e["device_id"] not in res["_tenant_devices"]:
                fails.append(f"cross-tenant evidence {e['device_id']}")
    return fails


def run_case(store: TelemetryStore, case: dict, mode: str, workdir: Path) -> tuple[dict, list[dict], list[str]]:
    audit_path = workdir / f"{case['id']}.audit.jsonl"
    agent = FleetCopilot(store, case["company"], mode=mode, audit_path=audit_path)
    res = agent.ask(case["question"], approver_for(case.get("approve", "default")))
    res["_tenant_devices"] = set(agent.view.device_ids())
    audit = AuditLog(audit_path).read()
    return res, audit, check_case(case, res, audit)


def run_all(mode: str = "offline", only: str | None = None, verbose: bool = False,
            data: str | Path = DEFAULT_DATA_PATH) -> dict:
    store = TelemetryStore(data)
    cases = [c for c in load_cases() if not only or c["id"] == only]
    by_cat: dict[str, list[bool]] = defaultdict(list)
    failures: dict[str, list[str]] = {}
    cited_answers = answered = 0
    with tempfile.TemporaryDirectory() as tmp:
        for case in cases:
            res, audit, fails = run_case(store, case, mode, Path(tmp))
            ok = not fails
            by_cat[case["category"]].append(ok)
            if not res["refused"]:
                answered += 1
                cited_answers += bool(re.search(r"\[E\d+\]|cited snapshot values", res["answer"]))
            if fails:
                failures[case["id"]] = fails
            mark = "PASS" if ok else "FAIL"
            print(f"{mark}  {case['id']:<34} [{case['category']}]")
            if verbose or fails:
                for f in fails:
                    print(f"        - {f}")
            if verbose:
                print("        " + res["answer"].replace("\n", "\n        "))
    total = sum(len(v) for v in by_cat.values())
    passed = total - len(failures)
    print("\nResults by category:")
    for cat, oks in sorted(by_cat.items()):
        print(f"  {cat:<12} {sum(oks)}/{len(oks)}")
    print(f"  {'citation rate':<12} {cited_answers}/{answered} answered questions cite evidence")
    print(f"\n{passed}/{total} cases passed (mode={mode})")
    return {"total": total, "passed": passed, "failures": failures}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the Fleet Copilot eval suite")
    ap.add_argument("--mode", choices=["offline", "claude", "auto"], default="offline")
    ap.add_argument("--only", help="Run one case by id")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    summary = run_all(args.mode, args.only, args.verbose)
    return 0 if not summary["failures"] else 1


if __name__ == "__main__":
    sys.exit(main())
