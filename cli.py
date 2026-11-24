"""Command-line entry point.

Examples:
  python cli.py --company acme "Which devices are low on disk space?"
  python cli.py --company globex --approve ask "Order RAM upgrades for constrained devices"
  python cli.py --company initech            # interactive session
"""
from __future__ import annotations

import argparse
import json
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

from agent import FleetCopilot
from telemetry import DEFAULT_DATA_PATH, TelemetryStore, TenantAccessError


def make_approver(policy: str):
    def ask(pending: list[dict]) -> dict[str, str]:
        decisions = {}
        for p in pending:
            if policy == "all":
                decisions[p["id"]] = "approve"
                continue
            if policy == "none":
                decisions[p["id"]] = "reject"
                continue
            print(f"\nApproval needed: {p['action']} {json.dumps(p['args'])}")
            print(f"  why: {p['reason']}; policy: {'; '.join(p['policy_notes'])}")
            print(f"  evidence: {len(p['evidence'])} snapshot values")
            try:
                ans = input("  Approve? [y/N] ").strip().lower()
            except EOFError:  # no human at the terminal: default deny
                ans = "n"
            decisions[p["id"]] = "approve" if ans in ("y", "yes") else "reject"
        return decisions
    return ask


def print_result(res: dict, show_trace: bool, show_evidence: bool) -> None:
    print(f"\n[{res['company_id']} | mode={res['mode']}]")
    print(res["answer"])
    if show_evidence:
        from planner import number_evidence
        print("\nEvidence:")
        for e in number_evidence(res["results"]):
            print(f"  [{e['ref']}] {e['device_id']} @ {e['collected_at']}: {e['field']} = {e['value']}")
    if show_trace:
        print("\nTrace:")
        for line in res["trace"]:
            print(f"  {line}")
    if res["grounding_problems"]:
        print("\nGrounding warnings:", *res["grounding_problems"], sep="\n  ")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Rayda Fleet Copilot")
    ap.add_argument("question", nargs="?", help="Question to ask. Omit for an interactive session.")
    ap.add_argument("--company", required=True, help="Tenant for this session: acme, globex or initech.")
    ap.add_argument("--mode", choices=["auto", "offline", "claude"], default="auto",
                    help="offline = deterministic rule planner; claude = Claude plans and writes. "
                         "auto uses claude when ANTHROPIC_API_KEY is set.")
    ap.add_argument("--approve", choices=["ask", "all", "none"], default="ask",
                    help="How to answer approval requests (default: ask on the terminal).")
    ap.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    ap.add_argument("--audit-log", default="audit_log.jsonl")
    ap.add_argument("--trace", action="store_true", help="Print the agent's step-by-step trace.")
    ap.add_argument("--evidence", action="store_true", help="Print every cited evidence item.")
    args = ap.parse_args(argv)

    store = TelemetryStore(args.data)
    try:
        agent = FleetCopilot(store, args.company, mode=args.mode, audit_path=args.audit_log)
    except (ValueError, TenantAccessError) as exc:  # unknown company, or Claude mode without a key
        print(f"error: {exc}", file=sys.stderr)
        return 2
    approver = make_approver(args.approve)

    if args.question:
        print_result(agent.ask(args.question, approver), args.trace, args.evidence)
        return 0

    print(f"Fleet Copilot for {agent.view.alias.title()} ({agent.mode} mode). Type 'exit' to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("exit", "quit"):
            break
        if q:
            print_result(agent.ask(q, approver), args.trace, args.evidence)
    return 0


if __name__ == "__main__":
    sys.exit(main())
