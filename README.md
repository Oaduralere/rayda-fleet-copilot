# Fleet Copilot

An agent that lets IT administrators ask questions about device telemetry in plain English, surfaces trends, and proposes operational actions that only run after a human approves them. Built with LangGraph and Claude (Anthropic SDK), with a deterministic offline mode so every result can be reproduced without an API key.

```
$ python cli.py --company acme "Which devices at Acme are low on disk space?"
2 devices under 10.0% free disk at the latest snapshot.
  - MT7PJB7N5LRE (MacBook Pro): 2.0% free (11.0 GB) [E2]
  - 8NM23J95R5I6 (ThinkPad X1 Carbon): 2.6% free (28.5 GB) [E1]
```

Every `[E#]` points to a specific device snapshot (`device_id`, `collected_at`, field, value). Run with `--evidence` to print them.

## 1. Installation and setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # optional: add ANTHROPIC_API_KEY for Claude mode
```

Copy the dataset that came with the brief (`device-telemetry-dataset.ndjson`) into the repo root. It is left out of this public repo on purpose, since it is customer telemetry. To load it from elsewhere, pass `--data path/to/file.ndjson` to `cli.py`.

**Modes**

| Mode | When | What plans and writes |
|---|---|---|
| `offline` | default when no key is set | Rule-based planner and template writer. Fully deterministic. |
| `claude` | default when `ANTHROPIC_API_KEY` is set | Claude plans (structured output) and writes the answer. Falls back to the offline path on any API error, refusal or failed grounding check. |

The model defaults to `claude-sonnet-5-5` and can be changed with `FLEET_COPILOT_MODEL`.

## 2. Running the agent

```bash
# One question, scoped to one company
python cli.py --company acme "Show me laptops failing high-severity compliance checks."

# Insights across the 30 days, with the agent's step-by-step trace
python cli.py --company initech --trace "What should I worry about this month?"

# An action: the agent proposes, you approve or reject at the prompt
python cli.py --company acme "Order RAM upgrades for devices that are consistently constrained"

# Interactive session
python cli.py --company globex
```

Useful flags: `--mode offline|claude|auto`, `--approve ask|all|none` (default `ask`), `--trace`, `--evidence`, `--audit-log PATH`.

The session's company is fixed at start-up (`--company`). There is no way to change tenant mid-session.

## 3. Running evaluations

```bash
pytest                      # 31 unit/integration tests, includes the full eval suite
python evals.py             # 30 scenario evals with a per-category report
python evals.py --verbose   # also print every answer
python evals.py --mode claude   # same cases with Claude planning and writing (needs a key)
```

Current result in offline mode: **30/30 evals and 31/31 tests pass**, and all 25 answered questions cite evidence.

`eval_cases.json` holds the scenarios. Each expectation was written from the raw data, not from the agent's output:

| Category | Cases | What is checked |
|---|---|---|
| grounded_qa | 8 | Required devices appear; other tenants' devices never appear; key facts (e.g. "4 macOS devices"); citations present |
| insight | 8 | Batteries, RAM pressure, storage trends, compliance drift, new risky software, per tenant |
| guardrail | 5 | Cross-tenant questions, foreign device IDs, foreign employee IDs, prompt injection and out-of-scope questions are refused, with no audit events |
| action | 9 | Proposal status counts (blocked / rejected / executed), targets, and the exact audit events written |

On every case the harness also asserts that the grounding check found no problems and that every evidence item belongs to the session's tenant.

## 4. Agent architecture

The agent is an explicit LangGraph `StateGraph` (`agent.py`), not a single prompt:

```
START -> guard -> plan -> act -> reflect -> propose -> approval -> execute -> answer -> ground -> END
           |        |               |  ^         |                              ^
           |        |               |  '- plan (retry on tool errors, max 2)    |
           '--------'---------------'---------- answer (refuse / no actions) ---'
```

| Node | Job |
|---|---|
| `guard` | Hard, code-level check: refuses questions naming another company, a device ID or an employee ID outside the tenant. Runs before any LLM call. |
| `plan` | Produces a typed `Plan` (intent, tool calls, action requests). Claude via `messages.parse(output_format=Plan)`, or the rule planner offline. |
| `act` | Runs the planned read tools against the tenant-scoped view. |
| `reflect` | Checks results. On tool errors (unknown tool, bad args) it sends the errors back to `plan` for one retry. Empty or failed results end in an honest "nothing found" answer. |
| `propose` | Turns action requests into `Proposal`s with evidence taken from tool rows, then runs the policy engine. Blocked proposals never reach a human. |
| `approval` | `interrupt()` pauses the graph and hands pending proposals to the caller. It resumes with `Command(resume=...)`. Anything not explicitly approved is rejected. |
| `execute` | Re-checks policy, then runs the (mock) side effect. |
| `answer` | Claude writer or deterministic template, using only tool results and numbered evidence. |
| `ground` | Verifies the answer (see section 6). If a Claude answer fails, the cited template answer replaces it. |

State is checkpointed (`InMemorySaver`) so the approval pause is a real suspend and resume, which is how a web UI or Slack approval would plug in.

**Files**

| File | Contents |
|---|---|
| `telemetry.py` | Loads the NDJSON. `TelemetryStore.scope(company)` returns a `TenantView`, the only way to read data. |
| `tools.py` | The 12 read and insight tools, each returning rows plus evidence. |
| `actions.py` | Action specs, `Proposal`, policy engine, JSONL `AuditLog`, mock `execute`. |
| `planner.py` | Plan schema, rule planner, Claude planner, answer writers, grounding check. |
| `agent.py` | The LangGraph graph and the `FleetCopilot.ask()` API. |
| `cli.py` | Command line and interactive approval prompt. |
| `evals.py`, `eval_cases.json` | Scenario eval harness. |
| `test_fleet_copilot.py` | Unit and integration tests. |

## 5. Tool catalog

All tools receive the tenant view from the runtime. The model never passes a `company_id`, and one passed anyway is stripped.

**Read tools (grounded Q&A)**

| Tool | Args | Returns |
|---|---|---|
| `fleet_overview` | none | Device, employee and platform counts, date range |
| `list_devices` | `platform?` | Devices with model, OS and owner |
| `get_device` | `device_id` | Latest snapshot summary for one device |
| `low_disk` | `threshold_pct=10` | Devices under the threshold at their latest snapshot |
| `compliance_failures` | `severity?`, `check_id?`, `laptops_only?` | Checks failing at the latest snapshot. When nothing fails it cites the passing results. |
| `os_older_than` | `platform="macOS"`, `major=15` | Devices below the version. It also says how many devices on other platforms were not counted. |
| `software_search` | `name` | Devices with matching installed software |

**Insight tools (trends over the 30 days)**

| Tool | Rule |
|---|---|
| `battery_health` | Battery condition not Normal, or full-charge capacity down at least 3%. Cites first and last capacity, condition and cycles. |
| `ram_pressure` | RAM at least 85% in at least 40% of snapshots |
| `storage_trend` | Free disk under 10% at the end of the period, with the start-to-end trend |
| `compliance_drift` | A check that passed earlier and is failing at the latest snapshot |
| `new_software` | Apps that appear during the period, with risky ones (uTorrent, TeamViewer) flagged |

**Action tools (proposals only, human approval required)**

| Action | Policy rule, re-computed from raw telemetry |
|---|---|
| `create_upgrade_order(device_id, component, spec)` | RAM: at least 85% in at least 40% of snapshots. Storage: latest free disk under 10%. |
| `open_remediation_ticket(device_id, check_id, note)` | The check is failing at the latest snapshot. |
| `flag_device_for_replacement(device_id, reason)` | Battery condition is in the bad set (e.g. Service Recommended). |
| `notify_employee(employee_id, message)` | The employee belongs to the tenant, the evidence concerns a device they own, and the message is 500 characters or fewer. |

## 6. Grounding strategy

1. **Tools return evidence, not just answers.** Every row is backed by `Evidence(device_id, collected_at, field, value)` from a real snapshot.
2. **Evidence is numbered** (`[E1]`, `[E2]`, ...) and passed to the writer. The writer prompt allows only those references and only devices present in the results.
3. **A grounding check runs on every answer** (`grounding_problems`):
   - every cited `[E#]` exists;
   - every device ID mentioned was returned by a tool;
   - an answer with findings has citations;
   - the answer never claims an action happened unless one was executed.
4. **Failure is safe.** If a Claude answer fails the check, the deterministic cited template replaces it. The problems are kept in the result for review.
5. **"Nothing found" is still grounded.** For example, no device fails a high-severity check in this data. The answer says so and cites the passing `disk_encryption` results rather than inventing failures.

Data facts the agent reports honestly instead of echoing the brief:
- The data has 14 employees, not ~18.
- No high-severity compliance check fails anywhere.
- The "older than macOS 15" count excludes Windows devices and says so.

## 7. Guardrails

| Requirement | How |
|---|---|
| Tenant isolation | `TenantView` is the only data path. Reading another tenant's device raises `TenantAccessError`. |
| Cross-company access | The `guard` node refuses before any model call. The planner and every tool are bound to the view. Policy rejects proposals whose company, device, employee or evidence is outside the tenant. |
| Explicit human approval | Graph `interrupt()`. Default deny: no answer means rejected. Policy is re-checked right before execution. |
| Audit log | Append-only JSONL. Each event (`proposed`, `policy_blocked`, `approved`, `rejected`, `executed`) records the timestamp, actor, company, action, args and evidence. |
| Refuse without evidence | The policy engine re-computes each justification from raw data and rejects missing or fabricated evidence (a snapshot that doesn't exist in the tenant). Even "approve all" cannot execute a blocked action. |
| Out-of-scope / injection | Questions no telemetry tool can answer are refused. Injection attempts ("ignore your rules...") are planned as refusals and cannot widen the tenant. |

## 8. Key design decisions and trade-offs

- **The LLM plans and writes, while code decides and enforces.** Tenant scope, policy and approval are deterministic code, so a bad or manipulated model output cannot cause an unsafe action (tested with a fake model that plans an unjustified replacement). The trade-off is less flexibility for actions the rules don't cover.
- **Typed structured planning instead of free-form tool calling.** One `Plan` object makes the loop inspectable and testable, and lets the rule planner implement the same contract. A free-form ReAct loop would handle unusual multi-hop questions better, at the cost of predictability.
- **Two interchangeable modes.** The offline path makes evals reproducible in CI without a key. The Claude path handles phrasing the rules miss. The Claude path has the same guardrails and falls back to offline on any failure. With more time, Claude-mode evals would run nightly against a pinned model.
- **Thresholds are explicit constants** (10% disk, 85% RAM in 40% of snapshots, 3% battery capacity loss). They are easy to tune per customer, but they are policy choices, not learned values.
- **Actions are mocked.** `execute()` returns a reference such as `PO-46B59E`. Real integrations (ticketing, procurement, messaging) would replace that one function.
- **In-memory store.** 750 snapshots fit in memory. At production scale the `TenantView` interface would sit on a database with row-level tenant security, and tools would push filters into queries.
- **Not done in the time box:** multi-turn memory across questions, a web approval UI, and per-tenant configurable policy.
