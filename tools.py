"""Read and insight tools the agent can call.

Every tool:
  * takes a TenantView as its first argument (injected by the runtime, never
    chosen by the LLM), so it physically cannot see another company's data;
  * returns a ToolResult with `rows` (the answer data) and `evidence`
    (device_id + snapshot timestamp + field + value for every claim).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from telemetry import Evidence, TenantView

# Thresholds live in one place so the README, tools and evals agree.
LOW_DISK_PCT = 10.0          # free space below this = "low on disk"
RAM_PRESSURE_PCT = 85.0      # used RAM above this = "constrained" for that snapshot
RAM_CONSISTENT_SHARE = 0.4   # share of snapshots above the line = "consistently"
BATTERY_BAD_CONDITIONS = {"Service Recommended", "Replace Soon", "Replace Now", "Poor"}
RISKY_SOFTWARE = {"uTorrent": "peer-to-peer file sharing", "TeamViewer": "remote-access tool"}


def _n(count: int, word: str) -> str:
    """'1 device', '2 devices'."""
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


@dataclass
class ToolResult:
    tool: str
    args: dict
    summary: str
    rows: list[dict] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "tool": self.tool,
            "args": self.args,
            "summary": self.summary,
            "rows": self.rows,
            "evidence": [e.as_dict() for e in self.evidence],
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Read tools
# ---------------------------------------------------------------------------

def fleet_overview(view: TenantView) -> ToolResult:
    latest = view.latest_all()
    start, end = view.date_range()
    platforms: dict[str, int] = {}
    for s in latest:
        platforms[s.platform] = platforms.get(s.platform, 0) + 1
    rows = [{
        "company_id": view.company_id,
        "devices": len(latest),
        "employees": len(view.employees()),
        "platforms": platforms,
        "first_snapshot": start,
        "last_snapshot": end,
    }]
    ev = [s.evidence("model_name", s.model_name) for s in latest]
    return ToolResult("fleet_overview", {}, f"{_n(len(latest), 'device')}, {_n(len(view.employees()), 'employee')}, {start[:10]} to {end[:10]}.", rows, ev)


def list_devices(view: TenantView, platform: str | None = None) -> ToolResult:
    rows, ev = [], []
    for s in view.latest_all():
        if platform and s.platform.lower() != platform.lower():
            continue
        rows.append({"device_id": s.device_id, "model": s.model_name, "platform": s.platform,
                     "os_version": s.os_version, "employee_id": s.employee_id})
        ev.append(s.evidence("os.product_version", s.os_version))
    return ToolResult("list_devices", {"platform": platform}, f"{_n(len(rows), 'device')}.", rows, ev)


def get_device(view: TenantView, device_id: str) -> ToolResult:
    if not view.owns_device(device_id):
        return ToolResult("get_device", {"device_id": device_id}, "Device not found in this company.",
                          error="not_found")
    s = view.latest(device_id)
    b = s.battery
    row = {
        "device_id": s.device_id, "model": s.model_name, "platform": s.platform,
        "os_version": s.os_version, "employee_id": s.employee_id,
        "ram_used_pct": s.ram_used_pct, "disk_free_pct": s.disk_free_pct,
        "battery_condition": b["condition"] if b else None,
        "battery_cycles": b["cycle_count"] if b else None,
        "failing_checks": [c["check_id"] for c in s.failing_checks()],
        "software": sorted(s.software),
    }
    ev = [s.evidence("disk_free_pct", s.disk_free_pct), s.evidence("ram_used_pct", s.ram_used_pct)]
    return ToolResult("get_device", {"device_id": device_id}, f"Latest snapshot of {device_id}.", [row], ev)


def low_disk(view: TenantView, threshold_pct: float = LOW_DISK_PCT) -> ToolResult:
    rows, ev = [], []
    for s in view.latest_all():
        if s.disk_free_pct < threshold_pct:
            rows.append({"device_id": s.device_id, "model": s.model_name,
                         "disk_free_pct": s.disk_free_pct, "disk_free_gb": s.disk_free_gb,
                         "employee_id": s.employee_id})
            ev.append(s.evidence("disk_free_pct", s.disk_free_pct))
    rows.sort(key=lambda r: r["disk_free_pct"])
    return ToolResult("low_disk", {"threshold_pct": threshold_pct},
                      f"{_n(len(rows), 'device')} under {threshold_pct}% free disk at the latest snapshot.", rows, ev)


def compliance_failures(view: TenantView, severity: str | None = None, check_id: str | None = None,
                        laptops_only: bool = False) -> ToolResult:
    rows, ev = [], []
    checks_seen: dict[str, str] = {}
    for s in view.latest_all():
        for c in s.compliance:
            checks_seen[c["check_id"]] = c["severity"]
        if laptops_only and not s.is_laptop:
            continue
        for c in s.failing_checks(severity):
            if check_id and c["check_id"] != check_id:
                continue
            rows.append({"device_id": s.device_id, "check_id": c["check_id"], "severity": c["severity"],
                         "model": s.model_name})
            ev.append(s.evidence(f"compliance.{c['check_id']}", c["status"]))
    args = {"severity": severity, "check_id": check_id, "laptops_only": laptops_only}
    summary = f"{_n(len(rows), 'failing check result')} at the latest snapshot."
    if not rows and severity:
        same = sorted(k for k, v in checks_seen.items() if v == severity)
        summary = (f"No devices fail a {severity}-severity check. The {severity}-severity checks in this "
                   f"data are: {', '.join(same) or 'none'}, and they pass on every device's latest snapshot.")
        # Cite the passing results so "none" is evidence-backed too.
        for s in view.latest_all():
            for c in s.compliance:
                if c["severity"] == severity:
                    ev.append(s.evidence(f"compliance.{c['check_id']}", c["status"]))
    return ToolResult("compliance_failures", args, summary, rows, ev)


def os_older_than(view: TenantView, platform: str = "macOS", major: int = 15) -> ToolResult:
    rows, ev, excluded = [], [], 0
    for s in view.latest_all():
        if s.platform.lower() != platform.lower():
            excluded += 1
            continue
        if s.os_major is not None and s.os_major < major:
            rows.append({"device_id": s.device_id, "model": s.model_name, "os_version": s.os_version})
            ev.append(s.evidence("os.product_version", s.os_version))
    return ToolResult("os_older_than", {"platform": platform, "major": major},
                      f"{_n(len(rows), platform + ' device')} on a version older than {major}. "
                      f"{_n(excluded, 'device')} on other platforms not counted.", rows, ev)


def software_search(view: TenantView, name: str) -> ToolResult:
    rows, ev = [], []
    for s in view.latest_all():
        hits = [n for n in s.software if name.lower() in n.lower()]
        for n in hits:
            rows.append({"device_id": s.device_id, "software": n, "employee_id": s.employee_id})
            ev.append(s.evidence("installed_software", n))
    summary = f"{_n(len(rows), 'install')} matching '{name}'." if rows else f"No device has software matching '{name}' at its latest snapshot."
    return ToolResult("software_search", {"name": name}, summary, rows, ev)


# ---------------------------------------------------------------------------
# Insight / trend tools (look across all 30 days, not just the latest snapshot)
# ---------------------------------------------------------------------------

def battery_health(view: TenantView) -> ToolResult:
    rows, ev, skipped = [], [], []
    for d in view.device_ids():
        snaps = [s for s in view.snapshots(d) if s.battery]
        if not snaps:
            skipped.append(d)  # desktops (Mac mini) report no battery
            continue
        first, last = snaps[0], snaps[-1]
        cap0, cap1 = first.battery["full_charge_capacity"], last.battery["full_charge_capacity"]
        drop = round(100.0 * (cap0 - cap1) / cap0, 1) if cap0 else 0.0
        cond = last.battery["condition"]
        if cond in BATTERY_BAD_CONDITIONS or drop >= 3.0:
            rows.append({"device_id": d, "condition": cond, "cycle_count": last.battery["cycle_count"],
                         "capacity_first": cap0, "capacity_last": cap1, "capacity_drop_pct": drop,
                         "employee_id": last.employee_id})
            ev += [first.evidence("battery.full_charge_capacity", cap0),
                   last.evidence("battery.full_charge_capacity", cap1),
                   last.evidence("battery.condition", cond),
                   last.evidence("battery.cycle_count", last.battery["cycle_count"])]
    rows.sort(key=lambda r: -r["capacity_drop_pct"])
    if not rows:  # a "nothing found" answer still cites what was checked
        for d in view.device_ids():
            snaps = [s for s in view.snapshots(d) if s.battery]
            if snaps:
                ev.append(snaps[-1].evidence("battery.condition", snaps[-1].battery["condition"]))
    head = (f"{_n(len(rows), 'battery').replace('batterys', 'batteries')} approaching end of life."
            if rows else "No batteries are approaching end of life.")
    if skipped:
        head += f" {_n(len(skipped), 'device')} reported no battery data at all."
    return ToolResult("battery_health", {}, head, rows, ev)


def ram_pressure(view: TenantView, threshold_pct: float = RAM_PRESSURE_PCT,
                 min_share: float = RAM_CONSISTENT_SHARE) -> ToolResult:
    rows, ev = [], []
    for d in view.device_ids():
        snaps = view.snapshots(d)
        hot = [s for s in snaps if s.ram_used_pct >= threshold_pct]
        share = len(hot) / len(snaps)
        if share >= min_share:
            avg = round(sum(s.ram_used_pct for s in snaps) / len(snaps), 1)
            rows.append({"device_id": d, "snapshots_over_threshold": len(hot), "snapshots_total": len(snaps),
                         "avg_ram_used_pct": avg, "model": snaps[-1].model_name,
                         "total_memory": snaps[-1].raw["device_identity"]["total_memory"]})
            ev += [hot[0].evidence("ram_used_pct", hot[0].ram_used_pct),
                   hot[-1].evidence("ram_used_pct", hot[-1].ram_used_pct)]
    rows.sort(key=lambda r: -r["snapshots_over_threshold"])
    return ToolResult("ram_pressure", {"threshold_pct": threshold_pct, "min_share": min_share},
                      f"{_n(len(rows), 'device')} used {threshold_pct}%+ RAM in at least {int(min_share*100)}% of snapshots.",
                      rows, ev)


def storage_trend(view: TenantView, threshold_pct: float = LOW_DISK_PCT) -> ToolResult:
    rows, ev = [], []
    for d in view.device_ids():
        snaps = view.snapshots(d)
        first, last = snaps[0], snaps[-1]
        low_share = sum(1 for s in snaps if s.disk_free_pct < threshold_pct) / len(snaps)
        if last.disk_free_pct < threshold_pct:
            rows.append({"device_id": d, "free_pct_first": first.disk_free_pct, "free_pct_last": last.disk_free_pct,
                         "change_pts": round(last.disk_free_pct - first.disk_free_pct, 1),
                         "share_of_snapshots_low": round(low_share, 2)})
            ev += [first.evidence("disk_free_pct", first.disk_free_pct),
                   last.evidence("disk_free_pct", last.disk_free_pct)]
    rows.sort(key=lambda r: r["free_pct_last"])
    return ToolResult("storage_trend", {"threshold_pct": threshold_pct},
                      f"{_n(len(rows), 'device')} end the period under {threshold_pct}% free disk.", rows, ev)


def compliance_drift(view: TenantView) -> ToolResult:
    """Checks that passed earlier in the window and are failing at the end."""
    rows, ev = [], []
    for d in view.device_ids():
        snaps = view.snapshots(d)
        check_ids = {c["check_id"] for c in snaps[0].compliance}
        for cid in sorted(check_ids):
            statuses = [(s, next(c["status"] for c in s.compliance if c["check_id"] == cid)) for s in snaps]
            if statuses[0][1] == "pass" and statuses[-1][1] != "pass":
                first_fail = next(s for s, st in statuses if st != "pass")
                rows.append({"device_id": d, "check_id": cid, "passed_until": statuses[0][0].collected_at[:10],
                             "first_failed": first_fail.collected_at[:10]})
                ev += [statuses[0][0].evidence(f"compliance.{cid}", "pass"),
                       first_fail.evidence(f"compliance.{cid}", "fail"),
                       statuses[-1][0].evidence(f"compliance.{cid}", statuses[-1][1])]
    return ToolResult("compliance_drift", {}, f"{_n(len(rows), 'check')} drifted from pass to fail and stayed failing.", rows, ev)


def new_software(view: TenantView) -> ToolResult:
    rows, ev = [], []
    for d in view.device_ids():
        snaps = view.snapshots(d)
        baseline = snaps[0].software
        seen: set[str] = set()
        for s in snaps[1:]:
            for name in sorted(s.software - baseline - seen):
                seen.add(name)
                rows.append({"device_id": d, "software": name, "first_seen": s.collected_at[:10],
                             "risk": RISKY_SOFTWARE.get(name, "")})
                ev.append(s.evidence("installed_software", name))
    return ToolResult("new_software", {}, f"{_n(len(rows), 'app')} newly installed during the period.", rows, ev)


# ---------------------------------------------------------------------------
# Registry: what the planner is allowed to call, with descriptions for the LLM
# ---------------------------------------------------------------------------

READ_TOOLS: dict[str, tuple[Callable[..., ToolResult], str]] = {
    "fleet_overview": (fleet_overview, "Device, employee and platform counts and the date range. No args."),
    "list_devices": (list_devices, "List devices. Optional arg: platform ('macOS' or 'Windows')."),
    "get_device": (get_device, "Latest details for one device. Arg: device_id."),
    "low_disk": (low_disk, f"Devices under threshold_pct free disk at latest snapshot (default {LOW_DISK_PCT})."),
    "compliance_failures": (compliance_failures, "Failing compliance checks at latest snapshot. Optional args: "
                            "severity ('high'|'medium'), check_id, laptops_only (bool)."),
    "os_older_than": (os_older_than, "Devices on an OS major version below `major`. Args: platform, major."),
    "software_search": (software_search, "Devices with installed software matching `name`."),
    "battery_health": (battery_health, "Batteries approaching end of life: bad condition or capacity loss. No args."),
    "ram_pressure": (ram_pressure, "Devices consistently over 85% RAM across the 30 days. No args needed."),
    "storage_trend": (storage_trend, "Devices consistently/increasingly low on storage over time. No args needed."),
    "compliance_drift": (compliance_drift, "Compliance checks that went from pass to fail over time. No args."),
    "new_software": (new_software, "Apps installed during the period, flagging risky ones. No args."),
}


def run_tool(view: TenantView, name: str, args: dict | None = None) -> ToolResult:
    """Execute a tool by name. The TenantView is injected here, never by the LLM."""
    args = dict(args or {})
    args.pop("company_id", None)  # a planner can't redirect a tool to another tenant
    if name not in READ_TOOLS:
        return ToolResult(name, args, f"Unknown tool {name!r}.", error="unknown_tool")
    fn = READ_TOOLS[name][0]
    try:
        return fn(view, **args)
    except TypeError as exc:
        return ToolResult(name, args, f"Bad arguments: {exc}", error="bad_args")


def tool_catalog_text() -> str:
    return "\n".join(f"- {n}: {d}" for n, (_, d) in READ_TOOLS.items())
