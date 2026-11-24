"""Telemetry data layer.

Loads the NDJSON snapshots once and hands out *tenant-scoped* read-only views.
Nothing outside this module can read another company's devices: every query
goes through a TenantView that is bound to one company_id at creation time.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_DATA_PATH = Path(__file__).parent / "device-telemetry-dataset.ndjson"


class TenantAccessError(PermissionError):
    """Raised when code tries to read a device or employee outside its tenant."""


@dataclass(frozen=True)
class Evidence:
    """One cited fact: which device, which snapshot, which field, what value."""

    device_id: str
    collected_at: str
    field: str
    value: Any

    def as_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "collected_at": self.collected_at,
            "field": self.field,
            "value": self.value,
        }


class Snapshot:
    """A single point-in-time device report, with derived helper fields."""

    def __init__(self, raw: dict):
        self.raw = raw
        self.device_id: str = raw["device_id"]
        self.company_id: str = raw["company_id"]
        self.employee_id: str = raw["employee_id"]
        self.collected_at: str = raw["collected_at"]

    # --- identity / OS -------------------------------------------------
    @property
    def platform(self) -> str:
        return self.raw["os"]["product_name"]  # "macOS" or "Windows"

    @property
    def os_version(self) -> str:
        return self.raw["os"]["product_version"]

    @property
    def os_major(self) -> int | None:
        try:
            return int(self.os_version.split(".")[0].split()[0])
        except (ValueError, IndexError):
            return None

    @property
    def model_name(self) -> str:
        return self.raw["device_identity"]["model_name"]

    @property
    def hostname(self) -> str:
        return self.raw["os"].get("hostname", "")

    @property
    def is_laptop(self) -> bool:
        return self.model_name not in {"Mac mini"}

    # --- memory / disk -------------------------------------------------
    @property
    def ram_used_pct(self) -> float:
        mem = self.raw["memory"]
        return round(100.0 * mem["used_memory_bytes"] / mem["total_memory_bytes"], 1)

    @property
    def disk_free_pct(self) -> float:
        vol = self.raw["disk_volumes"][0]  # every device reports one system volume
        return round(100.0 * vol["available_bytes"] / vol["size_bytes"], 1)

    @property
    def disk_free_gb(self) -> float:
        return round(self.raw["disk_volumes"][0]["available_bytes"] / 1e9, 1)

    # --- battery -------------------------------------------------------
    @property
    def battery(self) -> dict | None:
        b = self.raw.get("battery")
        if not b or not b.get("battery_present", True):
            return None
        return b

    # --- compliance / software -----------------------------------------
    @property
    def compliance(self) -> list[dict]:
        return self.raw["compliance_results"]

    def failing_checks(self, severity: str | None = None) -> list[dict]:
        return [
            c
            for c in self.compliance
            if c["status"] != "pass" and (severity is None or c["severity"] == severity)
        ]

    @property
    def software(self) -> set[str]:
        return {s["name"] for s in self.raw["installed_software"]}

    def evidence(self, field: str, value: Any) -> Evidence:
        return Evidence(self.device_id, self.collected_at, field, value)


def company_alias(company_id: str) -> str:
    """'acme-001' -> 'acme' (the name an admin would type)."""
    return company_id.split("-")[0].lower()


class TelemetryStore:
    """Holds every snapshot. Only hands data out through `scope()`."""

    def __init__(self, path: str | Path = DEFAULT_DATA_PATH):
        self._by_company: dict[str, dict[str, list[Snapshot]]] = defaultdict(lambda: defaultdict(list))
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                snap = Snapshot(json.loads(line))
                self._by_company[snap.company_id][snap.device_id].append(snap)
        for devices in self._by_company.values():
            for snaps in devices.values():
                snaps.sort(key=lambda s: s.collected_at)

    @property
    def company_ids(self) -> list[str]:
        return sorted(self._by_company)

    def resolve_company(self, name_or_id: str) -> str:
        """Accept 'acme', 'Acme' or 'acme-001'. Raises if unknown."""
        key = name_or_id.strip().lower()
        for cid in self._by_company:
            if key in (cid, company_alias(cid)):
                return cid
        raise TenantAccessError(f"Unknown company: {name_or_id!r}")

    def scope(self, company: str) -> "TenantView":
        cid = self.resolve_company(company)
        return TenantView(cid, self._by_company[cid], other_aliases={
            company_alias(c) for c in self._by_company if c != cid
        })


class TenantView:
    """Read-only view of ONE company's telemetry. The only door to the data."""

    def __init__(self, company_id: str, devices: dict[str, list[Snapshot]], other_aliases: set[str]):
        self.company_id = company_id
        self.alias = company_alias(company_id)
        self._devices = devices
        self.other_company_aliases = frozenset(other_aliases)

    # --- devices ---------------------------------------------------------
    def device_ids(self) -> list[str]:
        return sorted(self._devices)

    def owns_device(self, device_id: str) -> bool:
        return device_id in self._devices

    def snapshots(self, device_id: str) -> list[Snapshot]:
        if device_id not in self._devices:
            raise TenantAccessError(f"Device {device_id} is not in tenant {self.company_id}")
        return self._devices[device_id]

    def latest(self, device_id: str) -> Snapshot:
        return self.snapshots(device_id)[-1]

    def latest_all(self) -> list[Snapshot]:
        return [self.latest(d) for d in self.device_ids()]

    def find_snapshot(self, device_id: str, collected_at: str) -> Snapshot | None:
        if not self.owns_device(device_id):
            return None
        for s in self._devices[device_id]:
            if s.collected_at == collected_at:
                return s
        return None

    # --- employees -------------------------------------------------------
    def employees(self) -> dict[str, list[str]]:
        """employee_id -> device_ids (based on each device's latest snapshot)."""
        out: dict[str, list[str]] = defaultdict(list)
        for s in self.latest_all():
            out[s.employee_id].append(s.device_id)
        return dict(out)

    def owns_employee(self, employee_id: str) -> bool:
        return employee_id in self.employees()

    def date_range(self) -> tuple[str, str]:
        all_ts = [s.collected_at for snaps in self._devices.values() for s in snaps]
        return min(all_ts), max(all_ts)
