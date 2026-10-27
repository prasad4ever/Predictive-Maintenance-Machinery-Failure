"""Build the hourly tag extract and the 24-hour failure labels."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from data.catalog import (
    COMPONENTS,
    DAYS,
    FAILURE_PRIORITIES,
    HORIZON_HOURS,
    MACHINES,
    NUMERIC_FEATURES,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TELEMETRY_PATH = DATA_DIR / "telemetry.csv"
FAILURES_PATH = DATA_DIR / "failures.csv"
START = pd.Timestamp("2026-01-01", tz="UTC")

PRIORITY_GAIN = {"P4": 0.14, "P3": 0.32, "P2": 0.52, "P1": 1.05}
WORST = {"OK": 0, "P4": 1, "P3": 2, "P2": 3, "P1": 4}

RAISED = {
    "drive_end_bearing": ("bearing_temp_c", "vibration_rms_mms", "vibration_kurtosis", "vibration_crest", "spectrum_10_mms", "spectrum_11_mms", "spectrum_12_mms"),
    "non_drive_bearing": ("bearing_temp_c", "vibration_rms_mms", "vibration_radial_mms", "spectrum_08_mms", "spectrum_09_mms"),
    "mechanical_seal": ("seal_leak_ppm", "acoustic_db"),
    "motor_winding": ("winding_temp_c", "motor_current_a", "power_kw"),
    "rotor": ("vibration_radial_mms", "vibration_rms_mms", "spectrum_04_mms", "spectrum_05_mms"),
    "lube_oil": ("oil_temp_c", "bearing_temp_c"),
    "coupling": ("vibration_axial_mms", "vibration_rms_mms", "spectrum_03_mms"),
    "fuel_system": ("fuel_gas_flow_kgh", "exhaust_temp_c", "power_kw", "spectrum_06_mms"),
}
LOWERED = {
    "mechanical_seal": ("suction_pressure_kpa", "flow_m3h"),
    "lube_oil": ("oil_pressure_kpa",),
    "fuel_system": ("fuel_gas_pressure_kpa",),
    "rotor": ("flow_m3h",),
}


def main() -> None:
    rng = np.random.default_rng(42)
    events = _schedule(rng)
    rows = _rows(rng, events)
    frame = pd.DataFrame(rows)
    frame = _trends(frame)
    frame = frame.dropna(
        subset=[
            "vibration_rms_slope_24h",
            "bearing_temp_slope_24h",
            "power_slope_24h",
            "seal_leak_slope_24h",
            "oil_pressure_slope_24h",
            "fuel_flow_slope_24h",
            "winding_temp_slope_24h",
        ]
    ).reset_index(drop=True)
    keep = [
        "timestamp",
        "hour_index",
        "day_index",
        "machine_id",
        "machine_name",
        "area",
        "component_id",
        "component_name",
        "priority",
        *NUMERIC_FEATURES,
    ]
    frame = frame[keep]
    TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TELEMETRY_PATH, index=False)
    failure_rows = []
    for event in events:
        failure_rows.append(
            {
                "timestamp": (START + pd.Timedelta(hours=event["hour"])).isoformat(),
                "machine_id": event["machine_id"],
                "component_id": event["component_id"],
                "priority": event["priority"],
            }
        )
    pd.DataFrame(failure_rows).sort_values("timestamp").to_csv(FAILURES_PATH, index=False)
    counts = frame["priority"].value_counts().to_dict()
    print(f"Wrote {len(frame)} hourly component rows to {TELEMETRY_PATH}")
    print(f"Wrote {len(failure_rows)} failure records to {FAILURES_PATH}")
    print("Label counts:", {name: int(counts.get(name, 0)) for name in ("OK", *FAILURE_PRIORITIES)})


def _schedule(_rng: np.random.Generator) -> list[dict[str, object]]:
    machines = [str(machine["machine_id"]) for machine in MACHINES if not machine["fuel"]]
    common = [str(component["component_id"]) for component in COMPONENTS if component["component_id"] != "fuel_system"]
    priorities = ("P1", "P2", "P3", "P4")
    planned: list[tuple[str, str, int, str]] = []
    for index, component_id in enumerate(common):
        for offset, priority in enumerate(priorities):
            planned.append(
                (
                    machines[(index * 2 + offset) % len(machines)],
                    component_id,
                    24 * (6 + index * 3) + 6 + offset * 5,
                    priority,
                )
            )
    planned.extend(
        (
            ("GT-210", "fuel_system", 24 * 10 + 8, "P1"),
            ("GT-210", "fuel_system", 24 * 20 + 8, "P2"),
            ("GT-210", "fuel_system", 24 * 27 + 8, "P4"),
            ("K-410", "mechanical_seal", 24 * 34 + 10, "P1"),
            ("P-305", "motor_winding", 24 * 35 + 12, "P1"),
            ("C-101", "lube_oil", 24 * 36 + 8, "P2"),
            ("P-201", "drive_end_bearing", 24 * 37 + 14, "P2"),
            ("P-512", "rotor", 24 * 33 + 9, "P3"),
            ("GT-210", "fuel_system", 24 * 34 + 18, "P3"),
            ("K-410", "coupling", 24 * 38 + 6, "P4"),
            ("P-201", "non_drive_bearing", 24 * 38 + 16, "P4"),
            ("P-305", "lube_oil", 24 * 42 + 11, "P1"),
            ("P-512", "non_drive_bearing", 24 * 44 + 9, "P1"),
            ("C-101", "mechanical_seal", 24 * 43 + 15, "P2"),
            ("GT-210", "rotor", 24 * 45 + 7, "P2"),
            ("K-410", "motor_winding", 24 * 42 + 20, "P3"),
            ("P-201", "coupling", 24 * 45 + 12, "P4"),
            ("P-201", "mechanical_seal", DAYS * 24 + 6, "P1"),
            ("C-101", "drive_end_bearing", DAYS * 24 + 10, "P2"),
            ("GT-210", "fuel_system", DAYS * 24 + 14, "P3"),
            ("K-410", "lube_oil", DAYS * 24 + 18, "P4"),
        )
    )
    return [
        {
            "machine_id": machine_id,
            "component_id": component_id,
            "hour": hour,
            "priority": priority,
        }
        for machine_id, component_id, hour, priority in planned
    ]


def _lookup(events: list[dict[str, object]]) -> dict[tuple[str, str], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for event in events:
        key = (str(event["machine_id"]), str(event["component_id"]))
        grouped.setdefault(key, []).append(event)
    return grouped


def _label_and_gain(
    grouped: dict[tuple[str, str], list[dict[str, object]]],
    machine_id: str,
    component_id: str,
    hour_index: int,
) -> tuple[str, float]:
    best = "OK"
    gain = 0.0
    best_rank = 0
    nearest = 10**9
    for event in grouped.get((machine_id, component_id), []):
        hours_until = int(event["hour"]) - hour_index
        if hours_until <= 0 or hours_until > HORIZON_HOURS:
            continue
        priority = str(event["priority"])
        rank = WORST[priority]
        if rank < best_rank or (rank == best_rank and hours_until >= nearest):
            continue
        best = priority
        best_rank = rank
        nearest = hours_until
        closeness = (HORIZON_HOURS - hours_until) / HORIZON_HOURS
        gain = PRIORITY_GAIN[priority] * (0.8 + 0.2 * closeness)
    return best, gain


def _machine_voltage(power_kw: float) -> float:
    return 6600.0 if power_kw >= 1000 else 400.0


def _rows(rng: np.random.Generator, events: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = _lookup(events)
    total_hours = DAYS * 24
    rows: list[dict[str, object]] = []
    for machine in MACHINES:
        voltage = _machine_voltage(float(machine["power_kw"]))
        current_mean = float(machine["power_kw"]) * 1000.0 / (math.sqrt(3.0) * voltage * 0.88 * 0.94)
        for component in COMPONENTS:
            component_id = str(component["component_id"])
            if component_id == "fuel_system" and not machine["fuel"]:
                continue
            for hour_index in range(total_hours):
                priority, gain = _label_and_gain(grouped, machine["machine_id"], component_id, hour_index)
                values = _healthy(rng, machine, component_id, current_mean, voltage, hour_index)
                if gain > 0:
                    _degrade(rng, values, component_id, gain)
                elif rng.random() < 0.035:
                    values["vibration_rms_mms"] *= float(rng.uniform(1.2, 1.55))
                    values["vibration_kurtosis"] += float(rng.uniform(0.2, 0.8))
                values.update(
                    {
                        "timestamp": (START + pd.Timedelta(hours=hour_index)).isoformat(),
                        "hour_index": hour_index,
                        "day_index": hour_index // 24,
                        "machine_id": machine["machine_id"],
                        "machine_name": machine["machine_name"],
                        "area": machine["area"],
                        "component_id": component_id,
                        "component_name": component["component_name"],
                        "priority": priority,
                        "hour_of_day": hour_index % 24,
                        "fuel_system_present": 1.0 if machine["fuel"] else 0.0,
                    }
                )
                rows.append(values)
    return rows


def _healthy(
    rng: np.random.Generator,
    machine: dict[str, object],
    component_id: str,
    current_mean: float,
    voltage: float,
    hour_index: int,
) -> dict[str, float]:
    def draw(mean: float, relative: float, low: float | None = None) -> float:
        value = float(rng.normal(mean, abs(mean) * relative + 1e-6))
        if low is not None:
            return max(low, value)
        return value

    cycle = 1.0 + 0.015 * math.sin(2 * math.pi * (hour_index % 24) / 24)
    power = draw(float(machine["power_kw"]) * cycle, 0.02, 0)
    flow = draw(float(machine["flow_m3h"]) * cycle, 0.02, 0)
    suction = draw(float(machine["suction_kpa"]), 0.015, 0)
    differential = draw(float(machine["differential_kpa"]), 0.02, 0)
    vibration = draw(float(machine["vibration_mms"]), 0.05, 0.05)
    bearing = draw(68 if "bearing" in component_id else 62, 0.015, 0)
    seal = draw(8 if component_id == "mechanical_seal" else 3, 0.08, 0)
    fuel_flow = draw(1650, 0.02, 0) if machine["fuel"] and component_id == "fuel_system" else draw(0.2, 0.2, 0)
    fuel_pressure = draw(2100, 0.015, 0) if machine["fuel"] and component_id == "fuel_system" else draw(0.2, 0.2, 0)
    exhaust = draw(545, 0.01, 0) if machine["fuel"] else draw(35, 0.02, 0)
    spectrum = [draw(vibration / 12, 0.08, 0) for _ in range(12)]
    if component_id == "drive_end_bearing":
        spectrum[9] *= 1.4
        spectrum[10] *= 1.6
        spectrum[11] *= 1.3
    return {
        "power_kw": power,
        "motor_current_a": max(0.0, current_mean * (power / float(machine["power_kw"])) * float(rng.normal(1, 0.01))),
        "motor_voltage_v": draw(voltage, 0.005, 0),
        "shaft_speed_rpm": draw(float(machine["speed_rpm"]), 0.004, 0),
        "suction_pressure_kpa": suction,
        "discharge_pressure_kpa": suction + differential,
        "flow_m3h": flow,
        "inlet_temp_c": draw(42 if not machine["fuel"] else 28, 0.02, -20),
        "outlet_temp_c": draw(96 if machine["machine_id"].startswith("C") or machine["machine_id"].startswith("K") else 48, 0.02, -20),
        "vibration_rms_mms": vibration,
        "vibration_axial_mms": draw(vibration * 0.45, 0.06, 0),
        "vibration_radial_mms": draw(vibration * 0.8, 0.05, 0),
        "vibration_kurtosis": draw(3.1, 0.04, 1.5),
        "vibration_crest": draw(3.3, 0.03, 1.5),
        "bearing_temp_c": bearing,
        "oil_temp_c": draw(54, 0.02, 0),
        "oil_pressure_kpa": draw(220 if component_id == "lube_oil" else 190, 0.03, 0),
        "winding_temp_c": draw(78 if component_id == "motor_winding" else 70, 0.02, 0),
        "seal_leak_ppm": seal,
        "acoustic_db": draw(82, 0.01, 40),
        "fuel_gas_flow_kgh": fuel_flow,
        "fuel_gas_pressure_kpa": fuel_pressure,
        "exhaust_temp_c": exhaust,
        "spectrum_01_mms": spectrum[0],
        "spectrum_02_mms": spectrum[1],
        "spectrum_03_mms": spectrum[2],
        "spectrum_04_mms": spectrum[3],
        "spectrum_05_mms": spectrum[4],
        "spectrum_06_mms": spectrum[5],
        "spectrum_07_mms": spectrum[6],
        "spectrum_08_mms": spectrum[7],
        "spectrum_09_mms": spectrum[8],
        "spectrum_10_mms": spectrum[9],
        "spectrum_11_mms": spectrum[10],
        "spectrum_12_mms": spectrum[11],
    }


def _degrade(rng: np.random.Generator, values: dict[str, float], component_id: str, gain: float) -> None:
    for name in RAISED.get(component_id, ()):
        values[name] = max(0.0, values[name] * (1.0 + gain) * float(rng.normal(1.0, 0.02)))
    for name in LOWERED.get(component_id, ()):
        values[name] = max(0.0, values[name] * (1.0 - min(0.75, 0.55 * gain)) * float(rng.normal(1.0, 0.02)))


def _trends(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["machine_id", "component_id", "hour_index"])
    pieces = []
    for _, group in frame.groupby(["machine_id", "component_id"], sort=False):
        group = group.copy()
        group["vibration_rms_mean_24h"] = group["vibration_rms_mms"].rolling(24, min_periods=12).mean()
        group["vibration_rms_slope_24h"] = (group["vibration_rms_mms"] - group["vibration_rms_mms"].shift(23)) / 23
        group["bearing_temp_mean_24h"] = group["bearing_temp_c"].rolling(24, min_periods=12).mean()
        group["bearing_temp_slope_24h"] = (group["bearing_temp_c"] - group["bearing_temp_c"].shift(23)) / 23
        group["power_mean_24h"] = group["power_kw"].rolling(24, min_periods=12).mean()
        group["power_slope_24h"] = (group["power_kw"] - group["power_kw"].shift(23)) / 23
        group["seal_leak_mean_24h"] = group["seal_leak_ppm"].rolling(24, min_periods=12).mean()
        group["seal_leak_slope_24h"] = (group["seal_leak_ppm"] - group["seal_leak_ppm"].shift(23)) / 23
        group["oil_pressure_slope_24h"] = (group["oil_pressure_kpa"] - group["oil_pressure_kpa"].shift(23)) / 23
        group["fuel_flow_slope_24h"] = (group["fuel_gas_flow_kgh"] - group["fuel_gas_flow_kgh"].shift(23)) / 23
        group["winding_temp_slope_24h"] = (group["winding_temp_c"] - group["winding_temp_c"].shift(23)) / 23
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


if __name__ == "__main__":
    main()
