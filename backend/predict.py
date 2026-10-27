"""Score hourly component rows with the saved classifier."""

from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from backend.config import get_settings
from data.catalog import CATEGORICAL_FEATURES, NUMERIC_FEATURES, PRIORITY_ACTION

FEATURE_LABELS = {
    "power_kw": "Power",
    "motor_current_a": "Motor current",
    "motor_voltage_v": "Motor voltage",
    "shaft_speed_rpm": "Shaft speed",
    "suction_pressure_kpa": "Suction pressure",
    "discharge_pressure_kpa": "Discharge pressure",
    "flow_m3h": "Flow",
    "inlet_temp_c": "Inlet temperature",
    "outlet_temp_c": "Outlet temperature",
    "vibration_rms_mms": "Vibration RMS",
    "vibration_axial_mms": "Axial vibration",
    "vibration_radial_mms": "Radial vibration",
    "vibration_kurtosis": "Vibration kurtosis",
    "vibration_crest": "Crest factor",
    "bearing_temp_c": "Bearing temperature",
    "oil_temp_c": "Oil temperature",
    "oil_pressure_kpa": "Oil pressure",
    "winding_temp_c": "Winding temperature",
    "seal_leak_ppm": "Seal leak",
    "acoustic_db": "Acoustics",
    "fuel_gas_flow_kgh": "Fuel-gas flow",
    "fuel_gas_pressure_kpa": "Fuel-gas pressure",
    "exhaust_temp_c": "Exhaust temperature",
    "seal_leak_slope_24h": "Seal-leak rise",
    "bearing_temp_slope_24h": "Bearing-temperature rise",
    "vibration_rms_slope_24h": "Vibration rise",
    "oil_pressure_slope_24h": "Oil-pressure change",
    "fuel_flow_slope_24h": "Fuel-flow change",
    "winding_temp_slope_24h": "Winding-temperature rise",
    "power_slope_24h": "Power change",
}

RANK = {"P1": 4, "P2": 3, "P3": 2, "P4": 1, "OK": 0}


class Predictor:
    def __init__(self, bundle: dict[str, object]) -> None:
        self.pipeline = bundle["pipeline"]
        self.model_name = str(bundle["model_name"])
        self._mean = np.asarray(bundle["numeric_mean"], dtype=np.float64)
        self._scale = np.asarray(bundle["numeric_scale"], dtype=np.float64)
        model = self.pipeline.named_steps["model"]
        self.classes = [str(name) for name in model.classes_]

    def score(self, frame: pd.DataFrame) -> list[dict[str, object]]:
        features = frame[list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)]
        probabilities = self.pipeline.predict_proba(features)
        chosen = probabilities.argmax(axis=1)
        numeric = frame[list(NUMERIC_FEATURES)].to_numpy(dtype=np.float64)
        centered = (numeric - self._mean) / self._scale
        findings: list[dict[str, object]] = []
        for index, row in enumerate(frame.itertuples(index=False)):
            priority = self.classes[int(chosen[index])]
            order = np.argsort(np.abs(centered[index]))[::-1]
            drivers = []
            for feature_index in order:
                name = NUMERIC_FEATURES[int(feature_index)]
                if name not in FEATURE_LABELS:
                    continue
                drivers.append(
                    {
                        "name": FEATURE_LABELS[name],
                        "z": round(float(centered[index, feature_index]), 2),
                    }
                )
                if len(drivers) == 3:
                    break
            findings.append(
                {
                    "machine_id": row.machine_id,
                    "machine_name": row.machine_name,
                    "area": row.area,
                    "component_id": row.component_id,
                    "component_name": row.component_name,
                    "priority": priority,
                    "action": PRIORITY_ACTION[priority],
                    "confidence": round(float(probabilities[index, chosen[index]]), 3),
                    "drivers": drivers,
                }
            )
        findings.sort(key=lambda item: (RANK[str(item["priority"])], float(item["confidence"])), reverse=True)
        return findings


def snapshot(findings: list[dict[str, object]], timestamp: str) -> dict[str, object]:
    counts = {name: 0 for name in ("P1", "P2", "P3", "P4", "OK")}
    machines: dict[str, dict[str, object]] = {}
    alarms = []
    for item in findings:
        counts[str(item["priority"])] += 1
        current = machines.get(str(item["machine_id"]))
        if current is None or RANK[str(item["priority"])] > RANK[str(current["priority"])]:
            machines[str(item["machine_id"])] = item
        if item["priority"] != "OK":
            alarms.append(item)
    ordered = sorted(machines.values(), key=lambda item: str(item["machine_id"]))
    return {
        "timestamp": timestamp,
        "counts": counts,
        "machines": ordered,
        "alarms": alarms,
    }


@lru_cache(maxsize=1)
def load_predictor() -> Predictor:
    settings = get_settings()
    path = settings.resolve(settings.model_path)
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found at {path}. Run python -m backend.train")
    bundle = joblib.load(path)
    return Predictor(bundle)
