"""Equipment, components, and the columns used for training."""

from __future__ import annotations

PRIORITIES = ("OK", "P4", "P3", "P2", "P1")
FAILURE_PRIORITIES = ("P4", "P3", "P2", "P1")

PRIORITY_ACTION = {
    "P1": "Immediate. Repair within minutes.",
    "P2": "Urgent. Repair within this shift.",
    "P3": "Plan the repair inside 24 hours.",
    "P4": "Schedule the next maintenance window.",
    "OK": "No failure expected in the next 24 hours.",
}

MACHINES = (
    {"machine_id": "C-101", "machine_name": "Hydrogen recycle compressor", "area": "Hydroprocessing", "fuel": False, "power_kw": 3150, "speed_rpm": 11840, "flow_m3h": 18500, "suction_kpa": 2100, "differential_kpa": 2500, "vibration_mms": 2.4},
    {"machine_id": "P-201", "machine_name": "Crude charge pump", "area": "Crude Distillation", "fuel": False, "power_kw": 305, "speed_rpm": 2985, "flow_m3h": 520, "suction_kpa": 280, "differential_kpa": 1650, "vibration_mms": 2.0},
    {"machine_id": "P-305", "machine_name": "Naphtha reflux pump", "area": "Naphtha Splitter", "fuel": False, "power_kw": 27, "speed_rpm": 2970, "flow_m3h": 95, "suction_kpa": 160, "differential_kpa": 640, "vibration_mms": 1.5},
    {"machine_id": "K-410", "machine_name": "Flare-gas compressor", "area": "Flare Recovery", "fuel": False, "power_kw": 78, "speed_rpm": 590, "flow_m3h": 860, "suction_kpa": 110, "differential_kpa": 780, "vibration_mms": 4.6},
    {"machine_id": "P-512", "machine_name": "Cooling-water pump", "area": "Utilities", "fuel": False, "power_kw": 190, "speed_rpm": 742, "flow_m3h": 2400, "suction_kpa": 130, "differential_kpa": 220, "vibration_mms": 2.7},
    {"machine_id": "GT-210", "machine_name": "Feed-gas turbine", "area": "Hydrocracker", "fuel": True, "power_kw": 8400, "speed_rpm": 6200, "flow_m3h": 24000, "suction_kpa": 1400, "differential_kpa": 2800, "vibration_mms": 3.1},
)

COMPONENTS = (
    {"component_id": "drive_end_bearing", "component_name": "Drive-end bearing"},
    {"component_id": "non_drive_bearing", "component_name": "Non-drive-end bearing"},
    {"component_id": "mechanical_seal", "component_name": "Mechanical seal"},
    {"component_id": "motor_winding", "component_name": "Motor winding"},
    {"component_id": "rotor", "component_name": "Rotor"},
    {"component_id": "lube_oil", "component_name": "Lube-oil system"},
    {"component_id": "coupling", "component_name": "Coupling"},
    {"component_id": "fuel_system", "component_name": "Fuel-gas system"},
)

NUMERIC_FEATURES = (
    "power_kw",
    "motor_current_a",
    "motor_voltage_v",
    "shaft_speed_rpm",
    "suction_pressure_kpa",
    "discharge_pressure_kpa",
    "flow_m3h",
    "inlet_temp_c",
    "outlet_temp_c",
    "vibration_rms_mms",
    "vibration_axial_mms",
    "vibration_radial_mms",
    "vibration_kurtosis",
    "vibration_crest",
    "bearing_temp_c",
    "oil_temp_c",
    "oil_pressure_kpa",
    "winding_temp_c",
    "seal_leak_ppm",
    "acoustic_db",
    "fuel_gas_flow_kgh",
    "fuel_gas_pressure_kpa",
    "exhaust_temp_c",
    "fuel_system_present",
    "spectrum_01_mms",
    "spectrum_02_mms",
    "spectrum_03_mms",
    "spectrum_04_mms",
    "spectrum_05_mms",
    "spectrum_06_mms",
    "spectrum_07_mms",
    "spectrum_08_mms",
    "spectrum_09_mms",
    "spectrum_10_mms",
    "spectrum_11_mms",
    "spectrum_12_mms",
    "vibration_rms_mean_24h",
    "vibration_rms_slope_24h",
    "bearing_temp_mean_24h",
    "bearing_temp_slope_24h",
    "power_mean_24h",
    "power_slope_24h",
    "seal_leak_mean_24h",
    "seal_leak_slope_24h",
    "oil_pressure_slope_24h",
    "fuel_flow_slope_24h",
    "winding_temp_slope_24h",
    "hour_of_day",
)

CATEGORICAL_FEATURES = ("machine_id", "component_id")
HORIZON_HOURS = 24
DAYS = 48
TRAIN_DAYS = 32
VALIDATION_DAYS = 8
