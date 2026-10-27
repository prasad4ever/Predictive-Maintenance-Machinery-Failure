# Predictive Maintenance — Machinery Failure

Hourly classification of which component on a refinery machine is likely to fail in the next 24 hours, and how urgent that failure is.

## Problem

A hydroprocessing unit runs compressors, pumps, and a feed-gas turbine. Each machine is made of several components: bearings, seal, winding, rotor, coupling, lube oil, and, on the turbine, the fuel-gas system. Tags arrive every hour and cover power, current, speed, pressures, flow, temperatures, vibration, oil, seal leak, acoustics, and fuel gas.

Most hours are healthy. The useful output is not a plant-wide health score. It is the component that is heading for a failure, on which machine, and the repair window:

| Priority | Meaning |
| --- | --- |
| P1 | Extreme urgency. Repair within minutes. |
| P2 | Repair within the shift. |
| P3 | Plan the repair inside 24 hours. |
| P4 | Take it at the next maintenance window. |
| OK | No failure expected in the next 24 hours. |

Past failures are the labels. A row is labeled with the priority of a failure of that same component inside the following 24 hours. If none is recorded, the label is OK.

## Solution

Each training row is one component on one machine for one hour. The features are that hour's tags, the trailing 24-hour level and slope of the main tags, and the machine and component identities.

The record is split by time: 32 days to train, the next 8 days to choose the model, and the last 8 days to report. Rows are not shuffled, so a model cannot train on hours that sit after the hour it is scoring.

Three classifiers were fit with class balancing, because healthy hours outnumber failures by a wide margin:

- logistic regression
- random forest
- histogram gradient boosting

Accuracy is the wrong way to pick a winner here. A model that always says OK is about 97% accurate and never catches a P1. The validation score is `0.55 * P1 F1 + 0.25 * P2 F1 + 0.20 * macro F1`.

Histogram gradient boosting won that score: **0.627**, against 0.614 for the random forest and 0.487 for logistic regression. On the validation window it matched the forest on P1 (precision 0.96, recall 0.50) and scored higher once P2 and the rest of the priority set were included. Logistic regression trailed on the urgent classes.

The last 8 days were not used to make that choice. On those days the selected model caught 80% of P1 hours at 96% precision (F1 0.88). Macro F1 across OK and P1–P4 was 0.70. Accuracy was 0.99 and is reported only as context.

The board shows, for each machine, the component with the highest predicted priority and the repair instruction. A P1 card reads "Immediate. Repair within minutes."

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m data.prepare
python -m backend.train
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The page reads `/api/latest` and then follows `/ws/live`.

Copy `.env.example` to `.env` to change the plant name, paths, or replay pace.

## Layout

| Path | Role |
| --- | --- |
| `data/` | Equipment list, hourly extract, failure records |
| `backend/train.py` | Fits the three classifiers and writes the chosen one |
| `backend/main.py` | Dashboard and live priority feed |
| `models/best.joblib` | Chosen classifier |
| `models/comparison.json` | Validation and test metrics for all three |
| `static/` | Dashboard assets |
