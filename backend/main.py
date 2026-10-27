"""Dashboard and live component-priority feed."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend.config import ROOT, get_settings
from backend.predict import load_predictor, snapshot
from data.catalog import NUMERIC_FEATURES

STATIC_DIR = ROOT / "static"
TEMPLATES = Jinja2Templates(directory=str(ROOT / "backend" / "templates"))
REPLAY_HOURS = 36


class Board:
    def __init__(self) -> None:
        self.model_name = ""
        self.frames: list[tuple[str, pd.DataFrame]] = []

    def load(self) -> None:
        settings = get_settings()
        predictor = load_predictor()
        self.model_name = predictor.model_name
        path = settings.resolve(settings.telemetry_path)
        frame = pd.read_csv(path)
        latest = sorted(frame["timestamp"].unique())[-REPLAY_HOURS:]
        window = frame[frame["timestamp"].isin(latest)]
        columns = ["timestamp", "machine_id", "machine_name", "area", "component_id", "component_name", *NUMERIC_FEATURES]
        self.frames = [
            (str(timestamp), group[columns])
            for timestamp, group in window.groupby("timestamp", sort=True)
        ]

    def hours(self) -> AsyncIterator[dict[str, object]]:
        predictor = load_predictor()

        async def _generate() -> AsyncIterator[dict[str, object]]:
            if not self.frames:
                return
            ordered = list(reversed(self.frames))
            while True:
                for timestamp, group in ordered:
                    findings = predictor.score(group)
                    payload = snapshot(findings, timestamp)
                    payload["model"] = self.model_name
                    yield payload

        return _generate()


BOARD = Board()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    BOARD.load()
    yield


app = FastAPI(title="Predictive Maintenance", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def dashboard(request: Request) -> object:
    settings = get_settings()
    return TEMPLATES.TemplateResponse(
        request,
        "dashboard.html",
        {
            "plant_name": settings.plant_name,
            "model_name": {
                "hist_gradient_boosting": "histogram gradient boosting",
                "random_forest": "random forest",
                "logistic_regression": "logistic regression",
            }.get(BOARD.model_name, BOARD.model_name),
        },
    )


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    return FileResponse(Path(STATIC_DIR) / "favicon.svg")


@app.get("/api/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model": BOARD.model_name, "hours": len(BOARD.frames)}


@app.get("/api/latest")
def latest() -> dict[str, object]:
    if not BOARD.frames:
        return {"model": BOARD.model_name, "machines": [], "alarms": [], "counts": {}}
    timestamp, group = BOARD.frames[-1]
    payload = snapshot(load_predictor().score(group), timestamp)
    payload["model"] = BOARD.model_name
    return payload


@app.websocket("/ws/live")
async def live(websocket: WebSocket) -> None:
    await websocket.accept()
    settings = get_settings()
    try:
        async for payload in BOARD.hours():
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(settings.replay_interval_seconds)
    except WebSocketDisconnect:
        return
