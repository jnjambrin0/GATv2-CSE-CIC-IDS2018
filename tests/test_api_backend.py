import json
import os
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nids.api.app import app


MODEL_PATH = Path("data/processed/training/best_model.pt")
FEATURE_COLS_PATH = Path("data/processed/feature_cols.json")
GRAPH_DIR = Path("data/graphs")
CSV_DIRS = [Path("data/flows"), Path("data/flows-improved")]


def _first_existing_csv() -> Optional[Path]:
    for base in CSV_DIRS:
        if not base.exists():
            continue
        files = sorted(base.glob("*.csv"))
        if files:
            return min(files, key=lambda p: p.stat().st_size)
    return None


def _sample_csv_path(tmp_path: Path, source: Path, nrows: int = 2000) -> Path:
    df = pd.read_csv(source, nrows=nrows, low_memory=False)
    out_path = tmp_path / source.name
    df.to_csv(out_path, index=False)
    return out_path


def _small_graph_path(max_candidates: int = 50) -> Optional[Path]:
    if not GRAPH_DIR.exists():
        return None
    files = sorted(GRAPH_DIR.glob("*.pt"))
    if not files:
        return None
    candidates = files[:max_candidates]
    return min(candidates, key=lambda p: p.stat().st_size)


def _iter_sse_events(response) -> Iterator[Dict[str, object]]:
    event = None
    data_lines: List[str] = []
    for raw_line in response.iter_lines():
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8")
        else:
            line = raw_line
        if line == "":
            if data_lines:
                payload = json.loads("\n".join(data_lines))
                yield {"event": event or "message", "data": payload}
            event = None
            data_lines = []
            continue
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].lstrip())
    if data_lines:
        payload = json.loads("\n".join(data_lines))
        yield {"event": event or "message", "data": payload}


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    missing = [p for p in (MODEL_PATH, FEATURE_COLS_PATH) if not p.exists()]
    if missing:
        pytest.skip(f"Faltan artefactos requeridos: {', '.join(str(p) for p in missing)}")
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_graph(client: TestClient) -> None:
    graph_path = _small_graph_path()
    if graph_path is None:
        pytest.skip("No hay grafos en data/graphs para probar /predict/graph.")

    with graph_path.open("rb") as f:
        resp = client.post(
            "/predict/graph",
            files={"file": (graph_path.name, f, "application/octet-stream")},
        )
    assert resp.status_code == 200
    payload = resp.json()
    assert "summary" in payload
    summary = payload["summary"]
    assert summary["total_nodes"] >= 1
    assert 0.0 <= summary["attack_rate"] <= 1.0
    assert "metadata" in payload


def test_predict_csv_stream(client: TestClient, tmp_path: Path) -> None:
    source_csv = _first_existing_csv()
    if source_csv is None:
        pytest.skip("No hay CSV en data/flows para probar /predict/csv.")

    full_csv = os.getenv("FULL_CSV_TEST") == "1"
    sample_path = source_csv if full_csv else _sample_csv_path(tmp_path, source_csv)
    start = time.time()
    with sample_path.open("rb") as f:
        with client.stream(
            "POST",
            "/predict/csv",
            files={"file": (sample_path.name, f, "text/csv")},
        ) as resp:
            assert resp.status_code == 200
            events = list(_iter_sse_events(resp))
    elapsed = time.time() - start

    event_names = [evt["event"] for evt in events]
    assert "error" not in event_names
    assert event_names, "No se recibieron eventos SSE."
    assert event_names[0] == "received"
    assert "preprocess" in event_names
    assert "completed" in event_names
    completed = next(evt for evt in events if evt["event"] == "completed")
    summary = completed["data"]
    assert summary["graphs"] >= 1
    assert summary["total_nodes"] >= summary["attack_nodes"]
    assert 0.0 <= summary["attack_rate"] <= 1.0
    assert summary["elapsed_seconds"] >= 0.0
    if full_csv:
        print(f"Tiempo total streaming CSV: {elapsed:.2f}s")
