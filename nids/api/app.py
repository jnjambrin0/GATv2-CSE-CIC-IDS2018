import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .pipeline import load_assets, predict_graph_file, stream_csv_predictions
from .state import ModelAssets


CHUNK_SIZE = 1024 * 1024
ALLOW_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    assets = load_assets()
    app.state.assets = assets
    yield
    app.state.assets = None


app = FastAPI(title="NIDS Backend", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_assets(app: FastAPI) -> ModelAssets:
    assets = getattr(app.state, "assets", None)
    if assets is None:
        raise RuntimeError("Model assets no estan disponibles.")
    return assets


async def _save_upload_to_temp(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix or ".bin"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    path = Path(tmp_path)
    try:
        with path.open("wb") as f:
            while True:
                chunk = await upload.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
    finally:
        await upload.close()
    return path


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict/graph")
async def predict_graph(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".pt"):
        raise HTTPException(status_code=400, detail="Se requiere un archivo .pt")

    assets = _ensure_assets(app)
    tmp_path = await _save_upload_to_temp(file)
    try:
        result = predict_graph_file(tmp_path, assets)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return JSONResponse(result)


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Se requiere un archivo .csv")

    assets = _ensure_assets(app)
    tmp_path = await _save_upload_to_temp(file)
    capture_name = Path(file.filename).stem

    def event_stream() -> Iterator[str]:
        try:
            yield from stream_csv_predictions(tmp_path, capture_name, assets)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
