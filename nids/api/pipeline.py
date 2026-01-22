import json
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from nids.graphs.builder import iter_graphs_from_dataframe
from nids.preprocessing.transform import clean_and_label, compute_feature_cols
from nids.training.data import load_graph
from nids.training.model import GATv2Net

from .config import (
    DEVICE,
    FEATURE_COLS_PATH,
    FEATURE_STATS_PATH,
    MODEL_PATH,
    THRESHOLD,
    build_graph_config,
    build_preprocess_config,
    PROGRESS_EVERY_GRAPHS,
)
from .sse import format_sse
from .state import ModelAssets


def _load_feature_cols(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta load.py primero.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_feature_stats(
    path: Path,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        mean = torch.tensor(payload["mean"], dtype=torch.float32)
        std = torch.tensor(payload["std"], dtype=torch.float32)
        return mean, std
    except Exception:
        return None


def _load_model(in_channels: int) -> torch.nn.Module:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No existe {MODEL_PATH}. Entrena el modelo primero.")
    # Debe reflejar los hiperparametros de train_gatv2.py para cargar el state dict sin errores.
    model = GATv2Net(
        in_channels,
        hidden_channels=32,
        heads=4,
        dropout=0.2,
        use_residual=True,
    )
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def load_assets() -> ModelAssets:
    feature_cols = _load_feature_cols(FEATURE_COLS_PATH)
    mean_std = _load_feature_stats(FEATURE_STATS_PATH)
    if mean_std is not None and len(mean_std[0]) != len(feature_cols):
        raise ValueError("feature_stats.json no coincide con feature_cols.json.")
    model = _load_model(in_channels=len(feature_cols))
    mean = mean_std[0] if mean_std else None
    std = mean_std[1] if mean_std else None
    return ModelAssets(
        model=model,
        feature_cols=feature_cols,
        mean=mean,
        std=std,
        device=DEVICE,
        threshold=THRESHOLD,
        graph_cfg=build_graph_config(),
        preprocess_cfg=build_preprocess_config(),
        progress_every_graphs=PROGRESS_EVERY_GRAPHS,
    )


def _align_feature_columns(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        for col in missing:
            df[col] = 0.0
        df[missing] = df[missing].astype("float32")
    return df


def _prepare_graph(data, assets: ModelAssets) -> torch.Tensor:
    data.x = data.x.float()
    if assets.mean is not None and assets.std is not None:
        data.x = (data.x - assets.mean) / assets.std
    data = data.to(assets.device)
    with torch.no_grad():
        logits = assets.model(data)
    return logits


def _summarize_probs_simple(
    probs: torch.Tensor, threshold: float
) -> Dict[str, float]:
    attack_nodes = int((probs >= threshold).sum().item())
    total_nodes = int(probs.numel())
    prob_sum = float(probs.sum().item())
    prob_max = float(probs.max().item()) if total_nodes > 0 else 0.0
    return {
        "total_nodes": total_nodes,
        "attack_nodes": attack_nodes,
        "prob_sum": prob_sum,
        "prob_max": prob_max,
    }


def _summarize_probs_detailed(
    probs: torch.Tensor, threshold: float
) -> Dict[str, object]:
    arr = probs.detach().cpu().numpy()
    total_nodes = int(arr.size)
    attack_nodes = int((arr >= threshold).sum())
    metrics = {
        "total_nodes": total_nodes,
        "attack_nodes": attack_nodes,
        "attack_rate": float(attack_nodes / total_nodes) if total_nodes else 0.0,
        "prob_mean": float(arr.mean()) if total_nodes else 0.0,
        "prob_p50": float(np.percentile(arr, 50)) if total_nodes else 0.0,
        "prob_p90": float(np.percentile(arr, 90)) if total_nodes else 0.0,
        "prob_p95": float(np.percentile(arr, 95)) if total_nodes else 0.0,
        "prob_max": float(arr.max()) if total_nodes else 0.0,
    }
    return metrics


def predict_graph_data(data, assets: ModelAssets) -> Dict[str, object]:
    logits = _prepare_graph(data, assets)
    probs = torch.sigmoid(logits)
    summary = _summarize_probs_detailed(probs, assets.threshold)
    metadata = {}
    for key in ("capture_name", "window_id", "subwindow_id", "num_nodes", "num_edges"):
        if hasattr(data, key):
            metadata[key] = getattr(data, key)
    return {"summary": summary, "metadata": metadata, "threshold": assets.threshold}


def predict_graph_file(path: Path, assets: ModelAssets) -> Dict[str, object]:
    data = load_graph(path)
    if not hasattr(data, "x") or not hasattr(data, "edge_index"):
        raise ValueError("El archivo .pt no contiene un grafo PyG valido.")
    return predict_graph_data(data, assets)


def stream_csv_predictions(
    csv_path: Path, capture_name: str, assets: ModelAssets
) -> Iterator[str]:
    start = time.time()
    try:
        yield format_sse("received", {"file": capture_name})

        df = pd.read_csv(csv_path, low_memory=False)
        yield format_sse("preprocess", {"step": "loaded", "rows": int(len(df))})

        df["capture_name"] = capture_name
        df = clean_and_label(df, assets.preprocess_cfg)
        compute_feature_cols(df, assets.preprocess_cfg)
        df = _align_feature_columns(df, assets.feature_cols)
        yield format_sse(
            "preprocess",
            {"step": "done", "rows": int(len(df)), "features": len(assets.feature_cols)},
        )

        graphs = 0
        total_nodes = 0
        attack_nodes = 0
        prob_sum = 0.0
        prob_max = 0.0

        for data in iter_graphs_from_dataframe(
            df, capture_name, assets.feature_cols, assets.graph_cfg
        ):
            graphs += 1
            logits = _prepare_graph(data, assets)
            probs = torch.sigmoid(logits)
            stats = _summarize_probs_simple(probs, assets.threshold)
            total_nodes += stats["total_nodes"]
            attack_nodes += stats["attack_nodes"]
            prob_sum += stats["prob_sum"]
            prob_max = max(prob_max, stats["prob_max"])

            if graphs % assets.progress_every_graphs == 0:
                yield format_sse(
                    "progress",
                    {
                        "graphs_processed": graphs,
                        "nodes_processed": total_nodes,
                        "attack_nodes": attack_nodes,
                    },
                )

        elapsed = time.time() - start
        summary = {
            "graphs": graphs,
            "total_nodes": total_nodes,
            "attack_nodes": attack_nodes,
            "attack_rate": float(attack_nodes / total_nodes) if total_nodes else 0.0,
            "prob_mean": float(prob_sum / total_nodes) if total_nodes else 0.0,
            "prob_max": float(prob_max),
            "elapsed_seconds": float(elapsed),
            "threshold": assets.threshold,
        }
        if graphs == 0:
            summary["warning"] = "No se generaron grafos con los parametros actuales."
        yield format_sse("completed", summary)
    except Exception as exc:
        yield format_sse("error", {"error": str(exc)})
