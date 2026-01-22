import json
import math
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch_geometric.data import Data
except Exception as exc:  # pragma: no cover - si no hay torch/pyg en el entorno, salimos con mensaje claro
    raise SystemExit(
        "Falta torch o torch-geometric. Instala las dependencias para generar grafos."
    ) from exc

from .config import GraphBuildConfig


def load_feature_cols(cfg: GraphBuildConfig) -> List[str]:
    if not cfg.feature_cols_json.exists():
        raise FileNotFoundError(
            f"No existe {cfg.feature_cols_json}. Ejecuta load.py primero."
        )
    with open(cfg.feature_cols_json, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_parquet_files(cfg: GraphBuildConfig) -> List[Path]:
    files = sorted(cfg.parquet_dir.glob("*.parquet"))
    if cfg.captures is None:
        return files
    cap_set = set(cfg.captures)
    return [f for f in files if f.stem in cap_set]


def _window_splits(n_nodes: int, cfg: GraphBuildConfig) -> Tuple[int, int]:
    splits = int(math.ceil(n_nodes / cfg.max_nodes_per_graph))
    subwindow_seconds = int(math.ceil(cfg.window_size_seconds / splits))
    return splits, max(subwindow_seconds, 1)


def _split_by_max_nodes(order: np.ndarray, cfg: GraphBuildConfig) -> List[np.ndarray]:
    splits = int(math.ceil(len(order) / cfg.max_nodes_per_graph))
    return list(np.array_split(order, splits))


def _init_stats() -> Dict[str, object]:
    return {
        "graphs": 0,
        "base_windows": 0,
        "base_windows_split": 0,
        "windows_skipped_min_nodes": 0,
        "subwindows_skipped_min_nodes": 0,
        "subwindows_split_by_chunk": 0,
        "nodes": [],
        "edges": [],
        "edges_per_node": [],
        "base_window_nodes": [],
        "subwindow_seconds": [],
    }


def _merge_stats(dst: Dict[str, object], src: Dict[str, object]) -> None:
    for key in (
        "graphs",
        "base_windows",
        "base_windows_split",
        "windows_skipped_min_nodes",
        "subwindows_skipped_min_nodes",
        "subwindows_split_by_chunk",
    ):
        dst[key] += src[key]
    for key in (
        "nodes",
        "edges",
        "edges_per_node",
        "base_window_nodes",
        "subwindow_seconds",
    ):
        dst[key].extend(src[key])


def _describe(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype="float64")
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def _summarize_stats(stats: Dict[str, object]) -> Dict[str, object]:
    return {
        "graphs": int(stats["graphs"]),
        "base_windows": int(stats["base_windows"]),
        "base_windows_split": int(stats["base_windows_split"]),
        "windows_skipped_min_nodes": int(stats["windows_skipped_min_nodes"]),
        "subwindows_skipped_min_nodes": int(stats["subwindows_skipped_min_nodes"]),
        "subwindows_split_by_chunk": int(stats["subwindows_split_by_chunk"]),
        "nodes": _describe(stats["nodes"]),
        "edges": _describe(stats["edges"]),
        "edges_per_node": _describe(stats["edges_per_node"]),
        "base_window_nodes": _describe(stats["base_window_nodes"]),
        "subwindow_seconds": _describe(stats["subwindow_seconds"]),
    }


def _build_edges(
    order: np.ndarray,
    src_ips: List[str],
    dst_ips: List[str],
    cfg: GraphBuildConfig,
) -> np.ndarray:
    edges = set()
    last_by_ip: Dict[str, List[int]] = {}

    for rank, node_idx in enumerate(order):
        for k in range(1, cfg.k_time + 1):
            if rank - k < 0:
                break
            prev_idx = order[rank - k]
            edges.add((prev_idx, node_idx))
            edges.add((node_idx, prev_idx))

        for ip in (src_ips[node_idx], dst_ips[node_idx]):
            if ip not in last_by_ip:
                last_by_ip[ip] = []
            for prev_idx in last_by_ip[ip]:
                edges.add((prev_idx, node_idx))
                edges.add((node_idx, prev_idx))
            last_by_ip[ip].append(node_idx)
            if len(last_by_ip[ip]) > cfg.k_ip:
                last_by_ip[ip] = last_by_ip[ip][-cfg.k_ip :]

    if not edges:
        return np.empty((2, 0), dtype="int64")
    return np.array(list(edges), dtype="int64").T


def _build_graph_data(
    x: np.ndarray,
    y: np.ndarray,
    edge_index: np.ndarray,
    metadata: Dict[str, object],
) -> Data:
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.long),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
    )
    for key, value in metadata.items():
        setattr(data, key, value)
    return data


def _save_graph(out_path: Path, data: Data) -> None:
    torch.save(data, out_path)


def _ensure_required_columns(df: pd.DataFrame, feature_cols: List[str]) -> None:
    required = {"Timestamp", "src_ip", "dst_ip", "label_bin"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    missing_feat = set(feature_cols) - set(df.columns)
    if missing_feat:
        raise ValueError(f"Faltan features: {sorted(missing_feat)[:10]}")


def _build_one_graph(
    dfs: pd.DataFrame,
    cap: str,
    feature_cols: List[str],
    base_id: int,
    sub_id: int,
    cfg: GraphBuildConfig,
    stats: Optional[Dict[str, object]] = None,
) -> int:
    if len(dfs) < cfg.min_nodes_per_graph:
        return 0

    src_ips = dfs["src_ip"].astype(str).tolist()
    dst_ips = dfs["dst_ip"].astype(str).tolist()
    order = np.argsort(dfs["_t_sec"].to_numpy())

    edge_index = _build_edges(order, src_ips, dst_ips, cfg)
    x = dfs[feature_cols].to_numpy(dtype="float32", copy=False)
    y = dfs["label_bin"].to_numpy(dtype="int64", copy=False)

    ts_min = int(dfs["_t_sec"].min())
    ts_max = int(dfs["_t_sec"].max())

    out_path = cfg.out_dir / f"{cap}_w{base_id:06d}_s{sub_id:03d}.pt"
    metadata = {
        "capture_name": cap,
        "window_id": int(base_id),
        "subwindow_id": int(sub_id),
        "t_start_sec": ts_min,
        "t_end_sec": ts_max,
        "num_nodes": int(len(dfs)),
        "num_edges": int(edge_index.shape[1]),
    }
    data = _build_graph_data(x, y, edge_index, metadata)
    _save_graph(out_path, data)
    if stats is not None:
        nodes = int(len(dfs))
        edges = int(edge_index.shape[1])
        stats["graphs"] += 1
        stats["nodes"].append(nodes)
        stats["edges"].append(edges)
        stats["edges_per_node"].append(edges / nodes if nodes > 0 else 0.0)
    return 1


def _build_graphs_for_capture(
    df: pd.DataFrame,
    cap: str,
    feature_cols: List[str],
    cfg: GraphBuildConfig,
    stats: Optional[Dict[str, object]] = None,
) -> int:
    _ensure_required_columns(df, feature_cols)

    if not np.issubdtype(df["Timestamp"].dtype, np.datetime64):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    df = df[df["Timestamp"].notna()].copy()
    if df.empty:
        return 0

    t0 = df["Timestamp"].min()
    t_sec = (df["Timestamp"] - t0).dt.total_seconds().astype("int64")
    base_win = (t_sec // cfg.window_size_seconds).astype("int64")
    df = df.assign(_base_win=base_win, _t_sec=t_sec)

    graphs_built = 0
    for base_id, dfw in df.groupby("_base_win", sort=True):
        n_nodes = len(dfw)
        if stats is not None:
            stats["base_windows"] += 1
            stats["base_window_nodes"].append(int(n_nodes))
        if n_nodes < cfg.min_nodes_per_graph:
            if stats is not None:
                stats["windows_skipped_min_nodes"] += 1
            continue

        splits, subwindow_seconds = _window_splits(n_nodes, cfg)
        if stats is not None:
            stats["subwindow_seconds"].append(int(subwindow_seconds))
            if splits > 1:
                stats["base_windows_split"] += 1
        base_start = int(base_id * cfg.window_size_seconds)
        sub_win = ((dfw["_t_sec"] - base_start) // subwindow_seconds).astype("int64")
        dfw = dfw.assign(_sub_win=sub_win)

        for sub_id, dfs in dfw.groupby("_sub_win", sort=True):
            if len(dfs) < cfg.min_nodes_per_graph:
                if stats is not None:
                    stats["subwindows_skipped_min_nodes"] += 1
                continue

            order = np.argsort(dfs["_t_sec"].to_numpy())
            if len(dfs) > cfg.max_nodes_per_graph:
                if stats is not None:
                    stats["subwindows_split_by_chunk"] += 1
                for chunk_idx, chunk in enumerate(_split_by_max_nodes(order, cfg)):
                    graphs_built += _build_one_graph(
                        dfs.iloc[chunk],
                        cap,
                        feature_cols,
                        base_id,
                        int(sub_id) * 100 + chunk_idx,
                        cfg,
                        stats,
                    )
                continue

            graphs_built += _build_one_graph(
                dfs, cap, feature_cols, base_id, sub_id, cfg, stats
            )

    return graphs_built


def build_graphs(cfg: GraphBuildConfig) -> None:
    if not cfg.parquet_dir.exists():
        raise FileNotFoundError(
            f"No existe {cfg.parquet_dir}. Ejecuta load.py en per_file_parquet."
        )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    feature_cols = load_feature_cols(cfg)

    total_graphs = 0
    report = {
        "captures": {},
        "overall": {},
        "config": {
            "window_size_seconds": cfg.window_size_seconds,
            "max_nodes_per_graph": cfg.max_nodes_per_graph,
            "min_nodes_per_graph": cfg.min_nodes_per_graph,
            "k_ip": cfg.k_ip,
            "k_time": cfg.k_time,
            "dev_mode": cfg.dev_mode,
            "max_rows_per_capture": cfg.max_rows_per_capture if cfg.dev_mode else None,
        },
    }
    overall_stats = _init_stats()
    for path in iter_parquet_files(cfg):
        cap = path.stem
        print(f"\nConstruyendo grafos para: {cap}")
        df = pd.read_parquet(path)
        if cfg.dev_mode and len(df) > cfg.max_rows_per_capture:
            df = df.sample(cfg.max_rows_per_capture, random_state=42)
        cap_stats = _init_stats() if cfg.debug_summary else None
        graphs = _build_graphs_for_capture(df, cap, feature_cols, cfg, cap_stats)
        print(f"  Grafos generados: {graphs}")
        total_graphs += graphs
        if cfg.debug_summary and cap_stats is not None:
            report["captures"][cap] = _summarize_stats(cap_stats)
            _merge_stats(overall_stats, cap_stats)

    print(f"\nTotal de grafos generados: {total_graphs}")
    if cfg.debug_summary:
        report["overall"] = _summarize_stats(overall_stats)
        cfg.debug_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.debug_summary_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Reporte guardado en: {cfg.debug_summary_path}")


def iter_graphs_from_dataframe(
    df: pd.DataFrame,
    cap: str,
    feature_cols: List[str],
    cfg: GraphBuildConfig,
) -> Iterator[Data]:
    _ensure_required_columns(df, feature_cols)

    if not np.issubdtype(df["Timestamp"].dtype, np.datetime64):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    df = df[df["Timestamp"].notna()].copy()
    if df.empty:
        return

    t0 = df["Timestamp"].min()
    t_sec = (df["Timestamp"] - t0).dt.total_seconds().astype("int64")
    base_win = (t_sec // cfg.window_size_seconds).astype("int64")
    df = df.assign(_base_win=base_win, _t_sec=t_sec)

    for base_id, dfw in df.groupby("_base_win", sort=True):
        if len(dfw) < cfg.min_nodes_per_graph:
            continue

        splits, subwindow_seconds = _window_splits(len(dfw), cfg)
        base_start = int(base_id * cfg.window_size_seconds)
        sub_win = ((dfw["_t_sec"] - base_start) // subwindow_seconds).astype("int64")
        dfw = dfw.assign(_sub_win=sub_win)

        for sub_id, dfs in dfw.groupby("_sub_win", sort=True):
            if len(dfs) < cfg.min_nodes_per_graph:
                continue

            order = np.argsort(dfs["_t_sec"].to_numpy())
            if len(dfs) > cfg.max_nodes_per_graph:
                for chunk_idx, chunk in enumerate(_split_by_max_nodes(order, cfg)):
                    chunk_df = dfs.iloc[chunk]
                    chunk_order = np.argsort(chunk_df["_t_sec"].to_numpy())
                    edge_index = _build_edges(
                        chunk_order,
                        chunk_df["src_ip"].astype(str).tolist(),
                        chunk_df["dst_ip"].astype(str).tolist(),
                        cfg,
                    )
                    metadata = {
                        "capture_name": cap,
                        "window_id": int(base_id),
                        "subwindow_id": int(sub_id) * 100 + chunk_idx,
                        "t_start_sec": int(chunk_df["_t_sec"].min()),
                        "t_end_sec": int(chunk_df["_t_sec"].max()),
                        "num_nodes": int(len(chunk_df)),
                        "num_edges": int(edge_index.shape[1]),
                    }
                    yield _build_graph_data(
                        chunk_df[feature_cols].to_numpy(dtype="float32", copy=False),
                        chunk_df["label_bin"].to_numpy(dtype="int64", copy=False),
                        edge_index,
                        metadata,
                    )
                continue

            src_ips = dfs["src_ip"].astype(str).tolist()
            dst_ips = dfs["dst_ip"].astype(str).tolist()
            order = np.argsort(dfs["_t_sec"].to_numpy())

            edge_index = _build_edges(order, src_ips, dst_ips, cfg)
            x = dfs[feature_cols].to_numpy(dtype="float32", copy=False)
            y = dfs["label_bin"].to_numpy(dtype="int64", copy=False)

            ts_min = int(dfs["_t_sec"].min())
            ts_max = int(dfs["_t_sec"].max())

            metadata = {
                "capture_name": cap,
                "window_id": int(base_id),
                "subwindow_id": int(sub_id),
                "t_start_sec": ts_min,
                "t_end_sec": ts_max,
                "num_nodes": int(len(dfs)),
                "num_edges": int(edge_index.shape[1]),
            }
            yield _build_graph_data(x, y, edge_index, metadata)
