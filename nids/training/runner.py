import json
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from .config import TrainingConfig
from .data import (
    GraphPTDataset,
    list_graph_files,
    load_graph,
    maybe_subsample,
    split_paths,
)
from .model import GATv2Net


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _save_pos_weight(value: float, cfg: TrainingConfig) -> None:
    cfg.pos_weight_cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.pos_weight_cache, "w", encoding="utf-8") as f:
        json.dump({"pos_weight": value}, f, indent=2)


def _load_pos_weight(cfg: TrainingConfig) -> Optional[float]:
    if not cfg.pos_weight_cache.exists():
        return None
    try:
        with open(cfg.pos_weight_cache, "r", encoding="utf-8") as f:
            return float(json.load(f)["pos_weight"])
    except Exception:
        return None


def _save_feature_stats(mean: torch.Tensor, std: torch.Tensor, cfg: TrainingConfig) -> None:
    cfg.feature_stats_cache.parent.mkdir(parents=True, exist_ok=True)
    payload = {"mean": mean.tolist(), "std": std.tolist()}
    with open(cfg.feature_stats_cache, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_feature_stats(cfg: TrainingConfig) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if not cfg.feature_stats_cache.exists():
        return None
    try:
        with open(cfg.feature_stats_cache, "r", encoding="utf-8") as f:
            payload = json.load(f)
        mean = torch.tensor(payload["mean"], dtype=torch.float32)
        std = torch.tensor(payload["std"], dtype=torch.float32)
        return mean, std
    except Exception:
        return None


def _compute_pos_weight(loader: DataLoader, cfg: TrainingConfig) -> float:
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    pos = 0.0
    neg = 0.0
    for idx, data in enumerate(loader, start=1):
        y = data.y.view(-1)
        pos += float(y.sum().item())
        neg += float(y.numel() - y.sum().item())
        if cfg.verbose and cfg.log_every_n and (idx % cfg.log_every_n == 0):
            msg = f"[pos_weight] batch {idx}"
            if total_batches:
                msg += f"/{total_batches}"
            print(msg, flush=True)
    if pos <= 0:
        return 1.0
    return neg / pos


def _compute_train_stats(
    loader: DataLoader, cfg: TrainingConfig
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    sum_x = None
    sum_x2 = None
    total_nodes = 0
    pos = 0.0
    neg = 0.0

    for idx, data in enumerate(loader, start=1):
        x = data.x
        if sum_x is None:
            sum_x = torch.zeros(x.size(1), dtype=torch.float64)
            sum_x2 = torch.zeros(x.size(1), dtype=torch.float64)
        x64 = x.double()
        sum_x += x64.sum(dim=0)
        sum_x2 += (x64 * x64).sum(dim=0)
        total_nodes += int(x.size(0))

        y = data.y.view(-1)
        pos += float(y.sum().item())
        neg += float(y.numel() - y.sum().item())

        if cfg.verbose and cfg.log_every_n and (idx % cfg.log_every_n == 0):
            msg = f"[stats] batch {idx}"
            if total_batches:
                msg += f"/{total_batches}"
            print(msg, flush=True)

    if total_nodes <= 0:
        raise ValueError("No hay nodos para calcular stats.")

    mean = sum_x / total_nodes
    var = (sum_x2 / total_nodes) - mean * mean
    var = torch.clamp(var, min=0.0)
    std = torch.sqrt(var)
    std[std < 1e-12] = 1.0

    pos_weight = neg / pos if pos > 0 else 1.0
    return mean.float(), std.float(), float(pos_weight)


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    cfg: TrainingConfig,
) -> float:
    model.train()
    total_loss = 0.0
    total_nodes = 0
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    for idx, data in enumerate(loader, start=1):
        data = data.to(cfg.device)
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * int(data.y.numel())
        total_nodes += int(data.y.numel())
        if cfg.verbose and cfg.log_every_n and (idx % cfg.log_every_n == 0):
            msg = f"[train] batch {idx}"
            if total_batches:
                msg += f"/{total_batches}"
            msg += f" avg_loss={(total_loss / max(total_nodes, 1)):.5f}"
            print(msg, flush=True)
    return total_loss / max(total_nodes, 1)


def _eval_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    cfg: TrainingConfig,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_nodes = 0
    tp = fp = tn = fn = 0
    scores = []
    targets = []
    remaining = cfg.max_eval_nodes

    with torch.no_grad():
        total_batches = len(loader) if hasattr(loader, "__len__") else None
        for idx, data in enumerate(loader, start=1):
            data = data.to(cfg.device)
            logits = model(data)
            loss = criterion(logits, data.y)
            total_loss += float(loss.item()) * int(data.y.numel())
            total_nodes += int(data.y.numel())

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).to(torch.int64)
            y = data.y.to(torch.int64)

            tp += int(((preds == 1) & (y == 1)).sum().item())
            fp += int(((preds == 1) & (y == 0)).sum().item())
            tn += int(((preds == 0) & (y == 0)).sum().item())
            fn += int(((preds == 0) & (y == 1)).sum().item())

            if remaining > 0:
                take = min(int(y.numel()), remaining)
                sample_idx = torch.randperm(int(y.numel()), device=y.device)[:take]
                scores.append(probs[sample_idx].detach().cpu().numpy())
                targets.append(y[sample_idx].detach().cpu().numpy())
                remaining -= take
            if cfg.verbose and cfg.log_every_n and (idx % cfg.log_every_n == 0):
                msg = f"[eval] batch {idx}"
                if total_batches:
                    msg += f"/{total_batches}"
                print(msg, flush=True)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / max((tp + tn + fp + fn), 1)

    auc = None
    ap = None
    if scores and targets:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            y_true = np.concatenate(targets, axis=0)
            y_score = np.concatenate(scores, axis=0)
            if len(np.unique(y_true)) > 1:
                auc = float(roc_auc_score(y_true, y_score))
                ap = float(average_precision_score(y_true, y_score))
        except Exception:
            pass

    return {
        "loss": total_loss / max(total_nodes, 1),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": auc,
        "pr_auc": ap,
        "nodes_eval": int(total_nodes),
        "nodes_auc_sample": int(cfg.max_eval_nodes - remaining),
    }


def train_and_evaluate(cfg: TrainingConfig) -> None:
    if not cfg.graph_dir.exists():
        raise FileNotFoundError(
            f"No existe {cfg.graph_dir}. Ejecuta build_graphs.py primero."
        )

    seed_everything(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.verbose:
        print(f"Device: {cfg.device}", flush=True)
        print("Cargando lista de grafos...", flush=True)
    paths = list_graph_files(cfg.graph_dir)
    train_paths, val_paths, test_paths = split_paths(
        paths, cfg.train_captures, cfg.val_captures, cfg.test_captures
    )
    train_paths = maybe_subsample(
        train_paths,
        dev_mode=cfg.dev_mode,
        max_graphs_per_split=cfg.max_graphs_per_split,
        seed=cfg.seed,
    )
    val_paths = maybe_subsample(
        val_paths,
        dev_mode=cfg.dev_mode,
        max_graphs_per_split=cfg.max_graphs_per_split,
        seed=cfg.seed,
    )
    test_paths = maybe_subsample(
        test_paths,
        dev_mode=cfg.dev_mode,
        max_graphs_per_split=cfg.max_graphs_per_split,
        seed=cfg.seed,
    )

    if cfg.verbose:
        print(
            "Splits -> train: "
            f"{len(train_paths)} | val: {len(val_paths)} | test: {len(test_paths)}",
            flush=True,
        )
    if not train_paths:
        raise ValueError("No hay grafos para entrenamiento. Revisa los splits.")

    if cfg.verbose:
        print("Inicializando loaders...", flush=True)
    train_dataset = GraphPTDataset(train_paths)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    if cfg.verbose:
        print("Cargando sample para inferir in_channels...", flush=True)
    sample = load_graph(train_paths[0])
    in_channels = int(sample.x.size(-1))

    mean_std = _load_feature_stats(cfg) if cfg.normalize_features else None
    pos_weight = _load_pos_weight(cfg)
    if cfg.normalize_features and mean_std is None:
        if cfg.verbose:
            print("Calculando media/std en train...", flush=True)
        mean, std, pos_weight_calc = _compute_train_stats(train_loader, cfg)
        _save_feature_stats(mean, std, cfg)
        mean_std = (mean, std)
        _save_pos_weight(pos_weight_calc, cfg)
        pos_weight = pos_weight_calc
        if cfg.verbose:
            print(f"Stats guardadas en {cfg.feature_stats_cache}", flush=True)
    if cfg.normalize_features and mean_std is not None:
        train_dataset.set_normalization(*mean_std)

    val_dataset = GraphPTDataset(
        val_paths, mean_std[0] if mean_std else None, mean_std[1] if mean_std else None
    )
    test_dataset = GraphPTDataset(
        test_paths, mean_std[0] if mean_std else None, mean_std[1] if mean_std else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    if cfg.verbose:
        print(
            "Inicializando modelo GATv2 (in_channels="
            f"{in_channels}, heads={cfg.heads}, hidden={cfg.hidden_channels})",
            flush=True,
        )
    model = GATv2Net(
        in_channels,
        hidden_channels=cfg.hidden_channels,
        heads=cfg.heads,
        dropout=cfg.dropout,
        use_residual=cfg.use_residual,
    ).to(cfg.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    if pos_weight is None:
        if cfg.verbose:
            print("Calculando pos_weight en train...", flush=True)
        pos_weight = _compute_pos_weight(train_loader, cfg)
        _save_pos_weight(pos_weight, cfg)
        if cfg.verbose:
            print(f"pos_weight calculado: {pos_weight:.4f}", flush=True)
    elif cfg.verbose:
        print(f"pos_weight cargado de cache: {pos_weight:.4f}", flush=True)
    pos_weight_t = torch.tensor([pos_weight], device=cfg.device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

    history = []
    best = {"metric": -1.0, "state_dict": None}

    for epoch in range(1, cfg.epochs + 1):
        if cfg.verbose:
            print(f"\nEpoch {epoch:03d}", flush=True)
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, cfg)
        if cfg.verbose:
            print(f"Epoch {epoch:03d} train_loss={train_loss:.5f}", flush=True)
        val_metrics = _eval_epoch(model, val_loader, criterion, cfg) if val_paths else {}
        test_metrics = _eval_epoch(model, test_loader, criterion, cfg) if test_paths else {}

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val": val_metrics,
            "test": test_metrics,
        }
        history.append(row)

        metric = val_metrics.get("pr_auc") if val_metrics else None
        metric = metric if metric is not None else -train_loss
        if metric > best["metric"]:
            best["metric"] = metric
            best["state_dict"] = {k: v.cpu() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | "
            f"val_pr_auc={val_metrics.get('pr_auc')}"
        )

    with open(cfg.out_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    if best["state_dict"] is not None:
        torch.save(best["state_dict"], cfg.out_dir / "best_model.pt")

    print(f"Entrenamiento completado. Logs en {cfg.out_dir}.")
