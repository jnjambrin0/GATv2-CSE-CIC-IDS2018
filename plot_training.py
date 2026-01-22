import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_history(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")
    with open(path, "r", encoding="utf-8") as f:
        history = json.load(f)
    if not isinstance(history, list):
        raise ValueError("train_history.json no es una lista de epochs.")
    return history


def _get_series(history: List[Dict[str, Any]], keys: List[str]) -> List[Optional[float]]:
    values: List[Optional[float]] = []
    for row in history:
        val: Any = row
        for k in keys:
            if not isinstance(val, dict) or k not in val:
                val = None
                break
            val = val[k]
        values.append(float(val) if val is not None else None)
    return values


def _plot_if(ax, x, y, label: str) -> None:
    if any(v is not None for v in y):
        ax.plot(x, y, label=label)


def _last_valid(values: List[Optional[float]]) -> Optional[float]:
    for v in reversed(values):
        if v is not None:
            return v
    return None


def _best_epoch(history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    best = None
    best_metric = None
    for row in history:
        val = row.get("val", {})
        pr_auc = val.get("pr_auc") if isinstance(val, dict) else None
        if pr_auc is None:
            continue
        if best is None or pr_auc > best_metric:
            best = row
            best_metric = pr_auc
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from train_history.json")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/training/train_history.json",
        help="Path al train_history.json",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/processed/training/plots",
        help="Directorio de salida para graficas",
    )
    args = parser.parse_args()

    history = _load_history(Path(args.input))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = [row.get("epoch", i + 1) for i, row in enumerate(history)]

    train_loss = _get_series(history, ["train_loss"])
    val_loss = _get_series(history, ["val", "loss"])
    test_loss = _get_series(history, ["test", "loss"])

    val_pr = _get_series(history, ["val", "pr_auc"])
    val_roc = _get_series(history, ["val", "roc_auc"])
    test_pr = _get_series(history, ["test", "pr_auc"])
    test_roc = _get_series(history, ["test", "roc_auc"])

    val_f1 = _get_series(history, ["val", "f1"])
    val_prec = _get_series(history, ["val", "precision"])
    val_rec = _get_series(history, ["val", "recall"])

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    _plot_if(ax, epochs, train_loss, "train_loss")
    _plot_if(ax, epochs, val_loss, "val_loss")
    _plot_if(ax, epochs, test_loss, "test_loss")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.savefig(out_dir / "loss_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    _plot_if(ax, epochs, val_pr, "val_pr_auc")
    _plot_if(ax, epochs, val_roc, "val_roc_auc")
    _plot_if(ax, epochs, test_pr, "test_pr_auc")
    _plot_if(ax, epochs, test_roc, "test_roc_auc")
    ax.set_title("AUC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend()
    fig.savefig(out_dir / "auc_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    _plot_if(ax, epochs, val_f1, "val_f1")
    _plot_if(ax, epochs, val_prec, "val_precision")
    _plot_if(ax, epochs, val_rec, "val_recall")
    ax.set_title("Val metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend()
    fig.savefig(out_dir / "val_metrics.png", dpi=160)
    plt.close(fig)

    best = _best_epoch(history)
    if best:
        print(
            f"Best epoch by val_pr_auc: epoch={best.get('epoch')} "
            f"val_pr_auc={best.get('val', {}).get('pr_auc')}"
        )
    print(f"Graficas guardadas en: {out_dir}")
    print(
        "Ultimos valores -> "
        f"train_loss={_last_valid(train_loss)} "
        f"val_pr_auc={_last_valid(val_pr)} "
        f"val_f1={_last_valid(val_f1)}"
    )


if __name__ == "__main__":
    main()
