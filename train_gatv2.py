from pathlib import Path

from nids.training.config import TrainingConfig
from nids.training.runner import train_and_evaluate

# =========================
# Configuracion principal
# =========================
GRAPH_DIR = Path("data/graphs")
OUT_DIR = Path("data/processed/training")

SEED = 42
'''
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
'''
DEVICE = "cpu"

# Split por captura; ajusta segun el tipo de generalizacion que busques.
TRAIN_CAPTURES = [
    "Friday-02-03-2018",
    "Friday-16-02-2018",
    "Friday-23-02-2018",
    "Thursday-01-03-2018",
    "Thursday-15-02-2018",
    "Thursday-22-02-2018",
    "Wednesday-14-02-2018",
    "Wednesday-28-02-2018",
]
VAL_CAPTURES = ["Tuesday-20-02-2018"]
TEST_CAPTURES = ["Wednesday-21-02-2018"]

DEV_MODE = False
MAX_GRAPHS_PER_SPLIT = 50

BATCH_SIZE = 1
NUM_WORKERS = 0

EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 5e-4

HEADS = 4
HIDDEN_CHANNELS = 32  # por head; total real = HIDDEN_CHANNELS * HEADS
DROPOUT = 0.2
USE_RESIDUAL = True

POS_WEIGHT_CACHE = OUT_DIR / "pos_weight.json"
MAX_EVAL_NODES = 2_000_000
VERBOSE = True
LOG_EVERY_N = 50
NORMALIZE_FEATURES = True
FEATURE_STATS_CACHE = OUT_DIR / "feature_stats.json"


def _build_config() -> TrainingConfig:
    return TrainingConfig(
        graph_dir=GRAPH_DIR,
        out_dir=OUT_DIR,
        seed=SEED,
        device=DEVICE,
        train_captures=TRAIN_CAPTURES,
        val_captures=VAL_CAPTURES,
        test_captures=TEST_CAPTURES,
        dev_mode=DEV_MODE,
        max_graphs_per_split=MAX_GRAPHS_PER_SPLIT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        heads=HEADS,
        hidden_channels=HIDDEN_CHANNELS,
        dropout=DROPOUT,
        use_residual=USE_RESIDUAL,
        pos_weight_cache=POS_WEIGHT_CACHE,
        max_eval_nodes=MAX_EVAL_NODES,
        verbose=VERBOSE,
        log_every_n=LOG_EVERY_N,
        normalize_features=NORMALIZE_FEATURES,
        feature_stats_cache=FEATURE_STATS_CACHE,
    )


def main() -> None:
    train_and_evaluate(_build_config())


if __name__ == "__main__":
    main()
