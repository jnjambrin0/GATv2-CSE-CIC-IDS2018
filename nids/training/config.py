from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class TrainingConfig:
    graph_dir: Path
    out_dir: Path
    seed: int
    device: str
    train_captures: Sequence[str]
    val_captures: Sequence[str]
    test_captures: Sequence[str]
    dev_mode: bool
    max_graphs_per_split: int
    batch_size: int
    num_workers: int
    epochs: int
    lr: float
    weight_decay: float
    heads: int
    hidden_channels: int
    dropout: float
    use_residual: bool
    pos_weight_cache: Path
    max_eval_nodes: int
    verbose: bool
    log_every_n: int
    normalize_features: bool
    feature_stats_cache: Path
