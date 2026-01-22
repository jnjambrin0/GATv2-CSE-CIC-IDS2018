import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch


def list_graph_files(graph_dir: Path) -> List[Path]:
    return sorted(graph_dir.glob("*.pt"))


def capture_name(path: Path) -> str:
    return path.name.split("_w")[0]


def split_paths(
    paths: Sequence[Path],
    train_caps: Iterable[str],
    val_caps: Iterable[str],
    test_caps: Iterable[str],
) -> Tuple[List[Path], List[Path], List[Path]]:
    train_set = set(train_caps)
    val_set = set(val_caps)
    test_set = set(test_caps)

    train, val, test = [], [], []
    for path in paths:
        cap = capture_name(path)
        if cap in test_set:
            test.append(path)
        elif cap in val_set:
            val.append(path)
        elif cap in train_set:
            train.append(path)
    return train, val, test


def maybe_subsample(
    paths: List[Path], *, dev_mode: bool, max_graphs_per_split: int, seed: int
) -> List[Path]:
    if not dev_mode or len(paths) <= max_graphs_per_split:
        return paths
    rng = random.Random(seed)
    return rng.sample(paths, max_graphs_per_split)


def load_graph(path: Path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


class GraphPTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        paths: Sequence[Path],
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
    ):
        self.paths = list(paths)
        self.mean = mean
        self.std = std

    def set_normalization(
        self, mean: Optional[torch.Tensor], std: Optional[torch.Tensor]
    ) -> None:
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        data = load_graph(self.paths[idx])
        data.x = data.x.float()
        if self.mean is not None and self.std is not None:
            data.x = (data.x - self.mean) / self.std
        data.y = data.y.view(-1).float()
        return data
