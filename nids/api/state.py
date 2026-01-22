from dataclasses import dataclass
from typing import List, Optional

import torch

from nids.graphs.config import GraphBuildConfig
from nids.preprocessing.config import PreprocessConfig


@dataclass
class ModelAssets:
    model: torch.nn.Module
    feature_cols: List[str]
    mean: Optional[torch.Tensor]
    std: Optional[torch.Tensor]
    device: str
    threshold: float
    graph_cfg: GraphBuildConfig
    preprocess_cfg: PreprocessConfig
    progress_every_graphs: int
