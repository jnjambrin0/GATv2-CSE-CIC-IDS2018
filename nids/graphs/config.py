from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class GraphBuildConfig:
    parquet_dir: Path
    feature_cols_json: Path
    out_dir: Path
    window_size_seconds: int
    max_nodes_per_graph: int
    min_nodes_per_graph: int
    k_ip: int
    k_time: int
    captures: Optional[Iterable[str]]
    dev_mode: bool
    max_rows_per_capture: int
    debug_summary: bool
    debug_summary_path: Path
