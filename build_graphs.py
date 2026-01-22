from pathlib import Path
from typing import Iterable, Optional

from nids.graphs.builder import build_graphs
from nids.graphs.config import GraphBuildConfig

# =========================
# Configuracion principal
# =========================
PARQUET_DIR = Path("data/processed/cleaned_parquet")
FEATURE_COLS_JSON = Path("data/processed/feature_cols.json")
OUT_DIR = Path("data/graphs")

WINDOW_SIZE_SECONDS = 60
MAX_NODES_PER_GRAPH = 25_000
MIN_NODES_PER_GRAPH = 50

K_IP = 5
K_TIME = 3

CAPTURES: Optional[Iterable[str]] = None  # None -> todas las capturas
DEV_MODE = False
MAX_ROWS_PER_CAPTURE = 1_000_000

# Debug y validacion; reporte opcional
DEBUG_SUMMARY = True
DEBUG_SUMMARY_PATH = Path("data/processed/graph_build_report.json")


def _build_config() -> GraphBuildConfig:
    return GraphBuildConfig(
        parquet_dir=PARQUET_DIR,
        feature_cols_json=FEATURE_COLS_JSON,
        out_dir=OUT_DIR,
        window_size_seconds=WINDOW_SIZE_SECONDS,
        max_nodes_per_graph=MAX_NODES_PER_GRAPH,
        min_nodes_per_graph=MIN_NODES_PER_GRAPH,
        k_ip=K_IP,
        k_time=K_TIME,
        captures=CAPTURES,
        dev_mode=DEV_MODE,
        max_rows_per_capture=MAX_ROWS_PER_CAPTURE,
        debug_summary=DEBUG_SUMMARY,
        debug_summary_path=DEBUG_SUMMARY_PATH,
    )


def main() -> None:
    build_graphs(_build_config())


if __name__ == "__main__":
    main()
