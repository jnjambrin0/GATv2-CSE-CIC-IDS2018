from pathlib import Path

from nids.graphs.config import GraphBuildConfig
from nids.preprocessing.config import PreprocessConfig

# =========================
# Assets principales de la API
# =========================
MODEL_PATH = Path("data/processed/training/best_model.pt")
FEATURE_COLS_PATH = Path("data/processed/feature_cols.json")
FEATURE_STATS_PATH = Path("data/processed/training/feature_stats.json")

DEVICE = "cpu"
THRESHOLD = 0.95

# =========================
# Config de preprocessing: mantener alineada con load.py
# =========================
ATTEMPTED_POLICY = "relabel_benign"
EXCLUDE_PORTS_AND_PROTOCOL_FROM_FEATURES = False
CONVERT_STR_TO_CATEGORY = True
CONVERT_IPS_TO_CATEGORY = True
DROP_FLOW_ID_COLUMN = True
DROP_ROW_ID_COLUMN = False
EXCLUDE_COLUMNS_AT_READ = True
DROP_ROWS_WITH_BAD_TIMESTAMP = True
NORMALIZE_LABEL_STRINGS = True
TIMESTAMP_DAYFIRST = True
TIMESTAMP_FORMAT = None
TIMESTAMP_NAT_FALLBACK_RATIO = 0.20
PRINT_MEMORY_USAGE = False

# =========================
# Config de grafos: mantener alineada con build_graphs.py
# =========================
WINDOW_SIZE_SECONDS = 60
MAX_NODES_PER_GRAPH = 25_000
MIN_NODES_PER_GRAPH = 50
K_IP = 5
K_TIME = 3

# Streaming: cada cuantos grafos enviamos progreso
PROGRESS_EVERY_GRAPHS = 10


def build_preprocess_config() -> PreprocessConfig:
    return PreprocessConfig(
        flow_dir=Path("data/flows"),
        process_mode="concat",
        per_file_out_dir=Path("data/processed/cleaned_parquet"),
        save_per_file_parquet=False,
        parquet_compression="snappy",
        dev_mode=False,
        max_rows_per_file=0,
        use_multiprocess_load=False,
        max_workers=1,
        drop_rows_with_bad_timestamp=DROP_ROWS_WITH_BAD_TIMESTAMP,
        normalize_label_strings=NORMALIZE_LABEL_STRINGS,
        timestamp_dayfirst=TIMESTAMP_DAYFIRST,
        timestamp_format=TIMESTAMP_FORMAT,
        timestamp_nat_fallback_ratio=TIMESTAMP_NAT_FALLBACK_RATIO,
        attempted_policy=ATTEMPTED_POLICY,
        exclude_ports_and_protocol_from_features=EXCLUDE_PORTS_AND_PROTOCOL_FROM_FEATURES,
        convert_str_to_category=CONVERT_STR_TO_CATEGORY,
        convert_ips_to_category=CONVERT_IPS_TO_CATEGORY,
        drop_flow_id_column=DROP_FLOW_ID_COLUMN,
        drop_row_id_column=DROP_ROW_ID_COLUMN,
        exclude_columns_at_read=EXCLUDE_COLUMNS_AT_READ,
        save_feature_cols_json=False,
        feature_cols_json_path=Path("data/processed/feature_cols.json"),
        save_clean_df=False,
        clean_df_out_path=Path("data/processed/flows_clean.parquet"),
        run_preflight_checks=False,
        preflight_window_stats=False,
        preflight_window_size_seconds=WINDOW_SIZE_SECONDS,
        preflight_max_captures_for_stats=0,
        preflight_sample_rows_per_capture=None,
        print_memory_usage=PRINT_MEMORY_USAGE,
    )


def build_graph_config() -> GraphBuildConfig:
    return GraphBuildConfig(
        parquet_dir=Path("data/processed/cleaned_parquet"),
        feature_cols_json=FEATURE_COLS_PATH,
        out_dir=Path("data/graphs"),
        window_size_seconds=WINDOW_SIZE_SECONDS,
        max_nodes_per_graph=MAX_NODES_PER_GRAPH,
        min_nodes_per_graph=MIN_NODES_PER_GRAPH,
        k_ip=K_IP,
        k_time=K_TIME,
        captures=None,
        dev_mode=False,
        max_rows_per_capture=0,
        debug_summary=False,
        debug_summary_path=Path("data/processed/graph_build_report.json"),
    )
