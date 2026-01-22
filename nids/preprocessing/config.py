from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PreprocessConfig:
    flow_dir: Path
    process_mode: str
    per_file_out_dir: Path
    save_per_file_parquet: bool
    parquet_compression: str
    dev_mode: bool
    max_rows_per_file: int
    use_multiprocess_load: bool
    max_workers: int
    drop_rows_with_bad_timestamp: bool
    normalize_label_strings: bool
    timestamp_dayfirst: bool
    timestamp_format: Optional[str]
    timestamp_nat_fallback_ratio: float
    attempted_policy: str
    exclude_ports_and_protocol_from_features: bool
    convert_str_to_category: bool
    convert_ips_to_category: bool
    drop_flow_id_column: bool
    drop_row_id_column: bool
    exclude_columns_at_read: bool
    save_feature_cols_json: bool
    feature_cols_json_path: Path
    save_clean_df: bool
    clean_df_out_path: Path
    run_preflight_checks: bool
    preflight_window_stats: bool
    preflight_window_size_seconds: int
    preflight_max_captures_for_stats: int
    preflight_sample_rows_per_capture: Optional[int]
    print_memory_usage: bool
