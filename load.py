import os
from pathlib import Path

from nids.preprocessing.config import PreprocessConfig
from nids.preprocessing.pipeline import run_preprocessing

# =========================
# Configuracion principal
# =========================
FLOW_DIR = Path("data/flows")

# Modo de ejecucion: elige el que mejor encaje en tu maquina.
# - "concat": junta todos los CSV en un solo DataFrame; es comodo, pero pide mucha RAM.
# - "per_file_parquet": procesa CSV uno a uno, limpia y tipa, y guarda cada captura a parquet; usa mucha menos RAM.
PROCESS_MODE = "per_file_parquet"  # valores: "concat" | "per_file_parquet"

# Salida por fichero: solo tiene efecto si PROCESS_MODE es "per_file_parquet".
PER_FILE_OUT_DIR = Path("data/processed/cleaned_parquet")
SAVE_PER_FILE_PARQUET = True
PARQUET_COMPRESSION = "snappy"  # "snappy" suele dar buen equilibrio tamano/velocidad

# Modo desarrollo: limita filas por fichero para iterar rapido.
DEV_MODE = False
MAX_ROWS_PER_FILE = 250_000  # solo se usa con DEV_MODE=True

# Carga paralela: suele acelerar, pero puede subir el pico de RAM.
USE_MULTIPROCESS_LOAD = False
MAX_WORKERS = min(4, (os.cpu_count() or 2))

# Limpieza y validacion basica
DROP_ROWS_WITH_BAD_TIMESTAMP = True
NORMALIZE_LABEL_STRINGS = True

# Parsing de Timestamp con tolerancia a formatos
TIMESTAMP_DAYFIRST = True  # util si el CSV viene en formato dd/mm/yyyy
TIMESTAMP_FORMAT = None  # si quieres forzar formato exacto, pon el string; si no, deja None
TIMESTAMP_NAT_FALLBACK_RATIO = 0.20  # si NaT supera el 20%, intenta parse alternativo invirtiendo dayfirst

# Politica para attempted_category: decide como tratar los flujos "attempted".
# - "ignore": no cambia etiquetas; solo normaliza attempted_category si existe.
# - "drop": elimina las filas con attempted_category != -1.
# - "relabel_benign": trata attempted como benign, aunque Label no sea BENIGN.
ATTEMPTED_POLICY = "relabel_benign"  # valores: "ignore" | "drop" | "relabel_benign"

# Para excluir puertos/protocolo como features numericas, opcional por si quieres evitar sesgos.
EXCLUDE_PORTS_AND_PROTOCOL_FROM_FEATURES = False

# Ajustes de memoria pensados para datasets grandes.
CONVERT_STR_TO_CATEGORY = True
CONVERT_IPS_TO_CATEGORY = True
DROP_FLOW_ID_COLUMN = True
DROP_ROW_ID_COLUMN = False

# Opcional: excluir columnas ya en read_csv; reduce RAM, pero requiere que existan con esos nombres.
EXCLUDE_COLUMNS_AT_READ = True

SAVE_FEATURE_COLS_JSON = True
FEATURE_COLS_JSON_PATH = Path("data/processed/feature_cols.json")

# Guardado del dataframe limpio; puede ser pesado si PROCESS_MODE="concat".
SAVE_CLEAN_DF = False
CLEAN_DF_OUT_PATH = Path("data/processed/flows_clean.parquet")

# Preflight checks: solo imprime en consola.
RUN_PREFLIGHT_CHECKS = True
PREFLIGHT_WINDOW_STATS = True
PREFLIGHT_WINDOW_SIZE_SECONDS = 60
PREFLIGHT_MAX_CAPTURES_FOR_STATS = 2
PREFLIGHT_SAMPLE_ROWS_PER_CAPTURE = None

PRINT_MEMORY_USAGE = True


def _build_config() -> PreprocessConfig:
    return PreprocessConfig(
        flow_dir=FLOW_DIR,
        process_mode=PROCESS_MODE,
        per_file_out_dir=PER_FILE_OUT_DIR,
        save_per_file_parquet=SAVE_PER_FILE_PARQUET,
        parquet_compression=PARQUET_COMPRESSION,
        dev_mode=DEV_MODE,
        max_rows_per_file=MAX_ROWS_PER_FILE,
        use_multiprocess_load=USE_MULTIPROCESS_LOAD,
        max_workers=MAX_WORKERS,
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
        save_feature_cols_json=SAVE_FEATURE_COLS_JSON,
        feature_cols_json_path=FEATURE_COLS_JSON_PATH,
        save_clean_df=SAVE_CLEAN_DF,
        clean_df_out_path=CLEAN_DF_OUT_PATH,
        run_preflight_checks=RUN_PREFLIGHT_CHECKS,
        preflight_window_stats=PREFLIGHT_WINDOW_STATS,
        preflight_window_size_seconds=PREFLIGHT_WINDOW_SIZE_SECONDS,
        preflight_max_captures_for_stats=PREFLIGHT_MAX_CAPTURES_FOR_STATS,
        preflight_sample_rows_per_capture=PREFLIGHT_SAMPLE_ROWS_PER_CAPTURE,
        print_memory_usage=PRINT_MEMORY_USAGE,
    )


if __name__ == "__main__":
    run_preprocessing(_build_config())
