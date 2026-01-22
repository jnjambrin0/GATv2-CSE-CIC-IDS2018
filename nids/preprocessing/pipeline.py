import json
from typing import List

import pandas as pd

from .config import PreprocessConfig
from .loader import load_flows, process_per_file_to_parquet
from .transform import clean_and_label, compute_feature_cols


def maybe_save_outputs(df: pd.DataFrame, feature_cols: List[str], cfg: PreprocessConfig) -> None:
    cfg.feature_cols_json_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg.save_feature_cols_json:
        with open(cfg.feature_cols_json_path, "w", encoding="utf-8") as f:
            json.dump(feature_cols, f, indent=2, ensure_ascii=False)
        print(f"\nGuardado feature_cols en: {cfg.feature_cols_json_path}")

    if cfg.save_clean_df:
        cfg.clean_df_out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(
                cfg.clean_df_out_path,
                index=False,
                compression=cfg.parquet_compression,
            )
            print(f"Guardado dataframe limpio en: {cfg.clean_df_out_path}")
        except Exception as exc:
            print("\nNo se pudo guardar a parquet. Motivo:")
            print(exc)
            print("Solucion tipica: `pip install pyarrow` o `pip install fastparquet`.")


def preflight_checks_concat(df: pd.DataFrame, cfg: PreprocessConfig) -> None:
    """
    Preflight checks before graph construction:
    - required columns
    - per-capture stats
    - window size distribution for a subset of captures
    """
    if not cfg.run_preflight_checks:
        return

    print("\n=========================")
    print("PREFLIGHT CHECKS")
    print("=========================")

    required = {"Timestamp", "src_ip", "dst_ip", "label_bin", "capture_name"}
    missing = required - set(df.columns)
    print("Missing required columns:", missing)

    try:
        attack_rate = df.groupby("capture_name")["label_bin"].mean().sort_values(
            ascending=False
        )
        counts = df.groupby("capture_name")["label_bin"].size().sort_values(
            ascending=False
        )
        print("\nAttack rate por captura (mean(label_bin)):")
        print(attack_rate)
        print("\nNum de flujos por captura:")
        print(counts)
    except Exception as exc:
        print("\nNo se pudo calcular attack rate por captura:", exc)

    try:
        tstats = df.groupby("capture_name")["Timestamp"].agg(["min", "max", "count"]).sort_values(
            "min"
        )
        print("\nRango temporal por captura:")
        print(tstats)
    except Exception as exc:
        print("\nNo se pudo calcular rango temporal por captura:", exc)

    if cfg.preflight_window_stats:
        try:
            caps = list(df["capture_name"].astype(str).unique())
            caps = caps[: cfg.preflight_max_captures_for_stats]

            print(
                "\nStats de nodos por ventana "
                f"(WINDOW_SIZE={cfg.preflight_window_size_seconds}s) en capturas: {caps}"
            )

            for cap in caps:
                dfc = df[df["capture_name"].astype(str) == cap]

                if (
                    cfg.preflight_sample_rows_per_capture is not None
                    and len(dfc) > cfg.preflight_sample_rows_per_capture
                ):
                    dfc = dfc.sample(cfg.preflight_sample_rows_per_capture, random_state=42)

                t0 = dfc["Timestamp"].min()
                t_sec = (dfc["Timestamp"] - t0).dt.total_seconds()
                win = (t_sec // cfg.preflight_window_size_seconds).astype("int64")

                sizes = win.value_counts()
                print(f"\n[{cap}] Ventanas: {sizes.size}, Nodos/ventana describe():")
                print(sizes.describe(percentiles=[0.5, 0.9, 0.95, 0.99]))
        except Exception as exc:
            print("\nNo se pudieron calcular stats de ventanas:", exc)


def run_preprocessing(cfg: PreprocessConfig) -> None:
    if cfg.process_mode == "per_file_parquet":
        process_per_file_to_parquet(cfg)
        return

    if cfg.process_mode != "concat":
        raise ValueError(f"PROCESS_MODE invalido: {cfg.process_mode}")

    full_df = load_flows(cfg)
    full_df = clean_and_label(full_df, cfg)
    feature_cols = compute_feature_cols(full_df, cfg)

    print("\nResumen final:")
    print("Shape final:", full_df.shape)
    print("Columnas totales:", len(full_df.columns))

    preflight_checks_concat(full_df, cfg)
    maybe_save_outputs(full_df, feature_cols, cfg)
