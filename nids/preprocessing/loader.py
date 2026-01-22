from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from .config import PreprocessConfig
from .transform import _drop_header_rows, _print_memory_usage, clean_and_label, compute_feature_cols


def _get_read_usecols_excluding(cfg: PreprocessConfig) -> Optional[Callable[[str], bool]]:
    """
    Return a usecols function for read_csv to exclude columns we will drop.
    """
    if not cfg.exclude_columns_at_read:
        return None

    exclude = set()
    if cfg.drop_flow_id_column:
        exclude.add("Flow ID")
    if cfg.drop_row_id_column:
        exclude.add("id")

    if not exclude:
        return None

    return lambda col: col not in exclude


def _read_one_csv(args) -> Tuple[str, pd.DataFrame]:
    fpath_str, cfg, usecols_fn = args
    fpath = Path(fpath_str)

    df = pd.read_csv(
        fpath,
        low_memory=False,
        nrows=(cfg.max_rows_per_file if cfg.dev_mode else None),
        usecols=usecols_fn,
    )
    df = _drop_header_rows(df)
    df["capture_name"] = fpath.stem
    return fpath_str, df


def load_flows(cfg: PreprocessConfig) -> pd.DataFrame:
    csv_files = sorted(cfg.flow_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron CSV en: {cfg.flow_dir.resolve()}")

    print("Num de ficheros de flujo:", len(csv_files))
    usecols_fn = _get_read_usecols_excluding(cfg)

    if not cfg.use_multiprocess_load:
        dfs = []
        for fpath in csv_files:
            print(f"\nLeyendo: {fpath.name}")
            df = pd.read_csv(
                fpath,
                low_memory=False,
                nrows=(cfg.max_rows_per_file if cfg.dev_mode else None),
                usecols=usecols_fn,
            )
            df = _drop_header_rows(df)
            df["capture_name"] = fpath.stem
            dfs.append(df)
            print(f"  - Shape parcial: {df.shape}")

        full_df = pd.concat(dfs, ignore_index=True)
        print("\nShape inicial (concat):", full_df.shape)
        _print_memory_usage(full_df, "Tras concat", cfg)
        return full_df

    print(f"Usando multiproceso con MAX_WORKERS={cfg.max_workers}")
    args = [(str(fpath), cfg, usecols_fn) for fpath in csv_files]

    results: Dict[str, pd.DataFrame] = {}
    with ProcessPoolExecutor(max_workers=cfg.max_workers) as ex:
        for fpath_str, df in ex.map(_read_one_csv, args):
            results[fpath_str] = df
            print(f"Leido: {Path(fpath_str).name} -> {df.shape}")

    dfs = [results[str(fpath)] for fpath in csv_files]
    full_df = pd.concat(dfs, ignore_index=True)
    print("\nShape inicial (concat):", full_df.shape)
    _print_memory_usage(full_df, "Tras concat (multiproceso)", cfg)
    return full_df


def process_per_file_to_parquet(cfg: PreprocessConfig) -> None:
    """
    Process each CSV independently (bounded RAM), apply clean_and_label +
    compute_feature_cols, and write parquet per capture.
    """
    csv_files = sorted(cfg.flow_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron CSV en: {cfg.flow_dir.resolve()}")

    cfg.per_file_out_dir.mkdir(parents=True, exist_ok=True)

    feature_cols_ref: Optional[List[str]] = None

    print("Num de ficheros de flujo:", len(csv_files))
    usecols_fn = _get_read_usecols_excluding(cfg)

    for fpath in csv_files:
        print(f"\nProcesando fichero: {fpath.name}")
        df = pd.read_csv(
            fpath,
            low_memory=False,
            nrows=(cfg.max_rows_per_file if cfg.dev_mode else None),
            usecols=usecols_fn,
        )
        df = _drop_header_rows(df)
        df["capture_name"] = fpath.stem

        df = clean_and_label(df, cfg)
        feature_cols = compute_feature_cols(df, cfg)

        if feature_cols_ref is None:
            feature_cols_ref = feature_cols
        else:
            missing = set(feature_cols_ref) - set(feature_cols)
            extra = set(feature_cols) - set(feature_cols_ref)
            if missing or extra:
                raise ValueError(
                    f"Inconsistencia de features en {fpath.name}. "
                    f"Missing={list(missing)[:10]} Extra={list(extra)[:10]}"
                )

        if cfg.save_per_file_parquet:
            out_path = cfg.per_file_out_dir / f"{fpath.stem}.parquet"
            try:
                df.to_parquet(out_path, index=False, compression=cfg.parquet_compression)
                print(f"Guardado: {out_path}")
            except Exception as exc:
                print("No se pudo guardar parquet (instala pyarrow/fastparquet):", exc)
                raise

        del df

    if feature_cols_ref is not None and cfg.save_feature_cols_json:
        cfg.feature_cols_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.feature_cols_json_path, "w", encoding="utf-8") as f:
            import json

            json.dump(feature_cols_ref, f, indent=2, ensure_ascii=False)
        print(f"\nGuardado feature_cols en: {cfg.feature_cols_json_path}")

    print("\nProcesado por fichero completado.")
