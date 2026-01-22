from typing import List, Optional

import numpy as np
import pandas as pd

from .config import PreprocessConfig


def _normalize_label(x: str) -> str:
    if pd.isna(x):
        return x
    x = str(x).strip()
    if x == "Label":
        return x
    return x


def _drop_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    masks = []
    if "Label" in df.columns:
        masks.append(df["Label"].astype(str) == "Label")
    if "Timestamp" in df.columns:
        masks.append(df["Timestamp"].astype(str) == "Timestamp")

    if not masks:
        return df

    header_like = masks[0]
    for mask in masks[1:]:
        header_like |= mask

    n_bad = int(header_like.sum())
    if n_bad > 0:
        print(f"  - Eliminando {n_bad} filas tipo header incrustado")
        df = df[~header_like].copy()
    return df


def _parse_timestamp(series: pd.Series, cfg: PreprocessConfig) -> pd.Series:
    s = series.astype(str).str.strip()

    if cfg.timestamp_format:
        return pd.to_datetime(s, errors="coerce", format=cfg.timestamp_format)

    iso_like = s.str.match(r"^\d{4}-\d{2}-\d{2}[ T]")
    if float(iso_like.mean()) >= 0.90:
        return pd.to_datetime(s, errors="coerce", dayfirst=False)

    ts = pd.to_datetime(s, errors="coerce", dayfirst=cfg.timestamp_dayfirst)

    nat_ratio = float(ts.isna().mean())
    if nat_ratio > cfg.timestamp_nat_fallback_ratio:
        ts_alt = pd.to_datetime(s, errors="coerce", dayfirst=not cfg.timestamp_dayfirst)
        if float(ts_alt.isna().mean()) < nat_ratio:
            ts = ts_alt

    return ts


def _print_memory_usage(df: pd.DataFrame, label: str, cfg: PreprocessConfig) -> None:
    if not cfg.print_memory_usage:
        return
    try:
        mem = df.memory_usage(deep=True).sum()
        gib = mem / (1024**3)
        print(f"[RAM] {label}: {gib:.2f} GiB")
    except Exception:
        # memory_usage con deep=True puede fallar con algunos dtypes; no dejamos que pare el pipeline.
        pass


def clean_and_label(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    expected = {"Label", "Timestamp", "Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol"}
    missing = [c for c in expected if c not in df.columns]
    if missing:
        print(f"AVISO: faltan columnas esperadas: {missing}")

    if "Label" not in df.columns:
        raise KeyError("No existe la columna 'Label' en el dataframe.")

    df = df[df["Label"].notna()].copy()
    if cfg.normalize_label_strings:
        df["Label"] = df["Label"].map(_normalize_label)
    df = df[df["Label"] != "Label"].copy()

    if "Timestamp" in df.columns:
        df["Timestamp"] = _parse_timestamp(df["Timestamp"], cfg)
        if cfg.drop_rows_with_bad_timestamp:
            before = len(df)
            df = df[df["Timestamp"].notna()].copy()
            dropped = before - len(df)
            if dropped > 0:
                print(f"Eliminadas {dropped} filas con Timestamp invalido (NaT).")
    else:
        print("Aviso: no existe 'Timestamp'. Sin esto no podras ventanear por tiempo.")

    rename_map = {
        "id": "row_id",
        "Flow ID": "flow_id",
        "Src IP": "src_ip",
        "Dst IP": "dst_ip",
        "Src Port": "src_port",
        "Dst Port": "dst_port",
        "Protocol": "protocol",
        "Attempted Category": "attempted_category",
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    if existing:
        df = df.rename(columns=existing)

    if "attempted_category" in df.columns:
        df["attempted_category"] = pd.to_numeric(df["attempted_category"], errors="coerce")
        df["attempted_category"] = df["attempted_category"].fillna(-1).astype(np.int8)

    df["Label"] = df["Label"].astype(str).str.strip()
    label_norm = df["Label"].str.casefold()

    df.loc[label_norm == "benign", "Label"] = "BENIGN"
    label_norm = df["Label"].str.casefold()

    is_benign = label_norm == "benign"

    attempted_mask: Optional[pd.Series] = None
    if "attempted_category" in df.columns:
        attempted_mask = df["attempted_category"] != -1

        if cfg.attempted_policy == "drop":
            n_drop = int(attempted_mask.sum())
            print(
                "ATTEMPTED_POLICY=drop -> eliminando "
                f"{n_drop} filas con attempted_category != -1"
            )
            df = df[~attempted_mask].copy()
            label_norm = df["Label"].astype(str).str.strip().str.casefold()
            is_benign = label_norm == "benign"
            attempted_mask = (
                df["attempted_category"] != -1 if "attempted_category" in df.columns else None
            )
        elif cfg.attempted_policy == "relabel_benign":
            is_benign = is_benign | attempted_mask
        elif cfg.attempted_policy == "ignore":
            pass
        else:
            raise ValueError(f"ATTEMPTED_POLICY invalida: {cfg.attempted_policy}")

        print("\nAttempted Category (top 10):")
        print(df["attempted_category"].value_counts().head(10))

    df["label_bin"] = (~is_benign).astype(np.int8)

    print("\nDistribucion label_bin (proporcion):")
    print(df["label_bin"].value_counts(normalize=True))

    if cfg.convert_str_to_category:
        for col in ["Label", "capture_name"]:
            if col in df.columns:
                df[col] = df[col].astype("category")

        if cfg.convert_ips_to_category:
            for col in ["src_ip", "dst_ip"]:
                if col in df.columns:
                    try:
                        df[col] = df[col].astype("category")
                    except Exception:
                        pass

    if cfg.drop_flow_id_column and "flow_id" in df.columns:
        df = df.drop(columns=["flow_id"])
    if cfg.drop_row_id_column and "row_id" in df.columns:
        df = df.drop(columns=["row_id"])

    if (cfg.attempted_policy == "relabel_benign") and (attempted_mask is not None):
        non_benign_by_label = int((label_norm != "benign").sum())
        attempted_count = int(attempted_mask.sum())
        attacks_expected = non_benign_by_label - attempted_count
        attacks_actual = int(df["label_bin"].sum())
        if abs(attacks_expected - attacks_actual) > 10:
            print(
                "AVISO coherencia relabel: attacks_expected="
                f"{attacks_expected}, attacks_actual={attacks_actual} "
                f"(diff={attacks_actual - attacks_expected}). "
                "Revisa attempted_category/Label."
            )

    print("\nDistribucion Label (top 20):")
    print(df["Label"].value_counts().head(20))

    if "Timestamp" in df.columns:
        print("\nRango temporal:")
        print(df["Timestamp"].min(), "->", df["Timestamp"].max())

    _print_memory_usage(df, "Tras clean_and_label", cfg)
    return df


def compute_feature_cols(df: pd.DataFrame, cfg: PreprocessConfig) -> List[str]:
    non_feature_cols = {
        "Label",
        "label_bin",
        "capture_name",
        "Timestamp",
        "row_id",
        "flow_id",
        "src_ip",
        "dst_ip",
        "attempted_category",
        "id",
        "Flow ID",
        "Src IP",
        "Dst IP",
        "Attempted Category",
    }

    if cfg.exclude_ports_and_protocol_from_features:
        non_feature_cols |= {"src_port", "dst_port", "protocol", "Src Port", "Dst Port", "Protocol"}

    feature_cols = [col for col in df.columns if col not in non_feature_cols]

    for col in feature_cols:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif str(df[col].dtype) == "category":
            df[col] = pd.to_numeric(df[col].astype(str), errors="coerce")

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(0.0)
    df[feature_cols] = df[feature_cols].astype("float32")

    print("\nNum de features numericas:", len(feature_cols))
    print("Ejemplo features:", feature_cols[:10])

    bad = [col for col in feature_cols if df[col].dtype == "object"]
    if bad:
        raise ValueError(f"Estas features siguen siendo 'object' tras la limpieza: {bad[:10]} ...")

    _print_memory_usage(df, "Tras compute_feature_cols", cfg)
    return feature_cols
