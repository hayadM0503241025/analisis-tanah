from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

pd.set_option("future.no_silent_downcasting", True)

try:
    from scipy import stats
except Exception:  # pragma: no cover - app still works without scipy
    stats = None

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except Exception:  # pragma: no cover - app still works without sklearn
    LinearRegression = None
    mean_absolute_error = None
    mean_squared_error = None
    r2_score = None


DEFAULT_FILE = "Data_Suleman hasil Analisis Tanah Lab dan Portabel.xlsx"
APP_TITLE = "Analisis Tanah: Laboratorium vs Alat Portabel"
PAGES = [
    "Ringkasan",
    "Korelasi Lab-Portabel",
    "Regresi Berganda",
]

LAB_REGRESSION_TARGETS = [
    {
        "label": "pH Lab",
        "parameter": "pH",
        "target": "pH_lab",
        "unit": "pH",
        "default_predictors": ["pH_portabel_1", "pH_portabel_2"],
    },
    {
        "label": "N-NH4 Lab",
        "parameter": "N-NH4",
        "target": "NH4_N_lab",
        "unit": "ppm",
        "default_predictors": ["N_portabel_1", "N_portabel_2"],
    },
    {
        "label": "N-NO3 Lab",
        "parameter": "N-NO3",
        "target": "NO3_N_lab",
        "unit": "ppm",
        "default_predictors": ["N_portabel_1", "N_portabel_2"],
    },
    {
        "label": "P Lab",
        "parameter": "P",
        "target": "P_lab",
        "unit": "ppm",
        "default_predictors": ["P_portabel_1", "P_portabel_2"],
    },
    {
        "label": "K Lab",
        "parameter": "K",
        "target": "K_lab",
        "unit": "ppm",
        "default_predictors": ["K_portabel_1", "K_portabel_2"],
    },
    {
        "label": "Kelembapan Lab",
        "parameter": "Kelembapan",
        "target": "kadar_air_lab_pct",
        "unit": "%",
        "default_predictors": ["kadar_air_portabel_pct"],
    },
]

PORTABLE_PREDICTORS = [
    "pH_portabel_1",
    "pH_portabel_2",
    "N_portabel_1",
    "N_portabel_2",
    "P_portabel_1",
    "P_portabel_2",
    "K_portabel_1",
    "K_portabel_2",
    "kadar_air_portabel_pct",
]

SOIL_NUMERIC_PREDICTORS = [
    "pasir_pct",
    "debu_pct",
    "liat_pct",
    "C_organik_pct",
]

DEVICE_NUMERIC_PREDICTORS = {
    "Portabel 1": [
        "pH_portabel_1",
        "N_portabel_1",
        "P_portabel_1",
        "K_portabel_1",
    ],
    "Portabel 2": [
        "pH_portabel_2",
        "N_portabel_2",
        "P_portabel_2",
        "K_portabel_2",
        "kadar_air_portabel_pct",
    ],
}

NUMERIC_PREDICTORS = [
    *PORTABLE_PREDICTORS,
    *SOIL_NUMERIC_PREDICTORS,
]

CATEGORY_PREDICTORS = [
    "SPT",
    "satuan_tanah",
    "landform",
    "bahan_induk",
    "tekstur_kelas",
    "C_organik_kelas",
    "vegetasi",
]


@dataclass(frozen=True)
class ComparisonSpec:
    parameter: str
    lab_col: str
    portable_col: str
    device: str
    unit: str


COMPARISONS = [
    ComparisonSpec("pH", "pH_lab", "pH_portabel_1", "Portabel 1", "pH"),
    ComparisonSpec("pH", "pH_lab", "pH_portabel_2", "Portabel 2", "pH"),
    ComparisonSpec("N-NH4", "NH4_N_lab", "N_portabel_1", "Portabel 1", "ppm"),
    ComparisonSpec("N-NH4", "NH4_N_lab", "N_portabel_2", "Portabel 2", "ppm"),
    ComparisonSpec("N-NO3", "NO3_N_lab", "N_portabel_1", "Portabel 1", "ppm"),
    ComparisonSpec("N-NO3", "NO3_N_lab", "N_portabel_2", "Portabel 2", "ppm"),
    ComparisonSpec("P", "P_lab", "P_portabel_1", "Portabel 1", "ppm"),
    ComparisonSpec("P", "P_lab", "P_portabel_2", "Portabel 2", "ppm"),
    ComparisonSpec("K", "K_lab", "K_portabel_1", "Portabel 1", "ppm"),
    ComparisonSpec("K", "K_lab", "K_portabel_2", "Portabel 2", "ppm"),
    ComparisonSpec("Kelembapan", "kadar_air_lab_pct", "kadar_air_portabel_pct", "Portabel 2", "%"),
]

ERROR_REGRESSION_TARGETS = [
    {
        "label": "Selisih pH - Portabel 1",
        "parameter": "pH",
        "device": "Portabel 1",
        "target": "pH_galat_1",
        "lab_col": "pH_lab",
        "portable_col": "pH_portabel_1",
        "unit": "pH",
        "default_predictors": ["pH_portabel_1"],
    },
    {
        "label": "Selisih pH - Portabel 2",
        "parameter": "pH",
        "device": "Portabel 2",
        "target": "pH_galat_2",
        "lab_col": "pH_lab",
        "portable_col": "pH_portabel_2",
        "unit": "pH",
        "default_predictors": ["pH_portabel_2"],
    },
    {
        "label": "Selisih N-NH4 - Portabel 1",
        "parameter": "N-NH4",
        "device": "Portabel 1",
        "target": "N_galat_1_vs_NH4",
        "lab_col": "NH4_N_lab",
        "portable_col": "N_portabel_1",
        "unit": "ppm",
        "default_predictors": ["N_portabel_1"],
    },
    {
        "label": "Selisih N-NH4 - Portabel 2",
        "parameter": "N-NH4",
        "device": "Portabel 2",
        "target": "N_galat_2_vs_NH4",
        "lab_col": "NH4_N_lab",
        "portable_col": "N_portabel_2",
        "unit": "ppm",
        "default_predictors": ["N_portabel_2"],
    },
    {
        "label": "Selisih N-NO3 - Portabel 1",
        "parameter": "N-NO3",
        "device": "Portabel 1",
        "target": "N_galat_1_vs_NO3",
        "lab_col": "NO3_N_lab",
        "portable_col": "N_portabel_1",
        "unit": "ppm",
        "default_predictors": ["N_portabel_1"],
    },
    {
        "label": "Selisih N-NO3 - Portabel 2",
        "parameter": "N-NO3",
        "device": "Portabel 2",
        "target": "N_galat_2_vs_NO3",
        "lab_col": "NO3_N_lab",
        "portable_col": "N_portabel_2",
        "unit": "ppm",
        "default_predictors": ["N_portabel_2"],
    },
    {
        "label": "Selisih P - Portabel 1",
        "parameter": "P",
        "device": "Portabel 1",
        "target": "P_galat_1",
        "lab_col": "P_lab",
        "portable_col": "P_portabel_1",
        "unit": "ppm",
        "default_predictors": ["P_portabel_1"],
    },
    {
        "label": "Selisih P - Portabel 2",
        "parameter": "P",
        "device": "Portabel 2",
        "target": "P_galat_2",
        "lab_col": "P_lab",
        "portable_col": "P_portabel_2",
        "unit": "ppm",
        "default_predictors": ["P_portabel_2"],
    },
    {
        "label": "Selisih K - Portabel 1",
        "parameter": "K",
        "device": "Portabel 1",
        "target": "K_galat_1",
        "lab_col": "K_lab",
        "portable_col": "K_portabel_1",
        "unit": "ppm",
        "default_predictors": ["K_portabel_1"],
    },
    {
        "label": "Selisih K - Portabel 2",
        "parameter": "K",
        "device": "Portabel 2",
        "target": "K_galat_2",
        "lab_col": "K_lab",
        "portable_col": "K_portabel_2",
        "unit": "ppm",
        "default_predictors": ["K_portabel_2"],
    },
    {
        "label": "Selisih Kelembapan - Portabel 2",
        "parameter": "Kelembapan",
        "device": "Portabel 2",
        "target": "kadar_air_galat",
        "lab_col": "kadar_air_lab_pct",
        "portable_col": "kadar_air_portabel_pct",
        "unit": "%",
        "default_predictors": ["kadar_air_portabel_pct"],
    },
]


META_COLS = ["SPT", "satuan_tanah", "landform", "bahan_induk", "unit_lahan", "vegetasi"]


def first_existing_file() -> Path | None:
    default = Path(DEFAULT_FILE)
    if default.exists():
        return default
    matches = sorted(Path(".").glob("*.xlsx"))
    return matches[0] if matches else None


def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_spt(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    text = text.replace("SPT", "").strip()
    return text


def read_raw_excel(file_obj: str | Path | object) -> dict[str, pd.DataFrame]:
    return pd.read_excel(file_obj, sheet_name=None, header=None, dtype=object, engine="openpyxl")


def data_rows(raw: pd.DataFrame, start_row: int) -> pd.DataFrame:
    data = raw.iloc[start_row:].copy()
    data = data.dropna(how="all")
    return data


def base_metadata(raw: pd.DataFrame, start_row: int) -> pd.DataFrame:
    data = data_rows(raw, start_row)
    meta = data.iloc[:, 1:7].copy()
    meta.columns = META_COLS
    for col in ["SPT", "satuan_tanah", "landform", "bahan_induk"]:
        meta[col] = meta[col].ffill()
    meta["SPT"] = meta["SPT"].map(normalize_spt)
    meta["unit_lahan"] = to_number(meta["unit_lahan"]).astype("Int64")
    meta["vegetasi"] = meta["vegetasi"].astype(str).str.strip()
    return meta.reset_index(drop=True)


def parse_sheet(
    sheets: dict[str, pd.DataFrame],
    sheet_name: str,
    start_row: int,
    value_cols: dict[str, int],
    required_numeric: str,
) -> pd.DataFrame:
    raw = sheets[sheet_name]
    data = data_rows(raw, start_row).reset_index(drop=True)
    out = base_metadata(raw, start_row)
    for name, index in value_cols.items():
        out[name] = data.iloc[:, index]

    out = out[out["unit_lahan"].notna() & out["vegetasi"].ne("")]
    for name in value_cols:
        if name.endswith("_class") or name in {"tekstur_kelas", "C_organik_kelas"}:
            out[name] = out[name].replace("", np.nan)
        else:
            out[name] = to_number(out[name])
    out = out[to_number(out[required_numeric]).notna()]
    return out.reset_index(drop=True)


def parse_kadar_air(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    raw = sheets["Kadar Air Portabel"].iloc[3:].copy().dropna(how="all")
    out = pd.DataFrame(
        {
            "SPT": raw.iloc[:, 1].ffill().map(normalize_spt),
            "unit_lahan": to_number(raw.iloc[:, 2]).astype("Int64"),
            "vegetasi": raw.iloc[:, 3].astype(str).str.strip(),
            "kadar_air_lab_pct": to_number(raw.iloc[:, 4]),
            "kadar_air_portabel_pct": to_number(raw.iloc[:, 5]),
            "kadar_air_galat": to_number(raw.iloc[:, 6]),
        }
    )
    return out[out["unit_lahan"].notna() & out["kadar_air_lab_pct"].notna()].reset_index(drop=True)


def parse_portable_replicates(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    raw = sheets["Alat Portabel"].iloc[5:].copy().dropna(how="all")
    cols = [
        "nomor",
        "label_sampel",
        "ulangan",
        "pH_portabel_1",
        "pH_portabel_1_mean",
        "N_portabel_1",
        "N_portabel_1_mean",
        "P_portabel_1",
        "P_portabel_1_mean",
        "K_portabel_1",
        "K_portabel_1_mean",
        "pH_portabel_2",
        "pH_portabel_2_mean",
        "N_portabel_2",
        "N_portabel_2_mean",
        "P_portabel_2",
        "P_portabel_2_mean",
        "K_portabel_2",
        "K_portabel_2_mean",
        "kelembapan_portabel_2",
        "kelembapan_portabel_2_mean",
    ]
    raw = raw.iloc[:, : len(cols)].copy()
    raw.columns = cols
    raw["label_sampel"] = raw["label_sampel"].ffill()
    raw["SPT"] = raw["label_sampel"].astype(str).str.extract(r"SPT\s*(\d+)")[0].fillna("")
    raw["unit_lahan"] = to_number(raw["label_sampel"].astype(str).str.extract(r"SPT\s*\d+\.(\d+)")[0]).astype("Int64")
    raw["vegetasi"] = raw["label_sampel"].astype(str).str.extract(r"\((.*)\)")[0].fillna("").str.strip()
    for col in cols:
        if col not in {"label_sampel"}:
            raw[col] = to_number(raw[col])
    return raw.reset_index(drop=True)


def summarize_replicates(replicates: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        "pH_portabel_1",
        "N_portabel_1",
        "P_portabel_1",
        "K_portabel_1",
        "pH_portabel_2",
        "N_portabel_2",
        "P_portabel_2",
        "K_portabel_2",
        "kelembapan_portabel_2",
    ]
    records = []
    for keys, group in replicates.groupby(["SPT", "unit_lahan", "vegetasi"], dropna=False):
        spt, unit, vegetation = keys
        for col in value_cols:
            values = to_number(group[col]).dropna()
            if values.empty:
                continue
            mean = values.mean()
            sd = values.std(ddof=1)
            records.append(
                {
                    "SPT": spt,
                    "unit_lahan": unit,
                    "vegetasi": vegetation,
                    "variabel": col,
                    "rata_rata": mean,
                    "simpangan_baku": sd,
                    "rentang": values.max() - values.min(),
                    "cv_pct": np.nan if mean == 0 else sd / mean * 100,
                }
            )
    return pd.DataFrame(records)


def add_error_regression_targets(clean: pd.DataFrame) -> pd.DataFrame:
    out = clean.copy()
    for target_spec in ERROR_REGRESSION_TARGETS:
        lab_col = target_spec["lab_col"]
        portable_col = target_spec["portable_col"]
        target_col = target_spec["target"]
        if lab_col in out.columns and portable_col in out.columns:
            out[target_col] = to_number(out[portable_col]) - to_number(out[lab_col])
    return out


@st.cache_data(show_spinner=False)
def load_data(file_obj: str | Path | object) -> dict[str, pd.DataFrame]:
    sheets = read_raw_excel(file_obj)

    ph = parse_sheet(
        sheets,
        "pH",
        4,
        {
            "pH_lab": 7,
            "pH_kelas": 8,
            "pH_portabel_1": 9,
            "pH_portabel_2": 10,
            "pH_galat_1": 11,
            "pH_galat_2": 12,
        },
        "pH_lab",
    )
    nitrogen = parse_sheet(
        sheets,
        "N",
        6,
        {
            "NH4_N_lab": 7,
            "NO3_N_lab": 8,
            "N_kelas": 9,
            "N_portabel_1": 10,
            "N_portabel_2": 11,
            "N_galat_1_vs_NH4": 12,
            "N_galat_2_vs_NH4": 13,
            "N_galat_1_vs_NO3": 14,
            "N_galat_2_vs_NO3": 15,
        },
        "NH4_N_lab",
    )
    phosphorus = parse_sheet(
        sheets,
        "P",
        4,
        {
            "P_lab": 7,
            "P_kelas": 8,
            "P_portabel_1": 9,
            "P_portabel_2": 10,
            "P_galat_1": 11,
            "P_galat_2": 12,
        },
        "P_lab",
    )
    potassium = parse_sheet(
        sheets,
        "K",
        5,
        {
            "K_lab": 7,
            "K_kelas": 8,
            "K_portabel_1": 9,
            "K_portabel_2": 10,
            "K_galat_1": 11,
            "K_galat_2": 12,
        },
        "K_lab",
    )
    texture = parse_sheet(
        sheets,
        "Tekstur",
        5,
        {
            "pasir_pct": 21,
            "debu_pct": 22,
            "liat_pct": 23,
            "tekstur_kelas": 24,
        },
        "liat_pct",
    )
    organic = parse_sheet(
        sheets,
        "C-organik",
        6,
        {
            "C_organik_pct": 19,
            "C_organik_kelas": 20,
            "C_organik_kadar_air_pct": 17,
        },
        "C_organik_pct",
    )
    moisture = parse_kadar_air(sheets)
    replicates = parse_portable_replicates(sheets)
    replicate_summary = summarize_replicates(replicates)

    clean = ph[META_COLS + ["pH_lab", "pH_kelas", "pH_portabel_1", "pH_portabel_2"]]
    clean = clean.merge(
        nitrogen[META_COLS + ["NH4_N_lab", "NO3_N_lab", "N_kelas", "N_portabel_1", "N_portabel_2"]],
        on=META_COLS,
        how="outer",
    )
    clean = clean.merge(
        phosphorus[META_COLS + ["P_lab", "P_kelas", "P_portabel_1", "P_portabel_2"]],
        on=META_COLS,
        how="outer",
    )
    clean = clean.merge(
        potassium[META_COLS + ["K_lab", "K_kelas", "K_portabel_1", "K_portabel_2"]],
        on=META_COLS,
        how="outer",
    )
    clean = clean.merge(
        texture[META_COLS + ["pasir_pct", "debu_pct", "liat_pct", "tekstur_kelas"]],
        on=META_COLS,
        how="outer",
    )
    clean = clean.merge(
        organic[META_COLS + ["C_organik_pct", "C_organik_kelas"]],
        on=META_COLS,
        how="outer",
    )
    clean = clean.merge(
        moisture[["SPT", "unit_lahan", "vegetasi", "kadar_air_lab_pct", "kadar_air_portabel_pct"]],
        on=["SPT", "unit_lahan", "vegetasi"],
        how="outer",
    )
    clean = add_error_regression_targets(clean)

    errors = make_error_table(clean)
    return {
        "clean": clean,
        "errors": errors,
        "replicates": replicates,
        "replicate_summary": replicate_summary,
        "ph": ph,
        "nitrogen": nitrogen,
        "phosphorus": phosphorus,
        "potassium": potassium,
        "texture": texture,
        "organic": organic,
        "moisture": moisture,
    }


def make_error_table(clean: pd.DataFrame) -> pd.DataFrame:
    records = []
    for spec in COMPARISONS:
        if spec.lab_col not in clean.columns or spec.portable_col not in clean.columns:
            continue
        subset = clean.copy()
        subset["lab_value"] = to_number(subset[spec.lab_col])
        subset["portable_value"] = to_number(subset[spec.portable_col])
        subset = subset[subset["lab_value"].notna() & subset["portable_value"].notna()]
        if subset.empty:
            continue
        subset["parameter"] = spec.parameter
        subset["alat"] = spec.device
        subset["satuan"] = spec.unit
        subset["galat"] = subset["portable_value"] - subset["lab_value"]
        subset["galat_absolut"] = subset["galat"].abs()
        subset["galat_relatif_pct"] = np.where(
            subset["lab_value"].abs() > 0,
            subset["galat_absolut"] / subset["lab_value"].abs() * 100,
            np.nan,
        )
        keep_cols = [
            "parameter",
            "alat",
            "satuan",
            "SPT",
            "satuan_tanah",
            "landform",
            "bahan_induk",
            "unit_lahan",
            "vegetasi",
            "lab_value",
            "portable_value",
            "galat",
            "galat_absolut",
            "galat_relatif_pct",
            "pasir_pct",
            "debu_pct",
            "liat_pct",
            "tekstur_kelas",
            "C_organik_pct",
            "C_organik_kelas",
            "kadar_air_lab_pct",
            "kadar_air_portabel_pct",
            "pH_lab",
        ]
        records.append(subset[[col for col in keep_cols if col in subset.columns]])
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def numeric_pair(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    pair = df[[x_col, y_col]].copy()
    pair[x_col] = to_number(pair[x_col])
    pair[y_col] = to_number(pair[y_col])
    return pair.dropna()


def pearson(x: pd.Series, y: pd.Series) -> tuple[float, float | None]:
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan, np.nan
    if stats is not None:
        result = stats.pearsonr(x, y)
        return float(result.statistic), float(result.pvalue)
    return float(x.corr(y, method="pearson")), np.nan


def spearman(x: pd.Series, y: pd.Series) -> tuple[float, float | None]:
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan, np.nan
    if stats is not None:
        result = stats.spearmanr(x, y)
        return float(result.statistic), float(result.pvalue)
    return float(x.corr(y, method="spearman")), np.nan


def linear_fit(x: pd.Series, y: pd.Series) -> dict[str, float]:
    if len(x) < 2 or x.nunique() < 2:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan}
    slope, intercept = np.polyfit(x.to_numpy(dtype=float), y.to_numpy(dtype=float), 1)
    pred = intercept + slope * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2)}


def add_regression_line(fig, df: pd.DataFrame, x_col: str, y_col: str, name: str = "Regresi linear") -> None:
    pair = numeric_pair(df, x_col, y_col)
    if len(pair) < 2 or pair[x_col].nunique() < 2:
        return
    fit = linear_fit(pair[x_col], pair[y_col])
    if pd.isna(fit["slope"]) or pd.isna(fit["intercept"]):
        return
    x_min = float(pair[x_col].min())
    x_max = float(pair[x_col].max())
    x_line = np.linspace(x_min, x_max, 50)
    y_line = fit["intercept"] + fit["slope"] * x_line
    fig.add_scatter(
        x=x_line,
        y=y_line,
        mode="lines",
        name=f"{name} (R²={fit['r2']:.3f})",
        line={"color": "#111827", "dash": "dash", "width": 2},
    )


def agreement_stats(df: pd.DataFrame, lab_col: str, portable_col: str) -> dict[str, float]:
    pair = numeric_pair(df, lab_col, portable_col)
    if pair.empty:
        return {}
    lab = pair[lab_col]
    portable = pair[portable_col]
    error = portable - lab
    pearson_r, pearson_p = pearson(lab, portable)
    spearman_r, spearman_p = spearman(lab, portable)
    fit_port_to_lab = linear_fit(portable, lab)
    fit_lab_to_port = linear_fit(lab, portable)
    lab_range = lab.max() - lab.min()
    rmse = float(np.sqrt(np.mean(error**2)))
    return {
        "n": int(len(pair)),
        "lab_mean": float(lab.mean()),
        "portable_mean": float(portable.mean()),
        "bias": float(error.mean()),
        "MAE": float(error.abs().mean()),
        "RMSE": rmse,
        "nRMSE_range_pct": np.nan if lab_range == 0 else float(rmse / lab_range * 100),
        "MAPE_pct": float((error.abs() / lab.abs().replace(0, np.nan) * 100).mean()),
        "Pearson_r": pearson_r,
        "Pearson_p": pearson_p,
        "Spearman_rho": spearman_r,
        "Spearman_p": spearman_p,
        "slope_portable_to_lab": fit_port_to_lab["slope"],
        "intercept_portable_to_lab": fit_port_to_lab["intercept"],
        "R2_portable_to_lab": fit_port_to_lab["r2"],
        "slope_lab_to_portable": fit_lab_to_port["slope"],
        "intercept_lab_to_portable": fit_lab_to_port["intercept"],
        "R2_lab_to_portable": fit_lab_to_port["r2"],
    }


def all_agreement_stats(clean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in COMPARISONS:
        stats_row = agreement_stats(clean, spec.lab_col, spec.portable_col)
        if not stats_row:
            continue
        rows.append(
            {
                "parameter": spec.parameter,
                "alat": spec.device,
                "satuan": spec.unit,
                **stats_row,
            }
        )
    return pd.DataFrame(rows)


def score_agreement(row: pd.Series) -> tuple[float, str]:
    corr = np.nanmax([abs(row.get("Pearson_r", np.nan)), abs(row.get("Spearman_rho", np.nan))])
    if np.isnan(corr):
        corr = 0.0
    nrmse = row.get("nRMSE_range_pct", np.nan)
    nrmse_score = 0.0 if pd.isna(nrmse) else max(0.0, min(1.0, 1 - nrmse / 100))
    bias_ratio = np.nan
    if pd.notna(row.get("lab_mean", np.nan)) and row.get("lab_mean", 0) != 0:
        bias_ratio = abs(row.get("bias", np.nan)) / abs(row.get("lab_mean", np.nan))
    bias_score = 0.0 if pd.isna(bias_ratio) else max(0.0, min(1.0, 1 - bias_ratio))
    score = 100 * (0.45 * corr + 0.35 * nrmse_score + 0.20 * bias_score)
    if score >= 75:
        label = "Layak estimasi setelah kalibrasi"
    elif score >= 55:
        label = "Layak screening, wajib kalibrasi lokal"
    elif score >= 35:
        label = "Terbatas, hanya indikasi kasar"
    else:
        label = "Tidak disarankan untuk keputusan dosis"
    return float(score), label


def recommendation_matrix(stats_df: pd.DataFrame) -> pd.DataFrame:
    if stats_df.empty:
        return stats_df
    matrix = stats_df.copy()
    scored = matrix.apply(score_agreement, axis=1, result_type="expand")
    matrix["skor_kelayakan"] = scored[0]
    matrix["rekomendasi"] = scored[1]
    matrix["persamaan_kalibrasi_lab"] = matrix.apply(
        lambda row: calibration_text(row.get("intercept_portable_to_lab"), row.get("slope_portable_to_lab")),
        axis=1,
    )
    return matrix.sort_values(["parameter", "skor_kelayakan"], ascending=[True, False])


SUBSCRIPT_TRANSLATION = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

ACADEMIC_COLUMN_LABELS = {
    "parameter": "Parameter",
    "alat": "Alat",
    "n": "n (jumlah sampel)",
    "bias": "ē (bias)",
    "MAE": "MAE (galat absolut rata-rata)",
    "RMSE": "RMSE (akar rerata kuadrat galat)",
    "MAPE_pct": "MAPE (%)",
    "nRMSE_range_pct": "nRMSE (%)",
    "Pearson_r": "r (Pearson)",
    "Pearson_p": "p_r (p-value Pearson)",
    "Spearman_rho": "ρ (Spearman)",
    "Spearman_p": "p_ρ (p-value Spearman)",
    "R2_portable_to_lab": "R² (kalibrasi ŷ_lab terhadap x_portabel)",
    "intercept_portable_to_lab": "β₀ (intersep)",
    "slope_portable_to_lab": "β₁ (koefisien x_portabel)",
    "persamaan_regresi_linear": "ŷ_lab = β₀ + β₁x (model regresi linear)",
    "skor_kelayakan": "Skor kelayakan",
    "rekomendasi": "Rekomendasi",
    "persamaan_kalibrasi_lab": "ŷ_lab = β₀ + β₁x (persamaan kalibrasi)",
    "Y": "Y (variabel respons)",
    "jumlah_prediktor": "p (jumlah prediktor)",
    "R2": "R² (koefisien determinasi)",
    "Adj_R2": "R²_adj (koefisien determinasi terkoreksi)",
    "catatan": "Catatan",
    "notasi": "βⱼ (notasi koefisien)",
    "variabel_model": "Xⱼ (notasi prediktor)",
    "suku_model": "βⱼXⱼ (suku model)",
    "prediktor": "Prediktor asli",
    "koefisien": "β̂ⱼ (estimasi koefisien)",
    "prediksi": "Ŷ (prediksi model)",
    "residual": "e = Y - Ŷ (residual)",
    "abs_residual": "|e| (galat absolut)",
}

LINEAR_REGRESSION_LATEX = r"\hat{y}_{lab} = \beta_0 + \beta_1 x_{portabel} + \varepsilon"
MULTIPLE_REGRESSION_LATEX = r"\hat{Y} = \beta_0 + \beta_1X_1 + \beta_2X_2 + \cdots + \beta_pX_p + \varepsilon"


def academic_subscript(value: int | str) -> str:
    return str(value).translate(SUBSCRIPT_TRANSLATION)


def beta_symbol(index: int) -> str:
    return f"β{academic_subscript(index)}"


def x_symbol(index: int) -> str:
    return f"X{academic_subscript(index)}"


def linear_regression_equation(intercept: float, slope: float) -> str:
    if pd.isna(intercept) or pd.isna(slope):
        return "Tidak cukup data"
    sign = "+" if slope >= 0 else "-"
    return f"ŷ_lab = β₀ + β₁x_portabel = {intercept:.3f} {sign} {abs(slope):.3f} × x_portabel"


def multiple_regression_formula(predictor_count: int) -> str:
    if predictor_count <= 0:
        return f"Ŷ = {beta_symbol(0)} + ε"
    if predictor_count <= 3:
        terms = " + ".join(f"{beta_symbol(index)}{x_symbol(index)}" for index in range(1, predictor_count + 1))
    else:
        first_terms = " + ".join(f"{beta_symbol(index)}{x_symbol(index)}" for index in range(1, 4))
        terms = f"{first_terms} + ... + {beta_symbol(predictor_count)}{x_symbol(predictor_count)}"
    return f"Ŷ = {beta_symbol(0)} + {terms} + ε"


def coefficient_table_from_model(predictors: Iterable[str], coefficients: Iterable[float], intercept: float) -> pd.DataFrame:
    records = []
    for index, (predictor, coefficient) in enumerate(zip(predictors, coefficients), start=1):
        beta = beta_symbol(index)
        variable = x_symbol(index)
        records.append(
            {
                "notasi": beta,
                "variabel_model": variable,
                "suku_model": f"{beta}{variable}",
                "prediktor": predictor,
                "koefisien": coefficient,
            }
        )
    records.append(
        {
            "notasi": beta_symbol(0),
            "variabel_model": "",
            "suku_model": beta_symbol(0),
            "prediktor": "intercept",
            "koefisien": intercept,
        }
    )
    return pd.DataFrame(records)


def calibration_text(intercept: float, slope: float) -> str:
    if pd.isna(intercept) or pd.isna(slope):
        return "Tidak cukup data"
    return f"{linear_regression_equation(intercept, slope)} (persamaan kalibrasi linear)"


def format_p(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def categorical_tests(df: pd.DataFrame, value_col: str, group_cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    for group_col in group_cols:
        if group_col not in df.columns:
            continue
        temp = df[[value_col, group_col]].dropna()
        temp = temp[temp[group_col].astype(str).str.strip().ne("")]
        groups = [group[value_col].to_numpy(dtype=float) for _, group in temp.groupby(group_col) if len(group) >= 2]
        total_groups = temp[group_col].nunique()
        if len(groups) < 2:
            rows.append(
                {
                    "faktor": group_col,
                    "jumlah_grup": int(total_groups),
                    "ANOVA_F": np.nan,
                    "ANOVA_p": np.nan,
                    "Kruskal_H": np.nan,
                    "Kruskal_p": np.nan,
                    "catatan": "Butuh minimal 2 grup dengan n >= 2.",
                }
            )
            continue
        if stats is None:
            rows.append(
                {
                    "faktor": group_col,
                    "jumlah_grup": int(total_groups),
                    "ANOVA_F": np.nan,
                    "ANOVA_p": np.nan,
                    "Kruskal_H": np.nan,
                    "Kruskal_p": np.nan,
                    "catatan": "Install scipy untuk ANOVA/Kruskal.",
                }
            )
            continue
        anova = stats.f_oneway(*groups)
        kruskal = stats.kruskal(*groups)
        rows.append(
            {
                "faktor": group_col,
                "jumlah_grup": int(total_groups),
                "ANOVA_F": float(anova.statistic),
                "ANOVA_p": float(anova.pvalue),
                "Kruskal_H": float(kruskal.statistic),
                "Kruskal_p": float(kruskal.pvalue),
                "catatan": "Signifikan bila p < 0.05; cek ukuran sampel per grup.",
            }
        )
    return pd.DataFrame(rows)


def numeric_factor_correlations(df: pd.DataFrame, value_col: str, factor_cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    for col in factor_cols:
        if col not in df.columns:
            continue
        pair = numeric_pair(df, col, value_col)
        if len(pair) < 3:
            continue
        pearson_r, pearson_p = pearson(pair[col], pair[value_col])
        spearman_r, spearman_p = spearman(pair[col], pair[value_col])
        rows.append(
            {
                "faktor": col,
                "n": len(pair),
                "Pearson_r": pearson_r,
                "Pearson_p": pearson_p,
                "Spearman_rho": spearman_r,
                "Spearman_p": spearman_p,
            }
        )
    return pd.DataFrame(rows)


def multiple_regression(
    df: pd.DataFrame,
    target_col: str,
    numeric_cols: list[str],
    category_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    if LinearRegression is None:
        return pd.DataFrame(), {"catatan": "Install scikit-learn untuk regresi berganda."}

    category_cols = category_cols or []
    use_cols = [target_col] + [col for col in numeric_cols if col in df.columns] + [col for col in category_cols if col in df.columns]
    temp = df[use_cols].copy()
    for col in numeric_cols:
        if col in temp:
            temp[col] = to_number(temp[col])
    temp = temp.dropna(subset=[target_col])
    temp = temp.dropna(axis=1, how="all")
    temp = temp.dropna()
    if len(temp) < 5:
        return pd.DataFrame(), {"catatan": "Data terlalu sedikit setelah drop missing value."}

    y = temp[target_col].astype(float)
    x_num = temp[[col for col in numeric_cols if col in temp.columns]].astype(float)
    parts = [x_num]
    if category_cols:
        x_cat = pd.get_dummies(temp[[col for col in category_cols if col in temp.columns]].astype(str), drop_first=True)
        parts.append(x_cat)
    x = pd.concat(parts, axis=1)
    x = x.loc[:, x.nunique(dropna=False) > 1]
    if x.empty or len(temp) <= len(x.columns) + 1:
        return pd.DataFrame(), {"catatan": "Jumlah prediktor terlalu banyak dibanding jumlah sampel."}

    model = LinearRegression()
    model.fit(x, y)
    pred = model.predict(x)
    r2 = r2_score(y, pred) if r2_score is not None else np.nan
    adj_r2 = 1 - (1 - r2) * (len(y) - 1) / max(1, len(y) - x.shape[1] - 1)
    rmse = np.sqrt(mean_squared_error(y, pred)) if mean_squared_error is not None else np.nan
    mae = mean_absolute_error(y, pred) if mean_absolute_error is not None else np.nan
    coef = coefficient_table_from_model(x.columns, model.coef_, model.intercept_)
    summary = {
        "n": len(y),
        "jumlah_prediktor": x.shape[1],
        "R2": r2,
        "Adj_R2": adj_r2,
        "MAE": mae,
        "RMSE": rmse,
        "formula": multiple_regression_formula(x.shape[1]),
    }
    return coef.sort_values("koefisien", key=lambda s: s.abs(), ascending=False).reset_index(drop=True), summary


def regression_with_prediction(
    df: pd.DataFrame,
    target_col: str,
    numeric_cols: list[str],
    category_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, float | str], pd.DataFrame]:
    if LinearRegression is None:
        return pd.DataFrame(), {"catatan": "Install scikit-learn untuk regresi berganda."}, pd.DataFrame()

    category_cols = category_cols or []
    use_cols = [
        "SPT",
        "unit_lahan",
        "vegetasi",
        target_col,
        *[col for col in numeric_cols if col in df.columns],
        *[col for col in category_cols if col in df.columns],
    ]
    use_cols = list(dict.fromkeys(use_cols))
    temp = df[use_cols].copy()
    for col in [target_col, *numeric_cols]:
        if col in temp:
            temp[col] = to_number(temp[col])
    temp = temp.dropna(subset=[target_col])
    temp = temp.dropna(axis=1, how="all")
    model_data = temp.dropna().copy()
    if len(model_data) < 6:
        return pd.DataFrame(), {"catatan": "Data terlalu sedikit setelah missing value dibuang."}, pd.DataFrame()

    y = model_data[target_col].astype(float)
    x_num_cols = [col for col in numeric_cols if col in model_data.columns and col != target_col]
    x_num = model_data[x_num_cols].astype(float)
    parts = [x_num]
    if category_cols:
        cat_cols = [col for col in category_cols if col in model_data.columns]
        if cat_cols:
            parts.append(pd.get_dummies(model_data[cat_cols].astype(str), drop_first=True))
    x = pd.concat(parts, axis=1)
    x = x.loc[:, x.nunique(dropna=False) > 1]
    if x.empty:
        return pd.DataFrame(), {"catatan": "Tidak ada prediktor valid setelah data dibersihkan."}, pd.DataFrame()
    if len(model_data) <= len(x.columns) + 1:
        return (
            pd.DataFrame(),
            {"catatan": "Jumlah prediktor terlalu banyak dibanding jumlah sampel. Kurangi prediktor atau kategori."},
            pd.DataFrame(),
        )

    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    residual = y - y_pred
    r2 = r2_score(y, y_pred) if r2_score is not None else np.nan
    adj_r2 = 1 - (1 - r2) * (len(y) - 1) / max(1, len(y) - x.shape[1] - 1)
    rmse = np.sqrt(mean_squared_error(y, y_pred)) if mean_squared_error is not None else np.nan
    mae = mean_absolute_error(y, y_pred) if mean_absolute_error is not None else np.nan

    coef = coefficient_table_from_model(x.columns, model.coef_, model.intercept_)
    coef = coef.sort_values("koefisien", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    prediction = model_data[["SPT", "unit_lahan", "vegetasi", target_col]].copy()
    prediction["prediksi"] = y_pred
    prediction["residual"] = residual.to_numpy()
    prediction["abs_residual"] = prediction["residual"].abs()
    summary = {
        "n": len(y),
        "jumlah_prediktor": x.shape[1],
        "R2": r2,
        "Adj_R2": adj_r2,
        "MAE": mae,
        "RMSE": rmse,
        "formula": multiple_regression_formula(x.shape[1]),
    }
    return coef, summary, prediction


def filter_errors(errors: pd.DataFrame, parameters: list[str], devices: list[str], spts: list[str]) -> pd.DataFrame:
    filtered = errors.copy()
    if parameters:
        filtered = filtered[filtered["parameter"].isin(parameters)]
    if devices:
        filtered = filtered[filtered["alat"].isin(devices)]
    if spts:
        filtered = filtered[filtered["SPT"].isin(spts)]
    return filtered.reset_index(drop=True)


def clean_table_for_display(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan)


def academic_display_table(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    include_linear_equation: bool = False,
) -> pd.DataFrame:
    selected = df.copy() if columns is None else df[[col for col in columns if col in df.columns]].copy()
    if (
        include_linear_equation
        and "intercept_portable_to_lab" in selected.columns
        and "slope_portable_to_lab" in selected.columns
    ):
        selected["persamaan_regresi_linear"] = selected.apply(
            lambda row: linear_regression_equation(
                row["intercept_portable_to_lab"],
                row["slope_portable_to_lab"],
            ),
            axis=1,
        )
    return clean_table_for_display(selected).rename(columns=ACADEMIC_COLUMN_LABELS)


def coefficient_display_table(coef: pd.DataFrame) -> pd.DataFrame:
    display_cols = ["notasi", "variabel_model", "suku_model", "prediktor", "koefisien"]
    return academic_display_table(coef, display_cols)


def error_predictor_options(target_spec: dict[str, object], clean: pd.DataFrame) -> list[str]:
    device_predictors = DEVICE_NUMERIC_PREDICTORS.get(str(target_spec["device"]), [])
    options = [
        *device_predictors,
        *SOIL_NUMERIC_PREDICTORS,
    ]
    return [col for col in dict.fromkeys(options) if col in clean.columns and col != target_spec["target"]]


def stable_selectbox(container, label: str, options: list[str], key: str, index: int = 0):
    if not options:
        return None
    safe_index = min(max(index, 0), len(options) - 1)
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[safe_index]
    return container.selectbox(label, options, key=key)


def stable_multiselect(container, label: str, options: list[str], key: str, default: list[str] | None = None):
    default = options if default is None else default
    if key in st.session_state:
        st.session_state[key] = [item for item in st.session_state[key] if item in options]
    return container.multiselect(label, options, default=default, key=key)


def render_lab_distribution(clean: pd.DataFrame, title: str = "Distribusi Parameter Laboratorium") -> None:
    lab_cols = [
        "pH_lab",
        "NH4_N_lab",
        "NO3_N_lab",
        "P_lab",
        "K_lab",
        "kadar_air_lab_pct",
        "pasir_pct",
        "debu_pct",
        "liat_pct",
        "C_organik_pct",
    ]
    available = [col for col in lab_cols if col in clean.columns]
    if not available:
        return
    long_df = clean[["SPT", "unit_lahan", "vegetasi", *available]].melt(
        id_vars=["SPT", "unit_lahan", "vegetasi"],
        var_name="parameter",
        value_name="nilai",
    )
    long_df = long_df.dropna(subset=["nilai"])
    if long_df.empty:
        return
    fig = px.box(
        long_df,
        x="SPT",
        y="nilai",
        color="SPT",
        facet_col="parameter",
        facet_col_wrap=3,
        points="all",
        hover_data=["unit_lahan", "vegetasi"],
        title=title,
    )
    fig.update_yaxes(matches=None)
    fig.update_xaxes(matches=None)
    fig.update_layout(xaxis_title="SPT", yaxis_title="Nilai", showlegend=False)
    st.plotly_chart(fig, width="stretch")


def render_class_distribution(clean: pd.DataFrame) -> None:
    class_cols = {
        "pH_kelas": "pH",
        "N_kelas": "N",
        "P_kelas": "P",
        "K_kelas": "K",
        "tekstur_kelas": "Tekstur",
        "C_organik_kelas": "C-organik",
    }
    rows = []
    for col, label in class_cols.items():
        if col not in clean.columns:
            continue
        counts = clean[col].fillna("Tidak ada kelas").astype(str).str.strip().replace("", "Tidak ada kelas").value_counts()
        for kelas, jumlah in counts.items():
            rows.append({"parameter": label, "kelas": kelas, "jumlah": jumlah})
    if not rows:
        return
    fig = px.bar(
        pd.DataFrame(rows),
        x="kelas",
        y="jumlah",
        color="parameter",
        facet_col="parameter",
        facet_col_wrap=3,
        title="Sebaran Kelas/Kriteria Tanah",
        labels={"kelas": "Kelas", "jumlah": "Jumlah sampel"},
    )
    fig.update_xaxes(matches=None, tickangle=-25)
    fig.update_yaxes(matches=None)
    st.plotly_chart(fig, width="stretch")


def render_agreement_overview_charts(stats_df: pd.DataFrame) -> None:
    if stats_df.empty:
        return
    plot_df = stats_df.copy()
    plot_df["label"] = plot_df["parameter"] + " - " + plot_df["alat"]

    left, right = st.columns(2)
    fig_corr = px.bar(
        plot_df,
        x="label",
        y="Pearson_r",
        color="parameter",
        title="Korelasi Pearson per Parameter-Alat",
        labels={"label": "", "Pearson_r": "r (Pearson)"},
    )
    fig_corr.add_hline(y=0, line_dash="dot", line_color="#6B7280")
    fig_corr.update_xaxes(tickangle=-35)
    left.plotly_chart(fig_corr, width="stretch")

    fig_error = px.bar(
        plot_df,
        x="label",
        y="MAE",
        color="alat",
        title="MAE per Parameter-Alat",
        labels={"label": "", "MAE": "MAE (galat absolut rata-rata)"},
    )
    fig_error.update_xaxes(tickangle=-35)
    right.plotly_chart(fig_error, width="stretch")


def render_replicate_variability(rep_summary: pd.DataFrame) -> None:
    if rep_summary.empty:
        return
    plot_df = rep_summary.dropna(subset=["cv_pct"]).copy()
    if plot_df.empty:
        return
    top_df = plot_df.sort_values("cv_pct", ascending=False).head(20)
    top_df["sampel"] = "SPT " + top_df["SPT"].astype(str) + "." + top_df["unit_lahan"].astype(str) + " - " + top_df["vegetasi"]
    fig = px.bar(
        top_df,
        x="cv_pct",
        y="sampel",
        color="variabel",
        orientation="h",
        title="20 Pembacaan Portabel dengan Variasi Ulangan Tertinggi",
        labels={"cv_pct": "CV ulangan (%)", "sampel": "Sampel"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width="stretch")


def render_problem_map() -> None:
    st.subheader("Peta Analisis Sesuai Rumusan Masalah")
    st.caption("Bagian ini adalah peta kerja: setiap rumusan masalah dihubungkan dengan data yang dipakai, metode hitung, dan hasil yang dibaca.")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Rumusan masalah": "Korelasi alat portabel dan laboratorium",
                    "Pertanyaan analisis": "Seberapa dekat bacaan portabel dengan hasil laboratorium?",
                    "Data yang dipakai": "Pasangan nilai lab dan portabel untuk pH, N, P, K, dan kelembapan",
                    "Metode": "r Pearson, ρ Spearman, regresi linear ŷ = β₀ + β₁x, MAE, RMSE",
                    "Output yang dibaca": "r, ρ, β₀, β₁, R², galat, bias alat, dan kesesuaian alat",
                },
                {
                    "Rumusan masalah": "Pengaruh tekstur dan C-organik terhadap kinerja alat",
                    "Pertanyaan analisis": "Seberapa baik bacaan alat portabel dan faktor tanah dapat menduga selisih alat-lab?",
                    "Data yang dipakai": "Y (respons) = galat portabel - lab sesuai alat ukur; Xⱼ (prediktor) = bacaan portabel terpilih, tekstur, C-organik, SPT, vegetasi, dan faktor tanah",
                    "Metode": "Regresi berganda Ŷ = β₀ + β₁X₁ + ... + βₚXₚ + ε",
                    "Output yang dibaca": "βⱼ, Xⱼ, R², RMSE, prediksi galat, dan residual model selisih",
                },
                {
                    "Rumusan masalah": "Rekomendasi teknis penggunaan alat",
                    "Pertanyaan analisis": "Kapan alat portabel layak dipakai dan perlu dikalibrasi?",
                    "Data yang dipakai": "Hasil korelasi, galat, bias, skor kelayakan, dan persamaan kalibrasi",
                    "Metode": "Sintesis, matriks kelayakan, persamaan kalibrasi",
                    "Output yang dibaca": "Rekomendasi penggunaan alat dan arahan kalibrasi per parameter",
                },
            ]
        ),
        width="stretch",
        hide_index=True,
    )


def sidebar_data_source() -> str | Path | object | None:
    st.sidebar.header("Sumber Data")
    default_file = first_existing_file()
    uploaded = st.sidebar.file_uploader("Unggah workbook Excel", type=["xlsx"])
    if uploaded is not None:
        st.sidebar.success("Menggunakan file unggahan.")
        return uploaded
    if default_file:
        st.sidebar.info(f"Menggunakan file lokal: {default_file.name}")
        return default_file
    st.sidebar.error("Tidak ada file Excel di folder kerja.")
    return None


def render_overview(data: dict[str, pd.DataFrame]) -> None:
    clean = data["clean"]
    errors = data["errors"]
    stats_df = all_agreement_stats(clean)
    matrix = recommendation_matrix(stats_df)

    render_problem_map()

    st.subheader("Ringkasan Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sampel bersih", f"{len(clean):,}")
    c2.metric("SPT", clean["SPT"].nunique())
    c3.metric("Observasi galat", f"{len(errors):,}")
    c4.metric("Ulangan portabel", f"{len(data['replicates']):,}")

    st.markdown("#### Visualisasi Ringkasan")
    render_class_distribution(clean)
    render_lab_distribution(clean)
    render_replicate_variability(data["replicate_summary"])

    st.markdown("#### Statistik Inti Laboratorium")
    lab_cols = [
        "pH_lab",
        "NH4_N_lab",
        "NO3_N_lab",
        "P_lab",
        "K_lab",
        "kadar_air_lab_pct",
        "pasir_pct",
        "debu_pct",
        "liat_pct",
        "C_organik_pct",
    ]
    summary = clean[[col for col in lab_cols if col in clean.columns]].describe().T
    st.dataframe(summary, width="stretch")

    st.markdown("#### Matriks Kelayakan Awal")
    cols = [
        "parameter",
        "alat",
        "n",
        "Pearson_r",
        "Spearman_rho",
        "bias",
        "MAE",
        "RMSE",
        "nRMSE_range_pct",
        "skor_kelayakan",
        "rekomendasi",
        "persamaan_kalibrasi_lab",
    ]
    st.dataframe(academic_display_table(matrix, cols), width="stretch", hide_index=True)


def render_correlation(data: dict[str, pd.DataFrame]) -> None:
    clean = data["clean"]
    stats_df = all_agreement_stats(clean)

    st.subheader("Korelasi dan Kesesuaian Alat")
    st.caption("e = y_portabel - y_lab (bias). Persamaan kalibrasi memperkirakan ŷ_lab dari x_portabel.")
    render_agreement_overview_charts(stats_df)

    parameter_options = stats_df["parameter"].dropna().unique().tolist()
    col1, col2 = st.columns(2)
    parameter = stable_selectbox(col1, "Parameter", parameter_options, key="correlation_parameter")
    device_options = stats_df.loc[stats_df["parameter"].eq(parameter), "alat"].unique().tolist()
    device = stable_selectbox(col2, "Alat", device_options, key="correlation_device")

    spec = next(item for item in COMPARISONS if item.parameter == parameter and item.device == device)
    pair_source = clean.copy()
    pair_source["lab_value"] = to_number(pair_source[spec.lab_col])
    pair_source["portable_value"] = to_number(pair_source[spec.portable_col])
    pair_source = pair_source[pair_source["lab_value"].notna() & pair_source["portable_value"].notna()]
    pair_source["galat"] = pair_source["portable_value"] - pair_source["lab_value"]

    stats_row = stats_df[(stats_df["parameter"].eq(parameter)) & (stats_df["alat"].eq(device))].iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("r (Pearson)", f"{stats_row['Pearson_r']:.3f}", f"p_r={format_p(stats_row['Pearson_p'])}")
    m2.metric("ρ (Spearman)", f"{stats_row['Spearman_rho']:.3f}", f"p_ρ={format_p(stats_row['Spearman_p'])}")
    m3.metric("MAE (galat absolut)", f"{stats_row['MAE']:.3f} {spec.unit}")
    m4.metric("RMSE (akar galat kuadrat)", f"{stats_row['RMSE']:.3f} {spec.unit}")

    st.info(calibration_text(stats_row["intercept_portable_to_lab"], stats_row["slope_portable_to_lab"]))
    st.latex(LINEAR_REGRESSION_LATEX)

    fig = px.scatter(
        pair_source,
        x="portable_value",
        y="lab_value",
        color="SPT",
        hover_data=["unit_lahan", "vegetasi", "satuan_tanah", "galat"],
        labels={
            "portable_value": f"x_portabel ({parameter} {device}, {spec.unit})",
            "lab_value": f"y_lab ({parameter} lab, {spec.unit})",
        },
        title=f"{parameter}: Kalibrasi {device} ke Lab",
    )
    add_regression_line(fig, pair_source, "portable_value", "lab_value", name="Kalibrasi linear")
    st.plotly_chart(fig, width="stretch")

    fig_error = px.histogram(
        pair_source,
        x="galat",
        nbins=14,
        color="SPT",
        labels={"galat": f"e (galat, {spec.unit})"},
        title=f"Sebaran galat {parameter} - {device}",
    )
    st.plotly_chart(fig_error, width="stretch")

    fig_box = px.box(
        pair_source,
        x="SPT",
        y="galat",
        color="SPT",
        points="all",
        hover_data=["unit_lahan", "vegetasi", "satuan_tanah"],
        title=f"Galat {parameter} - {device} per SPT",
        labels={"galat": f"e (galat, {spec.unit})", "SPT": "SPT"},
    )
    st.plotly_chart(fig_box, width="stretch")

    st.markdown("#### Semua Hasil Statistik")
    show_cols = [
        "parameter",
        "alat",
        "n",
        "bias",
        "MAE",
        "RMSE",
        "MAPE_pct",
        "nRMSE_range_pct",
        "Pearson_r",
        "Pearson_p",
        "Spearman_rho",
        "Spearman_p",
        "R2_portable_to_lab",
        "intercept_portable_to_lab",
        "slope_portable_to_lab",
    ]
    st.dataframe(academic_display_table(stats_df, show_cols, include_linear_equation=True), width="stretch", hide_index=True)


def render_factor_analysis(data: dict[str, pd.DataFrame]) -> None:
    clean = data["clean"]
    st.subheader("Regresi Berganda untuk Menduga Selisih Alat-Lab")
    st.caption(
        "Y (variabel respons) adalah selisih/galat alat portabel terhadap hasil laboratorium (portabel - lab). "
        "X adalah bacaan alat portabel yang dipilih, dengan faktor tanah opsional sebagai variabel koreksi."
    )

    available_targets = [item for item in ERROR_REGRESSION_TARGETS if item["target"] in clean.columns]
    if not available_targets:
        st.warning("Kolom selisih alat-lab belum tersedia untuk regresi.")
        return

    left, right = st.columns([1, 2])
    device_options = list(dict.fromkeys(item["device"] for item in available_targets))
    selected_device = stable_selectbox(left, "Alat ukur", device_options, key="regression_device")
    target_labels = [item["label"] for item in available_targets if item["device"] == selected_device]
    target_label = stable_selectbox(right, "Y selisih yang diduga", target_labels, key="regression_target")
    target_spec = next(
        item for item in available_targets if item["label"] == target_label and item["device"] == selected_device
    )

    available_numeric = error_predictor_options(target_spec, clean)
    default_numeric = [col for col in target_spec["default_predictors"] if col in available_numeric]
    available_category = [col for col in CATEGORY_PREDICTORS if col in clean.columns]

    left, right = st.columns([2, 1])
    selected_numeric = stable_multiselect(
        left,
        "Prediktor numerik X",
        available_numeric,
        key=f"regression_numeric_predictors_{target_spec['target']}",
        default=default_numeric,
    )
    selected_category = stable_multiselect(
        right,
        "Prediktor kategori X",
        available_category,
        key="regression_category_predictors",
        default=[],
    )
    selected_spts = stable_multiselect(
        st,
        "Filter SPT",
        sorted(clean["SPT"].dropna().unique()),
        key="regression_spts",
        default=[],
    )

    model_df = clean.copy()
    if selected_spts:
        model_df = model_df[model_df["SPT"].isin(selected_spts)].copy()

    coef, summary, prediction = regression_with_prediction(
        model_df,
        target_spec["target"],
        selected_numeric,
        selected_category,
    )

    st.markdown("#### Ringkasan Model Terpilih")
    if coef.empty:
        st.warning(summary.get("catatan", "Regresi belum dapat dihitung."))
    else:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("n", int(summary["n"]))
        m2.metric("p (prediktor)", int(summary["jumlah_prediktor"]))
        m3.metric("R²", f"{summary['R2']:.3f}")
        m4.metric("R²_adj", f"{summary['Adj_R2']:.3f}")
        m5.metric("RMSE (akar galat kuadrat)", f"{summary['RMSE']:.3f} {target_spec['unit']}")

        st.info(f"{summary['formula']} ({target_spec['label']} sebagai Y selisih; {selected_device} sebagai alat ukur).")
        st.latex(MULTIPLE_REGRESSION_LATEX)
        st.caption(
            "Residual model = Y_galat aktual - Y_galat prediksi. "
            "Residual positif berarti selisih aktual lebih besar daripada prediksi model; residual negatif berarti lebih kecil. "
            "Semakin dekat residual ke 0, semakin baik model membaca pola selisih alat-lab."
        )

        coef_plot = coef[coef["prediktor"].ne("intercept")].copy()
        if not coef_plot.empty:
            fig_coef = px.bar(
                coef_plot,
                x="koefisien",
                y="prediktor",
                orientation="h",
                title=f"βⱼ: Koefisien Regresi Berganda untuk {target_spec['label']}",
                labels={"koefisien": "β̂ⱼ (estimasi koefisien)", "prediktor": "Xⱼ (prediktor asli)"},
            )
            fig_coef.add_vline(x=0, line_dash="dot", line_color="#6B7280")
            fig_coef.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_coef, width="stretch")

        fig_pred = px.scatter(
            prediction,
            x=target_spec["target"],
            y="prediksi",
            color="SPT",
            hover_data=["unit_lahan", "vegetasi", "residual"],
            title=f"Y_galat Aktual vs Prediksi: {target_spec['label']}",
            labels={
                target_spec["target"]: f"Y_galat aktual ({target_spec['unit']})",
                "prediksi": f"Y_galat prediksi ({target_spec['unit']})",
            },
        )
        min_val = float(min(prediction[target_spec["target"]].min(), prediction["prediksi"].min()))
        max_val = float(max(prediction[target_spec["target"]].max(), prediction["prediksi"].max()))
        fig_pred.add_scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Garis 1:1",
            line={"color": "#111827", "dash": "dash"},
        )
        st.plotly_chart(fig_pred, width="stretch")

        fig_resid = px.bar(
            prediction.sort_values("abs_residual", ascending=False),
            x="vegetasi",
            y="residual",
            color="SPT",
            hover_data=["unit_lahan", target_spec["target"], "prediksi", "abs_residual"],
            title=f"Residual Model Selisih: {target_spec['label']}",
            labels={"residual": f"e (residual, {target_spec['unit']})", "vegetasi": "Vegetasi"},
        )
        fig_resid.add_hline(y=0, line_dash="dot", line_color="#6B7280")
        fig_resid.update_xaxes(tickangle=-35)
        st.plotly_chart(fig_resid, width="stretch")

        st.markdown("#### Koefisien Model")
        st.dataframe(coefficient_display_table(coef), width="stretch", hide_index=True)

        st.markdown("#### Data Selisih Aktual, Prediksi, dan Residual")
        prediction_display = prediction.sort_values("abs_residual", ascending=False).rename(
            columns={target_spec["target"]: f"Y_galat aktual ({target_spec['label']})"}
        )
        st.dataframe(academic_display_table(prediction_display), width="stretch", hide_index=True)

    st.markdown("#### Ringkasan Regresi Semua Y Selisih")
    rows = []
    for item in ERROR_REGRESSION_TARGETS:
        if item["target"] not in clean.columns:
            continue
        predictors = [col for col in item["default_predictors"] if col in clean.columns]
        _, item_summary, _ = regression_with_prediction(clean, item["target"], predictors, [])
        if "R2" not in item_summary:
            rows.append({"Y": item["label"], "alat": item["device"], "catatan": item_summary.get("catatan", "-")})
            continue
        rows.append(
            {
                "Y": item["label"],
                "alat": item["device"],
                "n": item_summary["n"],
                "jumlah_prediktor": item_summary["jumlah_prediktor"],
                "R2": item_summary["R2"],
                "Adj_R2": item_summary["Adj_R2"],
                "MAE": item_summary["MAE"],
                "RMSE": item_summary["RMSE"],
                "catatan": "",
            }
        )
    summary_df = pd.DataFrame(rows)
    st.dataframe(academic_display_table(summary_df), width="stretch", hide_index=True)
    if "R2" in summary_df.columns:
        fig_all = px.bar(
            summary_df.dropna(subset=["R2"]),
            x="Y",
            y="R2",
            color="Y",
            title="Perbandingan R² Regresi untuk Semua Selisih Alat-Lab",
            labels={"Y": "Y_galat (target respons)", "R2": "R²"},
        )
        fig_all.update_xaxes(tickangle=-30)
        st.plotly_chart(fig_all, width="stretch")


def render_recommendations(data: dict[str, pd.DataFrame]) -> None:
    clean = data["clean"]
    stats_df = all_agreement_stats(clean)
    matrix = recommendation_matrix(stats_df)

    st.subheader("Rekomendasi Teknis dan SOP Penggunaan Alat")
    fig_score = px.bar(
        matrix,
        x="parameter",
        y="skor_kelayakan",
        color="alat",
        barmode="group",
        text=matrix["skor_kelayakan"].round(1),
        title="Skor Kelayakan Alat per Parameter",
        labels={"skor_kelayakan": "Skor kelayakan", "parameter": "Parameter"},
    )
    fig_score.add_hline(y=75, line_dash="dash", line_color="#16A34A", annotation_text="Layak estimasi")
    fig_score.add_hline(y=55, line_dash="dash", line_color="#F59E0B", annotation_text="Screening")
    fig_score.add_hline(y=35, line_dash="dash", line_color="#DC2626", annotation_text="Terbatas")
    st.plotly_chart(fig_score, width="stretch")

    fig_tradeoff = px.scatter(
        matrix,
        x="RMSE",
        y="Pearson_r",
        size="skor_kelayakan",
        color="rekomendasi",
        hover_data=["parameter", "alat", "bias", "MAE", "persamaan_kalibrasi_lab"],
        title="Trade-off Korelasi dan RMSE untuk Menilai Kelayakan",
        labels={"Pearson_r": "r (Pearson)", "RMSE": "RMSE"},
    )
    fig_tradeoff.add_hline(y=0, line_dash="dot", line_color="#6B7280")
    st.plotly_chart(fig_tradeoff, width="stretch")

    st.markdown("#### Matriks Kelayakan Per Parameter")
    display_cols = [
        "parameter",
        "alat",
        "n",
        "skor_kelayakan",
        "rekomendasi",
        "bias",
        "MAE",
        "RMSE",
        "Pearson_r",
        "Spearman_rho",
        "persamaan_kalibrasi_lab",
    ]
    st.dataframe(academic_display_table(matrix, display_cols), width="stretch", hide_index=True)

    best = matrix.sort_values("skor_kelayakan", ascending=False).groupby("parameter", as_index=False).first()
    st.markdown("#### Alat yang Paling Disarankan per Parameter")
    st.dataframe(
        academic_display_table(best, ["parameter", "alat", "skor_kelayakan", "rekomendasi", "persamaan_kalibrasi_lab"]),
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### SOP Ringkas")
    st.markdown(
        """
1. Ambil minimal 5 ulangan portabel pada titik yang sama, lalu gunakan rata-ratanya.
2. Catat kelembapan, tekstur, C-organik, SPT, vegetasi, dan kondisi lahan untuk setiap sampel.
3. Untuk keputusan pemupukan, gunakan nilai portabel sebagai screening awal, lalu koreksi dengan persamaan kalibrasi parameter terkait.
4. Bila skor kelayakan berada pada kategori terbatas atau tidak disarankan, jangan gunakan alat sebagai dasar tunggal keputusan dosis.
5. Lakukan kalibrasi lokal ulang ketika pindah SPT, tekstur dominan, atau rentang kelembapan tanah berubah nyata.
6. Prioritaskan validasi laboratorium pada sampel dengan galat absolut tinggi, tanah sangat liat, C-organik ekstrem, atau kelembapan sangat tinggi.
        """
    )

    st.markdown("#### Catatan Interpretasi Otomatis")
    for _, row in best.iterrows():
        st.write(
            f"- {row['parameter']}: gunakan {row['alat']} sebagai opsi utama. "
            f"Status: {row['rekomendasi']}. {row['persamaan_kalibrasi_lab']}."
        )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Aplikasi analisis komprehensif data tanah Suleman: lab, alat portabel, tekstur, C-organik, dan kelembapan.")

    data_source = sidebar_data_source()
    if data_source is None:
        st.stop()

    with st.spinner("Membaca dan membersihkan workbook..."):
        data = load_data(data_source)

    st.sidebar.header("Navigasi Analisis")
    if st.session_state.get("active_page") not in PAGES:
        st.session_state["active_page"] = PAGES[0]
    page = st.sidebar.radio("Pilih halaman", PAGES, key="active_page")
    st.sidebar.caption("Pilihan halaman disimpan, jadi klik filter atau grafik tidak akan memindahkan Anda ke halaman lain.")

    renderers = {
        "Ringkasan": render_overview,
        "Korelasi Lab-Portabel": render_correlation,
        "Regresi Berganda": render_factor_analysis,
    }
    renderers[page](data)


if __name__ == "__main__":
    main()
