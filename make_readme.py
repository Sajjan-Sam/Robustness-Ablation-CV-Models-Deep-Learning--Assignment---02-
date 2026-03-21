from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Optional, Iterable

import pandas as pd


# ============================================================
# Utility helpers
# ============================================================

ROOT = Path.cwd()
README_PATH = ROOT / "README.md"
ASSET_DIR = ROOT / "README_assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)


def find_first(filename: str, search_root: Path = ROOT) -> Optional[Path]:
    matches = list(search_root.rglob(filename))
    return matches[0] if matches else None


def fmt_pct(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "N/A"
    return f"{100.0 * float(x):.2f}%"


def fmt_num(x, digits=4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "N/A"
    return f"{float(x):.{digits}f}"


def df_to_markdown(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    if df is None or df.empty:
        return "_No data available._"

    if max_rows is not None:
        df = df.head(max_rows).copy()

    df = df.copy()
    cols = list(df.columns)

    # convert all values to string safely
    for c in cols:
        df[c] = df[c].apply(lambda v: "N/A" if pd.isna(v) else str(v))

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [
        "| " + " | ".join(str(row[c]) for c in cols) + " |"
        for _, row in df.iterrows()
    ]
    return "\n".join([header, sep] + rows)


def safe_copy(src: Path, dst: Path) -> Optional[Path]:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return dst
    except Exception:
        return None


def clean_name(x: str) -> str:
    return (
        str(x)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(",", "")
        .replace(":", "")
    )


# ============================================================
# Locate required CSV files
# ============================================================

csv_paths = {
    "all_selectors": find_first("full_assignment_all_selectors.csv"),
    "clean_rows": find_first("full_assignment_clean_rows.csv"),
    "best_corrupt_rows": find_first("full_assignment_best_corrupt_rows.csv"),
    "comparison": find_first("clean_vs_best_corrupt_comparison.csv"),
    "selector_robustness": find_first("report_all_selector_robustness.csv"),
    "clean_comparison": find_first("report_clean_accuracy_comparison.csv"),
    "robustness_comparison": find_first("report_robustness_comparison.csv"),
    "feature_master": find_first("feature_perturbation_master.csv"),
    "feature_errors": find_first("feature_perturbation_errors.csv"),
}

dfs = {}
for key, path in csv_paths.items():
    if path and path.exists():
        dfs[key] = pd.read_csv(path)
    else:
        dfs[key] = pd.DataFrame()

all_selectors_df = dfs["all_selectors"]
clean_rows_df = dfs["clean_rows"]
best_corrupt_rows_df = dfs["best_corrupt_rows"]
comparison_df = dfs["comparison"]
selector_robustness_df = dfs["selector_robustness"]
clean_comparison_df = dfs["clean_comparison"]
robustness_comparison_df = dfs["robustness_comparison"]
feature_master_df = dfs["feature_master"]
feature_errors_df = dfs["feature_errors"]


# ============================================================
# Prepare report-ready tables
# ============================================================

def build_clean_accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in ["test_clean_acc_clean_selector", "test_clean_acc_best_corrupt_selector"]:
        if col in out.columns:
            out[col] = out[col].map(fmt_pct)
    return out


def build_robustness_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in [
        "test_corrupt_mean_acc_clean_selector",
        "test_corrupt_mean_acc_best_corrupt_selector",
        "robustness_gap_clean_selector",
        "robustness_gap_best_corrupt_selector",
    ]:
        if col in out.columns:
            out[col] = out[col].map(fmt_pct)
    return out


def build_selector_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in ["clean", "corr_k1_s2", "corr_k3_s2", "corr_k5_s2"]:
        if col in out.columns:
            out[col] = out[col].map(fmt_pct)
    return out


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    keep_cols = [
        "dataset", "model", "where", "selector",
        "test_clean_acc", "test_corrupt_mean_acc", "robustness_gap"
    ]
    keep_cols = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols].copy()
    for col in ["test_clean_acc", "test_corrupt_mean_acc", "robustness_gap"]:
        if col in out.columns:
            out[col] = out[col].map(fmt_pct)
    return out


clean_accuracy_table = build_clean_accuracy_table(clean_comparison_df)
robustness_table = build_robustness_table(robustness_comparison_df)
selector_table = build_selector_table(selector_robustness_df)
feature_table = build_feature_table(feature_master_df)

# summary winners
dataset_clean_winners = pd.DataFrame()
if not clean_comparison_df.empty:
    rows = []
    for dataset_name, g in clean_comparison_df.groupby("dataset"):
        best_row = None
        best_val = -1.0
        for _, r in g.iterrows():
            cand1 = float(r["test_clean_acc_clean_selector"])
            cand2 = float(r["test_clean_acc_best_corrupt_selector"])
            if cand1 >= best_val:
                best_val = cand1
                best_row = (r["model"], "clean_selector", cand1)
            if cand2 >= best_val:
                best_val = cand2
                best_row = (r["model"], "best_corrupt_selector", cand2)
        rows.append({
            "dataset": dataset_name,
            "best_model": best_row[0],
            "winner_source": best_row[1],
            "best_clean_acc": fmt_pct(best_row[2]),
        })
    dataset_clean_winners = pd.DataFrame(rows)

dataset_robust_winners = pd.DataFrame()
if not robustness_comparison_df.empty:
    rows = []
    for dataset_name, g in robustness_comparison_df.groupby("dataset"):
        best_row = None
        best_val = -1.0
        for _, r in g.iterrows():
            cand1 = float(r["test_corrupt_mean_acc_clean_selector"])
            cand2 = float(r["test_corrupt_mean_acc_best_corrupt_selector"])
            if cand1 >= best_val:
                best_val = cand1
                best_row = (r["model"], "clean_selector", cand1)
            if cand2 >= best_val:
                best_val = cand2
                best_row = (r["model"], "best_corrupt_selector", cand2)
        rows.append({
            "dataset": dataset_name,
            "best_model": best_row[0],
            "winner_source": best_row[1],
            "best_corrupt_mean_acc": fmt_pct(best_row[2]),
        })
    dataset_robust_winners = pd.DataFrame(rows)


# ============================================================
# Collect representative figures
# ============================================================

def resolve_selector_plot_dir(exp_dir: Path, selector: str) -> Optional[Path]:
    p = exp_dir / "plots" / selector
    return p if p.exists() else None


def collect_pair_assets(row: pd.Series) -> dict:
    """
    Copy representative clean and best-corrupt plots for one dataset/model pair
    into README_assets/ and return relative paths.
    """
    assets = {}

    dataset_name = row.get("dataset", "unknown")
    model_name = row.get("model", "unknown")

    # clean selector
