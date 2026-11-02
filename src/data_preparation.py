"""
data_preparation.py
====================
Stage 1 — LUCAS 2018 TOPSOIL data ingestion, cleaning & merging.

Datasets
--------
1. LUCAS-SOIL-2018.csv        — 18,984 core soil chemistry points
2. BulkDensity_2018_final-2.csv — bulk density measurements
3. LUCAS_Text_All_10032025.csv  — texture data from 2009/2012/2015/2018 surveys

Key design decisions
--------------------
* POINT_ID is the join key across all three files (stored as POINTID in raw CSVs).
* Below-Detection-Limit values marked as '< LOD' are handled via `lod_strategy`:
    'zero' (default) → replace with 0.0 and set <col>_BDL flag = 1
    'nan'            → replace with NaN and set flag
    'half'           → replace with half the numeric value that follows (rare)
* Texture is multi-campaign; the best available year (2018 > 2015 > 2012 > 2009)
  is selected per point via `TextureFromSurvey`.
* Dates in the core file follow '%d-%m-%y' format (e.g., '06-07-18').
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# ---------------------------------------------------------------------------
# Column name constants (actual headers in the raw CSVs)
# ---------------------------------------------------------------------------
_CORE_ID       = "POINTID"
_BD_ID         = "POINTID"
_TEX_ID        = "POINTID"

_BD_COLS = {
    "BD_0_10":  ["BD 0-10",  "BD_0_10",  "BD0_10"],
    "BD_10_20": ["BD 10-20", "BD_10_20", "BD10_20"],
    "BD_0_20":  ["BD 0-20",  "BD_0_20",  "BD0_20"],
}

_TEXTURE_YEAR_COL = "TextureFromSurvey"
_TEXTURE_PRIORITY = {2018: 4, 2015: 3, 2012: 2, 2009: 1}

_CHEMISTRY_COLS = [
    "pH_H2O", "pH_CaCl2", "EC", "OC", "CaCO3", "P", "N", "K",
    "Ox_Al", "Ox_Fe",
]

_LOD_PATTERN = re.compile(r"^\s*<\s*LOD\s*$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _parse_lod(series: pd.Series, strategy: str) -> tuple[pd.Series, pd.Series]:
    """
    Parse a chemistry column that may contain '< LOD' strings.

    Returns
    -------
    values : pd.Series  — numeric values (float64)
    flag   : pd.Series  — binary BDL indicator (0/1, int8)
    """
    raw    = series.astype(str).str.strip()
    is_bdl = raw.str.match(_LOD_PATTERN).fillna(False)

    numeric = pd.to_numeric(series, errors="coerce")

    if strategy == "zero":
        numeric[is_bdl] = 0.0
    elif strategy == "nan":
        numeric[is_bdl] = np.nan
    elif strategy == "half":
        # Try to extract the threshold value from adjacent metadata (not always available)
        # Fall back to 0 for this dataset
        numeric[is_bdl] = 0.0
    else:
        raise ValueError(f"Unknown lod_strategy: {strategy!r}. Choose 'zero', 'nan', or 'half'.")

    flag = is_bdl.astype(np.int8)
    return numeric, flag


def _select_best_texture(group: pd.DataFrame) -> pd.Series:
    """
    For a group of texture rows belonging to the same POINT_ID,
    return the row with the highest-priority survey year.
    """
    if _TEXTURE_YEAR_COL not in group.columns:
        return group.iloc[0]

    def _priority(row):
        try:
            y = int(row[_TEXTURE_YEAR_COL])
            return _TEXTURE_PRIORITY.get(y, 0)
        except (ValueError, TypeError):
            return 0

    priorities = group.apply(_priority, axis=1)
    best_idx   = priorities.idxmax()
    return group.loc[best_idx]


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_core(path: str | Path, lod_strategy: str = "zero") -> pd.DataFrame:
    """
    Load the LUCAS 2018 core chemistry CSV.

    Parameters
    ----------
    path         : path to LUCAS-SOIL-2018.csv
    lod_strategy : how to handle '< LOD' values ('zero', 'nan', 'half')

    Returns
    -------
    DataFrame with POINT_ID as index, chemistry columns cleaned, BDL flags added.
    """
    df = pd.read_csv(path, low_memory=False)

    # Normalise the point-ID column
    if _CORE_ID in df.columns:
        df = df.rename(columns={_CORE_ID: "POINT_ID"})
    df["POINT_ID"] = df["POINT_ID"].astype(str).str.strip()
    df = df.set_index("POINT_ID")

    # Parse date column if present
    for date_col in ["SURVEY_DATE", "Date", "date"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], format="%d-%m-%y", errors="coerce")

    # Handle BDL chemistry values
    for col in _CHEMISTRY_COLS:
        if col not in df.columns:
            continue
        vals, flag = _parse_lod(df[col], lod_strategy)
        df[col]          = vals
        df[f"{col}_BDL"] = flag

    return df


def load_bulk_density(path: str | Path) -> pd.DataFrame:
    """
    Load the LUCAS 2018 bulk density CSV.

    Returns
    -------
    DataFrame with POINT_ID as index and standardised BD columns:
    BD_0_10, BD_10_20, BD_0_20 (where available).
    """
    df = pd.read_csv(path, low_memory=False)

    if _BD_ID in df.columns:
        df = df.rename(columns={_BD_ID: "POINT_ID"})
    df["POINT_ID"] = df["POINT_ID"].astype(str).str.strip()
    df = df.set_index("POINT_ID")

    # Standardise BD column names
    for standard, candidates in _BD_COLS.items():
        raw_col = _resolve_col(df, candidates)
        if raw_col and raw_col != standard:
            df = df.rename(columns={raw_col: standard})
        if standard in df.columns:
            df[standard] = pd.to_numeric(df[standard], errors="coerce")

    return df


def load_texture(path: str | Path) -> pd.DataFrame:
    """
    Load the multi-campaign LUCAS texture CSV and select the best year per point.

    Returns
    -------
    DataFrame with POINT_ID as index (one row per point, best survey year).
    """
    df = pd.read_csv(path, low_memory=False)

    if _TEX_ID in df.columns:
        df = df.rename(columns={_TEX_ID: "POINT_ID"})
    df["POINT_ID"] = df["POINT_ID"].astype(str).str.strip()

    # Numeric coercion for texture columns
    for col in ["Clay", "Sand", "Silt", "Coarse"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Select best year per point
    best = (
        df.groupby("POINT_ID", group_keys=False)
          .apply(_select_best_texture)
          .reset_index(drop=True)
    )
    best = best.set_index("POINT_ID")
    return best


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------

def build_master_dataframe(
    core_path:    str | Path,
    bd_path:      str | Path,
    texture_path: str | Path,
    lod_strategy: Literal["zero", "nan", "half"] = "zero",
) -> pd.DataFrame:
    """
    Ingest, clean and merge all three LUCAS 2018 datasets.

    Merge strategy
    --------------
    core (left) ← left-join → bulk density
                ← left-join → texture

    HAS_BD and HAS_TEXTURE flags are added (1/0) to indicate data availability.

    Parameters
    ----------
    core_path    : path to LUCAS-SOIL-2018.csv
    bd_path      : path to BulkDensity_2018_final-2.csv
    texture_path : path to LUCAS_Text_All_10032025.csv
    lod_strategy : BDL handling ('zero' | 'nan' | 'half')

    Returns
    -------
    master_df : pd.DataFrame — 18,984 rows, ~75 columns
    """
    print("Loading core chemistry …")
    core = load_core(core_path, lod_strategy)
    print(f"  Core      : {core.shape[0]:,} rows × {core.shape[1]} cols")

    print("Loading bulk density …")
    bd   = load_bulk_density(bd_path)
    print(f"  Bulk dens : {bd.shape[0]:,} rows × {bd.shape[1]} cols")

    print("Loading texture (best year per point) …")
    tex  = load_texture(texture_path)
    print(f"  Texture   : {tex.shape[0]:,} rows × {tex.shape[1]} cols")

    print("Merging …")
    master = core.copy()

    # Bulk density — avoid column clashes
    bd_cols = [c for c in bd.columns if c not in master.columns]
    master = master.join(bd[bd_cols], how="left")
    master["HAS_BD"] = master.index.isin(bd.index).astype(np.int8)

    # Texture — avoid column clashes
    tex_cols = [c for c in tex.columns if c not in master.columns]
    master = master.join(tex[tex_cols], how="left")
    master["HAS_TEXTURE"] = master.index.isin(tex.index).astype(np.int8)

    # Rename spatial columns if present
    for old, new in [("TH_LAT", "TH_LAT"), ("TH_LONG", "TH_LONG"), ("ELEV", "Elev")]:
        if old in master.columns and new not in master.columns:
            master = master.rename(columns={old: new})

    print(f"Master DataFrame: {master.shape[0]:,} rows × {master.shape[1]} columns")
    print(f"  Points with BD      : {master['HAS_BD'].sum():,}")
    print(f"  Points with Texture : {master['HAS_TEXTURE'].sum():,}")
    return master


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame summarising missing values per column (sorted descending)."""
    n    = len(df)
    miss = df.isna().sum()
    pct  = (miss / n * 100).round(2)
    rep  = pd.DataFrame({"missing_count": miss, "missing_pct": pct})
    return rep[rep["missing_count"] > 0].sort_values("missing_pct", ascending=False)


def bdl_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame summarising Below-Detection-Limit flags."""
    bdl_cols = [c for c in df.columns if c.endswith("_BDL")]
    if not bdl_cols:
        return pd.DataFrame(columns=["column", "bdl_count", "bdl_pct"])
    rows = []
    n = len(df)
    for col in bdl_cols:
        cnt = df[col].sum()
        rows.append({
            "column":   col.replace("_BDL", ""),
            "bdl_count": int(cnt),
            "bdl_pct":   round(cnt / n * 100, 2),
        })
    return pd.DataFrame(rows).sort_values("bdl_pct", ascending=False)


def describe_master(df: pd.DataFrame) -> None:
    """Print a human-readable summary of the master DataFrame."""
    print(f"\n{'='*60}")
    print(f"  LUCAS 2018 Master DataFrame Summary")
    print(f"{'='*60}")
    print(f"  Rows (soil points) : {df.shape[0]:,}")
    print(f"  Columns            : {df.shape[1]}")
    print(f"  Memory usage       : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    chem_present = [c for c in _CHEMISTRY_COLS if c in df.columns]
    print(f"\n  Chemistry columns  : {len(chem_present)}")
    for c in chem_present:
        nn = df[c].notna().sum()
        bdl = int(df[f"{c}_BDL"].sum()) if f"{c}_BDL" in df.columns else 0
        print(f"    {c:12s}  valid={nn:6,}  BDL={bdl:5,}  mean={df[c].mean():.3g}")

    print(f"\n  HAS_BD      : {df['HAS_BD'].sum():,}" if "HAS_BD" in df.columns else "")
    print(f"  HAS_TEXTURE : {df['HAS_TEXTURE'].sum():,}" if "HAS_TEXTURE" in df.columns else "")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI entry-point (intentionally excluded from Colab notebook embedding)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Build LUCAS 2018 master DataFrame")
    parser.add_argument("--core",    required=True, help="Path to LUCAS-SOIL-2018.csv")
    parser.add_argument("--bd",      required=True, help="Path to BulkDensity_2018_final-2.csv")
    parser.add_argument("--texture", required=True, help="Path to LUCAS_Text_All_10032025.csv")
    parser.add_argument("--out",     default="LUCAS_master.csv",  help="Output CSV path")
    parser.add_argument("--parquet", default="LUCAS_master.parquet", help="Output Parquet path")
    parser.add_argument("--lod",     default="zero", choices=["zero", "nan", "half"])
    args = parser.parse_args()

    master = build_master_dataframe(args.core, args.bd, args.texture, args.lod)
    describe_master(master)

    master.to_csv(args.out)
    master.to_parquet(args.parquet)
    print(f"Saved: {args.out}")
    print(f"Saved: {args.parquet}")
    sys.exit(0)
