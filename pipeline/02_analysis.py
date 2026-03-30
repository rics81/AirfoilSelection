# -*- coding: utf-8 -*-
"""
pipeline/02_analysis.py
=======================
Step 2 — Quality-check airfoils and compute aerodynamic indicators.

Each airfoil/Reynolds polar is validated and, if clean, a set of derived
indicators is calculated.  Airfoils that fail validation for *any* Reynolds
number are dropped entirely so that all three Reynolds entries are always
present for downstream steps.

Validation rules
----------------
* At least 3 Reynolds numbers must be available for the airfoil.
* All of alpha, Cl, Cd, Cm series must be fully numeric.
* The polar must contain a visible stall region (at least 7 points after
  the Cl maximum).

Indicators computed (per Reynolds number)
-----------------------------------------
max_cl_cd           Max Cl/Cd ratio          (higher = better)
max_cl_cd_alpha     Alpha at max Cl/Cd       (lower  = better → in 'L' set)
clMax               Maximum Cl               (higher = better)
clStdRate           Std of ΔCl before stall  (lower  = better → in 'L' set)
clMeanRate          Mean of ΔCl before stall (higher = better)
clStdStallRate      Std of ΔCl after stall   (lower  = better → in 'L' set)
clMeanStallRate     Mean |ΔCl| after stall   (lower  = better → in 'L' set)
cdMax               Max Cd (α ≥ 0)           (lower  = better → in 'L' set)
cdStdRate           Std of ΔCd (α ≥ 0)      (lower  = better → in 'L' set)
cdMeanRate          Mean of ΔCd (α ≥ 0)     (lower  = better → in 'L' set)
cmAlphaZero         |Cm| at α = 0            (lower  = better → in 'L' set)

Input
-----
Raw polars DataFrame (output of 01_scraper.py or loaded from parquet).

Output
------
Tuple (df_features_raw, failed_df):
  df_features_raw   DataFrame — one row per airfoil with all indicators
                    (columns: name + r2*/r5*/r10* prefixed indicators).
  failed_df         DataFrame — names and failure reasons for removed airfoils.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Indicator keys extracted from each Reynolds polar
INDICATOR_KEYS = [
    "max_cl_cd",
    "max_cl_cd_alpha",
    "clMax",
    "clStdRate",
    "clMeanRate",
    "clStdStallRate",
    "clMeanStallRate",
    "cdMax",
    "cdStdRate",
    "cdMeanRate",
    "cmAlphaZero",
]

REYNOLDS_PREFIX = {
    "200000": "r2",
    "500000": "r5",
    "1000000": "r10",
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_airfoil(name: str, re_rows: pd.DataFrame) -> str | None:
    """
    Return an error string if the airfoil fails validation, else None.

    Parameters
    ----------
    name : str
    re_rows : pd.DataFrame
        Subset of the raw polars DataFrame for this airfoil (all Reynolds).
    """
    if len(re_rows) < 3:
        return "fewer than 3 Reynolds numbers available"

    for _, row in re_rows.iterrows():
        al = pd.Series(row["alpha"])
        cl = pd.Series(row["cl"])
        cd = pd.Series(row["cd"])
        cm = pd.Series(row["cm"])

        if not (
            al.apply(pd.to_numeric, errors="coerce").notna().all()
            and cl.apply(pd.to_numeric, errors="coerce").notna().all()
            and cd.apply(pd.to_numeric, errors="coerce").notna().all()
            and cm.apply(pd.to_numeric, errors="coerce").notna().all()
        ):
            return f"non-numeric values in polar (Re={row['reynolds']})"

        max_idx = int(np.argmax(cl))
        if len(cl) - max_idx < 7:
            return f"insufficient stall region (Re={row['reynolds']})"

    return None


# ---------------------------------------------------------------------------
# Indicator calculation
# ---------------------------------------------------------------------------

def _compute_indicators(row: pd.Series) -> dict:
    """
    Compute all aerodynamic indicators for a single airfoil/Reynolds row.

    Parameters
    ----------
    row : pd.Series
        One row from the raw polars DataFrame.

    Returns
    -------
    dict with keys matching INDICATOR_KEYS.
    """
    al = np.array(row["alpha"], dtype=float)
    cl = np.array(row["cl"],    dtype=float)
    cd = np.array(row["cd"],    dtype=float)
    cm = np.array(row["cm"],    dtype=float)

    max_idx = int(np.argmax(cl))

    # --- Cl indicators ---
    cl_max = float(np.max(cl))

    # Pre-stall: positive-Cl portion before the peak
    valid_pre = np.where(cl[:max_idx] > 0)[0]
    cl_valid = cl[valid_pre]
    cl_diff = np.diff(cl_valid)

    cl_std_rate  = float(cl_diff.std())   if len(cl_diff) > 0 else np.nan
    cl_mean_rate = float(cl_diff.mean())  if len(cl_diff) > 0 else np.nan

    # Post-stall
    cl_stall_diff = np.diff(cl[max_idx:])
    cl_std_stall  = float(cl_stall_diff.std())        if len(cl_stall_diff) > 0 else np.nan
    cl_mean_stall = float(abs(cl_stall_diff.mean()))  if len(cl_stall_diff) > 0 else np.nan

    # --- Cd indicators (alpha >= 0, pre-stall) ---
    valid_cd = np.where(al[:max_idx] >= 0)[0]
    cd_valid = cd[valid_cd]
    cd_diff  = np.diff(cd_valid)

    cd_max      = float(np.max(cd_valid)) if len(cd_valid) > 0 else np.nan
    cd_std_rate = float(cd_diff.std())    if len(cd_diff)  > 0 else np.nan
    cd_mean_rate= float(cd_diff.mean())   if len(cd_diff)  > 0 else np.nan

    # --- Cm indicator ---
    cm_alpha_zero = float(abs(cm[al >= 0][0])) if (al >= 0).any() else np.nan

    return {
        "max_cl_cd":        row["max_cl_cd"],
        "max_cl_cd_alpha":  row["max_cl_cd_alpha"],
        "clMax":            cl_max,
        "clStdRate":        cl_std_rate,
        "clMeanRate":       cl_mean_rate,
        "clStdStallRate":   cl_std_stall,
        "clMeanStallRate":  cl_mean_stall,
        "cdMax":            cd_max,
        "cdStdRate":        cd_std_rate,
        "cdMeanRate":       cd_mean_rate,
        "cmAlphaZero":      cm_alpha_zero,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate airfoils and compute indicators.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Output of the scraper step (or loaded from raw_polars.parquet).

    Returns
    -------
    df_feat : pd.DataFrame
        Wide feature table — one row per airfoil with prefixed indicator
        columns (r2*, r5*, r10*) plus 'name'.
    df_failed : pd.DataFrame
        Table of removed airfoils with columns ['name', 'reason'].
    """
    failed_records = []
    feature_records = []

    for name, group in df_raw.groupby("name"):
        error = _validate_airfoil(name, group)
        if error:
            logger.warning("Dropping %s: %s", name, error)
            failed_records.append({"name": name, "reason": error})
            continue

        row_dict = {"name": name}
        for _, re_row in group.iterrows():
            prefix = REYNOLDS_PREFIX.get(re_row["reynolds"])
            if prefix is None:
                continue
            indicators = _compute_indicators(re_row)
            for key, val in indicators.items():
                row_dict[f"{prefix}{key}"] = val

        feature_records.append(row_dict)

    df_feat   = pd.DataFrame(feature_records)
    df_failed = pd.DataFrame(failed_records) if failed_records \
                else pd.DataFrame(columns=["name", "reason"])

    # Drop rows with any NaN (ensures clean downstream analysis)
    before = len(df_feat)
    df_feat = df_feat.dropna().reset_index(drop=True)
    after  = len(df_feat)
    if before != after:
        logger.warning(
            "Dropped %d airfoils due to NaN indicators.", before - after
        )

    logger.info(
        "Analysis complete: %d valid airfoils, %d removed.",
        len(df_feat), len(df_failed),
    )
    return df_feat, df_failed
