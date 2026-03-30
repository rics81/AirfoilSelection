# -*- coding: utf-8 -*-
"""
pipeline/03_features.py
=======================
Step 3 — Apply geometry pre-filter and save the final features table.

This step joins the aerodynamic indicators produced by 02_analysis.py with
the geometry metadata (thickness, camber) stored in the raw polars parquet,
then applies the user-defined thickness/camber bounds from config.yaml.

Geometry filter (config.yaml → filter section)
-----------------------------------------------
thickness_min / thickness_max   Max-thickness range (% chord).
camber_min    / camber_max      Max-camber range    (% chord).
Any bound set to ``null`` in the config is treated as ±∞ (no limit).

Column definitions added by this step
--------------------------------------
thickness_value   float   Max thickness  (% chord)
thickness_chord   float   Chord location of max thickness
camber_value      float   Max camber     (% chord)
camber_chord      float   Chord location of max camber

Output
------
data/features.parquet
    Final wide feature table used by all ranking steps.  One row per
    airfoil that passed both quality checks (02_analysis) and geometry
    filter (this step).
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _apply_geometry_filter(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter rows by thickness and camber bounds.

    Parameters
    ----------
    df : pd.DataFrame
        Feature table with geometry columns already merged.
    cfg : dict
        Full pipeline configuration.

    Returns
    -------
    df_pass : pd.DataFrame   Airfoils that pass the filter.
    df_fail : pd.DataFrame   Airfoils that were removed, with reason column.
    """
    filt = cfg["filter"]

    mask = pd.Series(True, index=df.index)
    reasons = pd.Series("", index=df.index)

    def _apply(bound_key: str, col: str, op: str):
        nonlocal mask, reasons
        val = filt.get(bound_key)
        if val is None:
            return  # no limit defined
        if op == "min":
            failed = df[col] < val
            label  = f"{col} < {val}"
        else:
            failed = df[col] > val
            label  = f"{col} > {val}"
        reasons[failed & mask] = label
        mask &= ~failed

    _apply("thickness_min", "thickness_value", "min")
    _apply("thickness_max", "thickness_value", "max")
    _apply("camber_min",    "camber_value",    "min")
    _apply("camber_max",    "camber_value",    "max")

    df_pass = df[mask].copy()
    df_fail = df[~mask][["name"]].copy()
    df_fail["reason"] = reasons[~mask].values

    return df_pass, df_fail


def run(
    df_feat: pd.DataFrame,
    df_raw: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """
    Merge geometry data, apply the pre-filter and persist the result.

    Parameters
    ----------
    df_feat : pd.DataFrame
        Output of 02_analysis.run() — wide indicator table.
    df_raw : pd.DataFrame
        Raw polars DataFrame (from 01_scraper or loaded parquet) — used
        to obtain geometry columns (thickness/camber) per airfoil.
    cfg : dict
        Full pipeline configuration.

    Returns
    -------
    pd.DataFrame
        Filtered feature table, also saved to
        ``cfg['data']['features_path']``.
    """
    output_path = cfg["data"]["features_path"]

    # --- Attach geometry (one row per airfoil: take first Reynolds row) ---
    geom_cols = ["name", "thickness_value", "thickness_chord",
                 "camber_value", "camber_chord"]

    df_geom = (
        df_raw[geom_cols]
        .drop_duplicates(subset="name")
        .reset_index(drop=True)
    )

    df_merged = df_feat.merge(df_geom, on="name", how="left")

    # Warn about airfoils with missing geometry (will fail numeric filter)
    missing_geom = df_merged[df_merged["thickness_value"].isna()]["name"].tolist()
    if missing_geom:
        logger.warning(
            "%d airfoils have no geometry data and will be removed: %s",
            len(missing_geom), missing_geom[:10],
        )
    df_merged = df_merged.dropna(subset=["thickness_value", "camber_value"])

    # --- Geometry pre-filter ---
    filt = cfg["filter"]
    logger.info(
        "Applying geometry filter — thickness: [%s, %s]  camber: [%s, %s]",
        filt.get("thickness_min"), filt.get("thickness_max"),
        filt.get("camber_min"),    filt.get("camber_max"),
    )

    df_pass, df_fail = _apply_geometry_filter(df_merged, cfg)

    if not df_fail.empty:
        logger.info(
            "Geometry filter removed %d airfoils:", len(df_fail)
        )
        for _, r in df_fail.iterrows():
            logger.info("  ✗  %-30s  (%s)", r["name"], r["reason"])

    logger.info(
        "%d airfoils remain after geometry filter (started with %d).",
        len(df_pass), len(df_merged),
    )

    # --- Save ---
    df_pass.to_parquet(output_path, index=False)
    logger.info("Features table saved to %s", output_path)

    return df_pass
