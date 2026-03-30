# -*- coding: utf-8 -*-
"""
pipeline/05_ranking_ahp.py
==========================
Step 4b — Rank airfoils using AHP (Analytic Hierarchy Process) with
          entropy-based weights.

Method
------
1. Min-Max scale all indicators to [0, 1].
2. For "lower-is-better" indicators invert the value (1 / x) so that a
   higher normalised score always means better.
3. Column-normalise the matrix (each value ÷ column sum).
4. Derive weights via the Coefficient of Variation (CV = std / mean):
   indicators with higher variability across airfoils get more weight
   because they carry more discriminating power.
5. Compute a final score as the dot-product of the normalised matrix and
   the weight vector.

Output
------
data/results/ranking_ahp.parquet
    Columns: name, score_ahp, rank_ahp
    Sorted by rank_ahp ascending (1 = best).
"""

import logging

import pandas as pd
from sklearn import preprocessing as skl_pp

logger = logging.getLogger(__name__)

# Indicator suffixes that are "lower-is-better"
L_SUFFIXES = [
    "max_cl_cd_alpha",
    "clStdRate",
    "clStdStallRate",
    "clMeanStallRate",
    "cdMax",
    "cdStdRate",
    "cdMeanRate",
    "cmAlphaZero",
]


def _get_L_columns(columns: pd.Index) -> list[str]:
    return [
        c for c in columns
        if any(c.endswith(s) or c.endswith(s.replace("_", "")) for s in L_SUFFIXES)
           and c != "name"
    ]


def run(df_features: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Rank airfoils using entropy-weighted AHP.

    Parameters
    ----------
    df_features : pd.DataFrame
        Output of 03_features.run().
    cfg : dict
        Full pipeline configuration.

    Returns
    -------
    pd.DataFrame
        Ranking table sorted by rank_ahp ascending.
        Also saved to ``data/results/ranking_ahp.parquet``.
    """
    output_path = f"{cfg['data']['results_dir']}/ranking_ahp.parquet"

    names = df_features["name"].reset_index(drop=True)

    indicator_cols = [
        c for c in df_features.columns
        if c not in ("name", "thickness_value", "thickness_chord",
                     "camber_value", "camber_chord")
    ]
    df_ahp = df_features[indicator_cols].reset_index(drop=True).copy()
    df_ahp = df_ahp.apply(pd.to_numeric, errors="coerce").dropna()
    names  = names.loc[df_ahp.index].reset_index(drop=True)
    df_ahp = df_ahp.reset_index(drop=True)

    # 1. Min-Max scaling to [0, 1]
    scaler = skl_pp.MinMaxScaler()
    df_ahp = pd.DataFrame(
        scaler.fit_transform(df_ahp),
        columns=df_ahp.columns,
        index=df_ahp.index,
    )

    # 2. Replace exact zeros to avoid division by zero on inversion
    for col in df_ahp.columns:
        zero_mask = df_ahp[col] == 0
        if zero_mask.any():
            nonzero_min = df_ahp.loc[~zero_mask, col].min()
            df_ahp.loc[zero_mask, col] = nonzero_min

    # 3. Invert lower-is-better indicators
    l_cols = _get_L_columns(df_ahp.columns)
    for col in l_cols:
        df_ahp[col] = 1.0 / df_ahp[col]

    logger.info("AHP ranking: %d L-columns inverted.", len(l_cols))

    # 4. Column-normalise
    df_ahp = df_ahp.div(df_ahp.sum())

    # 5. Entropy-based weights via CV
    stats = pd.DataFrame({
        "mean": df_ahp.mean(),
        "std":  df_ahp.std(),
    })
    stats["cv"]   = stats["std"] / stats["mean"]
    stats["peso"] = stats["cv"]  / stats["cv"].sum()

    # 6. Weighted score
    scores = df_ahp.dot(stats["peso"])

    df_result = pd.DataFrame({
        "name":      names,
        "score_ahp": scores.values,
    })
    df_result["rank_ahp"] = df_result["score_ahp"].rank(ascending=False, method="min")
    df_result = df_result.sort_values("rank_ahp").reset_index(drop=True)

    df_result.to_parquet(output_path, index=False)
    logger.info("AHP ranking saved to %s", output_path)

    top = df_result.head(cfg["ranking"]["top_n_per_method"])
    logger.info("Top %d by AHP:\n%s", len(top), top[["name", "rank_ahp"]].to_string(index=False))

    return df_result
