# -*- coding: utf-8 -*-
"""
pipeline/06_ranking_ahp_pca.py
==============================
Step 4c — Rank airfoils using AHP applied on top of PCA factor scores.

Method
------
Instead of applying AHP directly to the raw 33 indicator columns (as in
05_ranking_ahp.py), this step first reduces dimensionality by running a
joint factor analysis on all indicators, then applies the entropy-weighted
AHP procedure to the resulting factor scores.

This tends to reduce the impact of correlated indicator groups (e.g. all
three Reynolds entries of the same indicator behaving alike) by compressing
them into a common latent factor before weighting.

Steps
-----
1. Factor-analyse all indicators jointly → factor score matrix.
2. Designate the *first* factor as G (higher-is-better by convention in
   the original code) and all remaining factors as L (lower-is-better).
3. Min-Max scale, invert L factors, column-normalise.
4. Derive CV-based weights and compute the AHP score.

Output
------
data/results/ranking_ahp_pca.parquet
    Columns: name, score_ahp_pca, rank_ahp_pca
    Sorted by rank_ahp_pca ascending (1 = best).
"""

import logging

import numpy as np
import pandas as pd
from sklearn import preprocessing as skl_pp

logger = logging.getLogger(__name__)


def _choose_n_factors(df: pd.DataFrame, max_n: int) -> int:
    """
    Kaiser rule via numpy — avoids the factor_analyzer/sklearn
    force_all_finite incompatibility (sklearn >= 1.6).
    """
    corr = np.corrcoef(df.values, rowvar=False)
    eigenvalues = np.linalg.eigh(corr)[0][::-1]
    n = max(1, int((eigenvalues > 1).sum()))
    return min(n, max_n)


def run(df_features: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Rank airfoils using AHP on PCA factor scores.

    Parameters
    ----------
    df_features : pd.DataFrame
        Output of 03_features.run().
    cfg : dict
        Full pipeline configuration.

    Returns
    -------
    pd.DataFrame
        Ranking table sorted by rank_ahp_pca ascending.
        Also saved to ``data/results/ranking_ahp_pca.parquet``.
    """
    output_path = f"{cfg['data']['results_dir']}/ranking_ahp_pca.parquet"

    names = df_features["name"].reset_index(drop=True)

    indicator_cols = [
        c for c in df_features.columns
        if c not in ("name", "thickness_value", "thickness_chord",
                     "camber_value", "camber_chord")
    ]
    df_ind = df_features[indicator_cols].reset_index(drop=True).copy()
    df_ind = df_ind.apply(pd.to_numeric, errors="coerce").dropna()
    names  = names.loc[df_ind.index].reset_index(drop=True)
    df_ind = df_ind.reset_index(drop=True)

    # --- Factor analysis (pure numpy — no scikit-learn dependency) ---
    n_factors = _choose_n_factors(df_ind, max_n=min(10, len(indicator_cols)))
    logger.info("AHP-PCA: extracting %d factors from %d indicators.", n_factors, len(indicator_cols))

    X = (df_ind.values - df_ind.values.mean(axis=0)) / (df_ind.values.std(axis=0, ddof=1) + 1e-12)

    corr        = np.corrcoef(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    idx          = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx][:, :n_factors]
    eigenvalues  = eigenvalues[idx][:n_factors]
    loadings     = eigenvectors * np.sqrt(eigenvalues)
    scores       = X @ np.linalg.pinv(corr) @ loadings

    factor_names = [f"Fator {i+1}" for i in range(n_factors)]
    df_factors   = pd.DataFrame(scores, columns=factor_names)

    # Convention from original code: Factor 1 is G (higher=better),
    # all others are L (lower=better / inverted).
    l_factors = factor_names[1:]

    # --- AHP on factors ---
    df_ahp = df_factors.copy()

    scaler = skl_pp.MinMaxScaler()
    df_ahp = pd.DataFrame(
        scaler.fit_transform(df_ahp),
        columns=df_ahp.columns,
    )

    for col in df_ahp.columns:
        zero_mask = df_ahp[col] == 0
        if zero_mask.any():
            nonzero_min = df_ahp.loc[~zero_mask, col].min()
            df_ahp.loc[zero_mask, col] = nonzero_min

    for col in l_factors:
        df_ahp[col] = 1.0 / df_ahp[col]

    df_ahp = df_ahp.div(df_ahp.sum())

    stats = pd.DataFrame({
        "mean": df_ahp.mean(),
        "std":  df_ahp.std(),
    })
    stats["cv"]   = stats["std"] / stats["mean"]
    stats["peso"] = stats["cv"]  / stats["cv"].sum()

    scores = df_ahp.dot(stats["peso"])

    df_result = pd.DataFrame({
        "name":         names,
        "score_ahp_pca": scores.values,
    })
    df_result["rank_ahp_pca"] = df_result["score_ahp_pca"].rank(ascending=False, method="min")
    df_result = df_result.sort_values("rank_ahp_pca").reset_index(drop=True)

    df_result.to_parquet(output_path, index=False)
    logger.info("AHP-PCA ranking saved to %s", output_path)

    top = df_result.head(cfg["ranking"]["top_n_per_method"])
    logger.info("Top %d by AHP-PCA:\n%s", len(top), top[["name", "rank_ahp_pca"]].to_string(index=False))

    return df_result