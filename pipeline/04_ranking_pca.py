# -*- coding: utf-8 -*-
"""
pipeline/04_ranking_pca.py
==========================
Step 4a — Rank airfoils using Factor Analysis (PCA-based).

The method mirrors the original pca_analysis.py logic:

1. Separate features into two groups:
   - ``L`` (lower-is-better indicators, e.g. drag, stall severity)
   - ``G`` (higher-is-better indicators, e.g. lift, efficiency)
2. Run a principal-factor analysis on each group independently to extract
   latent factors and their explained-variance weights.
3. Compute a composite score per airfoil by weighting factor scores with
   the proportion of variance explained by each factor.
4. Combine the ``L`` and ``G`` rankings into a final position sum.

The ``L`` group ranking is ascending (lower raw score = better).
The ``G`` group ranking is descending (higher raw score = better).

Output
------
data/results/ranking_pca.parquet
    Columns: name, score_L, score_G, rank_L, rank_G, rank_final
    Sorted by rank_final ascending (1 = best).
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature group definitions
# ---------------------------------------------------------------------------

# Indicators where a LOWER value is better (L = "less is better")
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
    """Return feature columns whose suffix matches an L indicator."""
    return [
        c for c in columns
        if any(c.endswith(s) or c.endswith(s.replace("_", "")) for s in L_SUFFIXES)
           and c != "name"
    ]


def _pca_factor_scores(X: np.ndarray, n_factors: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pure-numpy principal-factor analysis.

    Computes loadings and factor scores via eigendecomposition of the
    correlation matrix — identical in result to FactorAnalyzer with
    method="principal" and rotation=None, but with no scikit-learn
    dependency so it is immune to the force_all_finite/ensure_all_finite
    rename that broke factor_analyzer on scikit-learn >= 1.6.

    Parameters
    ----------
    X        : (n_samples, n_features) standardised data array.
    n_factors: Number of factors to retain.

    Returns
    -------
    scores    : (n_samples, n_factors) factor score matrix.
    loadings  : (n_features, n_factors) loading matrix.
    variance  : (n_factors,) proportion of total variance per factor.
    """
    corr = np.corrcoef(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)

    # eigh returns ascending order — reverse to descending
    idx          = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Retain top n_factors
    eigenvalues  = eigenvalues[:n_factors]
    eigenvectors = eigenvectors[:, :n_factors]

    # Loadings: eigenvector * sqrt(eigenvalue)
    loadings = eigenvectors * np.sqrt(eigenvalues)

    # Factor scores via regression method
    corr_inv = np.linalg.pinv(corr)
    scores   = X @ corr_inv @ loadings

    # Variance explained (proportion of total variance = n_features)
    variance = eigenvalues / corr.shape[0]

    return scores, loadings, variance


def _factor_score(df_feat: pd.DataFrame, n_factors: int) -> tuple[pd.Series, pd.DataFrame]:
    """
    Fit a principal-factor model and return:
      - composite score Series (weighted sum of factor scores × explained variance)
      - variance table DataFrame
    """
    # Standardise (z-score) before factoring
    X = (df_feat.values - df_feat.values.mean(axis=0)) / (df_feat.values.std(axis=0, ddof=1) + 1e-12)

    scores, _, variance = _pca_factor_scores(X, n_factors)

    factor_names   = [f"Factor {i+1}" for i in range(n_factors)]
    variance_table = pd.DataFrame({
        "Eigenvalue": variance * df_feat.shape[1],
        "Variance":   variance,
        "Cumulative Variance": np.cumsum(variance),
    }, index=factor_names)

    factor_scores = pd.DataFrame(scores, columns=factor_names, index=df_feat.index)

    composite = pd.Series(0.0, index=df_feat.index)
    for factor in factor_names:
        composite += factor_scores[factor] * variance_table.loc[factor, "Variance"]

    return composite, variance_table


def _choose_n_factors(df: pd.DataFrame, max_n: int) -> int:
    """
    Select number of factors via the Kaiser rule (eigenvalues > 1),
    bounded by max_n.

    Eigenvalues are computed directly from the correlation matrix using
    numpy instead of calling FactorAnalyzer.fit() for the count step.
    This avoids a TypeError caused by a breaking change in scikit-learn
    (>=1.6) that removed the ``force_all_finite`` argument that older
    versions of factor_analyzer pass to check_array() internally.
    """
    corr = np.corrcoef(df.values, rowvar=False)
    eigenvalues = np.linalg.eigh(corr)[0][::-1]   # descending order
    n = max(1, int((eigenvalues > 1).sum()))
    return min(n, max_n)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(df_features: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Rank airfoils by factor analysis.

    Parameters
    ----------
    df_features : pd.DataFrame
        Output of 03_features.run() (filtered feature table).
    cfg : dict
        Full pipeline configuration.

    Returns
    -------
    pd.DataFrame
        Ranking table sorted by rank_final ascending.
        Also saved to ``data/results/ranking_pca.parquet``.
    """
    output_path = f"{cfg['data']['results_dir']}/ranking_pca.parquet"

    names = df_features["name"].reset_index(drop=True)

    # Keep only numeric indicator columns (drop geometry & name)
    indicator_cols = [
        c for c in df_features.columns
        if c not in ("name", "thickness_value", "thickness_chord",
                     "camber_value", "camber_chord")
    ]
    df_ind = df_features[indicator_cols].reset_index(drop=True)
    df_ind = df_ind.apply(pd.to_numeric, errors="coerce").dropna()

    # Re-align names after dropna
    names = names.loc[df_ind.index].reset_index(drop=True)
    df_ind = df_ind.reset_index(drop=True)

    # Split into L and G groups
    l_cols = _get_L_columns(df_ind.columns)
    g_cols = [c for c in df_ind.columns if c not in l_cols]

    logger.info("PCA ranking: %d L-features, %d G-features", len(l_cols), len(g_cols))

    df_L = df_ind[l_cols]
    df_G = df_ind[g_cols]

    # Choose number of factors automatically (Kaiser rule, capped)
    n_L = _choose_n_factors(df_L, max_n=min(8, len(l_cols)))
    n_G = _choose_n_factors(df_G, max_n=min(3, len(g_cols)))
    logger.info("Factors selected: L=%d  G=%d", n_L, n_G)

    score_L, var_table_L = _factor_score(df_L, n_L)
    score_G, var_table_G = _factor_score(df_G, n_G)

    logger.info(
        "L variance explained: %.1f%%   G variance explained: %.1f%%",
        var_table_L["Variance"].sum() * 100,
        var_table_G["Variance"].sum() * 100,
    )

    # Rank: L ascending (lower score = better), G descending (higher = better)
    df_result = pd.DataFrame({
        "name":    names,
        "score_L": score_L,
        "score_G": score_G,
    })
    df_result["rank_L"] = df_result["score_L"].rank(ascending=True,  method="min")
    df_result["rank_G"] = df_result["score_G"].rank(ascending=False, method="min")
    df_result["rank_final"] = df_result["rank_L"] + df_result["rank_G"]
    df_result = df_result.sort_values("rank_final").reset_index(drop=True)

    df_result.to_parquet(output_path, index=False)
    logger.info("PCA ranking saved to %s", output_path)

    top = df_result.head(cfg["ranking"]["top_n_per_method"])
    logger.info("Top %d by PCA:\n%s", len(top), top[["name", "rank_final"]].to_string(index=False))

    return df_result