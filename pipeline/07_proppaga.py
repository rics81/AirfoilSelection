# -*- coding: utf-8 -*-
"""
pipeline/07_proppaga.py
=======================
Step 5 — Final ranking via the PrOPPAGA method.

PrOPPAGA (Processo de Ordenação Por Pesos e Ganhos Acumulados) applies a
z-score normalisation, inverts lower-is-better indicators, re-normalises,
and then computes a weighted dot-product score.

Candidate selection
-------------------
The top-N airfoils from each of the three ranking methods (PCA, AHP,
AHP-PCA) are collected into a unique candidate pool.  Duplicates across
methods are kept only once.  N is controlled by
``cfg['ranking']['top_n_per_method']``.

Weights
-------
The 33-element weight vector defined in ``config.yaml`` (section
``proppaga.weights``) maps directly to the 33 indicator columns
(11 indicators × 3 Reynolds numbers) in the order they appear in the
features table.  Weights are normalised to sum to 1 internally.

Lower-is-better indicators
--------------------------
The same ``L_SUFFIXES`` list used by the other ranking modules is applied
here: matching columns are multiplied by −1 before the second z-score
normalisation so that a higher final score always means a better airfoil.

Output
------
data/results/ranking_proppaga.parquet
    Columns: name, score_proppaga, rank_proppaga, source_methods
    Sorted by rank_proppaga ascending (1 = best).

data/results/final_summary.parquet
    Wide table merging all four rankings for every candidate airfoil.
"""

import logging

import pandas as pd
from scipy.stats import zscore

logger = logging.getLogger(__name__)

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


def _collect_candidates(
    df_features: pd.DataFrame,
    rank_pca: pd.DataFrame,
    rank_ahp: pd.DataFrame,
    rank_ahp_pca: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """
    Build the candidate pool from the top-N of each ranking method.

    Returns a DataFrame with an extra column ``source_methods`` that lists
    which methods nominated each airfoil.
    """
    sources: dict[str, list[str]] = {}

    for label, df_rank, rank_col in [
        ("PCA",     rank_pca,     "rank_final"),
        ("AHP",     rank_ahp,     "rank_ahp"),
        ("AHP-PCA", rank_ahp_pca, "rank_ahp_pca"),
    ]:
        top_names = df_rank.nsmallest(top_n, rank_col)["name"].tolist()
        for name in top_names:
            sources.setdefault(name, []).append(label)

    candidate_names = list(sources.keys())
    logger.info(
        "PrOPPAGA candidate pool: %d unique airfoils from top-%d of each method.",
        len(candidate_names), top_n,
    )

    df_cand = df_features[df_features["name"].isin(candidate_names)].copy()
    df_cand["source_methods"] = df_cand["name"].map(
        lambda n: ", ".join(sources.get(n, []))
    )
    return df_cand


def run(
    df_features: pd.DataFrame,
    rank_pca: pd.DataFrame,
    rank_ahp: pd.DataFrame,
    rank_ahp_pca: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """
    Apply PrOPPAGA to the top-N candidates from all ranking methods.

    Parameters
    ----------
    df_features : pd.DataFrame
        Full filtered feature table (output of 03_features.run()).
    rank_pca : pd.DataFrame
        Output of 04_ranking_pca.run().
    rank_ahp : pd.DataFrame
        Output of 05_ranking_ahp.run().
    rank_ahp_pca : pd.DataFrame
        Output of 06_ranking_ahp_pca.run().
    cfg : dict
        Full pipeline configuration.

    Returns
    -------
    pd.DataFrame
        Final PrOPPAGA ranking table.
    """
    top_n       = cfg["ranking"]["top_n_per_method"]
    weights_raw = cfg["proppaga"]["weights"]
    results_dir = cfg["data"]["results_dir"]

    # --- Candidate pool ---
    df_cand = _collect_candidates(df_features, rank_pca, rank_ahp, rank_ahp_pca, top_n)

    indicator_cols = [
        c for c in df_features.columns
        if c not in ("name", "thickness_value", "thickness_chord",
                     "camber_value", "camber_chord", "source_methods")
    ]

    if len(weights_raw) != len(indicator_cols):
        raise ValueError(
            f"PrOPPAGA weight vector length ({len(weights_raw)}) does not match "
            f"indicator column count ({len(indicator_cols)}).  "
            "Update config.yaml → proppaga.weights."
        )

    # --- Build weight Series ---
    w = pd.Series(weights_raw, index=indicator_cols, dtype=float)
    w = w / w.sum()   # normalise to sum = 1

    # --- Step 1: z-score normalisation ---
    df_scores = df_cand[indicator_cols].apply(zscore, ddof=1)
    df_scores.index = df_cand.index

    # --- Step 2: Invert lower-is-better indicators ---
    l_cols = _get_L_columns(df_scores.columns)
    for col in l_cols:
        df_scores[col] = df_scores[col] * -1

    # --- Step 3: Re-normalise after inversion ---
    df_scores = df_scores.apply(zscore, ddof=1)

    # --- Step 4: Weighted dot-product ---
    df_cand = df_cand.copy()
    df_cand["score_proppaga"] = df_scores.dot(w)
    df_cand["rank_proppaga"]  = df_cand["score_proppaga"].rank(ascending=False, method="min")
    df_cand = df_cand.sort_values("rank_proppaga").reset_index(drop=True)

    # --- Save PrOPPAGA result ---
    out_cols = ["name", "score_proppaga", "rank_proppaga", "source_methods"]
    df_result = df_cand[out_cols].copy()
    df_result.to_parquet(f"{results_dir}/ranking_proppaga.parquet", index=False)
    logger.info("PrOPPAGA ranking saved to %s/ranking_proppaga.parquet", results_dir)

    # --- Save combined summary ---
    df_summary = df_result.merge(
        rank_pca[["name", "rank_final"]].rename(columns={"rank_final": "rank_pca"}),
        on="name", how="left",
    ).merge(
        rank_ahp[["name", "rank_ahp"]],
        on="name", how="left",
    ).merge(
        rank_ahp_pca[["name", "rank_ahp_pca"]],
        on="name", how="left",
    )
    df_summary.to_parquet(f"{results_dir}/final_summary.parquet", index=False)

    logger.info("\n=== FINAL PROPPAGA RANKING ===")
    logger.info("\n%s", df_summary.to_string(index=False))

    return df_result
