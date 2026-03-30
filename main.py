# -*- coding: utf-8 -*-
"""
main.py
=======
AirfoilSelection Pipeline Orchestrator
=======================================

Runs the full airfoil selection pipeline from data acquisition to final
ranking.  Each step is a separate module under ``pipeline/`` and can be
run independently if needed.

Pipeline steps
--------------
1. Scraper       — Download polar + geometry data from airfoiltools.com
2. Analysis      — Quality-check airfoils and compute aerodynamic indicators
3. Features      — Apply geometry pre-filter; save features.parquet
4. Rankings      — Factor-analysis (PCA), AHP and AHP-PCA rankings
5. PrOPPAGA      — Final weighted ranking of top-N candidates

Usage
-----
    python main.py                    # full pipeline, prompts for scrape decision
    python main.py --force-scrape     # always re-scrape, ignoring cached data
    python main.py --skip-scrape      # always use cached data, error if missing
    python main.py --steps 3 4 5      # run only specific steps (1-indexed)

Configuration
-------------
All parameters live in config.yaml (geometry filter bounds, weights,
Reynolds numbers, top-N, output paths …).  Edit that file — do not
hard-code values here.
"""

import argparse
import datetime
import logging
import os
import sys

import pandas as pd
import yaml

from pipeline.loader import load_step

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("Configuration loaded from %s", path)
    return cfg


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _parquet_exists(path: str) -> bool:
    return os.path.isfile(path)


def _ask_update(path: str, n_saved: int, n_total: int) -> str:
    """
    Ask the user what to do when a parquet file already exists.

    Parameters
    ----------
    path    : Path to the existing parquet.
    n_saved : Number of airfoils already in the parquet.
    n_total : Total airfoils on airfoiltools (from the current list fetch).

    Returns
    -------
    "fresh"  → delete parquet and start from scratch.
    "resume" → call scraper.run() so it skips done airfoils and continues.
    "skip"   → parquet is complete, load it directly without running scraper.
    """
    mtime    = os.path.getmtime(path)
    ts       = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    complete = (n_saved >= n_total)

    print("\n" + "─" * 60)
    print(f"  Cached data found : {path}")
    print(f"  Last modified     : {ts}")
    print(f"  Airfoils saved    : {n_saved} / {n_total}"
          + ("  ✓ complete" if complete else "  ⚠ incomplete — resume available"))
    print("─" * 60)

    if complete:
        while True:
            ans = input("  Re-scrape from scratch? [y/N]: ").strip().lower()
            if ans in ("", "n", "no"):
                return "skip"
            if ans in ("y", "yes"):
                return "fresh"
            print("  Please enter 'y' or 'n'.")
    else:
        print("  Options:")
        print("    r = resume from airfoil", n_saved + 1, "(recommended)")
        print("    f = start fresh (deletes saved progress)")
        print("    s = skip scraping entirely (use incomplete data as-is)")
        while True:
            ans = input("  Choice [R/f/s]: ").strip().lower()
            if ans in ("", "r", "resume"):
                return "resume"
            if ans in ("f", "fresh"):
                return "fresh"
            if ans in ("s", "skip"):
                return "skip"
            print("  Please enter r, f or s.")


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def step1_scrape(cfg: dict, force: bool = False, skip: bool = False) -> pd.DataFrame:
    """Step 1 — Scrape or load raw polar data.

    Decision matrix
    ---------------
    --force-scrape          : always delete checkpoint and start fresh.
    --skip-scrape           : always load parquet as-is (error if missing).
    no flags + no parquet   : run scraper from scratch.
    no flags + complete     : ask user — re-scrape fresh or use cached.
    no flags + incomplete   : ask user — resume / start fresh / use as-is.
    """
    raw_path = cfg["data"]["raw_polars_path"]
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    if skip:
        if not _parquet_exists(raw_path):
            logger.error("--skip-scrape requested but %s not found.", raw_path)
            sys.exit(1)
        logger.info("Step 1 skipped — loading cached data from %s", raw_path)
        return pd.read_parquet(raw_path)

    if force:
        if _parquet_exists(raw_path):
            logger.info("--force-scrape: deleting %s and starting fresh.", raw_path)
            os.remove(raw_path)
        logger.info("Step 1 — Scraping airfoiltools.com …")
        scraper = load_step(1)
        return scraper.run(cfg)

    if _parquet_exists(raw_path):
        # Count how many unique airfoils are already saved so we can show
        # the user an informed resume prompt.  We do NOT fetch the full
        # airfoil list here (that is the scraper's job) — instead we use
        # the row count as a proxy; the scraper will reconcile exactly.
        df_existing  = pd.read_parquet(raw_path)
        n_saved      = df_existing["name"].nunique()
        # Use a rough total from config if available, else show "?" 
        n_total      = cfg.get("_airfoil_total", 1638)   # updated by scraper on first run

        decision = _ask_update(raw_path, n_saved, n_total)

        if decision == "skip":
            logger.info("Using cached data from %s  (%d airfoils)", raw_path, n_saved)
            return df_existing

        if decision == "fresh":
            logger.info("Starting fresh — deleting %s", raw_path)
            os.remove(raw_path)
            # fall through to scraper below

        # decision == "resume" → just call run(); it will skip done names
        if decision == "resume":
            logger.info("Resuming scrape from checkpoint (%d done) …", n_saved)

    logger.info("Step 1 — Scraping airfoiltools.com …")
    scraper = load_step(1)
    return scraper.run(cfg)


def step2_analysis(df_raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 2 — Quality-check and compute indicators."""
    logger.info("Step 2 — Running quality checks and computing indicators …")
    analysis = load_step(2)
    return analysis.run(df_raw)


def step3_features(
    df_feat: pd.DataFrame,
    df_raw: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """Step 3 — Geometry pre-filter and save features.parquet."""
    os.makedirs(cfg["data"]["results_dir"], exist_ok=True)
    logger.info("Step 3 — Applying geometry filter …")
    features = load_step(3)
    return features.run(df_feat, df_raw, cfg)


def step4_rankings(df_features: pd.DataFrame, cfg: dict) -> tuple:
    """Steps 4a-c — PCA, AHP and AHP-PCA rankings."""
    pca_mod, ahp_mod, ahp_pca_mod = load_step(4)

    logger.info("Step 4a — Factor-analysis (PCA) ranking …")
    rank_pca = pca_mod.run(df_features, cfg)

    logger.info("Step 4b — AHP ranking …")
    rank_ahp = ahp_mod.run(df_features, cfg)

    logger.info("Step 4c — AHP-PCA ranking …")
    rank_ahp_pca = ahp_pca_mod.run(df_features, cfg)

    return rank_pca, rank_ahp, rank_ahp_pca


def step5_proppaga(
    df_features: pd.DataFrame,
    rank_pca: pd.DataFrame,
    rank_ahp: pd.DataFrame,
    rank_ahp_pca: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """Step 5 — PrOPPAGA final ranking."""
    logger.info("Step 5 — PrOPPAGA final ranking …")
    proppaga = load_step(5)
    return proppaga.run(df_features, rank_pca, rank_ahp, rank_ahp_pca, cfg)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AirfoilSelection pipeline orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    scrape_group = parser.add_mutually_exclusive_group()
    scrape_group.add_argument(
        "--force-scrape", action="store_true",
        help="Always re-scrape; ignore any cached parquet.",
    )
    scrape_group.add_argument(
        "--skip-scrape", action="store_true",
        help="Always use cached parquet; error if not found.",
    )
    parser.add_argument(
        "--steps", nargs="+", type=int, choices=range(1, 6),
        metavar="N",
        help=(
            "Run only specific steps (1–5).  Earlier step outputs must "
            "already exist as parquet files when skipping steps."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args  = parse_args()
    cfg   = load_config(args.config)
    steps = set(args.steps) if args.steps else {1, 2, 3, 4, 5}

    print("\n" + "═" * 60)
    print("  ✈   AirfoilSelection Pipeline")
    print("═" * 60 + "\n")

    results_dir = cfg["data"]["results_dir"]

    # ── Step 1: Scrape ──────────────────────────────────────────────────────
    if 1 in steps:
        df_raw = step1_scrape(cfg, force=args.force_scrape, skip=args.skip_scrape)
    else:
        raw_path = cfg["data"]["raw_polars_path"]
        if not _parquet_exists(raw_path):
            logger.error("Step 1 skipped but %s not found.  Run step 1 first.", raw_path)
            sys.exit(1)
        df_raw = pd.read_parquet(raw_path)
        logger.info("Step 1 skipped — loaded %s", raw_path)

    # ── Step 2: Analysis ────────────────────────────────────────────────────
    if 2 in steps:
        df_feat, df_failed = step2_analysis(df_raw, cfg)
        if not df_failed.empty:
            os.makedirs(results_dir, exist_ok=True)
            failed_path = f"{results_dir}/failed_airfoils.parquet"
            df_failed.to_parquet(failed_path, index=False)
            logger.info(
                "%d airfoils removed — saved to %s", len(df_failed), failed_path
            )
    else:
        feat_path = cfg["data"]["features_path"]
        if not _parquet_exists(feat_path):
            logger.error("Step 2 skipped but %s not found.", feat_path)
            sys.exit(1)
        df_feat = pd.read_parquet(feat_path)
        logger.info("Step 2 skipped — loaded %s", feat_path)

    # ── Step 3: Features + geometry filter ──────────────────────────────────
    if 3 in steps:
        df_features = step3_features(df_feat, df_raw, cfg)
    else:
        feat_path = cfg["data"]["features_path"]
        if not _parquet_exists(feat_path):
            logger.error("Step 3 skipped but %s not found.", feat_path)
            sys.exit(1)
        df_features = pd.read_parquet(feat_path)
        logger.info("Step 3 skipped — loaded %s", feat_path)

    # ── Steps 4a-c: Rankings ────────────────────────────────────────────────
    if 4 in steps:
        rank_pca, rank_ahp, rank_ahp_pca = step4_rankings(df_features, cfg)
    else:
        for fname in ("ranking_pca.parquet", "ranking_ahp.parquet", "ranking_ahp_pca.parquet"):
            path = f"{results_dir}/{fname}"
            if not _parquet_exists(path):
                logger.error("Step 4 skipped but %s not found.", path)
                sys.exit(1)
        rank_pca     = pd.read_parquet(f"{results_dir}/ranking_pca.parquet")
        rank_ahp     = pd.read_parquet(f"{results_dir}/ranking_ahp.parquet")
        rank_ahp_pca = pd.read_parquet(f"{results_dir}/ranking_ahp_pca.parquet")
        logger.info("Step 4 skipped — loaded existing ranking parquets.")

    # ── Step 5: PrOPPAGA ────────────────────────────────────────────────────
    if 5 in steps:
        df_final = step5_proppaga(df_features, rank_pca, rank_ahp, rank_ahp_pca, cfg)

        print("\n" + "═" * 60)
        print("  ✅  Pipeline complete!")
        print(f"  Final PrOPPAGA ranking ({len(df_final)} candidates):")
        print("─" * 60)
        print(
            df_final[["rank_proppaga", "name", "score_proppaga", "source_methods"]]
            .to_string(index=False)
        )
        print("═" * 60 + "\n")
    else:
        print("\n✅  Requested steps complete.\n")


if __name__ == "__main__":
    main()