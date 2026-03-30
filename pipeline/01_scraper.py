# -*- coding: utf-8 -*-
"""
pipeline/01_scraper.py
======================
Step 1 — Scrape aerodynamic polar data from airfoiltools.com.

For every airfoil listed on the search page the scraper downloads:
  • Polar curves (alpha, Cl, Cd, Cdp, Cm, Top_Xtr, Bot_Xtr) for each
    Reynolds number defined in config.yaml.
  • Geometry metadata (max thickness, max camber and their chord positions)
    from the airfoil detail page.

Resilience features
-------------------
Retry with exponential backoff
    Every HTTP request is retried up to ``scraper.max_retries`` times
    (default 5) on transient errors: timeouts, connection resets, and
    5xx server errors.  Waits between retries grow as 2^attempt seconds
    (2 s, 4 s, 8 s, 16 s, 32 s) to avoid hammering the server.
    404 responses are NOT retried — they mean the polar simply does not
    exist for that Reynolds number.

Incremental checkpoint save
    After each airfoil is fully scraped (geometry + all Reynolds polars)
    the result is immediately appended to the parquet checkpoint file
    ``data/raw_polars.parquet``.  A crash or keyboard interrupt never
    loses more than the single airfoil currently in progress.

Automatic resume
    On startup, if the checkpoint parquet already exists, the scraper
    reads which airfoil names are already saved and skips them.  A run
    interrupted at airfoil 800/1638 will resume from 801 automatically.

Output
------
data/raw_polars.parquet
    One row per (airfoil × Reynolds number).  Polar curves are stored as
    Python lists inside each cell (supported by Parquet via pyarrow).

Columns
-------
name            str     Airfoil identifier (e.g. "naca2412-il")
desc            str     Human-readable description
reynolds        str     Reynolds number string ("200000" / "500000" / "1000000")
max_cl_cd       float   Max Cl/Cd ratio from header
max_cl_cd_alpha float   Angle of attack at max Cl/Cd
alpha           list    Alpha series
cl              list    Lift coefficient series
cd              list    Drag coefficient series
cdp             list    Pressure drag coefficient series
cm              list    Pitching moment coefficient series
top_xtr         list    Top transition location series
bot_xtr         list    Bottom transition location series
thickness_value float   Max thickness (% chord)
thickness_chord float   Chord position of max thickness (% chord)
camber_value    float   Max camber (% chord)
camber_chord    float   Chord position of max camber (% chord)
"""

import os
import re
import time
import logging
from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# HTTP status codes that are worth retrying (server-side / transient)
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _delete_first_n_lines(text: str, n: int) -> str:
    """Drop the first *n* lines from a multi-line string."""
    return "".join(text.splitlines(True)[n:])


def _request_with_retry(
    url: str,
    headers: dict,
    timeout: int,
    max_retries: int,
    allow_404: bool = False,
) -> requests.Response | None:
    """
    GET *url* with exponential-backoff retries on transient failures.

    Parameters
    ----------
    url         : URL to fetch.
    headers     : HTTP headers dict.
    timeout     : Per-request timeout in seconds.
    max_retries : Maximum number of retry attempts after the first failure.
    allow_404   : If True, return None silently on 404 instead of warning.

    Returns
    -------
    requests.Response on success, or None if the resource does not exist
    (404) or all retries are exhausted.
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)

            # 404 → resource genuinely missing, no point retrying
            if response.status_code == 404:
                if not allow_404:
                    logger.warning("Not found (404): %s", url)
                return None

            # Retryable server error
            if response.status_code in _RETRYABLE_STATUSES:
                if attempt < max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "HTTP %d for %s — retrying in %ds (attempt %d/%d)",
                        response.status_code, url, wait, attempt + 1, max_retries,
                    )
                    time.sleep(wait)
                    continue
                else:
                    logger.error("HTTP %d for %s — max retries reached.", response.status_code, url)
                    return None

            # Any other non-200 that is not worth retrying
            if response.status_code != 200:
                logger.warning(
                    "Polar not found: %s (status %s)", url, response.status_code
                )
                return None

            return response  # success

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning(
                    "Connection error on %s: %s — retrying in %ds (attempt %d/%d)",
                    url, exc, wait, attempt + 1, max_retries,
                )
                time.sleep(wait)
            # else: fall through to the give-up log below

    logger.error(
        "Giving up on %s after %d attempts. Last error: %s",
        url, max_retries + 1, last_exc,
    )
    return None


def _get_airfoil_list(cfg: dict) -> pd.DataFrame:
    """
    Fetch the airfoil search page and return a DataFrame with columns
    ['Name', 'Link_Detail', 'Link_Polar', 'airfoil_id'].
    """
    url        = cfg["scraper"]["base_url"]
    headers    = cfg["scraper"]["headers"]
    timeout    = cfg["scraper"]["request_timeout"]
    max_retries= cfg["scraper"].get("max_retries", 5)

    response = _request_with_retry(url, headers, timeout, max_retries)
    if response is None:
        raise RuntimeError(f"Could not fetch airfoil list from {url}")

    soup   = BeautifulSoup(response.text, "html.parser")
    tables = soup.find_all("table")
    links  = tables[1].find_all("a")

    rows = []
    for link in links:
        name = link.get_text(strip=True)
        href = link.get("href", "")
        if name and href:
            rows.append({"Name": name, "Link_Detail": href})

    df = pd.DataFrame(rows)

    # Link_Detail is a relative path like /airfoil/details?airfoil=ag19-il
    # Extract the bare airfoil id and build URLs cleanly.
    polar_base = cfg["scraper"]["polar_base_url"]

    df["airfoil_id"] = df["Link_Detail"].str.extract(r"airfoil=(.+)$", expand=False)
    df["Link_Polar"]  = polar_base + df["airfoil_id"]   # Reynolds suffix appended later
    df["Link_Detail"] = "http://airfoiltools.com" + df["Link_Detail"]
    return df


def _fetch_polar(
    url: str,
    headers: dict,
    timeout: int,
    max_retries: int,
) -> dict | None:
    """
    Download a single polar CSV and return parsed data, or None on failure.
    """
    response = _request_with_retry(
        url, headers, timeout, max_retries, allow_404=True
    )
    if response is None:
        return None

    csv_file  = StringIO(response.text)
    df_header = pd.read_csv(csv_file, nrows=9, sep=",", on_bad_lines="skip", index_col=0)

    csv_file = StringIO(_delete_first_n_lines(response.text, 10))
    df_data  = pd.read_csv(csv_file, on_bad_lines="skip")

    required = {"Alpha", "Cl", "Cd", "Cdp", "Cm", "Top_Xtr", "Bot_Xtr"}
    if not required.issubset(df_data.columns):
        logger.warning("Missing columns in polar at %s", url)
        return None

    def _to_float_list(series: pd.Series) -> list:
        """
        Coerce a CSV column to a plain Python list of floats.
        Some airfoils ship values as quoted strings (e.g. '-0.2548');
        pd.to_numeric handles those cleanly so pyarrow does not raise
        ArrowInvalid when writing the checkpoint parquet.
        """
        return pd.to_numeric(series, errors="coerce").tolist()

    return {
        "max_cl_cd":        float(df_header.loc["Max Cl/Cd"].item()),
        "max_cl_cd_alpha":  float(df_header.loc["Max Cl/Cd alpha"].item()),
        "alpha":   _to_float_list(df_data["Alpha"]),
        "cl":      _to_float_list(df_data["Cl"]),
        "cd":      _to_float_list(df_data["Cd"]),
        "cdp":     _to_float_list(df_data["Cdp"]),
        "cm":      _to_float_list(df_data["Cm"]),
        "top_xtr": _to_float_list(df_data["Top_Xtr"]),
        "bot_xtr": _to_float_list(df_data["Bot_Xtr"]),
    }


def _fetch_geometry(
    url: str,
    headers: dict,
    timeout: int,
    max_retries: int,
) -> dict:
    """
    Scrape the airfoil detail page for thickness and camber.
    Returns a dict with keys: thickness_value, thickness_chord,
    camber_value, camber_chord.  Values are None when not found.
    """
    response = _request_with_retry(url, headers, timeout, max_retries)
    if response is None:
        return {k: None for k in
                ["thickness_value", "thickness_chord",
                 "camber_value",    "camber_chord"]}

    soup = BeautifulSoup(response.text, "html.parser")
    text = " ".join(soup.get_text(separator="\n", strip=True).split("\n"))

    thickness = re.search(r"Max thickness ([\d\.]+)% at ([\d\.]+)% chord", text)
    camber    = re.search(r"Max camber ([\d\.]+)% at ([\d\.]+)% chord", text)

    return {
        "thickness_value": float(thickness.group(1)) if thickness else None,
        "thickness_chord": float(thickness.group(2)) if thickness else None,
        "camber_value":    float(camber.group(1))    if camber    else None,
        "camber_chord":    float(camber.group(2))    if camber    else None,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(path: str) -> tuple[pd.DataFrame, set[str]]:
    """
    Load an existing checkpoint parquet if present.

    Returns
    -------
    df_existing : DataFrame of already-scraped rows (empty if no checkpoint).
    done_names  : Set of airfoil names already fully saved.
    """
    if os.path.isfile(path):
        df_existing = pd.read_parquet(path)
        done_names  = set(df_existing["name"].unique())
        logger.info(
            "Checkpoint found: %d rows, %d airfoils already scraped — resuming.",
            len(df_existing), len(done_names),
        )
        return df_existing, done_names

    return pd.DataFrame(), set()


def _append_checkpoint(new_rows: list[dict], path: str, existing: pd.DataFrame) -> pd.DataFrame:
    """
    Append *new_rows* to the checkpoint parquet atomically.

    Writes to a temp file first, then replaces the original, so a crash
    mid-write never corrupts the checkpoint.

    Returns the updated combined DataFrame.
    """
    df_new      = pd.DataFrame(new_rows)
    df_combined = pd.concat([existing, df_new], ignore_index=True)

    tmp_path = path + ".tmp"
    df_combined.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)   # atomic on all major OS

    return df_combined


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(cfg: dict) -> pd.DataFrame:
    """
    Scrape all airfoils and return a tidy DataFrame.

    Automatically resumes from a partial checkpoint if one exists.
    Retries transient network errors with exponential backoff.
    Saves progress after every airfoil so interruptions lose minimal work.

    Parameters
    ----------
    cfg : dict
        Full configuration loaded from config.yaml.

    Returns
    -------
    pd.DataFrame
        Tidy polar data — one row per (airfoil × Reynolds number).
        Saved incrementally to ``cfg['data']['raw_polars_path']``.
    """
    headers      = cfg["scraper"]["headers"]
    timeout      = cfg["scraper"]["request_timeout"]
    reynolds_list= cfg["scraper"]["reynolds_numbers"]
    max_retries  = cfg["scraper"].get("max_retries", 5)
    output_path  = cfg["data"]["raw_polars_path"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- Resume support ---
    df_existing, done_names = _load_checkpoint(output_path)

    logger.info("Fetching airfoil list from %s", cfg["scraper"]["base_url"])
    df_urls = _get_airfoil_list(cfg)
    total   = len(df_urls)

    remaining = df_urls[~df_urls["airfoil_id"].isin(done_names)]
    skipped   = total - len(remaining)
    if skipped:
        logger.info(
            "Skipping %d already-scraped airfoils. %d remaining.",
            skipped, len(remaining),
        )

    scraped_count = skipped

    for _, row in remaining.iterrows():
        airfoil_id   = row["airfoil_id"]
        name_parts   = row["Name"].split(" - ", 1)
        desc         = name_parts[1] if len(name_parts) > 1 else ""
        scraped_count += 1

        logger.info("[%d/%d] Scraping %s", scraped_count, total, airfoil_id)

        # --- Geometry (once per airfoil) ---
        geom = _fetch_geometry(row["Link_Detail"], headers, timeout, max_retries)
        if any(v is None for v in geom.values()):
            logger.warning("Geometry incomplete for %s — will still attempt polars.", airfoil_id)

        # --- Polars (once per Reynolds number) ---
        new_rows = []
        for re_str in reynolds_list:
            polar_url = row["Link_Polar"] + "-" + re_str
            polar     = _fetch_polar(polar_url, headers, timeout, max_retries)
            if polar is None:
                continue

            new_rows.append({
                "name":    airfoil_id,
                "desc":    desc,
                "reynolds": re_str,
                **polar,
                **geom,
            })

        # --- Checkpoint: save this airfoil even if it had zero polars ---
        # (zero-polar airfoils will be dropped in step 2 validation anyway,
        #  but recording them prevents re-fetching them on resume)
        if new_rows:
            df_existing = _append_checkpoint(new_rows, output_path, df_existing)
        else:
            # Write a sentinel row so this airfoil is marked done in the
            # checkpoint and won't be re-fetched on resume.
            sentinel = [{"name": airfoil_id, "desc": desc, "reynolds": None}]
            df_existing = _append_checkpoint(sentinel, output_path, df_existing)
            logger.debug("No polars found for %s — sentinel written.", airfoil_id)

    # Drop sentinel rows (reynolds=None) before returning
    df_final = df_existing[df_existing["reynolds"].notna()].reset_index(drop=True)

    # Overwrite parquet with clean version (no sentinels)
    df_final.to_parquet(output_path, index=False)
    logger.info(
        "Scrape complete. %d rows for %d airfoils saved to %s",
        len(df_final),
        df_final["name"].nunique(),
        output_path,
    )

    return df_final