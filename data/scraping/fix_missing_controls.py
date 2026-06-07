#!/usr/bin/env python3
"""
fix_missing_controls.py

Adds matched controls for the rupture players in feature_matrix.csv
that currently have zero controls in their season.

Root cause of the original failures:
  - leaguedashplayerstats with player_position_abbreviation_nullable
    returns empty JSON for many NBA seasons (silent empty response).
  - 30s timeout was too short for some seasons.

Fix:
  - Fetch the full season player list (no position filter).
  - Filter by position in Python using commonplayerinfo.
  - 60s timeout, 3 retries with 5s back-off per call.
  - Disk-cached bio and season results so re-runs skip already-fetched data.

After appending controls this script runs the ACWR pipeline automatically.

Usage:
    python data/scraping/fix_missing_controls.py
"""

from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import commonplayerinfo, leaguedashplayerstats
from nba_api.stats.static import players as nba_static

ROOT        = Path(__file__).resolve().parents[2]
PROCESSED   = ROOT / "data" / "processed"
RAW         = ROOT / "data" / "raw"
FM_CSV      = PROCESSED / "feature_matrix.csv"
GT_CSV      = PROCESSED / "achilles_ground_truth.csv"
BIO_CACHE   = RAW / "nba_api_bio_cache.json"
SEA_CACHE   = RAW / "nba_api_season_cache.json"

DELAY             = 3.0    # seconds between API calls
TIMEOUT           = 60     # nba_api request timeout in seconds
MAX_RETRIES       = 3
RETRY_BACKOFF     = 5.0
CONTROLS_PER_EVENT = 3
RNG_SEED          = 42


# ── Disk-cached bio store ─────────────────────────────────────────────────────

def _load_bio_cache() -> dict:
    if BIO_CACHE.exists():
        return json.loads(BIO_CACHE.read_text())
    return {}


def _save_bio_cache(cache: dict) -> None:
    BIO_CACHE.write_text(json.dumps(cache))


def _load_season_cache() -> dict:
    if SEA_CACHE.exists():
        return json.loads(SEA_CACHE.read_text())
    return {}


def _save_season_cache(cache: dict) -> None:
    SEA_CACHE.write_text(json.dumps(cache))


# ── Helpers ───────────────────────────────────────────────────────────────────

def season_str(year: int) -> str:
    return f"{year - 1}-{str(year)[2:]}"


def get_season_year(date: pd.Timestamp) -> int:
    return date.year + 1 if date.month >= 10 else date.year


def encode_position(pos: str) -> int:
    p = str(pos).strip().lower()
    if p.startswith("center"):
        return 2
    if p.startswith("guard"):
        return 0
    return 1


def parse_height(h: str) -> float:
    try:
        feet, inches = str(h).split("-")
        return int(feet) * 12 + int(inches)
    except Exception:
        return 78.0


def parse_weight(w) -> float:
    try:
        return float(str(w).replace("lbs", "").strip())
    except Exception:
        return 220.0


# ── API wrappers with retry + disk cache ──────────────────────────────────────

def get_bio(player_id: int, bio_cache: dict) -> dict | None:
    key = str(player_id)
    if key in bio_cache:
        return bio_cache[key]

    time.sleep(DELAY)
    for attempt in range(MAX_RETRIES):
        try:
            row = commonplayerinfo.CommonPlayerInfo(
                player_id=player_id, timeout=TIMEOUT
            ).get_data_frames()[0].iloc[0]

            draft_yr_raw = str(row.get("DRAFT_YEAR", ""))
            draft_year = int(draft_yr_raw) if draft_yr_raw.isdigit() else None

            bio = {
                "player_id":       player_id,
                "height_inches":   parse_height(row.get("HEIGHT", "6-6")),
                "weight_lbs":      parse_weight(row.get("WEIGHT", 220)),
                "birth_date":      str(row.get("BIRTHDATE", "")),
                "position":        str(row.get("POSITION", "")),
                "position_encoded": encode_position(str(row.get("POSITION", ""))),
                "draft_year":      draft_year,
            }
            bio_cache[key] = bio
            return bio

        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF)
            else:
                print(f"    bio failed player_id={player_id}: {str(exc)[:80]}")
                return None

    return None


def get_season_players(season_year: int, sea_cache: dict) -> pd.DataFrame:
    """
    Fetch all players who appeared in a season.
    No position filter — player_position_abbreviation_nullable returns empty
    JSON for many seasons; filter by position in Python instead.
    """
    key = str(season_year)
    if key in sea_cache:
        return pd.DataFrame(sea_cache[key])

    season = season_str(season_year)
    time.sleep(DELAY)

    for attempt in range(MAX_RETRIES):
        try:
            df = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                per_mode_detailed="PerGame",
                timeout=TIMEOUT,
            ).get_data_frames()[0][["PLAYER_ID", "PLAYER_NAME"]]

            sea_cache[key] = df.to_dict("records")
            print(f"  [api] {season}: {len(df)} players")
            return df

        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF)
            else:
                print(f"  [api] season {season} failed: {str(exc)[:80]}")
                return pd.DataFrame(columns=["PLAYER_ID", "PLAYER_NAME"])

    return pd.DataFrame(columns=["PLAYER_ID", "PLAYER_NAME"])


# ── Row builder ───────────────────────────────────────────────────────────────

def make_control_row(
    player_id: int,
    player_name: str,
    bio: dict,
    season_year: int,
) -> dict:
    season_start = pd.Timestamp(f"{season_year - 1}-10-01")
    season_end   = pd.Timestamp(f"{season_year}-06-15")

    birth_date = pd.to_datetime(bio.get("birth_date", ""), errors="coerce")
    age = (season_end - birth_date).days / 365.25 if pd.notna(birth_date) else np.nan

    draft_yr = bio.get("draft_year")
    years = (season_year - draft_yr) if draft_yr else np.nan

    return {
        "player_id":          player_id,
        "player_name":        player_name,
        "observation_date":   season_end.date(),
        "event":              0,
        "time_to_event_days": (season_end - season_start).days,
        "age_at_observation": round(age, 2) if not np.isnan(age) else np.nan,
        "position_encoded":   bio["position_encoded"],
        "height_inches":      bio["height_inches"],
        "weight_lbs":         bio["weight_lbs"],
        "years_in_league":    int(years) if (draft_yr and not np.isnan(years)) else np.nan,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    rng = random.Random(RNG_SEED)

    # ── 1. Identify rupture players missing controls ──────────────────────────
    fm = pd.read_csv(FM_CSV, parse_dates=["observation_date"])
    fm["season_year"] = fm["observation_date"].apply(
        lambda d: d.year + 1 if d.month >= 10 else d.year
    )
    controls_by_season = fm[fm["event"] == 0].groupby("season_year").size().to_dict()

    gt = pd.read_csv(GT_CSV, parse_dates=["date"])
    gt["player_name"] = gt["player_name"].str.lstrip("•").str.strip()
    gt["player_name"] = gt["player_name"].apply(
        lambda n: min(
            [re.sub(r"\s*\(.*?\)\s*$", "", s.strip()) for s in n.split("/")],
            key=len,
        )
        if "/" in n
        else re.sub(r"\s*\(.*?\)\s*$", "", n).strip()
    )

    ruptures = (
        gt[gt["severity"] == "rupture"]
        .sort_values("date")
        .groupby("player_name")
        .first()
        .reset_index()
    )
    ruptures["season_year"] = ruptures["date"].apply(get_season_year)

    missing = ruptures[
        ruptures["season_year"].map(lambda sy: controls_by_season.get(sy, 0)) == 0
    ].copy()

    print(f"Rupture players with no controls: {len(missing)}")
    print(f"Target: {CONTROLS_PER_EVENT} controls each\n")

    # ── 2. Build nba_api ID lookup ────────────────────────────────────────────
    all_players = nba_static.get_players()
    name_to_id: dict[str, int] = {p["full_name"].lower(): p["id"] for p in all_players}

    def find_id(name: str) -> int | None:
        key = name.lower()
        if key in name_to_id:
            return name_to_id[key]
        hits = [pid for n, pid in name_to_id.items() if key in n or n in key]
        if len(hits) == 1:
            return hits[0]
        if hits:
            return hits[0]
        parts = key.split()
        if len(parts) >= 2:
            last, initial = parts[-1], parts[0][0]
            hits = [pid for n, pid in name_to_id.items()
                    if last in n and n.startswith(initial)]
            if len(hits) == 1:
                return hits[0]
        return None

    # ── 3. Load caches ────────────────────────────────────────────────────────
    bio_cache = _load_bio_cache()
    sea_cache = _load_season_cache()
    print(f"Bio cache: {len(bio_cache)} entries | Season cache: {len(sea_cache)} entries\n")

    # IDs of existing rupture players (exclude from controls)
    rupture_ids: set[int] = set()
    for _, r in ruptures.iterrows():
        pid = find_id(r["player_name"])
        if pid:
            rupture_ids.add(pid)

    # ── 4. Find controls for each missing player ──────────────────────────────
    new_rows: list[dict] = []
    results: dict[str, int] = {}   # player_name → n_controls_found

    for _, rupt in missing.iterrows():
        name       = rupt["player_name"]
        season_year = int(rupt["season_year"])

        # Get rupture player's position from feature_matrix
        fm_row = fm[(fm["player_name"] == name) & (fm["event"] == 1)]
        if fm_row.empty:
            print(f"[skip] {name} not in feature_matrix")
            results[name] = 0
            continue
        target_pos = int(fm_row.iloc[0]["position_encoded"])
        pos_name   = {0: "Guard", 1: "Forward", 2: "Center"}[target_pos]

        print(f"\n{name}  season={season_str(season_year)}  pos={pos_name}")

        season_df = get_season_players(season_year, sea_cache)
        _save_season_cache(sea_cache)

        if season_df.empty:
            print(f"  → no season data, skipping")
            results[name] = 0
            continue

        candidates = season_df[
            ~season_df["PLAYER_ID"].isin(rupture_ids)
        ]["PLAYER_ID"].tolist()
        rng.shuffle(candidates)

        matched = 0
        attempts = 0
        for ctrl_id in candidates:
            if matched >= CONTROLS_PER_EVENT:
                break
            attempts += 1

            bio = get_bio(int(ctrl_id), bio_cache)
            if bio is None:
                continue
            if bio["position_encoded"] != target_pos:
                continue

            ctrl_name = season_df.loc[
                season_df["PLAYER_ID"] == ctrl_id, "PLAYER_NAME"
            ].iloc[0]
            row = make_control_row(int(ctrl_id), ctrl_name, bio, season_year)
            new_rows.append(row)
            matched += 1
            print(f"  + {ctrl_name} ({bio['position']})  [attempt {attempts}]")

        results[name] = matched
        _save_bio_cache(bio_cache)

        if matched < CONTROLS_PER_EVENT:
            print(f"  WARNING: only {matched}/{CONTROLS_PER_EVENT} found "
                  f"after checking {attempts} candidates")

    # ── 5. Append new rows to feature_matrix.csv ─────────────────────────────
    if not new_rows:
        print("\nNo new controls found — nothing to append.")
        return

    df_new = pd.DataFrame(new_rows)
    # Align columns to existing feature_matrix (ACWR cols will be added by pipeline)
    existing_cols = [c for c in fm.columns if c in df_new.columns]
    df_new = df_new[existing_cols]

    df_out = pd.concat([fm.drop(columns=["season_year"]), df_new], ignore_index=True)
    df_out.to_csv(FM_CSV, index=False)

    total_ruptures  = int((df_out["event"] == 1).sum())
    total_controls  = int((df_out["event"] == 0).sum())
    full_matches    = sum(1 for v in results.values() if v >= CONTROLS_PER_EVENT)
    partial_matches = sum(1 for v in results.values() if 0 < v < CONTROLS_PER_EVENT)
    no_matches      = sum(1 for v in results.values() if v == 0)

    print(f"\n{'='*60}")
    print(f"Controls added       : {len(new_rows)}")
    print(f"Full matches (3/3)   : {full_matches}/{len(missing)}")
    print(f"Partial matches      : {partial_matches}/{len(missing)}")
    print(f"No match             : {no_matches}/{len(missing)}")
    print(f"feature_matrix total : {len(df_out)} rows "
          f"({total_ruptures} ruptures, {total_controls} controls)")
    print(f"{'='*60}")

    # ── 6. Re-run ACWR pipeline ───────────────────────────────────────────────
    print("\nRunning ACWR pipeline for new players…")
    result = subprocess.run(
        [sys.executable, "-u", "data/scraping/scrape_bball_reference.py", "acwr"],
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        print("WARNING: ACWR pipeline exited with non-zero code")


if __name__ == "__main__":
    main()
