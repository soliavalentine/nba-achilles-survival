#!/usr/bin/env python3
"""
Build a minimal smoke-test feature_matrix.csv for training DeepHit.

Uses nba_api (already a project dependency) instead of basketball-reference-scraper,
which gives cleaner structured data without HTML parsing fragility. 3-second delays
between every API call respect the NBA stats endpoint rate limit.

For each confirmed Achilles rupture in achilles_ground_truth.csv (severity == "rupture"),
adds CONTROLS_PER_EVENT same-position players from the same NBA season who did not rupture.

Approximate runtime: 5–15 minutes.

Usage:
    python data/scraping/build_minimal_feature_matrix.py
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import commonplayerinfo, leaguedashplayerstats
from nba_api.stats.static import players as nba_static

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
ACHILLES_CSV = PROCESSED_DIR / "achilles_ground_truth.csv"
OUTPUT_CSV = PROCESSED_DIR / "feature_matrix.csv"

DELAY = 3.0          # seconds between API calls
CONTROLS_PER_EVENT = 3
RNG_SEED = 42

# nba_api only has reliable season data from 1996-97 onward.
# Dominique Wilkins (1992) will be skipped with a warning.
MIN_SEASON_YEAR = 1997


def get_season_year(date: pd.Timestamp) -> int:
    """Return the ending calendar year of the NBA season containing date."""
    return date.year + 1 if date.month >= 10 else date.year


def season_str(year: int) -> str:
    return f"{year - 1}-{str(year)[2:]}"


def encode_position(pos: str) -> int:
    """Broad position bucket: Guard=0, Forward=1, Center=2."""
    p = str(pos).strip().lower()
    if p.startswith("center"):
        return 2
    if p.startswith("guard"):
        return 0
    return 1  # forward / forward-* / empty → forward bucket


def parse_height(h: str) -> float:
    """Convert '6-7' → 79.0 inches. Returns 78.0 on parse failure."""
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


# ── Cached API helpers ────────────────────────────────────────────────────────

_bio_cache: dict[int, dict] = {}
# cache key: (season, pos_abbr)
_season_cache: dict[tuple, pd.DataFrame] = {}

# Maps position_encoded (0/1/2) → nba_api position abbreviation filter
_POS_ABBR = {0: "G", 1: "F", 2: "C"}


def get_bio(player_id: int) -> dict | None:
    if player_id in _bio_cache:
        return _bio_cache[player_id]
    time.sleep(DELAY)
    try:
        row = commonplayerinfo.CommonPlayerInfo(
            player_id=player_id
        ).get_data_frames()[0].iloc[0]

        draft_yr_raw = str(row.get("DRAFT_YEAR", ""))
        draft_year = int(draft_yr_raw) if draft_yr_raw.isdigit() else None

        bio = {
            "player_id": player_id,
            "height_inches": parse_height(row.get("HEIGHT", "6-6")),
            "weight_lbs": parse_weight(row.get("WEIGHT", 220)),
            "birth_date": pd.to_datetime(row.get("BIRTHDATE"), errors="coerce"),
            "position": str(row.get("POSITION", "")),
            "position_encoded": encode_position(str(row.get("POSITION", ""))),
            "draft_year": draft_year,
        }
        _bio_cache[player_id] = bio
        return bio
    except Exception as e:
        print(f"    bio fetch failed for player_id={player_id}: {e}")
        return None


def get_season_players(season: str, pos_abbr: str = "") -> pd.DataFrame:
    """Return PLAYER_ID + PLAYER_NAME for a season, optionally filtered by position."""
    key = (season, pos_abbr)
    if key in _season_cache:
        return _season_cache[key]
    time.sleep(DELAY)
    try:
        df = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed="PerGame",
            player_position_abbreviation_nullable=pos_abbr,
        ).get_data_frames()[0][["PLAYER_ID", "PLAYER_NAME"]]
        _season_cache[key] = df
        return df
    except Exception as e:
        print(f"    season player fetch failed for {season} pos={pos_abbr!r}: {e}")
        return pd.DataFrame(columns=["PLAYER_ID", "PLAYER_NAME"])


# ── Row builder ───────────────────────────────────────────────────────────────

def make_row(
    player_id: int,
    player_name: str,
    bio: dict,
    event: int,
    obs_date: pd.Timestamp,
    season_year: int,
) -> dict:
    season_start = pd.Timestamp(f"{season_year - 1}-10-01")
    age = (
        (obs_date - bio["birth_date"]).days / 365.25
        if pd.notna(bio["birth_date"])
        else np.nan
    )
    draft_yr = bio.get("draft_year")
    years = (season_year - draft_yr) if draft_yr else np.nan
    return {
        "player_id": player_id,
        "player_name": player_name,
        "observation_date": obs_date.date(),
        "event": event,
        "time_to_event_days": max((obs_date - season_start).days, 0),
        "age_at_observation": round(age, 2) if not np.isnan(age) else np.nan,
        "position_encoded": bio["position_encoded"],
        "height_inches": bio["height_inches"],
        "weight_lbs": bio["weight_lbs"],
        "years_in_league": int(years) if (draft_yr and not np.isnan(years)) else np.nan,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    rng = random.Random(RNG_SEED)

    print("[build_minimal] loading ground truth…")
    gt = pd.read_csv(ACHILLES_CSV, parse_dates=["date"])
    gt["player_name"] = gt["player_name"].str.lstrip("•").str.strip()

    # Clean ProSports slash-separated name variants: "Jeff Taylor / Jeffery Taylor (b)"
    # Take the shortest token (usually the common NBA name) before any slash.
    def clean_name(name: str) -> str:
        if "/" in name:
            candidates = [s.strip() for s in name.split("/")]
            # Strip parenthetical suffixes like "(b)" and pick shortest clean name
            candidates = [re.sub(r"\s*\(.*?\)\s*$", "", c).strip() for c in candidates]
            return min(candidates, key=len)
        # Strip trailing parentheticals: "John Wall (Hildred)" → "John Wall"
        return re.sub(r"\s*\(.*?\)\s*$", "", name).strip()

    import re
    gt["player_name"] = gt["player_name"].apply(clean_name)

    # Deduplicate: ProSports logs multiple follow-up IL entries per injury event
    # ("recovering from surgery to repair..."). Keep only FIRST event date per player.
    ruptures_all = gt[gt["severity"] == "rupture"].copy()
    ruptures = (
        ruptures_all.sort_values("date")
        .groupby("player_name", sort=False)
        .first()
        .reset_index()
    )
    print(f"  raw rupture rows  : {len(ruptures_all)}")
    print(f"  unique rupture players : {len(ruptures)}")

    ruptures["season_year"] = ruptures["date"].apply(get_season_year)
    ruptures["season"] = ruptures["season_year"].apply(season_str)

    # Warn about pre-nba_api era events but still attempt them
    old = ruptures[ruptures["season_year"] < MIN_SEASON_YEAR]
    if not old.empty:
        print(
            f"  WARNING: {list(old['player_name'])} are pre-{MIN_SEASON_YEAR} — "
            "nba_api may not have season data; will attempt bio only"
        )

    # Build name→id lookup from nba_api static player list
    all_players = nba_static.get_players()
    name_to_id: dict[str, int] = {p["full_name"].lower(): p["id"] for p in all_players}

    def find_id(name: str) -> int | None:
        key = name.lower()
        if key in name_to_id:
            return name_to_id[key]
        # Partial containment match
        hits = [pid for n, pid in name_to_id.items() if key in n or n in key]
        if len(hits) == 1:
            return hits[0]
        if hits:
            print(f"    ambiguous match for '{name}': {len(hits)} candidates, using first")
            return hits[0]
        # Last resort: match on last name + first initial
        parts = key.split()
        if len(parts) >= 2:
            last = parts[-1]
            initial = parts[0][0]
            hits = [pid for n, pid in name_to_id.items()
                    if last in n and n.startswith(initial)]
            if len(hits) == 1:
                return hits[0]
        return None

    rows: list[dict] = []
    rupture_ids: set[int] = set()

    # ── Pass 1: rupture player bios ──────────────────────────────────────────
    print("\n[build_minimal] fetching rupture player bios…")
    resolved: list[tuple] = []  # (name, pid, bio, rupture_row)

    for _, r in ruptures.iterrows():
        name = r["player_name"]
        pid = find_id(name)
        if pid is None:
            print(f"  SKIP: no nba_api match for '{name}'")
            continue

        bio = get_bio(pid)
        if bio is None:
            print(f"  SKIP: bio unavailable for '{name}' (id={pid})")
            continue

        season_year = int(r["season_year"])
        obs_date = r["date"]
        row = make_row(pid, name, bio, event=1, obs_date=obs_date, season_year=season_year)
        rows.append(row)
        rupture_ids.add(pid)
        resolved.append((name, pid, bio, r))
        print(
            f"  + rupture  {name:<22} age={row['age_at_observation']:.1f}"
            f"  pos={bio['position']:<16}  {r['season']}"
        )

    # ── Pass 2: controls ─────────────────────────────────────────────────────
    print(f"\n[build_minimal] fetching controls ({CONTROLS_PER_EVENT} per rupture)…")

    for name, pid, bio, r in resolved:
        season = r["season"]
        season_year = int(r["season_year"])
        target_pos = bio["position_encoded"]
        season_end = pd.Timestamp(f"{season_year}-06-15")

        pos_abbr = _POS_ABBR.get(target_pos, "")
        season_df = get_season_players(season, pos_abbr=pos_abbr)
        if season_df.empty:
            print(f"  WARNING: no season data for {season}, skipping controls for {name}")
            continue

        candidates = season_df[
            ~season_df["PLAYER_ID"].isin(rupture_ids)
        ].copy()
        candidate_ids = candidates["PLAYER_ID"].tolist()
        rng.shuffle(candidate_ids)

        # Only bio-fetch the players we actually select — much fewer API calls
        matched = 0
        for ctrl_id in candidate_ids:
            if matched >= CONTROLS_PER_EVENT:
                break
            ctrl_bio = get_bio(int(ctrl_id))
            if ctrl_bio is None:
                continue

            ctrl_name = candidates.loc[
                candidates["PLAYER_ID"] == ctrl_id, "PLAYER_NAME"
            ].iloc[0]
            row = make_row(
                int(ctrl_id), ctrl_name, ctrl_bio,
                event=0, obs_date=season_end, season_year=season_year,
            )
            rows.append(row)
            matched += 1
            print(f"  + control  {ctrl_name:<22} pos={ctrl_bio['position']:<16}  ← {name}")

        if matched < CONTROLS_PER_EVENT:
            print(
                f"  WARNING: only {matched}/{CONTROLS_PER_EVENT} controls found for {name}"
            )

    # ── Save ─────────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    feature_cols = [
        c for c in df.columns
        if c not in {"player_id", "player_name", "observation_date",
                     "event", "time_to_event_days"}
    ]

    print(f"\n{'=' * 60}")
    print(f"  Saved → {OUTPUT_CSV}")
    print(f"  Total rows    : {len(df)}")
    print(f"  Ruptures      : {int((df['event'] == 1).sum())}")
    print(f"  Censored      : {int((df['event'] == 0).sum())}")
    print(f"  Feature cols  : {feature_cols}")
    print(f"  Shape         : {df.shape}")
    print("=" * 60)
    print("\nNext step: python -m models.train --no-hpo --epochs 20")


if __name__ == "__main__":
    main()
