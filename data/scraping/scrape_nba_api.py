"""
NBA Stats API scraper using the nba_api library.

Pulls: player bio, season-level stats, play-type breakdowns.
All responses cached to data/raw/nba_api/ as JSON.
Resume-safe: file existence check before every call.

The nba_api library already imposes a default 1-second delay between
requests; we add our own jitter on top to be conservative.
"""

from __future__ import annotations

import json
import time
import random
from pathlib import Path

from nba_api.stats.endpoints import (
    CommonAllPlayers,
    PlayerCareerStats,
    PlayerDashboardByGameSplits,
)
from nba_api.stats.static import players as nba_players_static

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "nba_api"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MIN_DELAY = 1.0
MAX_DELAY = 3.0


def _jitter() -> None:
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))


def _cache_path(name: str) -> Path:
    return RAW_DIR / f"{name}.json"


def fetch_all_players(season: str = "2023-24") -> list[dict]:
    out = _cache_path(f"all_players_{season.replace('-', '_')}")
    if out.exists():
        return json.loads(out.read_text())
    _jitter()
    df = CommonAllPlayers(
        is_only_current_season=0,
        league_id="00",
        season=season,
        timeout=30,
    ).get_data_frames()[0]
    records = df.to_dict("records")
    out.write_text(json.dumps(records, indent=2))
    print(f"[nba_api] fetched {len(records)} players → {out.name}")
    return records


def fetch_player_career_stats(player_id: int) -> dict | None:
    out = _cache_path(f"career_{player_id}")
    if out.exists():
        return json.loads(out.read_text())
    _jitter()
    try:
        frames = PlayerCareerStats(player_id=player_id, timeout=30).get_data_frames()
        data = {f"frame_{i}": df.to_dict("records") for i, df in enumerate(frames)}
        out.write_text(json.dumps(data, indent=2))
        return data
    except Exception as exc:
        print(f"  [warn] player_id={player_id}: {exc}")
        return None


def fetch_player_game_splits(player_id: int, season: str) -> dict | None:
    season_slug = season.replace("-", "_")
    out = _cache_path(f"splits_{player_id}_{season_slug}")
    if out.exists():
        return json.loads(out.read_text())
    _jitter()
    try:
        frames = PlayerDashboardByGameSplits(
            player_id=player_id,
            season=season,
            timeout=30,
        ).get_data_frames()
        data = {f"frame_{i}": df.to_dict("records") for i, df in enumerate(frames)}
        out.write_text(json.dumps(data, indent=2))
        return data
    except Exception as exc:
        print(f"  [warn] splits player_id={player_id} season={season}: {exc}")
        return None


def run_full_scrape(min_season: int = 1990) -> None:
    import datetime
    current_year = datetime.date.today().year
    # season strings: "1990-91" through current
    seasons = [
        f"{y}-{str(y + 1)[-2:]}" for y in range(min_season, current_year)
    ]

    for season in seasons:
        print(f"\n[season] {season}")
        players = fetch_all_players(season=season)
        for p in players:
            pid = p.get("PERSON_ID")
            if not pid:
                continue
            fetch_player_career_stats(pid)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-season", type=int, default=1996)
    args = parser.parse_args()
    run_full_scrape(min_season=args.min_season)
