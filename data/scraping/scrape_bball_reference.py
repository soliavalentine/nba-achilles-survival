"""
Basketball-Reference scraper for NBA Achilles rupture survival analysis.

Scrapes: game logs (1990-present), player bio, injury designations, career stats.
All raw HTML saved to data/raw/bball_reference/ immediately on first fetch.
Resume-safe: checks for cached file before any network request.

IP safety follows the pattern established in nba-star-predictor:
  - 3 s base delay + jitter between every request
  - per-domain rate limit (10 req/min)
  - exponential back-off on 429
  - rotating User-Agent pool
  - session cooldown after 150 requests
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "bball_reference"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.basketball-reference.com"

# ---------------------------------------------------------------------------
# IP safety configuration  (mirrored from nba-star-predictor/scrape_bbref_fixed.py)
# ---------------------------------------------------------------------------

class IPSafeConfig:
    MIN_DELAY = 3.0
    MAX_DELAY = 6.0
    REQUESTS_PER_DOMAIN_PER_MINUTE = 10
    INITIAL_BACKOFF = 15
    MAX_BACKOFF = 300
    BACKOFF_MULTIPLIER = 2
    MAX_REQUESTS_PER_SESSION = 150
    COOLDOWN_AFTER_MAX = 600  # 10-minute cooldown

    USER_AGENTS = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    ]
    TIMEOUT = 30
    MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Request manager (disk-cached, rate-limited)
# ---------------------------------------------------------------------------

class IPSafeRequestManager:
    """Fetches URLs with disk-level caching, rate limiting, and back-off."""

    def __init__(self, cache_dir: Path, config: IPSafeConfig | None = None):
        self.config = config or IPSafeConfig()
        self.cache_dir = cache_dir
        self.session = requests.Session()
        self.domain_requests: dict[str, list[float]] = defaultdict(list)
        self.domain_backoff: dict[str, float] = defaultdict(
            lambda: self.config.INITIAL_BACKOFF
        )
        self.total_requests = 0
        self.session_start = time.time()

    # -- Helpers ----------------------------------------------------------------

    @staticmethod
    def _url_to_cache_path(cache_dir: Path, url: str) -> Path:
        slug = hashlib.md5(url.encode()).hexdigest()
        return cache_dir / f"{slug}.html"

    @staticmethod
    def _domain(url: str) -> str:
        return urlparse(url).netloc

    def _enforce_rate_limit(self, domain: str) -> None:
        now = time.time()
        self.domain_requests[domain] = [
            t for t in self.domain_requests[domain] if now - t < 60
        ]
        if len(self.domain_requests[domain]) >= self.config.REQUESTS_PER_DOMAIN_PER_MINUTE:
            oldest = min(self.domain_requests[domain])
            wait = 60 - (now - oldest)
            if wait > 0:
                print(f"  [rate-limit] sleeping {wait:.1f}s for {domain}")
                time.sleep(wait)

    def _check_session_limit(self) -> None:
        if self.total_requests < self.config.MAX_REQUESTS_PER_SESSION:
            return
        elapsed = time.time() - self.session_start
        remaining = self.config.COOLDOWN_AFTER_MAX - elapsed
        if remaining > 0:
            print(
                f"\n[session-limit] {self.total_requests} requests — "
                f"cooling down {remaining / 60:.1f} min...\n"
            )
            time.sleep(remaining)
        self.total_requests = 0
        self.session_start = time.time()

    # -- Public fetch -----------------------------------------------------------

    def fetch(self, url: str) -> str | None:
        """Return HTML string, using disk cache when available."""
        cache_path = self._url_to_cache_path(self.cache_dir, url)

        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")

        self._check_session_limit()
        domain = self._domain(url)
        self._enforce_rate_limit(domain)

        delay = random.uniform(self.config.MIN_DELAY, self.config.MAX_DELAY)
        print(f"  [fetch] sleeping {delay:.1f}s  →  {url}")
        time.sleep(delay)

        for attempt in range(self.config.MAX_RETRIES + 1):
            try:
                headers = {
                    "User-Agent": random.choice(self.config.USER_AGENTS),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": BASE_URL,
                }
                resp = self.session.get(url, headers=headers, timeout=self.config.TIMEOUT)

                if resp.status_code == 200:
                    html = resp.text
                    cache_path.write_text(html, encoding="utf-8")
                    self.domain_requests[domain].append(time.time())
                    self.domain_backoff[domain] = self.config.INITIAL_BACKOFF
                    self.total_requests += 1
                    print(f"  [ok] #{self.total_requests}  cached → {cache_path.name}")
                    return html

                elif resp.status_code == 404:
                    print(f"  [404] {url}")
                    return None

                elif resp.status_code == 429:
                    backoff = self.domain_backoff[domain]
                    print(f"  [429] back-off {backoff}s")
                    time.sleep(backoff)
                    self.domain_backoff[domain] = min(
                        backoff * self.config.BACKOFF_MULTIPLIER,
                        self.config.MAX_BACKOFF,
                    )

                else:
                    print(f"  [error] HTTP {resp.status_code} for {url}")
                    return None

            except requests.Timeout:
                print(f"  [timeout] attempt {attempt + 1}/{self.config.MAX_RETRIES + 1}")
                if attempt < self.config.MAX_RETRIES:
                    time.sleep(self.config.INITIAL_BACKOFF * (attempt + 1))

            except requests.RequestException as exc:
                print(f"  [request-error] {exc}")
                return None

        return None


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_player_index(html: str) -> list[dict]:
    """Parse /players/<letter>/ index page → list of {name, url, years_active}."""
    soup = BeautifulSoup(html, "html.parser")
    players = []
    table = soup.find("table", {"id": "players"})
    if not table:
        return players
    for row in table.find("tbody").find_all("tr"):
        th = row.find("th", {"data-stat": "player"})
        if not th:
            continue
        a = th.find("a")
        if not a:
            continue
        name = a.get_text(strip=True)
        url = BASE_URL + a["href"]
        # active range from the "year_min"/"year_max" cells
        year_min = row.find("td", {"data-stat": "year_min"})
        year_max = row.find("td", {"data-stat": "year_max"})
        players.append(
            {
                "name": name,
                "url": url,
                "year_min": int(year_min.get_text()) if year_min else None,
                "year_max": int(year_max.get_text()) if year_max else None,
            }
        )
    return players


def parse_player_bio(html: str, player_url: str) -> dict:
    """Extract bio fields from a player profile page."""
    soup = BeautifulSoup(html, "html.parser")
    bio: dict = {"player_url": player_url}

    meta = soup.find("div", {"id": "meta"})
    if not meta:
        return bio

    # Height / Weight  e.g. "6-9, 234lb"
    hw_match = re.search(r"(\d+)-(\d+),\s*(\d+)lb", meta.get_text())
    if hw_match:
        ft, ins, lbs = hw_match.groups()
        bio["height_inches"] = int(ft) * 12 + int(ins)
        bio["weight_lbs"] = int(lbs)

    # Date of birth
    dob = meta.find("span", {"id": "necro-birth"})
    if dob:
        bio["birth_date"] = dob.get("data-birth", "")

    # Position
    for p in meta.find_all("p"):
        txt = p.get_text(" ", strip=True)
        if txt.startswith("Position:"):
            bio["position"] = txt.replace("Position:", "").strip().split("▪")[0].strip()
        if "College:" in txt:
            a = p.find("a")
            if a:
                bio["college"] = a.get_text(strip=True)
        if "Draft:" in txt:
            m = re.search(r"(\d{4}).*?(\d+)(?:st|nd|rd|th) pick", txt)
            if m:
                bio["draft_year"] = int(m.group(1))
                bio["draft_pick"] = int(m.group(2))

    return bio


def parse_career_stats(html: str) -> dict:
    """Extract career totals row from a player profile page."""
    soup = BeautifulSoup(html, "html.parser")
    stats: dict = {}

    table = soup.find("table", {"id": "per_game"})
    if not table:
        return stats

    for row in table.find("tfoot", {}).find_all("tr"):
        th = row.find("th")
        if th and "Career" in th.get_text():
            for td in row.find_all("td"):
                stat = td.get("data-stat", "")
                val = td.get_text(strip=True)
                if stat and val:
                    try:
                        stats[f"career_{stat}"] = float(val)
                    except ValueError:
                        stats[f"career_{stat}"] = val
    return stats


def parse_game_log_season(html: str, season: int) -> list[dict]:
    """Parse a single game-log page → list of game rows."""
    soup = BeautifulSoup(html, "html.parser")
    rows_out = []
    table = soup.find("table", {"id": "pgl_basic"})
    if not table:
        return rows_out
    for row in table.find("tbody").find_all("tr"):
        if row.get("class") and "thead" in row.get("class"):
            continue
        rank = row.find("td", {"data-stat": "ranker"})
        if not rank or not rank.get_text(strip=True).isdigit():
            continue
        game: dict = {"season": season}
        for td in row.find_all(["td", "th"]):
            stat = td.get("data-stat", "")
            if stat:
                game[stat] = td.get_text(strip=True)
        rows_out.append(game)
    return rows_out


def parse_injury_log(html: str) -> list[dict]:
    """Parse the transactions / injury log table for a player."""
    soup = BeautifulSoup(html, "html.parser")
    rows_out = []
    # BBRef injury data lives in a div with id="div_injuries" or similar
    table = soup.find("table", {"id": re.compile(r"injur|transaction", re.I)})
    if not table:
        return rows_out
    for row in table.find("tbody").find_all("tr"):
        cells: dict = {}
        for td in row.find_all(["td", "th"]):
            stat = td.get("data-stat", "")
            if stat:
                cells[stat] = td.get_text(strip=True)
        if cells:
            rows_out.append(cells)
    return rows_out


# ---------------------------------------------------------------------------
# High-level scraping functions
# ---------------------------------------------------------------------------

def scrape_all_player_urls(manager: IPSafeRequestManager, min_year: int = 1990) -> list[dict]:
    """Walk /players/<a-z>/ and collect every player active since min_year."""
    all_players = []
    for letter in "abcdefghijklmnopqrstuvwxyz":
        url = f"{BASE_URL}/players/{letter}/"
        html = manager.fetch(url)
        if not html:
            continue
        players = parse_player_index(html)
        # Keep players whose active window overlaps [min_year, now]
        filtered = [
            p for p in players
            if p.get("year_max") is None or p["year_max"] >= min_year
        ]
        all_players.extend(filtered)
        print(f"  [index] {letter}: {len(filtered)} players (≥{min_year})")
    return all_players


def scrape_player_profiles(
    manager: IPSafeRequestManager,
    players: list[dict],
    output_csv: Path,
) -> pd.DataFrame:
    """Fetch bio + career stats for every player; save incrementally."""
    records = []
    existing_urls: set[str] = set()

    if output_csv.exists():
        existing_df = pd.read_csv(output_csv)
        existing_urls = set(existing_df["player_url"].dropna())
        records = existing_df.to_dict("records")
        print(f"[resume] {len(existing_urls)} profiles already done")

    for p in players:
        url = p["url"]
        if url in existing_urls:
            continue
        html = manager.fetch(url)
        if not html:
            continue
        bio = parse_player_bio(html, url)
        bio["name"] = p["name"]
        career = parse_career_stats(html)
        bio.update(career)
        records.append(bio)
        existing_urls.add(url)
        # Incremental save every 10 records
        if len(records) % 10 == 0:
            pd.DataFrame(records).to_csv(output_csv, index=False)

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"[profiles] saved {len(df)} rows → {output_csv}")
    return df


def scrape_game_logs(
    manager: IPSafeRequestManager,
    players: list[dict],
    min_season: int = 1990,
    max_season: int | None = None,
) -> None:
    """Fetch season game logs for each player; each season saved separately."""
    if max_season is None:
        import datetime
        max_season = datetime.date.today().year

    for p in players:
        year_min = p.get("year_min") or min_season
        year_max = p.get("year_max") or max_season
        year_min = max(year_min, min_season)

        # Derive player slug from URL e.g. /players/j/jamesle01.html
        slug_match = re.search(r"/players/[a-z]/([a-z0-9]+)\.html", p["url"])
        if not slug_match:
            continue
        slug = slug_match.group(1)

        for season in range(year_min, year_max + 1):
            out_path = RAW_DIR / "gamelogs" / slug / f"{season}.json"
            if out_path.exists():
                continue  # already cached

            url = f"{BASE_URL}/players/{slug[0]}/{slug}/gamelog/{season}/"
            html = manager.fetch(url)
            if not html:
                continue

            games = parse_game_log_season(html, season)
            if games:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(games, indent=2), encoding="utf-8")
                print(f"  [gamelog] {slug}/{season}: {len(games)} games")


def scrape_injury_logs(
    manager: IPSafeRequestManager,
    players: list[dict],
) -> None:
    """Fetch injury/transaction log for each player."""
    for p in players:
        slug_match = re.search(r"/players/[a-z]/([a-z0-9]+)\.html", p["url"])
        if not slug_match:
            continue
        slug = slug_match.group(1)
        out_path = RAW_DIR / "injuries" / f"{slug}.json"
        if out_path.exists():
            continue

        # BBRef injury page is the main player page; data is in a sub-table
        html = manager.fetch(p["url"])
        if not html:
            continue

        injuries = parse_injury_log(html)
        if injuries:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(injuries, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI entry point — general scraper
# ---------------------------------------------------------------------------

def main_general(args) -> None:
    manager = IPSafeRequestManager(cache_dir=RAW_DIR)

    print("=" * 60)
    print("BASKETBALL-REFERENCE SCRAPER")
    print(f"  min_year={args.min_year}  cache={RAW_DIR}")
    print("=" * 60)

    players = scrape_all_player_urls(manager, min_year=args.min_year)
    print(f"\n[index] {len(players)} players active since {args.min_year}")

    if args.max_players:
        players = players[: args.max_players]
        print(f"[cap] testing on {len(players)} players\n")

    profiles_csv = PROCESSED_DIR / "bbref_player_profiles.csv"
    scrape_player_profiles(manager, players, profiles_csv)

    if not args.skip_gamelogs:
        print("\n[gamelogs] scraping season-by-season game logs…")
        scrape_game_logs(manager, players, min_season=args.min_year)

    print("\n[injuries] scraping injury/transaction logs…")
    scrape_injury_logs(manager, players)
    print("\nDone.")


# ===========================================================================
# ACWR PIPELINE  (--acwr flag)
#
# For every player in feature_matrix.csv:
#   - Scrape game logs via basketball_reference_scraper (disk-cached JSON)
#   - Compute EWMA-ACWR at 3 window scales using features/acwr.py
#   - Extract point-in-time features at each player's observation_date
#   - Emit data/processed/acwr_features.csv
#   - Rebuild data/processed/feature_matrix.csv with ACWR joined in
#
# Run: python data/scraping/scrape_bball_reference.py --acwr
# ===========================================================================

import sys as _sys
_sys.path.insert(0, str(ROOT))

import numpy as np
from nba_api.stats.endpoints import playergamelog as _nba_gamelog
from features.acwr import compute_acwr, add_workload_spikes

GAMELOG_CACHE = ROOT / "data" / "raw" / "gamelogs"
GAMELOG_CACHE.mkdir(parents=True, exist_ok=True)

_API_MIN_DELAY = 3.0   # seconds between nba_api calls
_API_MAX_DELAY = 5.0
_API_MAX_RETRIES = 3
_API_INIT_BACKOFF = 10.0
_API_MAX_BACKOFF  = 120.0

# Seasons to fetch for each player type
_RUPTURE_SEASONS_BACK  = 3   # 3 seasons of history for EWMA warm-up
_CONTROL_SEASONS_BACK  = 2   # current + prior season


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_mp(val) -> float:
    """Convert '35:42' or '35.7' or DNP strings → decimal minutes."""
    s = str(val).strip()
    if s in ("", "Did Not Play", "Did Not Dress", "Not With Team", "Inactive", "nan"):
        return 0.0
    try:
        if ":" in s:
            mm, ss = s.split(":", 1)
            return float(mm) + float(ss) / 60.0
        return float(s)
    except (ValueError, AttributeError):
        return 0.0


def _gamelog_cache_path(player_id: int, season_year: int) -> Path:
    return GAMELOG_CACHE / f"{player_id}_{season_year}.json"


def _get_season_year(date: pd.Timestamp) -> int:
    """Return the ending calendar year of the NBA season containing date."""
    return date.year + 1 if date.month >= 10 else date.year


def _normalize_gamelog(df: pd.DataFrame, player_id: int) -> pd.DataFrame:
    """
    Normalise an nba_api PlayerGameLog DataFrame to the standard schema:
        player_id, game_date, minutes_played, fga, fta, pts
    Drops rows where the player did not play (MIN == 0).

    nba_api column names: GAME_DATE, MIN, FGA, FTA, PTS
    MIN may be "35:47" (string) or 35.78 (float) depending on api version.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["game_date"]      = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    out["player_id"]      = player_id
    out["minutes_played"] = df["MIN"].apply(_parse_mp)
    out["fga"]  = pd.to_numeric(df.get("FGA", 0), errors="coerce").fillna(0)
    out["fta"]  = pd.to_numeric(df.get("FTA", 0), errors="coerce").fillna(0)
    out["pts"]  = pd.to_numeric(df.get("PTS", 0), errors="coerce").fillna(0)

    out = out.dropna(subset=["game_date"])
    out = out[out["minutes_played"] > 0].copy()
    return out.sort_values("game_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Disk-cached game log fetch via nba_api (one season at a time)
# ---------------------------------------------------------------------------

def _fetch_season(
    player_name: str,
    player_id: int,
    season_year: int,
) -> pd.DataFrame:
    """
    Return a normalised game-log DataFrame for one player-season.

    Checks JSON disk cache first (data/raw/gamelogs/{player_id}_{season_year}.json).
    Falls back to nba_api.stats.endpoints.playergamelog with 3–5 s delay and
    exponential back-off on failure.  Returns empty DataFrame on failure.

    Uses nba_api instead of basketball_reference_scraper: BBRef blocks automated
    requests; nba_api is the official stats source and does not block.
    """
    path = _gamelog_cache_path(player_id, season_year)

    if path.exists():
        try:
            cached = pd.read_json(path, convert_dates=["game_date"])
            if not cached.empty:
                return cached
        except Exception:
            path.unlink(missing_ok=True)

    season_str_val = f"{season_year - 1}-{str(season_year)[2:]}"
    backoff = _API_INIT_BACKOFF

    for attempt in range(_API_MAX_RETRIES):
        delay = random.uniform(_API_MIN_DELAY, _API_MAX_DELAY)
        print(
            f"  [nba_api] {delay:.1f}s → {player_name} {season_str_val}"
            + (f"  (retry {attempt})" if attempt else "")
        )
        time.sleep(delay)

        try:
            raw = _nba_gamelog.PlayerGameLog(
                player_id=player_id,
                season=season_str_val,
                season_type_all_star="Regular Season",
            ).get_data_frames()[0]

            df = _normalize_gamelog(raw, player_id)

            path.parent.mkdir(parents=True, exist_ok=True)
            if df.empty:
                # Cache the empty result to skip on future runs
                pd.DataFrame().to_json(path, orient="records")
                print(f"  [nba_api] no games found: {player_name} {season_str_val}")
            else:
                df.to_json(path, orient="records", date_format="iso")
                print(f"  [nba_api] {len(df)} games cached → {path.name}")
            return df

        except Exception as exc:
            print(
                f"  [nba_api] attempt {attempt + 1}/{_API_MAX_RETRIES} failed: "
                f"{str(exc)[:120]}"
            )
            time.sleep(backoff)
            backoff = min(backoff * 2.0, _API_MAX_BACKOFF)

    print(f"  [nba_api] giving up on {player_name} {season_str_val}")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Multi-season loader with observation-date filter
# ---------------------------------------------------------------------------

def _load_player_logs(
    player_name: str,
    player_id: int,
    obs_date: pd.Timestamp,
    event: int,
) -> pd.DataFrame:
    """
    Fetch and concatenate all required seasons for a player, then
    return only games on or before obs_date.

    Rupture players: 3 seasons back  (needs EWMA warm-up over full chronic window)
    Control players: 2 seasons back  (current season + prior for warm-up)
    """
    season_year = _get_season_year(obs_date)
    seasons_back = _RUPTURE_SEASONS_BACK if event == 1 else _CONTROL_SEASONS_BACK
    target_seasons = range(season_year - seasons_back + 1, season_year + 1)

    frames = []
    for sy in target_seasons:
        df = _fetch_season(player_name, player_id, sy)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    all_games = pd.concat(frames, ignore_index=True)
    all_games["game_date"] = pd.to_datetime(all_games["game_date"])
    all_games = all_games.sort_values("game_date").drop_duplicates("game_date")
    return all_games[all_games["game_date"] <= obs_date].copy()


# ---------------------------------------------------------------------------
# Point-in-time ACWR feature extraction
# ---------------------------------------------------------------------------

def _pit_features(
    player_id: int,
    player_name: str,
    game_df: pd.DataFrame,
    obs_date: pd.Timestamp,
) -> dict:
    """
    Given all games for a player up to obs_date, compute the ACWR feature
    vector as it would be observed at obs_date.
    """
    base: dict = {
        "player_id":          player_id,
        "observation_date":   obs_date.date(),
        "acwr_3_21":          np.nan,
        "acwr_7_28":          np.nan,
        "acwr_14_56":         np.nan,
        "acwr_spike_flag":    0,
        "days_since_last_game":  np.nan,
        "games_last_7_days":  0,
        "games_last_14_days": 0,
    }

    if game_df.empty:
        return base

    game_df = game_df.copy()
    game_df["game_date"] = pd.to_datetime(game_df["game_date"])

    last_game = game_df["game_date"].max()
    base["days_since_last_game"] = int((obs_date - last_game).days)
    base["games_last_7_days"]    = int(
        (game_df["game_date"] >= obs_date - pd.Timedelta(days=7)).sum()
    )
    base["games_last_14_days"]   = int(
        (game_df["game_date"] >= obs_date - pd.Timedelta(days=14)).sum()
    )

    try:
        acwr_df = compute_acwr(
            game_df,
            player_col="player_id",
            date_col="game_date",
            load_col="minutes_played",
            fill_gaps=True,
        )
        last = acwr_df.iloc[-1]
        base["acwr_3_21"]  = float(last.get("acwr_3_21",  np.nan))
        base["acwr_7_28"]  = float(last.get("acwr_7_28",  np.nan))
        base["acwr_14_56"] = float(last.get("acwr_14_56", np.nan))
        base["acwr_spike_flag"] = int(
            any(
                not np.isnan(v) and v > 1.5
                for v in (base["acwr_3_21"], base["acwr_7_28"], base["acwr_14_56"])
            )
        )
    except Exception as exc:
        print(f"  [acwr] computation failed for {player_name}: {exc}")

    return base


# ---------------------------------------------------------------------------
# Main pipeline functions
# ---------------------------------------------------------------------------

def build_acwr_features(
    feature_matrix_csv: Path = PROCESSED_DIR / "feature_matrix.csv",
    output_csv: Path = PROCESSED_DIR / "acwr_features.csv",
) -> pd.DataFrame:
    """
    For every player in feature_matrix.csv:
      1. Fetch game logs (disk-cached) via basketball_reference_scraper
      2. Compute point-in-time ACWR at observation_date
      3. Write acwr_features.csv and return DataFrame
    """
    fm = pd.read_csv(feature_matrix_csv, parse_dates=["observation_date"])

    rows = []
    n = len(fm)
    for i, row in fm.iterrows():
        pid   = int(row["player_id"])
        pname = str(row["player_name"])
        obs   = pd.Timestamp(row["observation_date"])
        event = int(row["event"])

        print(f"\n[{i+1}/{n}] {pname}  obs={obs.date()}  event={event}")
        game_df = _load_player_logs(pname, pid, obs, event)
        print(f"  → {len(game_df)} games loaded before obs date")

        feat = _pit_features(pid, pname, game_df, obs)
        rows.append(feat)

    acwr_df = pd.DataFrame(rows)
    acwr_df.to_csv(output_csv, index=False)

    n_valid = acwr_df["acwr_7_28"].notna().sum()
    print(f"\n[acwr] saved {len(acwr_df)} rows ({n_valid} with valid ACWR) → {output_csv}")
    return acwr_df


def rebuild_feature_matrix(
    feature_matrix_csv: Path = PROCESSED_DIR / "feature_matrix.csv",
    acwr_csv: Path = PROCESSED_DIR / "acwr_features.csv",
    output_csv: Path = PROCESSED_DIR / "feature_matrix.csv",
) -> pd.DataFrame:
    """
    Left-join acwr_features onto feature_matrix on (player_id, observation_date)
    and overwrite feature_matrix.csv.
    """
    fm   = pd.read_csv(feature_matrix_csv)
    acwr = pd.read_csv(acwr_csv)

    # Normalise join keys (format='mixed' handles both date-only and datetime strings)
    fm["observation_date"]   = pd.to_datetime(fm["observation_date"], format="mixed").dt.date.astype(str)
    acwr["observation_date"] = pd.to_datetime(acwr["observation_date"], format="mixed").dt.date.astype(str)
    acwr["player_id"]        = acwr["player_id"].astype(int)
    fm["player_id"]          = fm["player_id"].astype(int)

    acwr_cols = [
        c for c in acwr.columns
        if c not in ("player_id", "observation_date")
    ]
    # Drop stale ACWR columns from fm if re-running
    fm = fm.drop(columns=[c for c in acwr_cols if c in fm.columns], errors="ignore")

    merged = fm.merge(
        acwr[["player_id", "observation_date"] + acwr_cols],
        on=["player_id", "observation_date"],
        how="left",
    )
    merged.to_csv(output_csv, index=False)

    feature_cols = [
        c for c in merged.columns
        if c not in {
            "player_id", "player_name", "observation_date",
            "event", "time_to_event_days", "birth_date",
        }
    ]
    print(f"[rebuild] {len(merged)} rows  ·  {len(feature_cols)} features  → {output_csv}")
    print(f"  feature columns: {feature_cols}")
    return merged


def run_acwr_pipeline() -> None:
    """Orchestrate the full ACWR pipeline: scrape → compute → rebuild."""
    fm_csv   = PROCESSED_DIR / "feature_matrix.csv"
    acwr_csv = PROCESSED_DIR / "acwr_features.csv"

    print("=" * 60)
    print("ACWR PIPELINE")
    print(f"  feature_matrix : {fm_csv}")
    print(f"  gamelog cache  : {GAMELOG_CACHE}")
    print(f"  output         : {acwr_csv}")
    print("=" * 60)

    print("\n── Step 1: scrape game logs + compute ACWR features ──")
    build_acwr_features(fm_csv, acwr_csv)

    print("\n── Step 2: rebuild feature_matrix.csv ──")
    merged = rebuild_feature_matrix(fm_csv, acwr_csv, fm_csv)

    # Quick sanity check — run the same load_data path train.py uses
    from sklearn.preprocessing import StandardScaler
    non_feat = {
        "player_id", "player_name", "observation_date",
        "event", "time_to_event_days", "birth_date",
        "time_clipped", "time_bin",
    }
    feat_cols = [
        c for c in merged.columns
        if c not in non_feat and merged[c].dtype in (float, int, "float64", "int64")
    ]
    print(f"\n[sanity] {len(feat_cols)} numeric feature cols, "
          f"{merged['event'].sum()} events, "
          f"{(merged['event']==0).sum()} censored")
    print("\nACWR pipeline complete.")
    print("Next: python -m models.train --no-hpo --epochs 20")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Basketball-Reference scraper + ACWR pipeline"
    )
    subparsers = parser.add_subparsers(dest="mode")

    # ── mode: acwr (focused pipeline) ────────────────────────────────────
    subparsers.add_parser(
        "acwr",
        help="Scrape game logs + build ACWR features + rebuild feature_matrix.csv",
    )

    # ── mode: full (original general scraper) ────────────────────────────
    full_p = subparsers.add_parser("full", help="Full BBRef player/gamelog scrape")
    full_p.add_argument("--min-year",      type=int, default=1990)
    full_p.add_argument("--max-players",   type=int, default=None)
    full_p.add_argument("--skip-gamelogs", action="store_true")

    args = parser.parse_args()

    if args.mode == "acwr" or args.mode is None and len(_sys.argv) == 1:
        # Default with no args → print help
        if args.mode is None:
            parser.print_help()
            return
        run_acwr_pipeline()
    elif args.mode == "full":
        main_general(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
