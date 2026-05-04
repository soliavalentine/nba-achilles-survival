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
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Basketball-Reference for NBA Achilles survival study"
    )
    parser.add_argument(
        "--min-year", type=int, default=1990,
        help="Earliest season to include (default: 1990)"
    )
    parser.add_argument(
        "--max-players", type=int, default=None,
        help="Cap number of players (for testing)"
    )
    parser.add_argument(
        "--skip-gamelogs", action="store_true",
        help="Skip game-log scraping (profiles + injuries only)"
    )
    args = parser.parse_args()

    manager = IPSafeRequestManager(cache_dir=RAW_DIR)

    print("=" * 60)
    print("BASKETBALL-REFERENCE SCRAPER")
    print(f"  min_year={args.min_year}  cache={RAW_DIR}")
    print("=" * 60)

    # Step 1: collect player URLs
    players = scrape_all_player_urls(manager, min_year=args.min_year)
    print(f"\n[index] {len(players)} players active since {args.min_year}")

    if args.max_players:
        players = players[: args.max_players]
        print(f"[cap] testing on {len(players)} players\n")

    # Step 2: player profiles + career stats
    profiles_csv = PROCESSED_DIR / "bbref_player_profiles.csv"
    scrape_player_profiles(manager, players, profiles_csv)

    # Step 3: game logs
    if not args.skip_gamelogs:
        print("\n[gamelogs] scraping season-by-season game logs…")
        scrape_game_logs(manager, players, min_season=args.min_year)

    # Step 4: injury logs
    print("\n[injuries] scraping injury/transaction logs…")
    scrape_injury_logs(manager, players)

    print("\nDone.")


if __name__ == "__main__":
    main()
