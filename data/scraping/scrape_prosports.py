"""
ProSportsTransactions.com scraper — ground-truth Achilles rupture dataset.

Scrapes all NBA IL placements that mention "achilles" in the transaction note,
1990-present.  Saves raw HTML pages to data/raw/prosports/ with full caching
so any interrupted run resumes where it left off.

Output CSV: data/processed/achilles_ground_truth.csv
Columns: player_name, date, team, transaction_type, reason, source_url
"""

from __future__ import annotations

import csv
import hashlib
import random
import re
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from urllib.parse import urlencode, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "prosports"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.prosportstransactions.com"
SEARCH_ENDPOINT = "/basketball/Search/SearchResults.php"

# ProSportsTransactions paginates 25 rows at a time
PAGE_SIZE = 25

# ---------------------------------------------------------------------------
# IP safety  (same pattern as nba-star-predictor, tuned for this domain)
# ---------------------------------------------------------------------------

class IPSafeConfig:
    MIN_DELAY = 3.5
    MAX_DELAY = 7.0
    REQUESTS_PER_DOMAIN_PER_MINUTE = 8
    INITIAL_BACKOFF = 20
    MAX_BACKOFF = 300
    BACKOFF_MULTIPLIER = 2
    MAX_REQUESTS_PER_SESSION = 120
    COOLDOWN_AFTER_MAX = 600

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


class IPSafeRequestManager:
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
                print(f"  [rate-limit] {domain}: sleeping {wait:.1f}s")
                time.sleep(wait)

    def _check_session_limit(self) -> None:
        if self.total_requests < self.config.MAX_REQUESTS_PER_SESSION:
            return
        elapsed = time.time() - self.session_start
        remaining = self.config.COOLDOWN_AFTER_MAX - elapsed
        if remaining > 0:
            print(
                f"\n[session-limit] {self.total_requests} requests — "
                f"cooling down {remaining / 60:.1f} min\n"
            )
            time.sleep(remaining)
        self.total_requests = 0
        self.session_start = time.time()

    def fetch(self, url: str, *, force_refresh: bool = False) -> str | None:
        """Fetch URL with disk caching.  Set force_refresh=True to bypass cache."""
        cache_path = self._url_to_cache_path(self.cache_dir, url)
        if cache_path.exists() and not force_refresh:
            return cache_path.read_text(encoding="utf-8")

        self._check_session_limit()
        domain = self._domain(url)
        self._enforce_rate_limit(domain)

        delay = random.uniform(self.config.MIN_DELAY, self.config.MAX_DELAY)
        print(f"  [fetch] +{delay:.1f}s  {url}")
        time.sleep(delay)

        for attempt in range(self.config.MAX_RETRIES + 1):
            try:
                headers = {
                    "User-Agent": random.choice(self.config.USER_AGENTS),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": BASE_URL + "/basketball/",
                }
                resp = self.session.get(
                    url, headers=headers, timeout=self.config.TIMEOUT
                )

                if resp.status_code == 200:
                    html = resp.text
                    cache_path.write_text(html, encoding="utf-8")
                    self.domain_requests[domain].append(time.time())
                    self.domain_backoff[domain] = self.config.INITIAL_BACKOFF
                    self.total_requests += 1
                    print(f"  [ok] #{self.total_requests}")
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
                    print(f"  [http-error] {resp.status_code}")
                    return None

            except requests.Timeout:
                print(f"  [timeout] attempt {attempt + 1}/{self.config.MAX_RETRIES + 1}")
                if attempt < self.config.MAX_RETRIES:
                    time.sleep(self.config.INITIAL_BACKOFF * (attempt + 1))

            except requests.RequestException as exc:
                print(f"  [req-error] {exc}")
                return None

        return None


# ---------------------------------------------------------------------------
# ProSportsTransactions URL builder
# ---------------------------------------------------------------------------

def build_search_url(keyword: str, start_row: int, league: str = "NBA") -> str:
    """Build a paginated search URL for the given keyword."""
    params = {
        "Player": "",
        "Team": "",
        "BeginDate": "1990-01-01",
        "EndDate": date.today().isoformat(),
        "InjuriesChkBx": "yes",
        "PersonalChkBx": "yes",
        "DisciplinaryChkBx": "yes",
        "DraftChkBx": "yes",
        "ContractChkBx": "yes",
        "Submit": "Search",
        "QueryType": "And",
        "query": keyword,
        "start": str(start_row),
    }
    return BASE_URL + SEARCH_ENDPOINT + "?" + urlencode(params)


# ---------------------------------------------------------------------------
# HTML parser
# ---------------------------------------------------------------------------

def parse_results_page(html: str, source_url: str) -> tuple[list[dict], int]:
    """
    Parse a ProSportsTransactions search results page.

    Returns:
        (rows, total_results)
        rows: list of dicts with keys: player_name, date, team,
              transaction_type, reason, source_url
        total_results: total match count reported by the site
    """
    soup = BeautifulSoup(html, "html.parser")
    rows_out: list[dict] = []

    # Total result count is in a string like "1 - 25 of 847 transactions"
    total = 0
    count_text = soup.find(string=re.compile(r"\d+ - \d+ of \d+"))
    if count_text:
        m = re.search(r"of\s+(\d+)", count_text)
        if m:
            total = int(m.group(1))

    table = soup.find("table", {"class": re.compile(r"result", re.I)})
    if not table:
        # Sometimes the table has no explicit class — try the main data table
        tables = soup.find_all("table")
        for t in tables:
            headers = [th.get_text(strip=True).lower() for th in t.find_all("th")]
            if "date" in headers and "player" in headers:
                table = t
                break

    if not table:
        return rows_out, total

    header_cells = table.find("tr").find_all(["th", "td"])
    col_names = [c.get_text(strip=True).lower() for c in header_cells]

    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        # Map by column index, falling back to positional defaults
        try:
            date_idx = next(
                (i for i, c in enumerate(col_names) if "date" in c), 0
            )
            team_idx = next(
                (i for i, c in enumerate(col_names) if "team" in c), 1
            )
            acquired_idx = next(
                (i for i, c in enumerate(col_names) if "acquir" in c or "placed" in c), 2
            )
            relinquished_idx = next(
                (i for i, c in enumerate(col_names) if "relinquish" in c), 3
            )
            notes_idx = next(
                (i for i, c in enumerate(col_names) if "note" in c or "reason" in c), 4
            )

            txn_date = cells[date_idx].get_text(strip=True) if date_idx < len(cells) else ""
            team = cells[team_idx].get_text(strip=True) if team_idx < len(cells) else ""
            acquired = cells[acquired_idx].get_text(strip=True) if acquired_idx < len(cells) else ""
            relinquished = cells[relinquished_idx].get_text(strip=True) if relinquished_idx < len(cells) else ""
            notes = cells[notes_idx].get_text(strip=True) if notes_idx < len(cells) else ""

        except (IndexError, StopIteration):
            continue

        # Determine player name and transaction direction
        if relinquished:
            player_name = relinquished
            txn_type = "placed_on_IL"
        elif acquired:
            player_name = acquired
            txn_type = "activated_from_IL"
        else:
            continue

        # Filter: only keep rows mentioning achilles in the notes
        combined = (notes + " " + player_name).lower()
        if "achilles" not in combined:
            continue

        rows_out.append(
            {
                "player_name": player_name,
                "date": txn_date,
                "team": team,
                "transaction_type": txn_type,
                "reason": notes,
                "source_url": source_url,
            }
        )

    return rows_out, total


# ---------------------------------------------------------------------------
# Main scraping pipeline
# ---------------------------------------------------------------------------

def scrape_achilles_transactions(
    keywords: list[str] | None = None,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """
    Paginate through all ProSportsTransactions results for each keyword
    and assemble the ground-truth Achilles IL dataset.

    Keywords tried: "achilles tendon", "achilles rupture", "achilles tear"
    """
    if keywords is None:
        keywords = ["achilles tendon", "achilles rupture", "achilles tear", "achilles"]
    if output_csv is None:
        output_csv = PROCESSED_DIR / "achilles_ground_truth.csv"

    manager = IPSafeRequestManager(cache_dir=RAW_DIR)
    all_rows: list[dict] = []
    seen_keys: set[tuple] = set()  # dedup by (player, date, team)

    for kw in keywords:
        print(f"\n[keyword] '{kw}'")
        start = 0

        # Fetch first page to learn total count
        url = build_search_url(kw, start)
        html = manager.fetch(url)
        if not html:
            print(f"  [skip] no response for keyword '{kw}'")
            continue

        rows, total = parse_results_page(html, url)
        print(f"  [total] {total} transactions found")
        all_rows, seen_keys = _merge_rows(all_rows, seen_keys, rows)

        # Paginate through remaining pages
        start += PAGE_SIZE
        while start < total:
            url = build_search_url(kw, start)
            html = manager.fetch(url)
            if not html:
                print(f"  [warn] failed at offset {start}, stopping pagination")
                break
            rows, _ = parse_results_page(html, url)
            all_rows, seen_keys = _merge_rows(all_rows, seen_keys, rows)
            start += PAGE_SIZE
            print(f"  [progress] {min(start, total)}/{total}")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = _clean_dataframe(df)
        df.to_csv(output_csv, index=False)
        print(f"\n[saved] {len(df)} records → {output_csv}")
        _print_summary(df)
    else:
        print("\n[warn] no Achilles transactions found")

    return df


def _merge_rows(
    existing: list[dict],
    seen: set[tuple],
    new_rows: list[dict],
) -> tuple[list[dict], set[tuple]]:
    for r in new_rows:
        key = (r["player_name"], r["date"], r["team"])
        if key not in seen:
            existing.append(r)
            seen.add(key)
    return existing, seen


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Classify severity from reason text
    df["severity"] = df["reason"].apply(_classify_severity)

    # Sort
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _classify_severity(reason: str) -> str:
    """
    Coarse label from free-text reason field.
    rupture > tear > strain/tendinitis > unspecified
    """
    r = reason.lower()
    if re.search(r"ruptur|torn|compl[ei]te", r):
        return "rupture"
    if re.search(r"tear|partial", r):
        return "tear"
    if re.search(r"strain|tendinit|tendinop|sore|inflam", r):
        return "tendinopathy"
    return "unspecified"


def _print_summary(df: pd.DataFrame) -> None:
    print("\n--- Summary ---")
    print(f"Total records : {len(df)}")
    print(f"Date range    : {df['date'].min().date()} – {df['date'].max().date()}")
    print(f"\nSeverity breakdown:")
    print(df["severity"].value_counts().to_string())
    print(f"\nTransaction type breakdown:")
    print(df["transaction_type"].value_counts().to_string())
    top_teams = df["team"].value_counts().head(10)
    print(f"\nTop 10 teams by Achilles events:")
    print(top_teams.to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape ProSportsTransactions for NBA Achilles IL placements"
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=["achilles tendon", "achilles rupture", "achilles tear", "achilles"],
        help="Search terms (default: 4 achilles variants)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / "achilles_ground_truth.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PROSPORTSTRANSACTIONS ACHILLES SCRAPER")
    print(f"  keywords : {args.keywords}")
    print(f"  output   : {args.output}")
    print(f"  cache    : {RAW_DIR}")
    print("=" * 60)

    scrape_achilles_transactions(keywords=args.keywords, output_csv=args.output)


if __name__ == "__main__":
    main()
