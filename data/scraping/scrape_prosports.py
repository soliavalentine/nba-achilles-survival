"""
ProSportsTransactions.com scraper — ground-truth Achilles rupture dataset.

Scrapes all NBA IL placements that mention "achilles" in the transaction note,
1990-present.  Saves raw HTML pages to data/raw/prosports/ with full caching
so any interrupted run resumes where it left off.

Requires FlareSolverr running locally to bypass Cloudflare's JS challenge:
    docker run -d --name=flaresolverr -p 8191:8191 ghcr.io/flaresolverr/flaresolverr:latest

Output CSV: data/processed/achilles_ground_truth.csv
Columns: player_name, date, team, transaction_type, reason, source_url
"""

from __future__ import annotations

import hashlib
import random
import re
import sys
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
# Config
# ---------------------------------------------------------------------------

class IPSafeConfig:
    # Delay between PST page requests (FlareSolverr is already slow, so keep
    # these modest — the CF challenge resolution adds natural spacing)
    MIN_DELAY = 4.0
    MAX_DELAY = 8.0
    REQUESTS_PER_DOMAIN_PER_MINUTE = 6
    INITIAL_BACKOFF = 30
    MAX_BACKOFF = 300
    BACKOFF_MULTIPLIER = 2
    MAX_REQUESTS_PER_SESSION = 120
    COOLDOWN_AFTER_MAX = 600

    # FlareSolverr endpoint (Docker: ghcr.io/flaresolverr/flaresolverr:latest)
    FLARESOLVERR_URL = "http://localhost:8191/v1"
    # Max seconds to wait for FlareSolverr to solve a CF challenge and return
    FLARESOLVERR_TIMEOUT = 90

    MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Request manager — routes everything through FlareSolverr
# ---------------------------------------------------------------------------

class IPSafeRequestManager:
    def __init__(self, cache_dir: Path, config: IPSafeConfig | None = None):
        self.config = config or IPSafeConfig()
        self.cache_dir = cache_dir
        # Local HTTP client only talks to FlareSolverr (localhost), never PST directly
        self._http = requests.Session()
        self._fs_session_id: str | None = None
        self.domain_requests: dict[str, list[float]] = defaultdict(list)
        self.domain_backoff: dict[str, float] = defaultdict(
            lambda: self.config.INITIAL_BACKOFF
        )
        self.total_requests = 0
        self.session_start = time.time()

    # ------------------------------------------------------------------
    # FlareSolverr session lifecycle
    # ------------------------------------------------------------------

    def check_flaresolverr(self) -> None:
        """Raise RuntimeError if FlareSolverr isn't reachable."""
        health_url = self.config.FLARESOLVERR_URL.replace("/v1", "")
        try:
            r = self._http.get(health_url, timeout=5)
            if r.status_code != 200:
                raise RuntimeError(f"FlareSolverr returned {r.status_code}")
        except requests.RequestException as exc:
            raise RuntimeError(
                f"FlareSolverr not reachable at {health_url}.\n"
                "Start it with:\n"
                "  docker run -d --name=flaresolverr -p 8191:8191 "
                "ghcr.io/flaresolverr/flaresolverr:latest"
            ) from exc
        print("[flaresolverr] reachable ✓")

    def create_fs_session(self) -> None:
        """Create a persistent browser session in FlareSolverr for cookie reuse."""
        r = self._http.post(
            self.config.FLARESOLVERR_URL,
            json={"cmd": "sessions.create"},
            timeout=15,
        )
        data = r.json()
        self._fs_session_id = data.get("session")
        print(f"[flaresolverr] session created: {self._fs_session_id}")

    def destroy_fs_session(self) -> None:
        if not self._fs_session_id:
            return
        try:
            self._http.post(
                self.config.FLARESOLVERR_URL,
                json={"cmd": "sessions.destroy", "session": self._fs_session_id},
                timeout=10,
            )
        except requests.RequestException:
            pass
        self._fs_session_id = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

    def _fs_get(self, url: str) -> tuple[int, str]:
        """Send one GET through FlareSolverr. Returns (http_status, html)."""
        payload: dict = {
            "cmd": "request.get",
            "url": url,
            "maxTimeout": self.config.FLARESOLVERR_TIMEOUT * 1000,
        }
        if self._fs_session_id:
            payload["session"] = self._fs_session_id
        resp = self._http.post(
            self.config.FLARESOLVERR_URL,
            json=payload,
            timeout=self.config.FLARESOLVERR_TIMEOUT + 30,
        )
        data = resp.json()
        sol = data.get("solution", {})
        return sol.get("status", 0), sol.get("response", "")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def warm_up(self) -> None:
        """Visit homepage then basketball section to establish session cookies."""
        for url in [BASE_URL + "/", BASE_URL + "/basketball/"]:
            delay = random.uniform(3.0, 6.0)
            print(f"  [warm-up] +{delay:.1f}s  {url}")
            time.sleep(delay)
            try:
                status, html = self._fs_get(url)
                print(f"  [warm-up] {status}  len={len(html)}")
            except requests.RequestException as exc:
                print(f"  [warm-up] error: {exc}")

    def fetch(self, url: str, *, force_refresh: bool = False) -> str | None:
        """Fetch URL with disk caching. Set force_refresh=True to bypass cache."""
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
                status, html = self._fs_get(url)

                if status == 200:
                    cache_path.write_text(html, encoding="utf-8")
                    self.domain_requests[domain].append(time.time())
                    self.domain_backoff[domain] = self.config.INITIAL_BACKOFF
                    self.total_requests += 1
                    print(f"  [ok] #{self.total_requests}  len={len(html)}")
                    return html

                elif status == 404:
                    print(f"  [404] {url}")
                    return None

                elif status == 429:
                    backoff = self.domain_backoff[domain]
                    print(f"  [429] back-off {backoff}s")
                    time.sleep(backoff)
                    self.domain_backoff[domain] = min(
                        backoff * self.config.BACKOFF_MULTIPLIER,
                        self.config.MAX_BACKOFF,
                    )

                else:
                    print(f"  [http-error] {status} — attempt {attempt + 1}")
                    if attempt < self.config.MAX_RETRIES:
                        time.sleep(self.config.INITIAL_BACKOFF * (attempt + 1))
                    else:
                        return None

            except requests.RequestException as exc:
                print(f"  [req-error] {exc}")
                if attempt < self.config.MAX_RETRIES:
                    time.sleep(self.config.INITIAL_BACKOFF * (attempt + 1))
                else:
                    return None

        return None


# ---------------------------------------------------------------------------
# ProSportsTransactions URL builder
# ---------------------------------------------------------------------------

def build_search_url(start_row: int, begin_date: str, end_date: str) -> str:
    """
    Build a paginated search URL for all injury transactions in a date window.
    PST doesn't filter by notes server-side, so we fetch all injury transactions
    and filter client-side for 'achilles'.
    """
    params = {
        "Player": "",
        "Team": "",
        "BeginDate": begin_date,
        "EndDate": end_date,
        "InjuriesChkBx": "yes",
        "Submit": "Search",
        "start": str(start_row),
    }
    return BASE_URL + SEARCH_ENDPOINT + "?" + urlencode(params)


# ---------------------------------------------------------------------------
# HTML parser
# ---------------------------------------------------------------------------

def parse_results_page(html: str, source_url: str) -> tuple[list[dict], int, bool]:
    """
    Parse a ProSportsTransactions search results page.

    PST paginates via page-number links (start=N), not a "1-25 of X" count.
    We derive the apparent total from the highest start= value found in the page,
    and also return a has_next flag so the caller can stop when there's no Next link.

    Returns:
        (rows, apparent_total, has_next)
        rows: list of dicts with keys: player_name, date, team,
              transaction_type, reason, source_url
        apparent_total: max(start= links) + PAGE_SIZE, or 0 if no pagination found
        has_next: True if a "Next" page link exists in the HTML
    """
    soup = BeautifulSoup(html, "html.parser")
    rows_out: list[dict] = []

    # Derive the current start offset from the source URL so we can check whether
    # PST actually includes a link to the *next* page (start=current+PAGE_SIZE).
    # Searching for free-text "Next" is unreliable because it also matches nav links.
    current_start_m = re.search(r'[?&]start=(\d+)', source_url)
    current_start = int(current_start_m.group(1)) if current_start_m else 0
    next_start = current_start + PAGE_SIZE
    has_next = f"start={next_start}" in html

    start_vals = [int(m) for m in re.findall(r'[?&]start=(\d+)', html)]
    apparent_total = max(start_vals) + PAGE_SIZE if start_vals else 0

    # PST table has class="datatable center" with <td> (not <th>) header cells
    table = soup.find("table", class_="datatable")
    if not table:
        # Fallback: any table whose first row has Date + Acquired/Relinquished cells
        for t in soup.find_all("table"):
            first_row = t.find("tr")
            if not first_row:
                continue
            cell_texts = [
                re.sub(r"[\xa0\s]+", " ", c.get_text()).strip().lower()
                for c in first_row.find_all(["th", "td"])
            ]
            if "date" in cell_texts and any(
                "relinquish" in ct or "acquir" in ct for ct in cell_texts
            ):
                table = t
                break

    if not table:
        return rows_out, apparent_total, has_next

    first_row = table.find("tr")
    header_cells = first_row.find_all(["th", "td"])
    col_names = [
        re.sub(r"[\xa0\s]+", " ", c.get_text()).strip().lower()
        for c in header_cells
    ]

    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        try:
            date_idx = next((i for i, c in enumerate(col_names) if "date" in c), 0)
            team_idx = next((i for i, c in enumerate(col_names) if "team" in c), 1)
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

        if relinquished:
            player_name = relinquished
            txn_type = "placed_on_IL"
        elif acquired:
            player_name = acquired
            txn_type = "activated_from_IL"
        else:
            continue

        # Client-side filter: only keep rows mentioning achilles in the notes
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

    return rows_out, apparent_total, has_next


# ---------------------------------------------------------------------------
# Main scraping pipeline
# ---------------------------------------------------------------------------

def scrape_achilles_transactions(
    begin_year: int = 1990,
    end_year: int | None = None,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """
    Paginate through all PST injury transactions year-by-year and filter
    client-side for achilles mentions.  Year-by-year chunking keeps each
    query to a manageable number of pages (~10-20) so FlareSolverr sessions
    stay healthy.
    """
    if end_year is None:
        end_year = date.today().year
    if output_csv is None:
        output_csv = PROCESSED_DIR / "achilles_ground_truth.csv"

    manager = IPSafeRequestManager(cache_dir=RAW_DIR)

    # Verify FlareSolverr is up before doing anything else
    manager.check_flaresolverr()
    manager.create_fs_session()

    all_rows: list[dict] = []
    seen_keys: set[tuple] = set()

    try:
        print("\n[warm-up] Establishing browser session via homepage…")
        manager.warm_up()

        for year in range(begin_year, end_year + 1):
            begin_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            print(f"\n[year] {year}")
            start = 0
            has_next = True
            # Safety cap: NBA seasons have ~450 players × ~3 IL entries = ~1350 rows ≈ 55 pages.
            # 200 pages (5000 rows) is a generous ceiling that prevents runaway loops.
            MAX_PAGES_PER_YEAR = 200
            page_count = 0

            while has_next and page_count < MAX_PAGES_PER_YEAR:
                url = build_search_url(start, begin_date, end_date)
                html = manager.fetch(url)
                if not html:
                    print(f"  [warn] fetch failed at start={start}, skipping rest of {year}")
                    break

                rows, apparent_total, has_next = parse_results_page(html, url)
                all_rows, seen_keys = _merge_rows(all_rows, seen_keys, rows)
                page_count += 1

                if rows:
                    print(f"  [start={start}] {len(rows)} achilles rows this page, "
                          f"has_next={has_next}")

                start += PAGE_SIZE

            if page_count >= MAX_PAGES_PER_YEAR:
                print(f"  [warn] hit {MAX_PAGES_PER_YEAR}-page cap for {year}, stopping")

    finally:
        manager.destroy_fs_session()

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
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["severity"] = df["reason"].apply(_classify_severity)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _classify_severity(reason: str) -> str:
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
    print(f"\nTop 10 teams by Achilles events:")
    print(df["team"].value_counts().head(10).to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape ProSportsTransactions for NBA Achilles IL placements",
        epilog=(
            "Requires FlareSolverr running on localhost:8191.\n"
            "Start it with: docker run -d --name=flaresolverr -p 8191:8191 "
            "ghcr.io/flaresolverr/flaresolverr:latest"
        ),
    )
    parser.add_argument(
        "--begin-year", type=int, default=1990,
        help="First season year to scrape (default: 1990)",
    )
    parser.add_argument(
        "--end-year", type=int, default=None,
        help="Last season year to scrape (default: current year)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / "achilles_ground_truth.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Scrape only the current year to verify connectivity end-to-end",
    )
    parser.add_argument(
        "--flaresolverr",
        default=IPSafeConfig.FLARESOLVERR_URL,
        help="FlareSolverr endpoint (default: http://localhost:8191/v1)",
    )
    args = parser.parse_args()

    if args.test:
        args.begin_year = date.today().year
        args.end_year = date.today().year

    IPSafeConfig.FLARESOLVERR_URL = args.flaresolverr

    print("=" * 60)
    print("PROSPORTSTRANSACTIONS ACHILLES SCRAPER")
    print(f"  years        : {args.begin_year}–{args.end_year or date.today().year}")
    print(f"  output       : {args.output}")
    print(f"  cache        : {RAW_DIR}")
    print(f"  flaresolverr : {args.flaresolverr}")
    print("=" * 60)

    scrape_achilles_transactions(
        begin_year=args.begin_year,
        end_year=args.end_year,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
