#!/usr/bin/env python3
"""
scrape_bbref_injuries.py

Expand the Achilles rupture ground truth by scraping Basketball Reference
injury + transaction data for seasons 2009-10 through 2024-25.

DO NOT RUN without reviewing all three config sections:
  ── RATE LIMITING  (~line 70)
  ── TOR ROTATION   (~line 105)
  ── SEVERITY KEYWORDS (~line 145)

PRE-FLIGHT CHECKLIST:
    1. Install Tor:
           brew install tor && brew services start tor
    2. Add to ~/.torrc  (create if missing):
           ControlPort 9051
           CookieAuthentication 0
       Then restart: brew services restart tor
    3. Verify: curl --socks5 127.0.0.1:9050 https://check.torproject.org/api/ip
    4. Review the three config sections, then run:
           python data/scraping/scrape_bbref_injuries.py

──────────────────────────────────────────────────────────────
BACKGROUND RUNNING (so progress survives terminal close):

    mkdir -p logs
    nohup python data/scraping/scrape_bbref_injuries.py \\
        >> logs/bbref_injuries.log 2>&1 &
    echo $! > .scraper_pid
    tail -f logs/bbref_injuries.log    # follow in another terminal
    kill $(cat .scraper_pid)           # stop when needed

NOTE: nohup survives terminal close but NOT machine sleep/shutdown.
For scraping across shutdowns you need a cloud VPS:
    ssh user@vps "nohup python scrape_bbref_injuries.py >> out.log 2>&1 &"
──────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import re
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    from stem import Signal
    from stem.control import Controller
    STEM_AVAILABLE = True
except ImportError:
    STEM_AVAILABLE = False

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[2]
CACHE_DIR    = ROOT / "data" / "raw" / "bbref_injuries"
PROCESSED    = ROOT / "data" / "processed"
LOG_DIR      = ROOT / "logs"
GT_CSV       = PROCESSED / "achilles_ground_truth.csv"   # read + append
LOG_FILE     = LOG_DIR / "bbref_injuries.log"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.basketball-reference.com"

# Seasons to scrape: 2009-10 → 2024-25  (end years 2010 … 2025)
SEASON_END_YEARS = list(range(2010, 2026))


# ══════════════════════════════════════════════════════════════════════════════
# ── RATE LIMITING CONFIG ──────────────────────────────────────────────────────
# REVIEW BEFORE RUNNING
# ══════════════════════════════════════════════════════════════════════════════

RATE_LIMIT = {
    # Delay range between every request (seconds, randomised)
    "min_delay_s":           4.0,
    "max_delay_s":           8.0,

    # Exponential backoff on HTTP 429 or 503.
    # We wait each duration in sequence, then give up on that URL.
    "backoff_429_503_s":     [30, 60, 120],

    # On HTTP 403: rotate Tor exit node and retry exactly ONCE.
    # If retry also 403, log and skip — do not retry again.
    "stop_on_403_after_retry": True,

    # Hard session cap. After this many live requests, pause for cooldown.
    "max_requests_per_session": 50,
    "session_cooldown_s":        600,   # 10 minutes

    # Persist new rows to CSV this often (counts ONLY newly found ruptures).
    "save_every_n_new_rows":  5,

    # Warn (and prompt) if run during peak hours.
    "off_peak_warn_start_h": 9,    # 9 AM local time
    "off_peak_warn_end_h":   17,   # 5 PM local time
}

# ══════════════════════════════════════════════════════════════════════════════
# ── TOR ROTATION CONFIG ───────────────────────────────────────────────────────
# REVIEW BEFORE RUNNING
# ══════════════════════════════════════════════════════════════════════════════

TOR = {
    "enabled":           True,

    # SOCKS5 proxy — Tor's default
    "socks5_host":       "127.0.0.1",
    "socks5_port":       9050,

    # Control port — for NEWNYM (exit node rotation)
    "control_host":      "127.0.0.1",
    "control_port":      9051,

    # Leave empty if torrc has CookieAuthentication 0 and no HashedControlPassword
    "control_password":  "",

    # Rotate exit node every N *live* requests (cache hits don't count)
    "rotate_every_n":    10,

    # Wait after NEWNYM before continuing (Tor needs time to build new circuit)
    "wait_after_newnym_s": 5,

    # Shown in error message if Tor is not running
    "install_cmd": "brew install tor && brew services start tor",
    "torrc_hint": (
        "Add to ~/.torrc:\n"
        "  ControlPort 9051\n"
        "  CookieAuthentication 0\n"
        "Then: brew services restart tor"
    ),
}

# ══════════════════════════════════════════════════════════════════════════════
# ── SEVERITY KEYWORDS ─────────────────────────────────────────────────────────
# REVIEW BEFORE RUNNING
# Applied in order: first match wins. All matches are case-insensitive.
# ══════════════════════════════════════════════════════════════════════════════

# Keywords that must appear for the entry to even be considered Achilles-related
ACHILLES_FILTER = ["achilles", "calf", "heel"]

SEVERITY = {
    # Highest priority — any of these → rupture
    "rupture": [
        "rupture",
        "ruptured",
        "torn",
        "tear",
        "tore",
        "surgery",
        "surgical",
        "repair",              # "Achilles repair surgery"
        "out for the season",
        "out for season",
        "remainder of the season",
        "missed remainder",
        "out indefinitely",    # typically means structural damage
        "season-ending",
        "season ending",
    ],

    # Lower priority — matches only if no rupture keyword was found
    "tendinopathy": [
        "tendinopathy",
        "tendinitis",
        "tendinosis",
        "tendonitis",
        "tendon inflammation",
        "inflammation",
        "soreness",
        "tightness",
        "day-to-day",
        "dtd",
        "chronic",
        "discomfort",
        "irritation",
        "strain",              # calf strain ≠ rupture but is load-related
    ],

    # Everything else achilles/calf/heel → unspecified
    "unspecified": [],
}

# ══════════════════════════════════════════════════════════════════════════════
# User agents — 10 real Chrome/Firefox/Safari strings
# ══════════════════════════════════════════════════════════════════════════════

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
]


# ── Tor utilities ─────────────────────────────────────────────────────────────

def _tor_proxies() -> dict:
    return {
        "http":  f"socks5h://{TOR['socks5_host']}:{TOR['socks5_port']}",
        "https": f"socks5h://{TOR['socks5_host']}:{TOR['socks5_port']}",
    }


def check_tor() -> None:
    """Verify Tor is running and SOCKS5 proxy works. Exits with clear message if not."""
    if not TOR["enabled"]:
        _log("Tor disabled — running without IP rotation")
        return
    if not STEM_AVAILABLE:
        print("\nERROR: stem not installed.\n  pip install stem requests[socks]")
        sys.exit(1)

    # 1. Control port
    try:
        with Controller.from_port(
            address=TOR["control_host"], port=TOR["control_port"]
        ) as ctrl:
            # Pass password only if set; empty string → let stem auto-detect
            # (handles both CookieAuthentication and HashedControlPassword)
            pwd = TOR["control_password"]
            ctrl.authenticate(password=pwd) if pwd else ctrl.authenticate()
    except Exception as exc:
        print(f"\nERROR: Tor control port {TOR['control_port']} not accessible: {exc}")
        print(f"\nInstall Tor:  {TOR['install_cmd']}")
        print(f"\n{TOR['torrc_hint']}")
        sys.exit(1)

    # 2. SOCKS5 proxy
    try:
        resp = requests.get(
            "https://check.torproject.org/api/ip",
            proxies=_tor_proxies(),
            timeout=20,
        )
        info = resp.json()
        ip = info.get("IP", "?")
        is_tor = info.get("IsTor", False)
        if is_tor:
            _log(f"Tor OK — exit node IP: {ip}")
        else:
            _log(f"WARNING: proxy returned non-Tor IP {ip} — check torrc")
    except Exception as exc:
        print(f"\nERROR: Tor SOCKS5 proxy not accessible at "
              f"{TOR['socks5_host']}:{TOR['socks5_port']}: {exc}")
        print(f"\nInstall Tor:  {TOR['install_cmd']}")
        sys.exit(1)


def rotate_tor_exit(reason: str = "") -> None:
    """Signal Tor to build a new circuit (NEWNYM)."""
    if not TOR["enabled"] or not STEM_AVAILABLE:
        return
    try:
        with Controller.from_port(
            address=TOR["control_host"], port=TOR["control_port"]
        ) as ctrl:
            # Pass password only if set; empty string → let stem auto-detect
            # (handles both CookieAuthentication and HashedControlPassword)
            pwd = TOR["control_password"]
            ctrl.authenticate(password=pwd) if pwd else ctrl.authenticate()
            ctrl.signal(Signal.NEWNYM)
        wait = TOR["wait_after_newnym_s"]
        tag = f" ({reason})" if reason else ""
        _log(f"NEWNYM{tag} — waiting {wait}s for new circuit")
        time.sleep(wait)
    except Exception as exc:
        _log(f"Tor rotation failed: {exc}")


# ── Logging ───────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ── Off-peak warning ──────────────────────────────────────────────────────────

def check_off_peak(force: bool = False) -> None:
    now = datetime.datetime.now()
    start = RATE_LIMIT["off_peak_warn_start_h"]
    end   = RATE_LIMIT["off_peak_warn_end_h"]
    if start <= now.hour < end:
        _log(
            f"WARNING: running at {now.strftime('%H:%M')} (peak hours {start}:00–{end}:00). "
            "BBRef rate limits are stricter during business hours."
        )
        if not force:
            ans = input("Continue anyway? [y/N] ").strip().lower()
            if ans != "y":
                sys.exit(0)


# ── Disk-cached, rate-limited, Tor-routed fetcher ─────────────────────────────

class Fetcher:
    """
    Single-responsibility: fetch a URL and return HTML.

    Rules enforced:
      - Check disk cache first (never re-requests a cached page)
      - Random 4–8 s delay before every live request
      - Session cap: pause 10 min after 50 live requests
      - Tor exit node rotation every 10 live requests
      - HTTP 429/503: exponential backoff [30, 60, 120] s then give up
      - HTTP 403: rotate exit node, retry once; if still 403, self.blocked = True
      - HTTP 404: log and return None (no retry)
    """

    def __init__(self) -> None:
        self.session               = requests.Session()
        self.live_requests         = 0    # live (non-cached) requests this session
        self.requests_since_rotate = 0
        self.session_start         = time.time()
        self.blocked               = False

    @staticmethod
    def _cache_path(url: str) -> Path:
        slug = hashlib.md5(url.encode()).hexdigest()
        return CACHE_DIR / f"{slug}.html"

    @staticmethod
    def _headers() -> dict:
        return {
            "User-Agent":      random.choice(USER_AGENTS),
            "Referer":         "https://www.google.com/search?q=nba+player+achilles+injury+basketball+reference",
            "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT":             "1",
            "Connection":      "keep-alive",
        }

    def _enforce_session_cap(self) -> None:
        if self.live_requests < RATE_LIMIT["max_requests_per_session"]:
            return
        elapsed   = time.time() - self.session_start
        cooldown  = RATE_LIMIT["session_cooldown_s"]
        remaining = max(0.0, cooldown - elapsed)
        if remaining > 0:
            _log(
                f"Session cap ({self.live_requests} requests) — "
                f"cooling down {remaining / 60:.1f} min"
            )
            time.sleep(remaining)
        self.live_requests   = 0
        self.session_start   = time.time()

    def _maybe_rotate(self) -> None:
        self.requests_since_rotate += 1
        if self.requests_since_rotate >= TOR["rotate_every_n"]:
            rotate_tor_exit("scheduled")
            self.requests_since_rotate = 0

    def _do_get(self, url: str) -> requests.Response:
        proxies = _tor_proxies() if TOR["enabled"] else {}
        return self.session.get(
            url, headers=self._headers(), proxies=proxies,
            timeout=30, allow_redirects=True,
        )

    def fetch(self, url: str) -> str | None:
        """Return cached or freshly fetched HTML, or None on unrecoverable failure."""
        if self.blocked:
            return None

        # ── Cache hit ────────────────────────────────────────────────────────
        cache = self._cache_path(url)
        if cache.exists():
            return cache.read_text(encoding="utf-8")

        # ── Pre-request housekeeping ─────────────────────────────────────────
        self._enforce_session_cap()
        self._maybe_rotate()

        delay = random.uniform(RATE_LIMIT["min_delay_s"], RATE_LIMIT["max_delay_s"])
        _log(f"{delay:.1f}s → {url[:90]}")
        time.sleep(delay)

        # ── Request with 429/503 backoff ─────────────────────────────────────
        backoffs = RATE_LIMIT["backoff_429_503_s"]
        for attempt in range(len(backoffs) + 1):
            if attempt > 0:
                wait = backoffs[attempt - 1]
                _log(f"backoff {wait}s (attempt {attempt + 1}/{len(backoffs) + 1})")
                time.sleep(wait)

            try:
                resp = self._do_get(url)
            except requests.RequestException as exc:
                _log(f"request error: {exc}")
                if attempt == len(backoffs):
                    return None
                continue

            code = resp.status_code

            if code == 200:
                html = resp.text
                cache.write_text(html, encoding="utf-8")
                self.live_requests += 1
                _log(f"ok #{self.live_requests} → {cache.name}")
                return html

            if code == 403:
                _log(f"HTTP 403 — rotating Tor exit node, retrying once")
                rotate_tor_exit("403 block")
                try:
                    resp2 = self._do_get(url)
                    if resp2.status_code == 200:
                        html = resp2.text
                        cache.write_text(html, encoding="utf-8")
                        self.live_requests += 1
                        _log(f"ok #{self.live_requests} after 403 rotate → {cache.name}")
                        return html
                    _log(f"retry after rotate returned HTTP {resp2.status_code}")
                except requests.RequestException as exc:
                    _log(f"retry after rotate failed: {exc}")
                _log("403 on both exit nodes — marking blocked, stopping")
                self.blocked = True
                return None

            if code == 404:
                _log(f"HTTP 404 — {url[:70]}")
                return None

            if code in (429, 503):
                _log(f"HTTP {code} rate-limited")
                if attempt == len(backoffs):
                    _log(f"exhausted backoff — skipping {url[:70]}")
                    return None
                continue  # retry with next backoff

            _log(f"HTTP {code} unexpected — skipping {url[:70]}")
            return None

        return None


# ── Severity classifier ───────────────────────────────────────────────────────

def classify_severity(description: str) -> str:
    """
    Return 'rupture', 'tendinopathy', or 'unspecified'.
    Checks SEVERITY['rupture'] keywords first (highest priority).
    """
    desc = description.lower()
    for kw in SEVERITY["rupture"]:
        if kw in desc:
            return "rupture"
    for kw in SEVERITY["tendinopathy"]:
        if kw in desc:
            return "tendinopathy"
    return "unspecified"


def is_achilles_related(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in ACHILLES_FILTER)


# ── Date parsing ──────────────────────────────────────────────────────────────

_MONTH_PAT = (
    r"(?:January|February|March|April|May|June|"
    r"July|August|September|October|November|December)"
)
_DATE_RE = re.compile(rf"({_MONTH_PAT}\s+\d{{1,2}},\s+\d{{4}})")


def _parse_date(text: str) -> str | None:
    """Extract and normalise a date string to ISO format (YYYY-MM-DD)."""
    m = _DATE_RE.search(text)
    if not m:
        return None
    try:
        return str(pd.to_datetime(m.group(1)).date())
    except Exception:
        return None


# ── HTML parsers ──────────────────────────────────────────────────────────────

def parse_injury_table(html: str, season_year: int) -> list[dict]:
    """
    Parse /friv/injuries.fcgi style pages — structured table with
    columns: Player, Team, Date Updated, Description.
    """
    soup = BeautifulSoup(html, "html.parser")
    results = []

    table = soup.find("table", {"id": re.compile(r"injur", re.I)})
    if not table:
        return results

    for row in table.find("tbody").find_all("tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) < 3:
            continue
        desc = cells[-1].get_text(" ", strip=True)
        if not is_achilles_related(desc):
            continue

        player = cells[0].get_text(strip=True).lstrip("•").strip()
        team   = cells[1].get_text(strip=True) if len(cells) > 1 else ""
        date_raw = cells[2].get_text(strip=True) if len(cells) > 2 else ""
        date_iso = _parse_date(date_raw) or f"{season_year}-06-15"  # fall back to season end

        results.append({
            "player_name":      player,
            "date":             date_iso,
            "team":             team,
            "transaction_type": "placed_on_IL",
            "reason":           desc[:400],
            "source_url":       f"{BASE_URL}/friv/injuries.fcgi",
            "severity":         classify_severity(desc),
        })

    return results


def parse_transactions_page(html: str, season_year: int, source_url: str) -> list[dict]:
    """
    Parse /leagues/NBA_{year}_transactions.html.

    Actual BBRef structure (confirmed from cached pages):
      - All content is inside <div id="content">
      - Dates appear as bare <strong>Month D, YYYY</strong> nodes between entries
      - Each transaction is a <p> tag; player/team names are <a href=...> links
        e.g. <p>The <a href="/teams/...">Lakers</a> placed
             <a href="/players/...">LeBron James</a> on ...</p>

    Player extraction uses href="/players/" links — exact, no regex needed.
    Team extraction uses href="/teams/" links — same approach.
    """
    soup = BeautifulSoup(html, "html.parser")
    results = []
    current_date: str | None = None

    # Scope to the content div — avoids nav/footer noise
    content = soup.find("div", {"id": "content"})
    if not content:
        return results

    for tag in content.find_all(["strong", "p"]):
        text = tag.get_text(" ", strip=True)
        if not text:
            continue

        # ── Date tracking from <strong> date nodes ────────────────────────────
        if tag.name == "strong":
            parsed = _parse_date(text)
            if parsed:
                current_date = parsed
            continue

        # ── Transaction <p> tags ──────────────────────────────────────────────
        if not is_achilles_related(text):
            continue

        date_iso = current_date or f"{season_year - 1}-10-01"

        # Player name: first <a href="/players/..."> in the tag (exact match)
        player = ""
        for a in tag.find_all("a", href=True):
            if "/players/" in a["href"]:
                player = a.get_text(strip=True)
                break

        if not player or len(player) < 4:
            continue

        # Team name: first <a href="/teams/..."> in the tag
        team = ""
        for a in tag.find_all("a", href=True):
            if "/teams/" in a["href"]:
                team = a.get_text(strip=True)
                break

        results.append({
            "player_name":      player,
            "date":             date_iso,
            "team":             team,
            "transaction_type": "placed_on_IL",
            "reason":           text[:400],
            "source_url":       source_url,
            "severity":         classify_severity(text),
        })

    return results


# ── Deduplication ─────────────────────────────────────────────────────────────

def load_existing_keys() -> set[tuple[str, str]]:
    """Return set of (player_name_lower, date_iso) already in ground truth."""
    if not GT_CSV.exists():
        return set()
    df = pd.read_csv(GT_CSV, parse_dates=["date"])
    keys = set()
    for _, row in df.iterrows():
        name = str(row["player_name"]).lstrip("•").strip().lower()
        try:
            date = str(row["date"].date())
        except Exception:
            date = str(row["date"])[:10]
        keys.add((name, date))
    return keys


def is_duplicate(row: dict, existing: set[tuple[str, str]]) -> bool:
    name = row["player_name"].lower().strip()
    date = str(row.get("date", ""))[:10]
    return (name, date) in existing


# ── Incremental CSV save ──────────────────────────────────────────────────────

def append_rows(rows: list[dict], output_csv: Path) -> None:
    """Append new rows to ground truth CSV, preserving existing data."""
    df_new = pd.DataFrame(rows)
    if output_csv.exists():
        df_existing = pd.read_csv(output_csv)
        # Align columns
        for col in df_existing.columns:
            if col not in df_new.columns:
                df_new[col] = ""
        df_new = df_new[df_existing.columns]
        out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        out = df_new
    out.to_csv(output_csv, index=False)


# ── URL builders ──────────────────────────────────────────────────────────────

def injury_page_url(season_year: int) -> str:
    # BBRef injury page only shows the current season.
    # Try a year parameter; fall back gracefully if page content doesn't change.
    return f"{BASE_URL}/friv/injuries.fcgi"


def transactions_page_url(season_year: int) -> str:
    # Historical league transaction pages — one per season
    return f"{BASE_URL}/leagues/NBA_{season_year}_transactions.html"


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    _log("=" * 70)
    _log(f"BBRef Achilles injury scraper — seasons {SEASON_END_YEARS[0]-1}-{str(SEASON_END_YEARS[0])[2:]} "
         f"through {SEASON_END_YEARS[-1]-1}-{str(SEASON_END_YEARS[-1])[2:]}")
    _log(f"  cache dir : {CACHE_DIR}")
    _log(f"  output    : {GT_CSV}")
    _log(f"  Tor       : {'enabled' if TOR['enabled'] else 'DISABLED'}")
    _log("=" * 70)

    check_off_peak(force=args.force)
    if TOR["enabled"]:
        check_tor()

    fetcher  = Fetcher()
    existing = load_existing_keys()
    _log(f"Existing ground truth rows: {len(existing)}")

    new_rows:       list[dict] = []
    new_ruptures    = 0
    total_new       = 0
    save_threshold  = RATE_LIMIT["save_every_n_new_rows"]

    def maybe_save() -> None:
        """Flush new_rows to CSV if we've hit the save threshold."""
        nonlocal new_rows, total_new
        if len(new_rows) >= save_threshold:
            append_rows(new_rows, GT_CSV)
            _log(f"  Saved {len(new_rows)} rows to {GT_CSV.name}")
            existing.update(
                (r["player_name"].lower().strip(), str(r["date"])[:10])
                for r in new_rows
            )
            new_rows = []

    # ── Step 1: Current-season injury table ─────────────────────────────────
    _log("\n── Step 1: current season injury table (/friv/injuries.fcgi)")
    html = fetcher.fetch(injury_page_url(SEASON_END_YEARS[-1]))
    if html:
        entries = parse_injury_table(html, SEASON_END_YEARS[-1])
        _log(f"  parsed {len(entries)} achilles entries from injury table")
        for e in entries:
            if is_duplicate(e, existing):
                continue
            new_rows.append(e)
            total_new += 1
            if e["severity"] == "rupture":
                new_ruptures += 1
                _log(f"  NEW RUPTURE: {e['player_name']} {e['date']} — {e['reason'][:80]}")
            maybe_save()

    if fetcher.blocked:
        _log("Blocked after injury table. Run again later with a new Tor circuit.")
        if new_rows:
            append_rows(new_rows, GT_CSV)
        return

    # ── Step 2: Historical + current transaction pages ───────────────────────
    _log(f"\n── Step 2: transactions pages for {len(SEASON_END_YEARS)} seasons")

    for year in SEASON_END_YEARS:
        if fetcher.blocked:
            break

        url  = transactions_page_url(year)
        season_label = f"{year - 1}-{str(year)[2:]}"
        _log(f"\n  [{season_label}] {url}")

        html = fetcher.fetch(url)
        if not html:
            _log(f"  [{season_label}] skipped (fetch failed)")
            continue

        entries = parse_transactions_page(html, year, url)
        achilles = [e for e in entries if is_achilles_related(e["reason"])]
        _log(f"  [{season_label}] {len(entries)} total entries, {len(achilles)} achilles-related")

        season_new = 0
        for e in achilles:
            if is_duplicate(e, existing):
                continue
            new_rows.append(e)
            total_new += 1
            season_new += 1
            if e["severity"] == "rupture":
                new_ruptures += 1
                _log(
                    f"  NEW RUPTURE [{season_label}]: "
                    f"{e['player_name']} {e['date']} — {e['reason'][:80]}"
                )
            maybe_save()

        _log(f"  [{season_label}] +{season_new} new  |  {new_ruptures} ruptures found so far, {total_new} total new")

    # ── Step 3: Final save ───────────────────────────────────────────────────
    if new_rows:
        append_rows(new_rows, GT_CSV)
        _log(f"\nFinal save: {len(new_rows)} remaining rows flushed")

    _log("\n" + "=" * 70)
    _log(f"DONE")
    _log(f"  New ruptures found   : {new_ruptures}")
    _log(f"  Total new rows added : {total_new}")
    _log(f"  Ground truth CSV     : {GT_CSV}")
    if fetcher.blocked:
        _log("  STATUS: stopped early — IP blocked. Re-run after Tor cooldown.")
    _log("=" * 70)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BBRef Achilles injury scraper with Tor IP rotation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip off-peak-hours warning and run immediately",
    )
    parser.add_argument(
        "--no-tor",
        action="store_true",
        help="Disable Tor routing (lower protection, faster setup)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which URLs would be fetched, don't make requests",
    )
    args = parser.parse_args()

    if args.no_tor:
        TOR["enabled"] = False
        _log("WARNING: Tor disabled — all requests from your real IP")

    if args.dry_run:
        _log("DRY RUN — URLs that would be fetched:")
        _log(f"  {injury_page_url(SEASON_END_YEARS[-1])}")
        for year in SEASON_END_YEARS:
            _log(f"  {transactions_page_url(year)}")
        _log(f"  Total: {1 + len(SEASON_END_YEARS)} URLs")
        return

    run(args)


if __name__ == "__main__":
    main()
