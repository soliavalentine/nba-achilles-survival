"""
NBA official injury report scraper.

The league publishes PDF injury reports before each game at:
  https://ak-static.cms.nba.com/referee/injury/Injury-Report_<date>_<AMPM>.pdf

This scraper:
  1. Builds a date range of game dates from Basketball-Reference schedules.
  2. Downloads each PDF (both AM and PM editions) to data/raw/injury_reports/.
  3. Parses the PDF with pdfplumber and outputs a flat CSV.

Resume-safe: skips PDFs already on disk.
"""

from __future__ import annotations

import csv
import random
import re
import time
from datetime import date, timedelta
from pathlib import Path

import pdfplumber
import requests

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "injury_reports"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PDF_BASE = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{date}_{ampm}.pdf"
MIN_DELAY = 2.0
MAX_DELAY = 5.0

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]


def _date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _pdf_path(report_date: date, ampm: str) -> Path:
    return RAW_DIR / f"{report_date.isoformat()}_{ampm}.pdf"


def download_injury_report(report_date: date, ampm: str = "AM") -> Path | None:
    out = _pdf_path(report_date, ampm)
    if out.exists():
        return out

    url = PDF_BASE.format(date=report_date.strftime("%Y-%m-%d"), ampm=ampm)
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": random.choice(USER_AGENTS)},
            timeout=30,
        )
        if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith("application/pdf"):
            out.write_bytes(resp.content)
            print(f"  [pdf] {out.name}")
            return out
        return None
    except requests.RequestException as exc:
        print(f"  [error] {exc}")
        return None


def parse_injury_pdf(pdf_path: Path) -> list[dict]:
    """Extract rows from an official NBA injury-report PDF."""
    rows: list[dict] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if not table:
                    continue
                header = [str(c).strip().lower() for c in table[0]]
                for row in table[1:]:
                    if len(row) < len(header):
                        continue
                    record = dict(zip(header, [str(c).strip() if c else "" for c in row]))
                    record["source_pdf"] = pdf_path.name
                    rows.append(record)
    except Exception as exc:
        print(f"  [parse-error] {pdf_path.name}: {exc}")
    return rows


def run_full_scrape(
    start: date = date(2015, 10, 1),
    end: date | None = None,
    output_csv: Path | None = None,
) -> None:
    if end is None:
        end = date.today()
    if output_csv is None:
        output_csv = PROCESSED_DIR / "nba_official_injury_reports.csv"

    all_rows: list[dict] = []
    for d in _date_range(start, end):
        for ampm in ("AM", "PM"):
            pdf_path = download_injury_report(d, ampm)
            if pdf_path:
                all_rows.extend(parse_injury_pdf(pdf_path))

    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n[saved] {len(all_rows)} rows → {output_csv}")
    else:
        print("[warn] no rows parsed")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2015-10-01")
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end) if args.end else None
    run_full_scrape(start=start_date, end=end_date)
