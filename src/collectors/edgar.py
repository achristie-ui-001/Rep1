import datetime
import json
import os
import re
import sqlite3
import time
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.collectors.base import BaseCollector, TranscriptRecord
from src.utils.logger import get_logger
from src.utils.rate_limiter import RateLimiter

logger = get_logger("inflection_detector.edgar")

EDGAR_BASE = "https://data.sec.gov"
EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

TRANSCRIPT_FILENAME_HINTS = {"transcript", "call", "remarks", "q1", "q2", "q3", "q4", "earnings"}
TRANSCRIPT_CONTENT_PATTERNS = re.compile(
    r"(operator|good morning|good afternoon|good evening|thank you for (joining|participating)|"
    r"welcome to the|earnings (call|conference)|conference call)",
    re.IGNORECASE,
)


class EdgarCollector(BaseCollector):
    source_name = "edgar"

    def __init__(self, config: dict, db_path: str = "data/cache/edgar_filings.db"):
        self.user_agent = config.get("user_agent", "InflectionDetector research@example.com")
        self.rate_limiter = RateLimiter(config.get("rate_limit_rps", 8))
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent, "Accept": "application/json"})
        self._init_db(db_path)
        self._cik_map: dict[str, str] = {}

    def _init_db(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS filings (
                cik TEXT, accession TEXT, filing_date TEXT,
                has_transcript INTEGER DEFAULT 0,
                transcript_url TEXT DEFAULT '',
                PRIMARY KEY (cik, accession)
            )"""
        )
        self.db.commit()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _get(self, url: str, params: dict = None, headers: dict = None) -> requests.Response:
        self.rate_limiter.acquire()
        resp = self.session.get(url, params=params, headers=headers or {}, timeout=30)
        resp.raise_for_status()
        return resp

    def load_cik_map(self) -> dict[str, str]:
        if self._cik_map:
            return self._cik_map
        logger.info("Fetching EDGAR company-ticker map")
        resp = self._get(TICKERS_URL)
        data = resp.json()
        for entry in data.values():
            ticker = entry.get("ticker", "").upper()
            cik = str(entry.get("cik_str", "")).zfill(10)
            self._cik_map[ticker] = cik
        return self._cik_map

    def get_cik(self, ticker: str) -> Optional[str]:
        return self.load_cik_map().get(ticker.upper())

    def get_8k_filings(self, cik: str, start_date: str, end_date: str) -> list[dict]:
        url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
        try:
            resp = self._get(url)
        except Exception as e:
            logger.error(f"Failed to fetch submissions for CIK {cik}: {e}")
            return []

        data = resp.json()
        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accessions = filings.get("accessionNumber", [])

        results = []
        for form, date, accession in zip(forms, dates, accessions):
            if form == "8-K" and start_date <= date <= end_date:
                results.append({"date": date, "accession": accession, "cik": cik})

        # Also check older filings if available
        if "files" in data.get("filings", {}):
            for extra_file in data["filings"]["files"]:
                try:
                    extra_resp = self._get(f"{EDGAR_BASE}/submissions/{extra_file['name']}")
                    extra_data = extra_resp.json()
                    for form, date, accession in zip(
                        extra_data.get("form", []),
                        extra_data.get("filingDate", []),
                        extra_data.get("accessionNumber", []),
                    ):
                        if form == "8-K" and start_date <= date <= end_date:
                            results.append({"date": date, "accession": accession, "cik": cik})
                except Exception:
                    pass

        return results

    def find_transcript_exhibit(self, cik: str, accession: str) -> Optional[str]:
        # Check DB cache
        row = self.db.execute(
            "SELECT has_transcript, transcript_url FROM filings WHERE cik=? AND accession=?",
            (cik, accession),
        ).fetchone()
        if row:
            if row[0] == 1:
                return row[1]
            if row[0] == -1:
                return None  # Previously determined: no transcript

        accession_dashes = accession.replace("-", "")
        index_url = f"{EDGAR_ARCHIVES}/{cik}/{accession_dashes}/"
        try:
            resp = self._get(index_url, headers={"Accept": "text/html"})
        except Exception as e:
            logger.debug(f"Index fetch failed {index_url}: {e}")
            return None

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "lxml")
        rows = soup.select("table tr")
        candidates = []
        fallback_exhibits = []  # EX-99.x exhibits without keyword match — try as fallback
        for row_el in rows:
            cells = row_el.find_all("td")
            if len(cells) < 3:
                continue
            desc = cells[1].get_text(strip=True).lower()
            fname = cells[2].get_text(strip=True).lower()
            link = cells[2].find("a")
            if not link:
                continue
            href = link.get("href", "")
            if not href:
                continue
            # Skip XML index files and XBRL
            if fname.endswith((".xml", ".xsd", ".json")) or "xbrl" in fname or "r9999" in fname:
                continue
            full_url = f"https://www.sec.gov{href}" if href.startswith("/") else href
            score = 0
            for hint in TRANSCRIPT_FILENAME_HINTS:
                if hint in fname or hint in desc:
                    score += 1
            if score > 0:
                candidates.append((score, full_url))
            elif ("ex-99" in fname or "ex99" in fname or "exhibit" in desc) and fname.endswith((".htm", ".html", ".txt")):
                fallback_exhibits.append((0, full_url))

        candidates.sort(key=lambda x: x[0], reverse=True)
        # If no keyword matches, try EX-99 fallbacks (they may be transcripts with generic names)
        all_to_try = candidates + fallback_exhibits[:5]

        transcript_url = None
        for _, url in all_to_try:
            if self._verify_transcript(url):
                transcript_url = url
                break

        has = 1 if transcript_url else -1
        self.db.execute(
            "INSERT OR REPLACE INTO filings (cik, accession, has_transcript, transcript_url) VALUES (?, ?, ?, ?)",
            (cik, accession, has, transcript_url or ""),
        )
        self.db.commit()
        return transcript_url

    def _verify_transcript(self, url: str) -> bool:
        try:
            resp = self._get(url, headers={"Accept": "text/html,text/plain"})
            text = resp.text[:5000]
            return bool(TRANSCRIPT_CONTENT_PATTERNS.search(text)) or len(text.split()) > 1500
        except Exception:
            return False

    def download_text(self, url: str) -> Optional[str]:
        try:
            resp = self._get(url, headers={"Accept": "text/html,text/plain"})
            return resp.text
        except Exception as e:
            logger.error(f"Download failed {url}: {e}")
            return None

    def collect(self, ticker: str, company_name: str, start_date: str, end_date: str, cache=None) -> list[TranscriptRecord]:
        from src.parsers.html_parser import parse_html_transcript
        from src.parsers.transcript_normalizer import normalize_transcript

        cik = self.get_cik(ticker)
        if not cik:
            logger.warning(f"No CIK found for {ticker}")
            return []

        filings = self.get_8k_filings(cik, start_date, end_date)
        logger.info(f"{ticker}: found {len(filings)} 8-K filings in range")

        records = []
        for filing in filings:
            from src.utils.cache import TranscriptCache
            cache_key = None
            if cache:
                cache_key = cache.make_key(ticker, filing["date"], "edgar")
                if cache.exists(cache_key):
                    data = cache.get(cache_key)
                    if data and data.get("parse_status") == "ok":
                        records.append(TranscriptRecord(**data))
                        continue

            transcript_url = self.find_transcript_exhibit(cik, filing["accession"])
            if not transcript_url:
                continue

            raw_html = self.download_text(transcript_url)
            if not raw_html:
                continue

            parsed = parse_html_transcript(raw_html)
            if not parsed or parsed.get("word_count", 0) < 500:
                continue

            quarter = self._date_to_quarter(filing["date"], parsed.get("raw_text", ""))
            record = normalize_transcript(
                raw_data=parsed,
                ticker=ticker,
                company_name=company_name,
                date=filing["date"],
                quarter=quarter,
                source="edgar",
                url=transcript_url,
            )

            if cache and cache_key:
                cache.put(cache_key, record.__dict__)
            records.append(record)
            time.sleep(0.1)

        return records

    @staticmethod
    def _date_to_quarter(date_str: str, text: str) -> str:
        try:
            dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return ""
        # Try to detect fiscal quarter from text
        q_match = re.search(r"\b(first|second|third|fourth|Q[1-4])\s+(quarter|fiscal quarter)\b", text[:2000], re.IGNORECASE)
        if q_match:
            token = q_match.group(1).upper()
            q_map = {"FIRST": "Q1", "SECOND": "Q2", "THIRD": "Q3", "FOURTH": "Q4"}
            q = q_map.get(token, token)
            return f"{q} {dt.year}"
        # Fallback: calendar quarter
        q = (dt.month - 1) // 3 + 1
        return f"Q{q} {dt.year}"
