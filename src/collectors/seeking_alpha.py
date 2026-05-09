"""
Seeking Alpha transcript collector.
Rate-limited to ~1 req/6s. Backs off on 429 responses.
Personal use only — do not distribute scraped content.
"""
import datetime
import re
import time
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.collectors.base import BaseCollector, TranscriptRecord
from src.utils.logger import get_logger
from src.utils.rate_limiter import RateLimiter

logger = get_logger("inflection_detector.seeking_alpha")

SA_BASE = "https://seekingalpha.com"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://seekingalpha.com/",
}

QUARTER_RE = re.compile(r"\b(Q[1-4])\s+(\d{4})\b", re.IGNORECASE)


class SeekingAlphaCollector(BaseCollector):
    source_name = "seeking_alpha"

    def __init__(self, config: dict):
        delay = config.get("delay_between_requests_s", 6)
        self.rate_limiter = RateLimiter(1 / delay)
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)
        self._warm_session()

    def _warm_session(self):
        try:
            self.rate_limiter.acquire()
            self.session.get(SA_BASE, timeout=15)
        except Exception:
            pass

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=10, max=120))
    def _get_json(self, url: str, params: dict = None) -> Optional[dict]:
        self.rate_limiter.acquire()
        resp = self.session.get(url, params=params, timeout=20)
        if resp.status_code == 429:
            raise requests.HTTPError("Rate limited", response=resp)
        if resp.status_code == 403:
            logger.warning(f"403 from Seeking Alpha: {url}")
            return None
        resp.raise_for_status()
        return resp.json()

    def get_transcript_list(self, ticker: str) -> list[dict]:
        url = f"{SA_BASE}/api/v3/symbol/{ticker.upper()}/transcripts"
        params = {"filter[type]": "earnings", "page[size]": "40", "page[number]": "1"}
        try:
            data = self._get_json(url, params)
        except Exception as e:
            logger.error(f"SA transcript list failed for {ticker}: {e}")
            return []
        if not data or "data" not in data:
            return []
        results = []
        for item in data["data"]:
            attrs = item.get("attributes", {})
            results.append({
                "id": item.get("id"),
                "title": attrs.get("title", ""),
                "publish_date": attrs.get("publishOn", "")[:10],
            })
        return results

    def get_transcript_content(self, article_id: str) -> Optional[str]:
        url = f"{SA_BASE}/api/v3/articles/{article_id}"
        params = {"include": "author,primaryTickers,content"}
        try:
            data = self._get_json(url, params)
        except Exception as e:
            logger.error(f"SA article fetch failed {article_id}: {e}")
            return None
        if not data:
            return None
        attrs = data.get("data", {}).get("attributes", {})
        return attrs.get("content", "") or attrs.get("body", "")

    def collect(self, ticker: str, company_name: str, start_date: str, end_date: str, cache=None) -> list[TranscriptRecord]:
        from src.parsers.html_parser import parse_html_transcript
        from src.parsers.transcript_normalizer import normalize_transcript

        transcript_list = self.get_transcript_list(ticker)
        records = []

        for item in transcript_list:
            date = item.get("publish_date", "")
            if not date or not (start_date <= date <= end_date):
                continue

            cache_key = None
            if cache:
                from src.utils.cache import TranscriptCache
                cache_key = cache.make_key(ticker, date, "seeking_alpha")
                if cache.exists(cache_key):
                    data = cache.get(cache_key)
                    if data and data.get("parse_status") == "ok":
                        records.append(TranscriptRecord(**data))
                        continue

            content_html = self.get_transcript_content(item["id"])
            if not content_html:
                continue

            parsed = parse_html_transcript(content_html)
            if not parsed or parsed.get("word_count", 0) < 500:
                continue

            quarter = self._extract_quarter(item.get("title", ""), date)
            url = f"{SA_BASE}/article/{item['id']}"

            record = normalize_transcript(
                raw_data=parsed,
                ticker=ticker,
                company_name=company_name,
                date=date,
                quarter=quarter,
                source="seeking_alpha",
                url=url,
            )

            if cache and cache_key:
                cache.put(cache_key, record.__dict__)
            records.append(record)

        return records

    @staticmethod
    def _extract_quarter(title: str, date: str) -> str:
        m = QUARTER_RE.search(title)
        if m:
            return f"{m.group(1).upper()} {m.group(2)}"
        try:
            dt = datetime.datetime.strptime(date, "%Y-%m-%d")
            q = (dt.month - 1) // 3 + 1
            return f"Q{q} {dt.year}"
        except ValueError:
            return ""
