"""
AlphaSense browser automation collector using Playwright.

IMPORTANT: AlphaSense's Terms of Service prohibit automated scraping and
bulk downloading of content. This module is intended for personal research
use ONLY on content the user is already licensed to access. It connects to
an existing authenticated browser session (uses stored cookies) rather than
handling credentials programmatically. Use judiciously with the configured
per-session transcript cap.

Do NOT distribute this module or content collected via it.
"""
import asyncio
import datetime
import re
import time
from typing import Optional

from src.collectors.base import BaseCollector, TranscriptRecord
from src.utils.logger import get_logger

logger = get_logger("inflection_detector.alphasense")

AS_BASE = "https://app.alpha-sense.com"
QUARTER_RE = re.compile(r"\b(Q[1-4])\s+(\d{4})\b", re.IGNORECASE)


class AlphaSenseCollector(BaseCollector):
    source_name = "alphasense"

    def __init__(self, config: dict):
        self.user_data_dir = config.get("user_data_dir", "")
        self.headless = config.get("headless", True)
        self.delay = config.get("delay_between_requests_s", 4)
        self.max_transcripts = config.get("max_transcripts_per_session", 50)
        self._session_count = 0

    def collect(self, ticker: str, company_name: str, start_date: str, end_date: str, cache=None) -> list[TranscriptRecord]:
        if self._session_count >= self.max_transcripts:
            logger.warning("AlphaSense session cap reached — skipping further fetches")
            return []
        try:
            return asyncio.run(self._collect_async(ticker, company_name, start_date, end_date, cache))
        except Exception as e:
            logger.error(f"AlphaSense collection failed for {ticker}: {e}")
            return []

    async def _collect_async(self, ticker: str, company_name: str, start_date: str, end_date: str, cache) -> list[TranscriptRecord]:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.error("Playwright not installed. Run: pip install playwright && playwright install chromium")
            return []

        from src.parsers.html_parser import parse_html_transcript
        from src.parsers.transcript_normalizer import normalize_transcript

        records = []

        async with async_playwright() as p:
            try:
                if self.user_data_dir:
                    ctx = await p.chromium.launch_persistent_context(
                        self.user_data_dir,
                        headless=self.headless,
                        args=["--disable-blink-features=AutomationControlled"],
                    )
                    page = await ctx.new_page()
                else:
                    browser = await p.chromium.launch(headless=self.headless)
                    ctx = await browser.new_context()
                    page = await ctx.new_page()
            except Exception as e:
                logger.error(f"Playwright browser launch failed: {e}")
                return []

            # Verify login
            try:
                await page.goto(AS_BASE, timeout=30000)
                await page.wait_for_load_state("networkidle", timeout=15000)
                if "login" in page.url.lower() or "sign-in" in page.url.lower():
                    logger.warning("AlphaSense: not logged in. Please log in manually first.")
                    await ctx.close()
                    return []
            except Exception as e:
                logger.error(f"AlphaSense navigation failed: {e}")
                await ctx.close()
                return []

            # Search for transcripts
            search_url = f"{AS_BASE}/search?q={company_name}+earnings+call&doc_type=earnings_call_transcript"
            try:
                await page.goto(search_url, timeout=30000)
                await page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                await ctx.close()
                return []

            await asyncio.sleep(self.delay)

            # Try to find result links
            links = await page.query_selector_all("a[href*='transcript'], a[href*='earnings']")
            transcript_urls = []
            for link in links[:10]:
                href = await link.get_attribute("href")
                if href and not href.startswith("http"):
                    href = AS_BASE + href
                if href:
                    transcript_urls.append(href)

            for t_url in transcript_urls[:5]:
                if self._session_count >= self.max_transcripts:
                    break
                try:
                    await page.goto(t_url, timeout=30000)
                    await page.wait_for_load_state("networkidle", timeout=15000)
                    await asyncio.sleep(self.delay)

                    # Try multiple selectors for transcript content
                    content = ""
                    for selector in [".document-viewer", ".transcript-content", ".article-content", "main", "article"]:
                        el = await page.query_selector(selector)
                        if el:
                            content = await el.inner_html()
                            if len(content) > 1000:
                                break

                    if not content:
                        continue

                    parsed = parse_html_transcript(content)
                    if not parsed or parsed.get("word_count", 0) < 500:
                        continue

                    # Try to extract date from page
                    date = self._extract_date_from_page(await page.title())
                    if not date or not (start_date <= date <= end_date):
                        continue

                    cache_key = None
                    if cache:
                        from src.utils.cache import TranscriptCache
                        cache_key = cache.make_key(ticker, date, "alphasense")
                        if cache.exists(cache_key):
                            cached = cache.get(cache_key)
                            if cached and cached.get("parse_status") == "ok":
                                records.append(TranscriptRecord(**cached))
                                continue

                    quarter = self._date_to_quarter(date)
                    record = normalize_transcript(
                        raw_data=parsed,
                        ticker=ticker,
                        company_name=company_name,
                        date=date,
                        quarter=quarter,
                        source="alphasense",
                        url=t_url,
                    )

                    if cache and cache_key:
                        cache.put(cache_key, record.__dict__)
                    records.append(record)
                    self._session_count += 1

                except Exception as e:
                    logger.warning(f"AlphaSense transcript fetch failed {t_url}: {e}")
                    continue

            await ctx.close()
        return records

    @staticmethod
    def _extract_date_from_page(title: str) -> str:
        date_re = re.search(r"(\d{4})[/-](\d{2})[/-](\d{2})", title or "")
        if date_re:
            return f"{date_re.group(1)}-{date_re.group(2)}-{date_re.group(3)}"
        return datetime.date.today().isoformat()

    @staticmethod
    def _date_to_quarter(date: str) -> str:
        try:
            dt = datetime.datetime.strptime(date, "%Y-%m-%d")
            q = (dt.month - 1) // 3 + 1
            return f"Q{q} {dt.year}"
        except ValueError:
            return ""
