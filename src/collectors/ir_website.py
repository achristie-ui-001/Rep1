"""
Generic Investor Relations website collector using Playwright.
Probes common IR URL patterns and extracts transcript links.
"""
import asyncio
import datetime
import re
from typing import Optional

from src.collectors.base import BaseCollector, TranscriptRecord
from src.utils.logger import get_logger
from src.utils.rate_limiter import RateLimiter

logger = get_logger("inflection_detector.ir_website")

TRANSCRIPT_LINK_RE = re.compile(
    r"(transcript|earnings.call|quarterly.result|investor.day|conference.call|q[1-4].20\d{2})",
    re.IGNORECASE,
)

IR_URL_PATTERNS = [
    "https://ir.{domain}/events-and-presentations",
    "https://ir.{domain}/financial-information/quarterly-results",
    "https://investor.{domain}/news-releases",
    "https://investors.{domain}/financial-information",
    "https://{domain}/investors",
    "https://{domain}/investor-relations",
]


class IRWebsiteCollector(BaseCollector):
    source_name = "ir"

    def __init__(self, config: dict):
        self.delay = config.get("delay_between_requests_s", 3)
        self.timeout_s = config.get("timeout_s", 30)
        self.rate_limiter = RateLimiter(1 / max(self.delay, 1))

    def collect(self, ticker: str, company_name: str, start_date: str, end_date: str,
                cache=None, ir_url: str = "") -> list[TranscriptRecord]:
        try:
            return asyncio.run(self._collect_async(ticker, company_name, start_date, end_date, cache, ir_url))
        except Exception as e:
            logger.error(f"IR website collection failed for {ticker}: {e}")
            return []

    async def _collect_async(self, ticker: str, company_name: str, start_date: str, end_date: str,
                              cache, ir_url: str) -> list[TranscriptRecord]:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.error("Playwright not installed")
            return []

        from src.parsers.html_parser import parse_html_transcript
        from src.parsers.transcript_normalizer import normalize_transcript

        records = []

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            ctx = await browser.new_context(
                user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            )
            page = await ctx.new_page()

            # Try the provided ir_url first, then probe candidates
            urls_to_try = []
            if ir_url:
                urls_to_try.append(ir_url)

            transcript_links = []
            for try_url in urls_to_try[:3]:
                try:
                    await page.goto(try_url, timeout=self.timeout_s * 1000)
                    await page.wait_for_load_state("domcontentloaded", timeout=10000)
                    links = await self._find_transcript_links(page, try_url)
                    transcript_links.extend(links)
                    if links:
                        break
                except Exception:
                    continue

            for t_url, link_text in transcript_links[:8]:
                try:
                    await page.goto(t_url, timeout=self.timeout_s * 1000)
                    await page.wait_for_load_state("domcontentloaded", timeout=10000)
                    await asyncio.sleep(self.delay)

                    content_type = ""
                    if t_url.endswith(".pdf"):
                        continue  # PDF handling requires separate download — skip for now

                    html = await page.content()
                    parsed = parse_html_transcript(html)
                    if not parsed or parsed.get("word_count", 0) < 500:
                        continue

                    date = self._extract_date(link_text, t_url)
                    if not date or not (start_date <= date <= end_date):
                        continue

                    cache_key = None
                    if cache:
                        cache_key = cache.make_key(ticker, date, "ir")
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
                        source="ir",
                        url=t_url,
                    )

                    if cache and cache_key:
                        cache.put(cache_key, record.__dict__)
                    records.append(record)

                except Exception as e:
                    logger.warning(f"IR page failed {t_url}: {e}")

            await browser.close()
        return records

    async def _find_transcript_links(self, page, base_url: str) -> list[tuple[str, str]]:
        links = await page.query_selector_all("a")
        results = []
        for link in links:
            href = await link.get_attribute("href") or ""
            text = (await link.inner_text()).strip()
            if TRANSCRIPT_LINK_RE.search(href) or TRANSCRIPT_LINK_RE.search(text):
                if not href.startswith("http"):
                    from urllib.parse import urljoin
                    href = urljoin(base_url, href)
                results.append((href, text))
        return results

    @staticmethod
    def _extract_date(text: str, url: str) -> str:
        for src in [text, url]:
            m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", src)
            if m:
                try:
                    dt = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                    return dt.strftime("%Y-%m-%d")
                except ValueError:
                    continue
        return datetime.date.today().isoformat()

    @staticmethod
    def _date_to_quarter(date: str) -> str:
        try:
            dt = datetime.datetime.strptime(date, "%Y-%m-%d")
            q = (dt.month - 1) // 3 + 1
            return f"Q{q} {dt.year}"
        except ValueError:
            return ""
