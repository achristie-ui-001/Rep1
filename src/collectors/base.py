from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TranscriptRecord:
    ticker: str
    company_name: str
    date: str               # ISO format YYYY-MM-DD
    source: str             # "edgar" | "alphasense" | "seeking_alpha" | "ir"
    quarter: str            # e.g. "Q1 2024"
    raw_text: str
    speakers: list          # [{name, title, role}]
    sections: dict          # {"prepared_remarks": [...], "qa": [...]}
    url: str
    fetched_at: str
    word_count: int = 0
    parse_status: str = "ok"
    error_message: str = ""

    def __post_init__(self):
        if self.word_count == 0 and self.raw_text:
            self.word_count = len(self.raw_text.split())


class BaseCollector:
    source_name: str = "base"

    def collect(self, ticker: str, company_name: str, **kwargs) -> list[TranscriptRecord]:
        raise NotImplementedError

    def _make_empty_record(self, ticker: str, company_name: str, url: str, error: str) -> TranscriptRecord:
        import datetime
        return TranscriptRecord(
            ticker=ticker,
            company_name=company_name,
            date="",
            source=self.source_name,
            quarter="",
            raw_text="",
            speakers=[],
            sections={},
            url=url,
            fetched_at=datetime.datetime.utcnow().isoformat(),
            parse_status="failed",
            error_message=error,
        )
