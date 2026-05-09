import datetime
from src.collectors.base import TranscriptRecord
from src.parsers.html_parser import _classify_role

SENIORITY_ORDER = ["CEO", "President", "CFO", "COO", "CTO", "EVP", "SVP", "VP", "Other", "Analyst", "Operator", "Unknown"]


def normalize_transcript(
    raw_data: dict,
    ticker: str,
    company_name: str,
    date: str,
    quarter: str,
    source: str,
    url: str,
) -> TranscriptRecord:
    """Convert parsed transcript dict into a canonical TranscriptRecord."""
    utterances = raw_data.get("utterances", [])
    speakers = raw_data.get("speakers", [])

    # Re-classify speaker roles if not already done
    for sp in speakers:
        if sp.get("role") in (None, "", "Unknown"):
            sp["role"] = _classify_role(sp.get("title", ""))

    # For utterances with speakers, enrich role
    for utt in utterances:
        sp = utt.get("speaker", {})
        if sp.get("role") in (None, "", "Unknown"):
            sp["role"] = _classify_role(sp.get("title", ""))

    raw_text = raw_data.get("raw_text", "")
    sections = raw_data.get("sections", {"prepared_remarks": [], "qa": []})

    return TranscriptRecord(
        ticker=ticker.upper(),
        company_name=company_name,
        date=date,
        source=source,
        quarter=quarter,
        raw_text=raw_text,
        speakers=speakers,
        sections=sections,
        url=url,
        fetched_at=datetime.datetime.utcnow().isoformat(),
        word_count=raw_data.get("word_count", len(raw_text.split())),
        parse_status="ok",
    )


def get_utterances_by_role(record: TranscriptRecord, roles: list[str]) -> list[dict]:
    """Return utterances from speakers matching given roles."""
    # Rebuild utterances from sections if not stored directly
    result = []
    for utt in _iter_utterances(record):
        if utt.get("speaker", {}).get("role") in roles:
            result.append(utt)
    return result


def _iter_utterances(record: TranscriptRecord):
    """Yield utterance dicts; works even if sections is text-only."""
    sections = record.sections or {}
    for section_name, items in sections.items():
        for item in items:
            if isinstance(item, dict):
                yield item
            elif isinstance(item, str):
                yield {"speaker": {"name": "Unknown", "title": "", "role": "Unknown"}, "text": item, "section": section_name}
