import re
from dataclasses import dataclass
from typing import Optional

import yaml


@dataclass
class SignalMatch:
    category: str
    pattern: str
    matched_text: str
    sentence: str
    weight: float
    char_offset: int
    speaker_name: str = "Unknown"
    speaker_role: str = "Unknown"
    section: str = "prepared_remarks"


def load_patterns(pattern_file: str = "config/signal_patterns.yaml") -> dict:
    with open(pattern_file, "r") as f:
        return yaml.safe_load(f)


def _compile_patterns(patterns_data: dict) -> dict:
    """Pre-compile all regex patterns. Returns {category: [(regex, weight, pattern_str)]}."""
    compiled = {}
    negative = []
    for category, tiers in patterns_data.items():
        if category == "negative_signals":
            for item in tiers:
                negative.append((re.compile(item["pattern"], re.IGNORECASE), item["weight"]))
            continue
        cat_patterns = []
        for tier_name in ("strong", "medium", "contextual"):
            tier_items = tiers.get(tier_name, [])
            for item in tier_items:
                p = item["pattern"]
                w = item["weight"]
                cat_patterns.append((re.compile(p, re.IGNORECASE | re.MULTILINE), w, p))
        compiled[category] = cat_patterns
    return compiled, negative


def _split_sentences(text: str) -> list[tuple[int, str]]:
    """Return list of (char_offset, sentence)."""
    sentence_re = re.compile(r"(?<=[.!?])\s+")
    parts = sentence_re.split(text)
    results = []
    offset = 0
    for part in parts:
        results.append((offset, part.strip()))
        offset += len(part) + 1
    return results


class KeywordMatcher:
    def __init__(self, pattern_file: str = "config/signal_patterns.yaml"):
        raw = load_patterns(pattern_file)
        self._compiled, self._negative = _compile_patterns(raw)

    def match(self, record) -> list[SignalMatch]:
        """Run keyword matching on a TranscriptRecord. Returns all matches."""
        all_matches = []
        raw_text = record.raw_text or ""
        sentences = _split_sentences(raw_text)

        # Build a sentence→speaker lookup from utterances (best effort)
        utterance_map = self._build_utterance_map(record)

        for category, patterns in self._compiled.items():
            for regex, weight, pattern_str in patterns:
                for m in regex.finditer(raw_text):
                    char_off = m.start()
                    sentence = _find_sentence_at(sentences, char_off)
                    neg_penalty = self._negative_penalty(sentence)
                    if neg_penalty < 0:
                        weight_adjusted = weight + neg_penalty
                    else:
                        weight_adjusted = weight

                    if weight_adjusted <= 0:
                        continue

                    speaker_info = utterance_map.get(char_off, {})
                    section = self._detect_section(char_off, record)

                    all_matches.append(SignalMatch(
                        category=category,
                        pattern=pattern_str,
                        matched_text=m.group(0),
                        sentence=sentence,
                        weight=weight_adjusted,
                        char_offset=char_off,
                        speaker_name=speaker_info.get("name", "Unknown"),
                        speaker_role=speaker_info.get("role", "Unknown"),
                        section=section,
                    ))

        return all_matches

    def _negative_penalty(self, sentence: str) -> float:
        total = 0.0
        for regex, w in self._negative:
            if regex.search(sentence):
                total += w
        return total

    # Pattern: "Name - Title:" or "Name - Title\n" speaker labels in raw text
    _SPEAKER_LABEL_RE = re.compile(
        r"^([A-Z][A-Za-z\s\-\.\']{2,40}?)\s*[-–]\s*([A-Za-z][^:\n]{0,80}?)[\s]*(?::\s*|\n)",
        re.MULTILINE,
    )

    def _build_utterance_map(self, record) -> dict:
        """Map char ranges to speaker info (approximate)."""
        result = {}
        raw_text = record.raw_text or ""
        sections = record.sections or {}

        # First pass: structured utterance dicts
        for section_name, items in sections.items():
            for item in items:
                if isinstance(item, dict):
                    text = item.get("text", "")
                    speaker = item.get("speaker", {})
                    idx = raw_text.find(text)
                    if idx >= 0:
                        for i in range(idx, idx + len(text)):
                            result[i] = speaker

        # Fallback: regex-based speaker label detection in raw text
        if not result:
            from src.parsers.html_parser import _classify_role
            current_speaker = {"name": "Unknown", "title": "", "role": "Unknown"}
            current_pos = 0
            for m in self._SPEAKER_LABEL_RE.finditer(raw_text):
                name = m.group(1).strip()
                title = (m.group(2) or "").strip()
                # Fill previous speaker's range
                for i in range(current_pos, m.start()):
                    if i not in result:
                        result[i] = current_speaker
                current_speaker = {"name": name, "title": title, "role": _classify_role(title)}
                current_pos = m.end()
            # Fill remainder
            for i in range(current_pos, len(raw_text)):
                if i not in result:
                    result[i] = current_speaker

        return result

    def _detect_section(self, char_off: int, record) -> str:
        raw_text = record.raw_text or ""
        sections = record.sections or {}
        pr_texts = sections.get("prepared_remarks", [])
        pr_text = " ".join(t if isinstance(t, str) else t.get("text", "") for t in pr_texts)
        if char_off < len(pr_text) + 500:
            return "prepared_remarks"
        return "qa"


def _find_sentence_at(sentences: list, char_off: int) -> str:
    best = ""
    for offset, sentence in sentences:
        if offset <= char_off:
            best = sentence
        else:
            break
    return best
