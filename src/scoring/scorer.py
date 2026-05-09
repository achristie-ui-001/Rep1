import re
from dataclasses import dataclass, field
from typing import Optional

from src.nlp.pass1_keywords import SignalMatch
from src.nlp.pass2_ner import NERResult
from src.nlp.pass3_classifier import ClassifierResult
from src.nlp.novelty_detector import NoveltyResult

ROLE_WEIGHTS = {
    "CEO": 1.0, "President": 1.0, "CFO": 0.85, "COO": 0.80, "CTO": 0.80,
    "EVP": 0.70, "SVP": 0.65, "VP": 0.55, "Other": 0.35,
    "Analyst": 0.15, "Operator": 0.05, "Unknown": 0.20,
}

GUIDANCE_PATTERNS = re.compile(
    r"\b(expect[s]?|anticipate[s]?|project[s]?|guid(e|ing|ance)|target[s]?|forecast[s]?)\b.{0,80}"
    r"\b(accelerat|significan|material|step[- ]change|outsized|dramatic|substantial)\b",
    re.IGNORECASE,
)

CATEGORY_DISPLAY = {
    "technology_adoption": "Technology Adoption",
    "regulatory": "Regulatory Tailwind",
    "market_expansion": "Market Expansion",
    "behavior_change": "Behavior Change",
    "operational": "Operational Transformation",
    "cross_industry": "Cross-Industry Adoption",
}


@dataclass
class InflectionSignal:
    ticker: str
    company_name: str
    sector: str
    date: str
    quarter: str
    source: str
    inflection_type: str
    tier: str
    score: float
    key_quote: str
    speaker_name: str
    speaker_role: str
    novelty_score: float
    has_dollar_qty: bool
    analyst_q_count: int
    transcript_url: str
    # Component scores for transparency
    keyword_score: float = 0.0
    frequency_score: float = 0.0
    seniority_score: float = 0.0
    guidance_score: float = 0.0
    qa_score: float = 0.0


class Scorer:
    def __init__(self, config: dict):
        w = config.get("weights", {})
        self.w_keyword = w.get("keyword", 0.20)
        self.w_freq = w.get("frequency", 0.10)
        self.w_seniority = w.get("seniority", 0.20)
        self.w_novelty = w.get("novelty", 0.25)
        self.w_dollar = w.get("dollar", 0.10)
        self.w_guidance = w.get("guidance", 0.10)
        self.w_qa = w.get("qa", 0.05)
        self.neg_penalty = config.get("negative_class_penalty", 0.40)
        tiers = config.get("tiers", {})
        self.tier1_min = tiers.get("tier1_min", 70)
        self.tier2_min = tiers.get("tier2_min", 45)
        self.tier3_min = tiers.get("tier3_min", 25)

    def score(
        self,
        record,
        keyword_matches: list[SignalMatch],
        ner_result: NERResult,
        classifier_result: Optional[ClassifierResult],
        novelty_result: NoveltyResult,
        sector: str = "",
    ) -> list[InflectionSignal]:
        if not keyword_matches:
            return []

        # Group matches by category
        by_category: dict[str, list[SignalMatch]] = {}
        for m in keyword_matches:
            by_category.setdefault(m.category, []).append(m)

        signals = []
        for category, matches in by_category.items():
            signal = self._score_category(
                record, category, matches, ner_result, classifier_result, novelty_result, sector
            )
            if signal:
                signals.append(signal)

        return signals

    def _score_category(
        self,
        record,
        category: str,
        matches: list[SignalMatch],
        ner_result: NERResult,
        classifier_result: Optional[ClassifierResult],
        novelty_result: NoveltyResult,
        sector: str,
    ) -> Optional[InflectionSignal]:
        # A: Keyword category score
        total_weight = sum(m.weight for m in matches)
        keyword_score = min(1.0, total_weight / 10.0)

        # B: Mention frequency
        freq_score = min(1.0, len(matches) / 15.0)

        # C: Speaker seniority — use highest-seniority speaker among matches
        seniority_score = max(ROLE_WEIGHTS.get(m.speaker_role, 0.20) for m in matches)

        # D: Novelty
        novelty_score = novelty_result.score if novelty_result else 0.65

        # E: Dollar quantification
        dollar_score = 1.0 if ner_result.has_dollar_quantification else 0.0

        # F: Guidance language
        prepared_text = " ".join(record.sections.get("prepared_remarks", []) if isinstance(record.sections.get("prepared_remarks", []), list) else [])
        prepared_text = " ".join(t if isinstance(t, str) else t.get("text", "") for t in record.sections.get("prepared_remarks", []))
        guidance_hits = len(GUIDANCE_PATTERNS.findall(prepared_text[-max(1, len(prepared_text)//5):]))
        guidance_score = min(1.0, guidance_hits / 5.0)

        # G: Analyst Q&A focus
        qa_items = record.sections.get("qa", [])
        analyst_q_count = self._count_analyst_questions(qa_items, matches)
        qa_score = min(1.0, analyst_q_count / 4.0)

        raw_score = (
            keyword_score * self.w_keyword
            + freq_score * self.w_freq
            + seniority_score * self.w_seniority
            + novelty_score * self.w_novelty
            + dollar_score * self.w_dollar
            + guidance_score * self.w_guidance
            + qa_score * self.w_qa
        ) * 100

        # Zero-shot validation penalty
        if classifier_result and classifier_result.is_negative:
            final_score = raw_score * self.neg_penalty
        else:
            final_score = raw_score

        final_score = round(final_score, 1)
        tier = self._assign_tier(final_score)

        # Pick the best key quote
        best_match = max(matches, key=lambda m: (ROLE_WEIGHTS.get(m.speaker_role, 0), m.weight))
        key_quote = best_match.sentence[:400]

        return InflectionSignal(
            ticker=record.ticker,
            company_name=record.company_name,
            sector=sector,
            date=record.date,
            quarter=record.quarter,
            source=record.source,
            inflection_type=CATEGORY_DISPLAY.get(category, category),
            tier=tier,
            score=final_score,
            key_quote=key_quote,
            speaker_name=best_match.speaker_name,
            speaker_role=best_match.speaker_role,
            novelty_score=round(novelty_score, 3),
            has_dollar_qty=ner_result.has_dollar_quantification,
            analyst_q_count=analyst_q_count,
            transcript_url=record.url,
            keyword_score=round(keyword_score, 3),
            frequency_score=round(freq_score, 3),
            seniority_score=round(seniority_score, 3),
            guidance_score=round(guidance_score, 3),
            qa_score=round(qa_score, 3),
        )

    def _assign_tier(self, score: float) -> str:
        if score >= self.tier1_min:
            return "Tier 1"
        if score >= self.tier2_min:
            return "Tier 2"
        if score >= self.tier3_min:
            return "Tier 3"
        return "Filtered"

    def _count_analyst_questions(self, qa_items: list, matches: list[SignalMatch]) -> int:
        match_keywords = set()
        for m in matches:
            for word in m.matched_text.lower().split():
                if len(word) > 4:
                    match_keywords.add(word)

        analyst_count = 0
        for item in qa_items:
            text = item if isinstance(item, str) else item.get("text", "")
            speaker = item.get("speaker", {}) if isinstance(item, dict) else {}
            role = speaker.get("role", "Unknown")
            if role == "Analyst":
                text_lower = text.lower()
                if any(kw in text_lower for kw in match_keywords):
                    analyst_count += 1
        return analyst_count
