import re
from dataclasses import dataclass, field
from typing import Optional

MONEY_RE = re.compile(r"\$[\d,.]+\s*(billion|million|trillion|B|M|T)\b|\b[\d,.]+\s*(billion|million|trillion)\s*(dollar|USD)", re.IGNORECASE)
PERCENT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(%|percent|basis points?|bps)\b", re.IGNORECASE)


@dataclass
class NERResult:
    orgs: list = field(default_factory=list)
    products: list = field(default_factory=list)
    money_mentions: list = field(default_factory=list)
    geographies: list = field(default_factory=list)
    percentages: list = field(default_factory=list)
    has_dollar_quantification: bool = False


class NERExtractor:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self._nlp = None
        self._model_name = spacy_model

    def _load(self):
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load(self._model_name)
            except OSError:
                # Try smaller fallback
                try:
                    import spacy
                    self._nlp = spacy.load("en_core_web_sm")
                except OSError:
                    self._nlp = None

    def extract(self, text: str, signal_sentences: list[str] = None) -> NERResult:
        self._load()
        result = NERResult()

        # Regex-based extractions (always available, no model needed)
        result.money_mentions = MONEY_RE.findall(text)
        result.has_dollar_quantification = bool(result.money_mentions)
        result.percentages = PERCENT_RE.findall(text)

        if self._nlp is None:
            return result

        # Run spaCy NER on signal sentences (faster than full text)
        target_text = " ".join(signal_sentences) if signal_sentences else text[:50000]
        try:
            doc = self._nlp(target_text[:100000])  # cap at 100k chars
        except Exception:
            return result

        for ent in doc.ents:
            if ent.label_ == "ORG" and ent.text not in result.orgs:
                result.orgs.append(ent.text)
            elif ent.label_ == "PRODUCT" and ent.text not in result.products:
                result.products.append(ent.text)
            elif ent.label_ in ("GPE", "LOC") and ent.text not in result.geographies:
                result.geographies.append(ent.text)
            elif ent.label_ == "MONEY":
                result.money_mentions.append(ent.text)
                result.has_dollar_quantification = True

        return result
