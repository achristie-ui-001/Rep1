from dataclasses import dataclass

CANDIDATE_LABELS = [
    "technology paradigm shift",
    "regulatory approval or mandate",
    "market size expansion",
    "consumer behavior change",
    "operational efficiency improvement",
    "unexpected new customer segment",
    "business as usual",
    "temporary or one-time event",
]

NEGATIVE_LABELS = {"business as usual", "temporary or one-time event"}

LABEL_TO_CATEGORY = {
    "technology paradigm shift": "technology_adoption",
    "regulatory approval or mandate": "regulatory",
    "market size expansion": "market_expansion",
    "consumer behavior change": "behavior_change",
    "operational efficiency improvement": "operational",
    "unexpected new customer segment": "cross_industry",
}


@dataclass
class ClassifierResult:
    top_label: str
    top_score: float
    is_negative: bool
    label_scores: dict


class ZeroShotClassifier:
    def __init__(self, model_name: str = "facebook/bart-large-mnli", threshold: float = 0.45, skip: bool = False):
        self._pipeline = None
        self._model_name = model_name
        self.threshold = threshold
        self.skip = skip

    def _load(self):
        if self._pipeline is not None or self.skip:
            return
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self._model_name,
                device=-1,  # CPU
            )
        except Exception as e:
            self._pipeline = None

    def classify_sentences(self, sentences: list[str], batch_size: int = 8) -> list[ClassifierResult]:
        if self.skip or not sentences:
            return [self._default_result() for _ in sentences]

        self._load()
        if self._pipeline is None:
            return [self._default_result() for _ in sentences]

        results = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i: i + batch_size]
            try:
                outputs = self._pipeline(batch, CANDIDATE_LABELS, multi_label=False)
                if isinstance(outputs, dict):
                    outputs = [outputs]
                for out in outputs:
                    top_label = out["labels"][0]
                    top_score = out["scores"][0]
                    label_scores = dict(zip(out["labels"], out["scores"]))
                    results.append(ClassifierResult(
                        top_label=top_label,
                        top_score=top_score,
                        is_negative=top_label in NEGATIVE_LABELS,
                        label_scores=label_scores,
                    ))
            except Exception:
                results.extend([self._default_result() for _ in batch])
        return results

    def classify_single(self, sentence: str) -> ClassifierResult:
        results = self.classify_sentences([sentence])
        return results[0]

    @staticmethod
    def _default_result() -> ClassifierResult:
        return ClassifierResult(
            top_label="technology paradigm shift",
            top_score=0.5,
            is_negative=False,
            label_scores={},
        )
