import hashlib
import json
import os
from typing import Optional


class TranscriptCache:
    def __init__(self, cache_dir: str = "data/cache"):
        self.transcript_dir = os.path.join(cache_dir, "transcripts")
        self.embedding_dir = os.path.join(cache_dir, "embeddings")
        os.makedirs(self.transcript_dir, exist_ok=True)
        os.makedirs(self.embedding_dir, exist_ok=True)
        self._index: set[str] = set()
        self._build_index()

    def _build_index(self):
        for fname in os.listdir(self.transcript_dir):
            if fname.endswith(".json"):
                self._index.add(fname[:-5])

    @staticmethod
    def make_key(ticker: str, date: str, source: str) -> str:
        raw = f"{ticker.upper()}_{date}_{source}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def exists(self, key: str) -> bool:
        return key in self._index

    def get(self, key: str) -> Optional[dict]:
        path = os.path.join(self.transcript_dir, f"{key}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def put(self, key: str, record: dict):
        path = os.path.join(self.transcript_dir, f"{key}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        self._index.add(key)

    def embedding_path(self, ticker: str, date: str) -> str:
        return os.path.join(self.embedding_dir, f"{ticker.upper()}_{date}.npy")

    def list_prior_embeddings(self, ticker: str, current_date: str, max_quarters: int = 4) -> list[str]:
        """Return up to max_quarters embedding .npy paths prior to current_date."""
        prefix = f"{ticker.upper()}_"
        candidates = []
        for fname in os.listdir(self.embedding_dir):
            if fname.startswith(prefix) and fname.endswith(".npy"):
                date = fname[len(prefix):-4]
                if date < current_date:
                    candidates.append((date, os.path.join(self.embedding_dir, fname)))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [path for _, path in candidates[:max_quarters]]
