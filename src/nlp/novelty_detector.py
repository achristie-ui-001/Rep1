from dataclasses import dataclass

import numpy as np


@dataclass
class NoveltyResult:
    score: float        # 0.0 = all seen before, 1.0 = completely new
    prior_count: int    # how many prior quarters were available
    is_cold_start: bool


class NoveltyDetector:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", default_score: float = 0.65):
        self._model = None
        self._model_name = model_name
        self.default_score = default_score

    def _load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        except Exception:
            self._model = None

    def compute(self, signal_sentences: list[str], prior_embedding_paths: list[str]) -> NoveltyResult:
        if not signal_sentences:
            return NoveltyResult(score=self.default_score, prior_count=len(prior_embedding_paths), is_cold_start=True)

        if not prior_embedding_paths:
            return NoveltyResult(score=self.default_score, prior_count=0, is_cold_start=True)

        self._load()
        if self._model is None:
            return NoveltyResult(score=self.default_score, prior_count=len(prior_embedding_paths), is_cold_start=True)

        try:
            current_embeddings = self._model.encode(signal_sentences, convert_to_numpy=True, show_progress_bar=False)
        except Exception:
            return NoveltyResult(score=self.default_score, prior_count=len(prior_embedding_paths), is_cold_start=True)

        prior_embeddings_list = []
        for path in prior_embedding_paths:
            try:
                emb = np.load(path)
                prior_embeddings_list.append(emb)
            except Exception:
                continue

        if not prior_embeddings_list:
            return NoveltyResult(score=self.default_score, prior_count=0, is_cold_start=True)

        prior_embeddings = np.vstack(prior_embeddings_list)

        # Normalize
        cur_norm = current_embeddings / (np.linalg.norm(current_embeddings, axis=1, keepdims=True) + 1e-8)
        pri_norm = prior_embeddings / (np.linalg.norm(prior_embeddings, axis=1, keepdims=True) + 1e-8)

        # Cosine similarity matrix: (N_current, N_prior)
        sim_matrix = cur_norm @ pri_norm.T
        max_sims = sim_matrix.max(axis=1)  # for each current sentence, best match in prior
        novelty = float(1.0 - np.mean(max_sims))
        novelty = max(0.0, min(1.0, novelty))

        return NoveltyResult(score=novelty, prior_count=len(prior_embedding_paths), is_cold_start=False)

    def save_embeddings(self, sentences: list[str], path: str):
        if not sentences:
            return
        self._load()
        if self._model is None:
            return
        try:
            embeddings = self._model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
            np.save(path, embeddings)
        except Exception:
            pass
