"""
reranker.py

Cross-encoder re-ranking of retrieved candidates.

The bi-encoder used for retrieval (embedder.py) encodes the question
and each chunk *independently* and compares the two vectors. That is
what makes it fast enough to search a whole corpus - every chunk
embedding is computed once, offline - but it also means the model
never sees a question and a chunk at the same time, so it cannot
reason about how they relate. It only measures whether they are about
similar things.

A cross-encoder sees the pair jointly: the question and the chunk go
through the transformer together, so attention runs across both, and
the model scores actual relevance rather than topical similarity. This
is far more accurate and far too slow to run over a whole corpus - one
forward pass per candidate - which is exactly why it belongs here, as
a second stage over the ~20 candidates the fast retrievers proposed
rather than over the thousands of chunks in the index.

The model is loaded lazily on first use, so an app that never turns
re-ranking on never pays the load cost, and a machine that cannot
reach the model hub can still run dense and hybrid retrieval.
"""

import os
from pathlib import Path
from typing import List, Optional

from src.vector_store import RetrievedChunk

# ~22M parameters, trained on MS MARCO passage ranking. Small enough to
# sit in VRAM next to both the bi-encoder and an 8B LLM on an 8GB card,
# which is the binding constraint on this project's model choices.
STOCK_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Re-measured on the 50-paper corpus: re-ranking lifts Hit@5 from 0.735 to
# 0.882, so it is load-bearing rather than a refinement. At 16 papers the
# first stage was already saturated at 0.974 and re-ranking looked
# marginal; that earlier reading was an artifact of corpus size.

# A re-ranker fine-tuned on this corpus, written by scripts.tune_reranker.
#
# This IS adopted automatically when present, which reverses an earlier
# decision. The first attempt at domain adaptation was rejected: trained
# on 565 examples from 12 papers with no validation split, it scored
# 1.000 on papers it had trained on and 0.818 on papers it had not, and
# auto-adopting it would have silently shipped a worse re-ranker.
#
# Repeated properly it wins. With 2,264 examples from the 30-paper train
# split, hyperparameters chosen on validation, and test read once after
# the choice was locked, the selected model beats stock on the held-out
# test split: Hit@5 0.803 against 0.737 and MRR 0.655 against 0.625, at
# the same 0.064 s per query. Four times the data, twenty extra unseen
# papers, and a protocol that cannot fit the choice to the score.
#
# A fresh checkout has no models/ directory and falls back to stock, so
# this costs nothing until someone trains one. Train on a different
# corpus and the gain is not guaranteed; measure before trusting it.
DOMAIN_RERANKER_PATH = Path("models/reranker-domain")


def default_reranker_model() -> str:
    """
    The re-ranker to use unless one is named explicitly.

    Resolution order: the RERANKER_MODEL environment variable, then a
    locally fine-tuned model if one has been trained, then the stock
    MS MARCO cross-encoder.
    """
    override = os.environ.get("RERANKER_MODEL", "").strip()
    if override:
        return override
    if (DOMAIN_RERANKER_PATH / "config.json").exists():
        return str(DOMAIN_RERANKER_PATH)
    return STOCK_RERANKER_MODEL


# Kept as a module-level name for callers that import it directly.
RERANKER_MODEL_NAME = STOCK_RERANKER_MODEL


class CrossEncoderReranker:
    """
    Re-scores retrieved chunks against the query with a cross-encoder.

    Scores are raw model logits, not probabilities: they are useful for
    ordering candidates against each other but are not comparable to the
    cosine similarities produced by dense retrieval, and are not
    calibrated to any absolute scale.
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or default_reranker_model()
        self.device = device
        self._model = None  # loaded on first rerank() call

    @property
    def model(self):
        """Loads the cross-encoder on first access."""
        if self._model is None:
            import torch
            from sentence_transformers import CrossEncoder

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = CrossEncoder(self.model_name, device=device)
            self.device = device
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: Optional[int] = None,
        batch_size: int = 16,
    ) -> List[RetrievedChunk]:
        """
        Returns the candidates re-ordered by cross-encoder relevance,
        truncated to top_k if given.

        The returned RetrievedChunk objects carry the cross-encoder
        score in `.score`, replacing whatever the first-stage retriever
        put there.
        """
        if not candidates:
            return []

        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

        rescored = [
            RetrievedChunk(chunk=candidate.chunk, score=float(score))
            for candidate, score in zip(candidates, scores)
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)

        return rescored[:top_k] if top_k is not None else rescored
