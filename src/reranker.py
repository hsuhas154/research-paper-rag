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

from typing import List, Optional

from src.vector_store import RetrievedChunk

# ~22M parameters, trained on MS MARCO passage ranking. Small enough to
# sit in VRAM next to both the bi-encoder and an 8B LLM on an 8GB card,
# which is the binding constraint on this project's model choices.
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Re-scores retrieved chunks against the query with a cross-encoder.

    Scores are raw model logits, not probabilities: they are useful for
    ordering candidates against each other but are not comparable to the
    cosine similarities produced by dense retrieval, and are not
    calibrated to any absolute scale.
    """

    def __init__(self, model_name: str = RERANKER_MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
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
