"""
retriever.py

The retrieval stage of the pipeline: runs dense and lexical search,
fuses their rankings, and optionally re-ranks the survivors with a
cross-encoder.

The full pipeline is:

    question
       |
       +--> dense search (FAISS, cosine)  --> ranked list A
       +--> lexical search (BM25)         --> ranked list B
                    |
            Reciprocal Rank Fusion
                    |
        ~20 fused candidates
                    |
          cross-encoder re-ranking
                    |
              top-k passages

Why Reciprocal Rank Fusion rather than a weighted score blend: cosine
similarities live in [-1, 1] and cluster tightly around 0.4-0.8 in
practice, while BM25 scores are unbounded and scale with query length
and corpus statistics. There is no principled constant that makes the
two comparable, and any normalization scheme has to be re-tuned per
corpus. RRF sidesteps the problem entirely by discarding the scores and
fusing on *rank* alone:

    RRF(d) = sum over retrievers of  1 / (k + rank(d))

A chunk ranked highly by both retrievers beats one ranked highly by
only one, and the constant k (60, the value from the original Cormack
et al. 2009 paper) damps the difference between ranks 1 and 2 so a
single retriever cannot dominate on overconfidence alone.

Retrieval modes are selectable rather than hardcoded, because being
able to run the same question through dense-only, hybrid, and
hybrid+rerank is what makes the comparison in scripts/compare_retrieval.py
possible - and that comparison is the evidence that any of this is
actually an improvement.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from src.bm25 import BM25Index
from src.embedder import Embedder
from src.reranker import CrossEncoderReranker
from src.vector_store import RetrievedChunk, VectorStore

# Retrieval strategies, in increasing order of sophistication.
MODE_DENSE = "dense"
MODE_BM25 = "bm25"
MODE_HYBRID = "hybrid"
MODE_HYBRID_RERANK = "hybrid+rerank"

RETRIEVAL_MODES = [MODE_DENSE, MODE_BM25, MODE_HYBRID, MODE_HYBRID_RERANK]

# The constant from Cormack et al. (2009), "Reciprocal Rank Fusion
# outperforms Condorcet and individual Rank Learning Methods".
RRF_K = 60


@dataclass
class RetrievalResult:
    """Retrieved passages plus a record of how they were obtained."""
    chunks: List[RetrievedChunk]
    mode: str
    n_dense_candidates: int = 0
    n_lexical_candidates: int = 0
    n_fused_candidates: int = 0


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[int]],
    k: int = RRF_K,
) -> List[int]:
    """
    Fuses several ranked lists of chunk indices into one.

    Args:
        ranked_lists: each inner sequence is chunk indices, best first
        k: RRF damping constant; larger values flatten the advantage of
            the top ranks relative to lower ones

    Returns:
        Chunk indices ordered by descending fused score. Ties are broken
        by best individual rank achieved, so fusion is deterministic and
        does not depend on dictionary iteration order.
    """
    fused_scores: Dict[int, float] = {}
    best_rank: Dict[int, int] = {}

    for ranked in ranked_lists:
        for rank, chunk_index in enumerate(ranked, start=1):
            fused_scores[chunk_index] = fused_scores.get(chunk_index, 0.0) + 1.0 / (k + rank)
            best_rank[chunk_index] = min(best_rank.get(chunk_index, rank), rank)

    return sorted(
        fused_scores,
        key=lambda index: (-fused_scores[index], best_rank[index], index),
    )


# Phrases that signal a question is asking across documents rather than
# about one. Kept deliberately narrow: a false positive splits the passage
# budget on a question that needed all of it on one paper, which measurably
# costs accuracy (Hit@5 0.97 -> 0.92 when the quota is applied to every
# query), while a false negative only leaves behaviour as it was.
_COMPARATIVE_PATTERNS = re.compile(
    r"\b("
    r"compare|comparison|contrast|versus|vs\.?|"
    r"each (?:paper|study|article|author)|"
    r"which (?:papers|studies|articles)|"
    r"any (?:two|other) papers?|"
    r"both papers|across (?:the )?papers|"
    r"between (?:the )?(?:two )?papers|"
    r"do (?:they|the papers) (?:agree|disagree)|"
    r"disagree"
    r")\b",
    re.IGNORECASE,
)


def looks_comparative(question: str) -> bool:
    """
    Whether a question appears to span several documents.

    Used to decide when to spend the passage budget across papers instead
    of on the single best-matching one. Measured on this project's own
    benchmark, applying the quota unconditionally costs accuracy on
    ordinary single-paper questions, so it is switched on by this signal
    rather than left on by default.
    """
    return bool(_COMPARATIVE_PATTERNS.search(question or ""))


def diversify_by_document(
    hits: Sequence[Tuple[int, float]],
    chunks: Sequence,
    top_k: int,
    max_per_document: Optional[int],
) -> List[Tuple[int, float]]:
    """
    Picks top_k results in rank order while allowing at most
    `max_per_document` of them to come from any single document.

    Why this exists: a question spanning two papers ("compare how Adam and
    the Transformer paper describe their optimizer settings") fills every
    passage slot from whichever paper matches more strongly, and the
    generator then correctly reports that it cannot compare, because the
    other paper's evidence was never retrieved. Re-ranking cannot fix
    that - it only reorders what the first stage proposed, and the
    missing passages are not in the pool at all. The budget has to be
    shared before ranking is allowed to spend it.

    Results that overflow a document's quota are not discarded; they are
    kept in order and used to backfill if there are not enough other
    documents to fill top_k. A corpus of one paper, or a question scoped
    to one paper, therefore behaves exactly as if no quota were set.

    `max_per_document` of None disables the quota entirely.
    """
    if max_per_document is None or max_per_document < 1:
        return list(hits)[:top_k]

    selected: List[Tuple[int, float]] = []
    overflow: List[Tuple[int, float]] = []
    taken: Dict[str, int] = {}

    for index, score in hits:
        doc_id = chunks[index].doc_id
        if taken.get(doc_id, 0) < max_per_document:
            selected.append((index, score))
            taken[doc_id] = taken.get(doc_id, 0) + 1
            if len(selected) == top_k:
                return selected
        else:
            overflow.append((index, score))

    # Not enough documents to fill top_k under the quota: backfill in rank
    # order rather than returning fewer passages than asked for.
    for item in overflow:
        if len(selected) == top_k:
            break
        selected.append(item)

    return selected


class HybridRetriever:
    """
    Owns the retrieval half of the RAG pipeline: the dense index, the
    lexical index, and the re-ranker.

    The BM25 index is derived from whatever chunks the vector store
    currently holds, so it has to be resynced whenever the corpus
    changes - call sync() after adding or removing a document.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        reranker: Optional[CrossEncoderReranker] = None,
        candidate_pool: int = 30,
    ):
        self.embedder = embedder
        self.store = store
        self.reranker = reranker if reranker is not None else CrossEncoderReranker()
        self.candidate_pool = candidate_pool
        self.bm25 = BM25Index()
        self.sync()

    def sync(self) -> None:
        """Rebuilds the lexical index to match the vector store's contents."""
        self.bm25.build(self.store.chunks)

    def _allowed_indices(self, doc_ids: Optional[Sequence[str]]) -> Optional[set]:
        """Chunk positions belonging to the requested documents, or None for all."""
        if doc_ids is None:
            return None
        wanted = set(doc_ids)
        return {i for i, c in enumerate(self.store.chunks) if c.doc_id in wanted}

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        mode: str = MODE_HYBRID_RERANK,
        doc_ids: Optional[Sequence[str]] = None,
        max_per_document: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Retrieves the top_k most relevant chunks for a question.

        Args:
            question: the user's natural-language question
            top_k: how many passages to hand to the generator
            mode: one of RETRIEVAL_MODES
            doc_ids: restrict retrieval to these documents; None searches
                the whole corpus
            max_per_document: at most this many of the returned passages
                may come from any one document. None means no quota, which
                is the right default for the ordinary case of a question
                about a single paper. See diversify_by_document.

        Returns:
            A RetrievalResult holding the passages and the candidate
            counts at each stage, which the UI surfaces so the retrieval
            path stays inspectable instead of being a black box.
        """
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"Unknown retrieval mode {mode!r}; expected one of {RETRIEVAL_MODES}")
        if not self.store.chunks:
            return RetrievalResult(chunks=[], mode=mode)

        # Single-strategy modes need no fusion and no candidate pool.
        # Single-strategy modes over-retrieve when a quota is in force, so
        # the quota has candidates from other documents to promote.
        single_limit = top_k if max_per_document is None else max(top_k, self.candidate_pool)

        if mode == MODE_DENSE:
            hits = self._dense(question, single_limit, doc_ids)
            return RetrievalResult(
                chunks=self._to_chunks(self._select(hits, top_k, max_per_document)),
                mode=mode,
                n_dense_candidates=len(hits),
            )

        if mode == MODE_BM25:
            hits = self._lexical(question, single_limit, doc_ids)
            return RetrievalResult(
                chunks=self._to_chunks(self._select(hits, top_k, max_per_document)),
                mode=mode,
                n_lexical_candidates=len(hits),
            )

        # Hybrid modes: over-retrieve from both arms, then fuse. The pool
        # is wider than top_k on purpose - fusion can only promote a
        # chunk that at least one retriever surfaced, and re-ranking can
        # only rescue a chunk that fusion kept.
        #
        # 30 is measured, not guessed: over a 16-paper corpus, widening
        # the pool from 20 to 30 lifts MRR@5 from 0.861 to 0.881, and 40
        # gives nothing further while costing more cross-encoder passes.
        pool = max(self.candidate_pool, top_k)
        dense_hits = self._dense(question, pool, doc_ids)
        lexical_hits = self._lexical(question, pool, doc_ids)

        fused_indices = reciprocal_rank_fusion(
            [[i for i, _ in dense_hits], [i for i, _ in lexical_hits]]
        )

        # Carry each chunk's best first-stage score forward, so a result
        # still shows a meaningful number when re-ranking is off.
        best_score: Dict[int, float] = {}
        for index, score in dense_hits + lexical_hits:
            best_score[index] = max(best_score.get(index, float("-inf")), score)

        candidates = self._to_chunks(
            [(i, best_score[i]) for i in fused_indices[:pool]]
        )

        if mode == MODE_HYBRID:
            fused_hits = [(i, best_score[i]) for i in fused_indices[:pool]]
            return RetrievalResult(
                chunks=self._to_chunks(self._select(fused_hits, top_k, max_per_document)),
                mode=mode,
                n_dense_candidates=len(dense_hits),
                n_lexical_candidates=len(lexical_hits),
                n_fused_candidates=len(candidates),
            )

        # Re-rank the whole pool, then apply the quota to the re-ranked
        # order, so the quota picks the best passage per document rather
        # than whatever the first stage happened to rank highest.
        reranked = self.reranker.rerank(question, candidates)
        position_of = {id(c.chunk): i for i, c in enumerate(candidates)}
        reranked_hits = [
            (fused_indices[position_of[id(r.chunk)]], r.score) for r in reranked
        ]
        return RetrievalResult(
            chunks=self._to_chunks(self._select(reranked_hits, top_k, max_per_document)),
            mode=mode,
            n_dense_candidates=len(dense_hits),
            n_lexical_candidates=len(lexical_hits),
            n_fused_candidates=len(candidates),
        )

    # Both retrieval arms return (chunk index, score) pairs rather than
    # RetrievedChunk objects. Rank fusion works on positions, so keeping
    # results in index form until the very end avoids having to map
    # chunk objects back to their positions in the store.

    def _dense(
        self, question: str, limit: int, doc_ids: Optional[Sequence[str]]
    ) -> List[Tuple[int, float]]:
        query_embedding = self.embedder.encode([question])[0]
        hits = self.store.search_indexed(query_embedding, top_k=limit, doc_ids=doc_ids)
        return hits

    def _lexical(
        self, question: str, limit: int, doc_ids: Optional[Sequence[str]]
    ) -> List[Tuple[int, float]]:
        allowed = self._allowed_indices(doc_ids)
        results = self.bm25.search(question, top_k=limit, allowed_indices=allowed)
        return [(r.chunk_index, r.score) for r in results]

    def _select(
        self,
        hits: List[Tuple[int, float]],
        top_k: int,
        max_per_document: Optional[int],
    ) -> List[Tuple[int, float]]:
        """Applies the per-document quota and truncates to top_k."""
        return diversify_by_document(hits, self.store.chunks, top_k, max_per_document)

    def _to_chunks(self, hits: List[Tuple[int, float]]) -> List[RetrievedChunk]:
        """Turns (chunk index, score) pairs back into citable results."""
        return [
            RetrievedChunk(chunk=self.store.chunks[index], score=score)
            for index, score in hits
        ]
