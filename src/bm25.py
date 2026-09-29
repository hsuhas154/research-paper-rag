"""
bm25.py

An Okapi BM25 lexical index over document chunks.

Why this exists: dense embedding retrieval matches on *meaning*, which
is exactly what you want for "what mechanism removes oxygen at 70 km"
but exactly what you don't want for "what value of Kzz was used". A
sentence transformer maps "1e4 cm2/s" and "1e7 cm2/s" to nearly the
same vector - the semantics are identical, only the number differs -
so the chunk holding the specific value a user asked about has no
particular reason to outrank its neighbours. This was the single
biggest retrieval failure noted at the end of Phase 1.

BM25 has the opposite bias: it scores on exact term overlap, weighted
so that rare terms count for more than common ones. Rare terms are
precisely what identifiers, symbols, and numeric values are. Running
both and fusing the rankings (see retriever.py) covers each method's
blind spot with the other's strength.

Implemented directly rather than pulled in as a dependency - BM25 is
about sixty lines and the scoring formula is the part of this project
most worth being able to explain.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from src.chunker import Chunk


# Extremely common English words carry almost no retrieval signal but do
# inflate document-length statistics. This is a deliberately short list:
# aggressive stopword removal hurts in a technical corpus, where words
# like "not", "between", and "above" are load-bearing.
_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or",
    "is", "are", "was", "were", "be", "been", "it", "its", "this", "that",
    "these", "those", "as", "by", "with", "from", "we", "our",
}

# Keeps alphanumeric runs together, so tokens like "1e4", "co2", "kzz",
# and "70km" survive tokenization intact. Splitting those apart would
# destroy exactly the rare-term signal BM25 exists to capture.
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def tokenize(text: str) -> List[str]:
    """Lowercases text and splits it into retrieval tokens."""
    return [t for t in _TOKEN_PATTERN.findall(text.lower()) if t not in _STOPWORDS]


@dataclass
class BM25Result:
    chunk_index: int  # position in the chunk list this index was built over
    score: float


class BM25Index:
    """
    Okapi BM25 over a fixed list of chunks.

    Parameters k1 and b are the standard Okapi defaults:
      - k1 (1.5) controls term-frequency saturation: seeing a term ten
        times in a chunk is better than once, but not ten times better.
      - b (0.75) controls length normalization: without it, long chunks
        win simply by containing more words.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_token_counts: List[Counter] = []
        self.doc_lengths: List[int] = []
        self.average_length: float = 0.0
        self.inverse_document_frequency: Dict[str, float] = {}

    def build(self, chunks: List[Chunk]) -> None:
        """Indexes a list of chunks, replacing any previous contents."""
        self.doc_token_counts = [Counter(tokenize(c.text)) for c in chunks]
        self.doc_lengths = [sum(counts.values()) for counts in self.doc_token_counts]

        n_docs = len(chunks)
        self.average_length = (sum(self.doc_lengths) / n_docs) if n_docs else 0.0

        document_frequency: Counter = Counter()
        for counts in self.doc_token_counts:
            document_frequency.update(counts.keys())

        # Standard BM25 IDF with the +0.5 smoothing terms. The outer +1
        # keeps the result positive for terms that appear in more than
        # half the corpus, which would otherwise score negative and let a
        # common term actively penalize a chunk that contains it.
        self.inverse_document_frequency = {
            term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def search(
        self,
        query: str,
        top_k: int = 10,
        allowed_indices: Optional[Set[int]] = None,
    ) -> List[BM25Result]:
        """
        Scores every indexed chunk against the query and returns the
        top_k highest scorers, best first.

        Args:
            query: the user's question, tokenized the same way as chunks
            top_k: how many results to return
            allowed_indices: if given, restricts scoring to these chunk
                positions, so a search can be scoped to a subset of the
                corpus. Corpus-wide IDF statistics are deliberately left
                untouched: they describe how rare a term is in the
                library as a whole, which is the more stable estimate.

        Chunks sharing no query terms score 0 and are dropped rather than
        returned as filler - a zero-score BM25 hit carries no lexical
        evidence at all, and passing it to rank fusion would add noise.
        """
        query_terms = tokenize(query)
        if not query_terms or not self.doc_token_counts:
            return []

        scored: List[BM25Result] = []
        for index, counts in enumerate(self.doc_token_counts):
            if allowed_indices is not None and index not in allowed_indices:
                continue

            length_norm = self.k1 * (
                1 - self.b + self.b * self.doc_lengths[index] / (self.average_length or 1)
            )

            score = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if frequency == 0:
                    continue
                idf = self.inverse_document_frequency.get(term, 0.0)
                score += idf * (frequency * (self.k1 + 1)) / (frequency + length_norm)

            if score > 0:
                scored.append(BM25Result(chunk_index=index, score=score))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self.doc_token_counts)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.pdf_processor import load_and_segment_pdf
    from src.chunker import chunk_segments

    if len(sys.argv) != 2:
        print("Usage: python -m src.bm25 <path_to_pdf>")
        sys.exit(1)

    doc_chunks = chunk_segments(load_and_segment_pdf(sys.argv[1]))
    index = BM25Index()
    index.build(doc_chunks)
    print(f"Indexed {len(index)} chunks, average length {index.average_length:.0f} tokens.\n")

    for query in [
        "What eddy diffusion coefficient Kzz was used?",
        "How is O2 eliminated at 70 km?",
    ]:
        print(f"=== {query} ===")
        for rank, result in enumerate(index.search(query, top_k=3), start=1):
            chunk = doc_chunks[result.chunk_index]
            print(f"[{rank}] score={result.score:.3f} | {chunk.citation()}")
            print(f"    {chunk.text[:180]}...")
        print()
