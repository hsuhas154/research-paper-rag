"""
vector_store.py

Builds and queries a FAISS vector index over chunk embeddings drawn
from one or more documents.

Uses IndexFlatIP (exact inner-product search). Because our embeddings
are L2-normalized (see embedder.py), inner product is mathematically
equivalent to cosine similarity - so this gives exact cosine-similarity
nearest-neighbor search. "Flat" means brute-force, exact search (no
approximation) - appropriate at this scale (tens of thousands of chunks
even for a sizeable library of papers). Approximate indexes (IVF, HNSW)
only start to pay off at corpus sizes this project will not reach, and
they trade away exact recall to get there.

Phase 2 changes:
  - The store holds many documents at once, and a search can be scoped
    to a subset of them (ask a question of one paper, or of all of them).
  - The raw embedding matrix is kept alongside the FAISS index. Holding
    both looks redundant, but a flat FAISS index is a write-mostly
    structure: removing a document, or rebuilding after a change of
    index type or embedding model, needs the vectors back. Keeping the
    matrix makes the index a derived artifact that can always be
    rebuilt in milliseconds, which is what makes remove_document() and
    on-disk persistence straightforward.
"""

import faiss
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from src.chunker import Chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # cosine similarity, higher = more relevant


class VectorStore:
    """
    A multi-document FAISS index plus the chunks it was built from, so a
    search returns actual chunk text and provenance rather than raw
    vector positions.
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.chunks: List[Chunk] = []
        self.embeddings = np.zeros((0, embedding_dim), dtype="float32")
        self.index = faiss.IndexFlatIP(embedding_dim)

    def add_document(self, doc_id: str, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """
        Adds one document's chunks and their embeddings to the store.

        Args:
            doc_id: corpus-assigned identifier, stamped onto every chunk
                so results can be traced back and scoped by document.
            chunks: list of Chunk objects (from chunker.chunk_segments)
            embeddings: np.ndarray of shape (len(chunks), embedding_dim),
                L2-normalized float32 (see embedder.Embedder.encode)
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {embeddings.shape[0]} embeddings"
            )
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: index expects {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )
        if doc_id in self.document_ids():
            raise ValueError(f"Document {doc_id!r} is already in the store")

        for chunk in chunks:
            chunk.doc_id = doc_id

        self.chunks.extend(chunks)
        self.embeddings = np.vstack([self.embeddings, embeddings.astype("float32")])
        self.index.add(embeddings.astype("float32"))

    def remove_document(self, doc_id: str) -> int:
        """
        Removes every chunk belonging to a document and rebuilds the
        index over what remains. Returns the number of chunks removed.
        """
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        removed = len(self.chunks) - len(keep)
        if removed == 0:
            return 0

        self.chunks = [self.chunks[i] for i in keep]
        self.embeddings = self.embeddings[keep] if keep else np.zeros(
            (0, self.embedding_dim), dtype="float32"
        )
        self._rebuild_index()
        return removed

    def set_contents(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """
        Installs a complete set of chunks and embeddings, replacing any
        current contents, and rebuilds the index over them.

        Used when restoring a saved corpus, where the chunks already
        carry their doc_ids and arrive as one flat batch rather than
        document by document.
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {embeddings.shape[0]} embeddings"
            )
        self.chunks = list(chunks)
        self.embeddings = embeddings.astype("float32")
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuilds the FAISS index from the retained embedding matrix."""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        if len(self.embeddings):
            self.index.add(self.embeddings)

    def document_ids(self) -> List[str]:
        """Distinct document ids currently in the store, in insertion order."""
        seen: Dict[str, None] = {}
        for chunk in self.chunks:
            seen.setdefault(chunk.doc_id, None)
        return list(seen)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        doc_ids: Optional[Sequence[str]] = None,
    ) -> List[RetrievedChunk]:
        """
        Finds the top_k chunks most similar to a query embedding.

        Returns:
            List of RetrievedChunk, sorted by descending similarity score.
        """
        return [
            RetrievedChunk(chunk=self.chunks[index], score=score)
            for index, score in self.search_indexed(query_embedding, top_k, doc_ids)
        ]

    def search_indexed(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        doc_ids: Optional[Sequence[str]] = None,
    ) -> List[Tuple[int, float]]:
        """
        Same search, returning (chunk position, score) pairs instead of
        chunk objects. Rank fusion in retriever.py works on positions, so
        this is the form it consumes.

        Args:
            query_embedding: shape (embedding_dim,) or (1, embedding_dim),
                a single L2-normalized query vector
            top_k: how many results to return
            doc_ids: if given, only chunks from these documents are
                considered. Implemented with a FAISS id selector rather
                than by over-fetching and filtering afterwards, which
                would silently return too few results whenever one
                document dominates the rankings.
        """
        if not self.chunks:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = np.ascontiguousarray(query_embedding, dtype="float32")

        # Both `selector` and the array backing it must outlive the
        # search call: FAISS holds them by reference, not by copy.
        params = None
        selector = None
        allowed_ids = None
        if doc_ids is not None:
            wanted = set(doc_ids)
            allowed_ids = np.array(
                [i for i, c in enumerate(self.chunks) if c.doc_id in wanted],
                dtype="int64",
            )
            if not len(allowed_ids):
                return []
            selector = faiss.IDSelectorBatch(allowed_ids)
            params = faiss.SearchParameters()
            params.sel = selector
            search_limit = min(top_k, len(allowed_ids))
        else:
            search_limit = min(top_k, len(self.chunks))

        scores, indices = self.index.search(query_embedding, search_limit, params=params)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS pads with -1 if fewer than top_k results exist
                continue
            results.append((int(idx), float(score)))

        return results

    def __len__(self) -> int:
        return len(self.chunks)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.pdf_processor import load_and_segment_pdf
    from src.chunker import chunk_segments
    from src.embedder import Embedder

    if len(sys.argv) != 2:
        print("Usage: python -m src.vector_store <path_to_pdf>")
        sys.exit(1)

    print("Loading embedding model...")
    embedder = Embedder()

    print("Processing PDF...")
    doc_chunks = chunk_segments(load_and_segment_pdf(sys.argv[1]))

    print(f"Embedding {len(doc_chunks)} chunks...")
    vectors = embedder.encode([c.text for c in doc_chunks])

    store = VectorStore(embedding_dim=vectors.shape[1])
    store.add_document("demo", doc_chunks, vectors)
    print(f"Index contains {len(store)} vectors across {len(store.document_ids())} document(s).\n")

    for query in [
        "What causes the depletion of SO2 in the upper clouds of Venus?",
        "What eddy diffusion coefficient was used in this model?",
    ]:
        print(f"=== Query: {query} ===")
        for rank, result in enumerate(store.search(embedder.encode([query])[0], top_k=3), start=1):
            print(f"\n[{rank}] score={result.score:.4f} | {result.chunk.citation()}")
            print(f"    {result.chunk.text[:200]}...")
        print("\n")
