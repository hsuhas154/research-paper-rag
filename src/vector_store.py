"""
vector_store.py

Builds and queries a FAISS vector index over document chunk
embeddings.

Uses IndexFlatIP (exact inner-product search). Because our embeddings
are L2-normalized (see embedder.py), inner product is mathematically
equivalent to cosine similarity - so this gives exact cosine-similarity
nearest-neighbor search. "Flat" means brute-force, exact search (no
approximation) - appropriate at this scale (tens to low thousands of
chunks per paper). Approximate indexes (IVF, HNSW) only start to pay
off once indexing far more vectors than a handful of research papers
will ever produce.
"""

import faiss
import numpy as np
from dataclasses import dataclass
from typing import List

from src.chunker import Chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # cosine similarity, higher = more relevant


class VectorStore:
    """
    Wraps a FAISS IndexFlatIP index plus the chunk objects it was
    built from, so a search returns actual chunk text/metadata, not
    just raw vector indices.
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """
        Adds a batch of chunks and their corresponding embeddings to
        the index. Called once per document for Phase 1.

        Args:
            chunks: list of Chunk objects (from chunker.chunk_text)
            embeddings: np.ndarray of shape (len(chunks), embedding_dim),
                must be L2-normalized float32 (see embedder.Embedder.encode)
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {embeddings.shape[0]} embeddings"
            )
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: index expects {self.embedding_dim}, got {embeddings.shape[1]}"
            )

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievedChunk]:
        """
        Finds the top_k chunks most similar to a query embedding.

        Args:
            query_embedding: np.ndarray of shape (embedding_dim,) or
                (1, embedding_dim) - a single L2-normalized query vector
            top_k: how many results to return

        Returns:
            List of RetrievedChunk, sorted by descending similarity score.
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        top_k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS pads with -1 if fewer than top_k results exist
                continue
            results.append(RetrievedChunk(chunk=self.chunks[idx], score=float(score)))

        return results

    def __len__(self) -> int:
        return len(self.chunks)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.pdf_processor import load_and_clean_pdf
    from src.chunker import chunk_text
    from src.embedder import Embedder

    if len(sys.argv) != 2:
        print("Usage: python src/vector_store.py <path_to_pdf>")
        sys.exit(1)

    print("Loading embedding model...")
    embedder = Embedder()

    print("Processing PDF...")
    cleaned = load_and_clean_pdf(sys.argv[1])
    chunks = chunk_text(cleaned)
    chunk_texts = [c.text for c in chunks]

    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embedder.encode(chunk_texts)

    print("Building FAISS index...")
    store = VectorStore(embedding_dim=embeddings.shape[1])
    store.build(chunks, embeddings)
    print(f"Index contains {len(store)} vectors.\n")

    test_queries = [
        "What causes the depletion of SO2 in the upper clouds of Venus?",
        "What eddy diffusion coefficient was used in this model?",
    ]

    for query in test_queries:
        print(f"=== Query: {query} ===")
        query_embedding = embedder.encode([query])[0]
        results = store.search(query_embedding, top_k=3)

        for rank, result in enumerate(results, start=1):
            preview = result.chunk.text[:200].replace("\n", " ")
            print(f"\n[{rank}] score={result.score:.4f} (chunk {result.chunk.chunk_id})")
            print(f"    {preview}...")
        print("\n")