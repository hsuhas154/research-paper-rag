"""
embedder.py

Wraps a pretrained Sentence Transformer model to convert text chunks
(or queries) into dense vector embeddings, ready for indexing or
similarity search with FAISS.

Embeddings are L2-normalized at generation time so that a plain dot
product at search time is mathematically equivalent to cosine
similarity - this lets us use FAISS's fastest index type
(IndexFlatIP, inner product) while still getting cosine similarity
search behavior.
"""

from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # fixed by this model's architecture


class Embedder:
    """
    Thin wrapper around a SentenceTransformer model. Instantiate this
    once (loading the model is the expensive part) and reuse it for
    both document chunk embedding and query embedding.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encodes a list of texts into L2-normalized embedding vectors.

        Args:
            texts: list of strings to embed (chunk texts, or a single
                query wrapped in a one-element list).
            batch_size: how many texts to embed per forward pass.

        Returns:
            np.ndarray of shape (len(texts), EMBEDDING_DIM), dtype float32,
            with each row normalized to unit length.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # this is what makes dot product == cosine similarity
            show_progress_bar=False,
        )
        return embeddings.astype("float32")  # FAISS expects float32 specifically


if __name__ == "__main__":
    import sys
    import time
    sys.path.insert(0, ".")
    from src.pdf_processor import load_and_clean_pdf
    from src.chunker import chunk_text

    if len(sys.argv) != 2:
        print("Usage: python src/embedder.py <path_to_pdf>")
        sys.exit(1)

    print("Loading embedding model...")
    embedder = Embedder()
    print(f"Model loaded on device: {embedder.device}\n")

    cleaned = load_and_clean_pdf(sys.argv[1])
    chunks = chunk_text(cleaned)
    chunk_texts = [c.text for c in chunks]

    print(f"Embedding {len(chunk_texts)} chunks...")
    start = time.time()
    embeddings = embedder.encode(chunk_texts)
    elapsed = time.time() - start

    print(f"Done in {elapsed:.2f}s")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"First vector's L2 norm (should be ~1.0): {np.linalg.norm(embeddings[0]):.4f}")

    # Sanity check: embed a query and compare it against the first two chunks,
    # just to confirm the similarity math produces sensible-looking numbers
    # (this isn't real retrieval yet — that's Stage 5 with FAISS).
    query_embedding = embedder.encode(["What is the SO2 dissolution rate in Venus clouds?"])
    sim_to_chunk0 = float(np.dot(query_embedding[0], embeddings[0]))
    sim_to_chunk_last = float(np.dot(query_embedding[0], embeddings[-1]))
    print(f"\nCosine similarity to chunk 0 (boilerplate): {sim_to_chunk0:.4f}")
    print(f"Cosine similarity to last chunk (likely references): {sim_to_chunk_last:.4f}")