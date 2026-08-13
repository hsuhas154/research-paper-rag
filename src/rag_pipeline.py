"""
rag_pipeline.py

Ties together PDF processing, chunking, embedding, vector search, and
LLM generation into a single RAGPipeline class. This is the object
the Gradio UI (app.py) will actually talk to - it shouldn't need to
know anything about chunks, embeddings, or FAISS directly.
"""

from typing import List, Optional, Tuple

from src.pdf_processor import load_and_clean_pdf
from src.chunker import chunk_text, Chunk
from src.embedder import Embedder
from src.vector_store import VectorStore, RetrievedChunk
from src.llm import generate_answer


class RAGPipeline:
    """
    Holds one loaded document's vector index and answers questions
    against it. The embedding model is loaded once at construction
    time (expensive) and reused across documents and queries (cheap).
    """

    def __init__(self, top_k: int = 5):
        self.embedder = Embedder()
        self.store: Optional[VectorStore] = None
        self.top_k = top_k
        self.document_name: Optional[str] = None

    def load_document(self, pdf_path: str, document_name: Optional[str] = None) -> int:
        """
        Processes a PDF end-to-end: extract -> clean -> chunk -> embed -> index.
        Calling this again replaces the currently loaded document.

        Returns the number of chunks indexed, so the UI can show
        something like "Indexed 77 chunks" as feedback.
        """
        cleaned = load_and_clean_pdf(pdf_path)
        chunks: List[Chunk] = chunk_text(cleaned)
        embeddings = self.embedder.encode([c.text for c in chunks])

        self.store = VectorStore(embedding_dim=embeddings.shape[1])
        self.store.build(chunks, embeddings)
        self.document_name = document_name or pdf_path

        return len(chunks)

    def answer_question(self, question: str) -> Tuple[str, List[RetrievedChunk]]:
        """
        Answers a question against the currently loaded document.

        Returns (answer_text, retrieved_chunks) so the caller can
        display both the generated answer and the source passages
        it was grounded in.
        """
        if self.store is None:
            raise ValueError("No document loaded. Call load_document() first.")

        query_embedding = self.embedder.encode([question])[0]
        retrieved = self.store.search(query_embedding, top_k=self.top_k)
        answer = generate_answer(question, retrieved)

        return answer, retrieved


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print('Usage: python src/rag_pipeline.py <path_to_pdf> "<question>"')
        sys.exit(1)

    pipeline = RAGPipeline(top_k=5)

    print("Loading document...")
    n_chunks = pipeline.load_document(sys.argv[1])
    print(f"Indexed {n_chunks} chunks.\n")

    answer, retrieved = pipeline.answer_question(sys.argv[2])

    print("=== ANSWER ===")
    print(answer)

    print("\n=== SOURCES ===")
    for i, r in enumerate(retrieved, start=1):
        preview = r.chunk.text[:120].replace("\n", " ")
        print(f"[{i}] score={r.score:.4f}: {preview}...")