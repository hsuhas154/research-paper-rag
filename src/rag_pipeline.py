"""
rag_pipeline.py

Ties the corpus, the hybrid retriever, and the LLM into a single
RAGPipeline object. This is what the Gradio UI (app.py) talks to - it
should not need to know anything about chunks, embeddings, FAISS, rank
fusion, or Ollama directly.

Phase 2 changes: the pipeline now sits on top of a persistent
multi-document corpus rather than one in-memory document, questions can
be scoped to a subset of papers, and the retrieval strategy is
selectable per question.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from src.corpus import Corpus, DocumentRecord
from src.embedder import Embedder
from src.llm import generate_answer
from src.reranker import CrossEncoderReranker
from src.retriever import MODE_HYBRID_RERANK, HybridRetriever
from src.vector_store import RetrievedChunk


@dataclass
class Answer:
    """A generated answer plus everything needed to verify it."""
    text: str
    sources: List[RetrievedChunk]
    mode: str
    document_titles: Dict[str, str]
    n_dense_candidates: int = 0
    n_lexical_candidates: int = 0
    n_fused_candidates: int = 0

    def retrieval_summary(self) -> str:
        """One line describing how the passages were found."""
        if self.mode in ("dense", "bm25"):
            return f"{self.mode} retrieval -> {len(self.sources)} passages"
        return (
            f"{self.n_dense_candidates} dense + {self.n_lexical_candidates} lexical "
            f"candidates -> {self.n_fused_candidates} fused -> {len(self.sources)} passages"
        )


class RAGPipeline:
    """
    The whole system behind one object.

    The embedding model is loaded once at construction (expensive) and
    reused for every document and query (cheap). The cross-encoder is
    constructed here too but loads lazily on its first use, so starting
    the app with re-ranking switched off costs nothing.
    """

    def __init__(self, top_k: int = 5, candidate_pool: int = 20):
        self.embedder = Embedder()
        self.corpus = Corpus(self.embedder)
        self.retriever = HybridRetriever(
            embedder=self.embedder,
            store=self.corpus.store,
            reranker=CrossEncoderReranker(),
            candidate_pool=candidate_pool,
        )
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Library management
    # ------------------------------------------------------------------

    def add_document(self, pdf_path: str, title: Optional[str] = None) -> DocumentRecord:
        """
        Indexes a PDF into the corpus and returns its registry entry.

        The retriever is resynced afterwards because its lexical index is
        derived from the corpus contents and would otherwise be blind to
        the new document.
        """
        record = self.corpus.add_pdf(pdf_path, title=title)
        self._resync()
        return record

    def remove_document(self, doc_id: str) -> bool:
        """Removes a document from the corpus. Returns False if unknown."""
        removed = self.corpus.remove_document(doc_id)
        if removed:
            self._resync()
        return removed

    def _resync(self) -> None:
        """Points the retriever at the corpus's current store and reindexes BM25."""
        self.retriever.store = self.corpus.store
        self.retriever.sync()

    @property
    def documents(self) -> List[DocumentRecord]:
        return self.corpus.documents

    def document_titles(self) -> Dict[str, str]:
        """doc_id -> short title, as used in citation labels."""
        return {d.doc_id: d.short_title for d in self.corpus.documents}

    # ------------------------------------------------------------------
    # Question answering
    # ------------------------------------------------------------------

    def answer_question(
        self,
        question: str,
        doc_ids: Optional[Sequence[str]] = None,
        mode: str = MODE_HYBRID_RERANK,
        top_k: Optional[int] = None,
    ) -> Answer:
        """
        Answers a question against the corpus.

        Args:
            question: natural-language question
            doc_ids: restrict to these papers; None searches all of them
            mode: retrieval strategy (see retriever.RETRIEVAL_MODES)
            top_k: passages to ground the answer in; defaults to the
                pipeline's configured value

        Returns:
            An Answer carrying the generated text, the source passages,
            and the retrieval statistics behind them.
        """
        if self.corpus.is_empty():
            raise ValueError("No documents in the corpus. Add a PDF first.")

        retrieval = self.retriever.retrieve(
            question,
            top_k=top_k or self.top_k,
            mode=mode,
            doc_ids=doc_ids,
        )
        titles = self.document_titles()
        text = generate_answer(question, retrieval.chunks, document_titles=titles)

        return Answer(
            text=text,
            sources=retrieval.chunks,
            mode=retrieval.mode,
            document_titles=titles,
            n_dense_candidates=retrieval.n_dense_candidates,
            n_lexical_candidates=retrieval.n_lexical_candidates,
            n_fused_candidates=retrieval.n_fused_candidates,
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('Usage: python -m src.rag_pipeline "<question>" [retrieval_mode]')
        sys.exit(1)

    print("Loading models and corpus...")
    pipeline = RAGPipeline(top_k=5)

    if pipeline.corpus.is_empty():
        print("Corpus is empty. Add a paper first: python -m src.corpus add <pdf>")
        sys.exit(1)

    print(f"Corpus: {len(pipeline.documents)} document(s), "
          f"{len(pipeline.corpus.store)} chunks.\n")

    retrieval_mode = sys.argv[2] if len(sys.argv) > 2 else MODE_HYBRID_RERANK
    answer = pipeline.answer_question(sys.argv[1], mode=retrieval_mode)

    print("=== ANSWER ===")
    print(answer.text)

    print(f"\n=== RETRIEVAL ({answer.retrieval_summary()}) ===")
    for i, source in enumerate(answer.sources, start=1):
        title = answer.document_titles.get(source.chunk.doc_id)
        print(f"[{i}] score={source.score:.4f} | {source.chunk.citation(title)}")
