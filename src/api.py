"""
api.py

FastAPI backend exposing the RAG pipeline over HTTP.

Phase 3 change: until now the Gradio UI called RAGPipeline directly, in
the same process. That works for one person on one machine and nothing
else: there is no way to put a different frontend in front of it, to run
the retrieval stack on a GPU box separate from the UI, or to let anything
other than Python talk to it. This module is the seam - the pipeline
behind a stable HTTP contract, with the UI becoming just one client of
it.

The pipeline is loaded once at application startup rather than per
request. Loading the embedding model, restoring the corpus, and building
the lexical index takes several seconds and holding one instance is the
whole reason requests after the first are fast.

Run with:
    uvicorn src.api:app --port 8000
    python -m src.api            (equivalent, for convenience)

Interactive documentation is generated automatically at /docs.
"""

import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.rag_pipeline import Answer, RAGPipeline
from src.retriever import MODE_HYBRID_RERANK, RETRIEVAL_MODES

# Populated during the startup lifespan event. A module-level handle is
# the simplest thing that works for a single-process deployment; a
# multi-worker setup would need each worker to build its own, which is
# what the lifespan hook already does per process.
pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads models and the corpus once, before the first request."""
    global pipeline
    pipeline = RAGPipeline(top_k=5)
    yield
    pipeline = None


app = FastAPI(
    title="Research Paper Q&A API",
    description=(
        "Retrieval-augmented question answering over a persistent library "
        "of research papers. Runs entirely locally."
    ),
    version="3.0.0",
    lifespan=lifespan,
)


def require_pipeline() -> RAGPipeline:
    """
    Returns the loaded pipeline, or fails loudly.

    A request arriving before startup finished is a server problem, not a
    client one, so it is a 503 rather than a 500: the caller can usefully
    retry.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is still starting up.")
    return pipeline


# ----------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------

class DocumentOut(BaseModel):
    doc_id: str
    title: str
    filename: str
    n_pages: int
    n_chunks: int
    added_at: str


class LibraryOut(BaseModel):
    documents: List[DocumentOut]
    n_documents: int
    n_chunks: int


class SourceOut(BaseModel):
    excerpt: int = Field(description="1-based index, matching [Excerpt N] in the answer")
    doc_id: str
    title: Optional[str] = None
    citation: str
    page_start: int
    page_end: int
    section: Optional[str] = None
    score: float
    text: str


class AskIn(BaseModel):
    question: str = Field(min_length=1)
    doc_ids: Optional[List[str]] = Field(
        default=None,
        description="Restrict to these documents. Omit to search the whole library.",
    )
    mode: str = Field(default=MODE_HYBRID_RERANK, description=f"One of {RETRIEVAL_MODES}")
    top_k: int = Field(default=5, ge=1, le=20)
    max_per_document: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Cap passages from any one paper. Omit to let the server decide, "
            "which applies a cap only to questions that read as comparative."
        ),
    )


class AskOut(BaseModel):
    answer: str
    sources: List[SourceOut]
    mode: str
    retrieval_summary: str
    max_per_document: Optional[int] = None


def to_sources(answer: Answer) -> List[SourceOut]:
    """Converts pipeline sources into the wire format."""
    return [
        SourceOut(
            excerpt=i,
            doc_id=s.chunk.doc_id,
            title=answer.document_titles.get(s.chunk.doc_id),
            citation=s.chunk.citation(answer.document_titles.get(s.chunk.doc_id)),
            page_start=s.chunk.page_start,
            page_end=s.chunk.page_end,
            section=s.chunk.section,
            score=s.score,
            text=s.chunk.text,
        )
        for i, s in enumerate(answer.sources, start=1)
    ]


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------

@app.get("/health", summary="Liveness and readiness")
def health() -> Dict[str, object]:
    """
    Reports whether the pipeline has finished loading, and how big the
    library is. Deliberately does not fail when the pipeline is still
    starting, so it can be used as a readiness probe.
    """
    if pipeline is None:
        return {"status": "starting", "ready": False}
    return {
        "status": "ok",
        "ready": True,
        "n_documents": len(pipeline.documents),
        "n_chunks": len(pipeline.corpus.store),
        "retrieval_modes": RETRIEVAL_MODES,
    }


@app.get("/documents", response_model=LibraryOut, summary="List indexed papers")
def list_documents() -> LibraryOut:
    p = require_pipeline()
    return LibraryOut(
        documents=[DocumentOut(**d.__dict__) for d in p.documents],
        n_documents=len(p.documents),
        n_chunks=len(p.corpus.store),
    )


@app.post("/documents", response_model=DocumentOut, status_code=201,
          summary="Index a PDF into the library")
async def add_document(file: UploadFile = File(...)) -> DocumentOut:
    """
    Accepts a PDF upload, indexes it, and returns its registry entry.

    The upload is staged in a temporary file rather than streamed into the
    corpus directly, because the whole extraction pipeline works from a
    path on disk. The corpus keeps its own archive copy, so the temporary
    file is always removed afterwards.
    """
    p = require_pipeline()

    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are accepted.")

    staged = Path(tempfile.mkdtemp()) / Path(file.filename).name
    try:
        with staged.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        record = p.add_document(str(staged))
    except ValueError as error:
        # Raised for scanned PDFs with no text layer, and for empty files.
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        shutil.rmtree(staged.parent, ignore_errors=True)

    return DocumentOut(**record.__dict__)


@app.delete("/documents/{doc_id}", status_code=204, summary="Remove a paper")
def remove_document(doc_id: str) -> None:
    p = require_pipeline()
    if not p.remove_document(doc_id):
        raise HTTPException(status_code=404, detail=f"No document {doc_id!r} in the library.")


@app.post("/ask", response_model=AskOut, summary="Ask a question")
def ask(request: AskIn) -> AskOut:
    """
    Answers a question against the library, returning the answer together
    with the passages it was grounded in.
    """
    p = require_pipeline()

    if request.mode not in RETRIEVAL_MODES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown retrieval mode {request.mode!r}; expected one of {RETRIEVAL_MODES}",
        )
    if p.corpus.is_empty():
        raise HTTPException(status_code=409, detail="The library is empty. Add a PDF first.")

    known = {d.doc_id for d in p.documents}
    unknown = sorted(set(request.doc_ids or []) - known)
    if unknown:
        raise HTTPException(status_code=404, detail=f"Unknown document ids: {unknown}")

    try:
        answer = p.answer_question(
            request.question,
            doc_ids=request.doc_ids,
            mode=request.mode,
            top_k=request.top_k,
            max_per_document=request.max_per_document,
        )
    except RuntimeError as error:
        # The local LLM is unreachable. That is an upstream dependency
        # failing, not a bad request, so it is a 502.
        raise HTTPException(status_code=502, detail=str(error)) from error

    return AskOut(
        answer=answer.text,
        sources=to_sources(answer),
        mode=answer.mode,
        retrieval_summary=answer.retrieval_summary(),
        max_per_document=answer.max_per_document,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=False)
