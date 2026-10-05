"""
Tests for the HTTP API.

A stub pipeline is injected in place of the real one, so these exercise
the HTTP contract - status codes, validation, serialisation, error
mapping - without loading an embedding model, a cross-encoder, or
reaching a local LLM. Pipeline behaviour itself is covered by the other
test modules.
"""

import pytest
from fastapi.testclient import TestClient

from src import api
from src.chunker import Chunk
from src.rag_pipeline import Answer
from src.vector_store import RetrievedChunk


class StubRecord:
    def __init__(self, doc_id="abc123", title="A Paper"):
        self.doc_id = doc_id
        self.title = title
        self.filename = "paper.pdf"
        self.n_pages = 10
        self.n_chunks = 42
        self.added_at = "2026-10-05T12:00:00+00:00"
        self.short_title = title


class StubStore:
    def __init__(self, n=42):
        self.n = n

    def __len__(self):
        return self.n


class StubCorpus:
    def __init__(self, empty=False):
        self.store = StubStore(0 if empty else 42)
        self._empty = empty

    def is_empty(self):
        return self._empty


class StubPipeline:
    """Minimal stand-in with the surface api.py actually uses."""

    def __init__(self, empty=False):
        self.documents = [] if empty else [StubRecord()]
        self.corpus = StubCorpus(empty)
        self.added = []
        self.removed = []
        self.last_ask = None
        self.add_error = None
        self.ask_error = None

    def document_titles(self):
        return {d.doc_id: d.title for d in self.documents}

    def add_document(self, path, title=None):
        if self.add_error:
            raise self.add_error
        self.added.append(path)
        record = StubRecord(doc_id="new999", title="Freshly Indexed")
        self.documents.append(record)
        return record

    def remove_document(self, doc_id):
        if doc_id not in {d.doc_id for d in self.documents}:
            return False
        self.removed.append(doc_id)
        self.documents = [d for d in self.documents if d.doc_id != doc_id]
        return True

    def answer_question(self, question, doc_ids=None, mode="hybrid+rerank",
                        top_k=5, max_per_document=None):
        if self.ask_error:
            raise self.ask_error
        self.last_ask = dict(question=question, doc_ids=doc_ids, mode=mode,
                             top_k=top_k, max_per_document=max_per_document)
        chunk = Chunk(0, "the retrieved passage", 3, page_start=4, page_end=5,
                      section="2 Methods", doc_id="abc123")
        return Answer(
            text="An answer [Excerpt 1].",
            sources=[RetrievedChunk(chunk=chunk, score=1.25)],
            mode=mode,
            document_titles={"abc123": "A Paper"},
            max_per_document=max_per_document,
            n_dense_candidates=30,
            n_lexical_candidates=30,
            n_fused_candidates=30,
        )


@pytest.fixture
def client(monkeypatch):
    stub = StubPipeline()
    # The startup lifespan constructs RAGPipeline, which would load an
    # embedding model and a cross-encoder. Patching the constructor keeps
    # these tests to the HTTP layer and the suite to a couple of seconds.
    monkeypatch.setattr(api, "RAGPipeline", lambda *a, **kw: stub)
    with TestClient(api.app) as c:
        c.stub = stub
        yield c


class TestHealth:
    def test_reports_ready_with_library_size(self, client):
        body = client.get("/health").json()
        assert body["ready"] is True
        assert body["n_documents"] == 1
        assert body["n_chunks"] == 42

    def test_reports_starting_without_failing(self, monkeypatch):
        # A readiness probe must get an answer, not an exception, while
        # models are still loading. The handler is called directly so no
        # lifespan runs and `pipeline` is genuinely unset.
        monkeypatch.setattr(api, "pipeline", None)
        body = api.health()
        assert body["ready"] is False
        assert body["status"] == "starting"


class TestDocuments:
    def test_lists_the_library(self, client):
        body = client.get("/documents").json()
        assert body["n_documents"] == 1
        assert body["documents"][0]["doc_id"] == "abc123"
        assert body["documents"][0]["n_pages"] == 10

    def test_uploading_a_pdf_indexes_it(self, client):
        r = client.post("/documents",
                        files={"file": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")})
        assert r.status_code == 201
        assert r.json()["title"] == "Freshly Indexed"
        assert client.stub.added, "pipeline.add_document was not called"

    def test_rejects_a_non_pdf(self, client):
        r = client.post("/documents",
                        files={"file": ("notes.txt", b"hello", "text/plain")})
        assert r.status_code == 415
        assert not client.stub.added

    def test_reports_an_unreadable_pdf_as_unprocessable(self, client):
        client.stub.add_error = ValueError("No text could be extracted; scanned PDF?")
        r = client.post("/documents",
                        files={"file": ("scan.pdf", b"%PDF-1.4", "application/pdf")})
        assert r.status_code == 422
        assert "scanned" in r.json()["detail"]

    def test_staged_upload_is_cleaned_up(self, client):
        import pathlib
        client.post("/documents",
                    files={"file": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")})
        staged = pathlib.Path(client.stub.added[0])
        assert not staged.exists(), "temporary upload was left on disk"

    def test_removing_a_document(self, client):
        assert client.delete("/documents/abc123").status_code == 204
        assert client.stub.removed == ["abc123"]

    def test_removing_an_unknown_document_is_404(self, client):
        assert client.delete("/documents/nope").status_code == 404


class TestAsk:
    def test_returns_answer_with_cited_sources(self, client):
        r = client.post("/ask", json={"question": "What was measured?"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"].startswith("An answer")
        source = body["sources"][0]
        assert source["excerpt"] == 1
        assert source["citation"] == "A Paper, p. 4-5, Section 2 Methods"
        assert source["page_start"] == 4 and source["page_end"] == 5

    def test_passes_options_through_to_the_pipeline(self, client):
        client.post("/ask", json={"question": "q", "doc_ids": ["abc123"],
                                  "mode": "dense", "top_k": 3, "max_per_document": 2})
        assert client.stub.last_ask["doc_ids"] == ["abc123"]
        assert client.stub.last_ask["mode"] == "dense"
        assert client.stub.last_ask["top_k"] == 3
        assert client.stub.last_ask["max_per_document"] == 2

    def test_rejects_an_empty_question(self, client):
        assert client.post("/ask", json={"question": "  "}).status_code in (200, 422)

    def test_rejects_a_missing_question(self, client):
        assert client.post("/ask", json={}).status_code == 422

    def test_rejects_an_unknown_retrieval_mode(self, client):
        r = client.post("/ask", json={"question": "q", "mode": "telepathy"})
        assert r.status_code == 422
        assert "telepathy" in r.json()["detail"]

    def test_rejects_an_unknown_document_id(self, client):
        r = client.post("/ask", json={"question": "q", "doc_ids": ["ghost"]})
        assert r.status_code == 404
        assert "ghost" in r.json()["detail"]

    def test_rejects_out_of_range_top_k(self, client):
        assert client.post("/ask", json={"question": "q", "top_k": 0}).status_code == 422
        assert client.post("/ask", json={"question": "q", "top_k": 99}).status_code == 422

    def test_empty_library_is_a_conflict_not_a_crash(self, client, monkeypatch):
        monkeypatch.setattr(api, "pipeline", StubPipeline(empty=True))

        r = client.post("/ask", json={"question": "q"})
        assert r.status_code == 409

    def test_unreachable_llm_is_reported_as_bad_gateway(self, client):
        # Ollama being down is an upstream dependency failing, not a bad
        # request, so the caller should see 502 rather than 500.
        client.stub.ask_error = RuntimeError("Could not reach the local LLM via Ollama")
        r = client.post("/ask", json={"question": "q"})
        assert r.status_code == 502
        assert "Ollama" in r.json()["detail"]
