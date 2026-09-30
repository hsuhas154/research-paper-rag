"""
Tests for corpus persistence and library management.

A fake embedder stands in for the real sentence transformer: these tests
are about whether the library round-trips through disk correctly, not
about embedding quality, and loading a real model would make the suite
hundreds of times slower for no added coverage.
"""

import json

import numpy as np
import pytest

from src.chunker import Chunk
from src.corpus import STORAGE_FORMAT_VERSION, Corpus, DocumentRecord


class FakeEmbedder:
    """Deterministic stand-in for Embedder with the same interface."""

    model_name = "fake-embedder"
    embedding_dim = 4

    def encode(self, texts, batch_size: int = 32) -> np.ndarray:
        # Hash each text into a fixed vector, then L2-normalize, matching
        # the real embedder's contract (unit vectors, float32).
        vectors = np.array(
            [[((hash(t) >> (8 * i)) & 0xFF) + 1 for i in range(4)] for t in texts],
            dtype="float32",
        )
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@pytest.fixture
def corpus(tmp_path):
    return Corpus(
        FakeEmbedder(),
        corpus_dir=tmp_path / "corpus",
        pdf_dir=tmp_path / "pdfs",
    )


def add_fake_document(corpus: Corpus, doc_id: str, title: str, n_chunks: int = 3) -> None:
    """Indexes a document directly, bypassing PDF parsing."""
    chunks = [
        Chunk(chunk_id=i, text=f"{doc_id} chunk {i}", word_count=3,
              page_start=i + 1, page_end=i + 1, section="1 Introduction")
        for i in range(n_chunks)
    ]
    embeddings = corpus.embedder.encode([c.text for c in chunks])
    corpus.store.add_document(doc_id, chunks, embeddings)
    corpus.documents.append(
        DocumentRecord(doc_id=doc_id, title=title, filename=f"{doc_id}.pdf",
                       n_pages=n_chunks, n_chunks=n_chunks, added_at="2026-01-01T00:00:00+00:00")
    )
    corpus.save()


class TestEmptyCorpus:
    def test_a_fresh_corpus_is_empty(self, corpus):
        assert corpus.is_empty()
        assert len(corpus) == 0
        assert len(corpus.store) == 0

    def test_loading_a_nonexistent_corpus_is_not_an_error(self, tmp_path):
        # First run has no corpus directory; that is a valid state, not a failure.
        fresh = Corpus(FakeEmbedder(), corpus_dir=tmp_path / "nothing_here",
                       pdf_dir=tmp_path / "pdfs")
        assert fresh.is_empty()


class TestPersistence:
    def test_documents_survive_a_reload(self, corpus, tmp_path):
        add_fake_document(corpus, "aaa", "First Paper")
        add_fake_document(corpus, "bbb", "Second Paper")

        reloaded = Corpus(FakeEmbedder(), corpus_dir=tmp_path / "corpus",
                          pdf_dir=tmp_path / "pdfs")

        assert len(reloaded) == 2
        assert [d.title for d in reloaded.documents] == ["First Paper", "Second Paper"]
        assert len(reloaded.store) == 6

    def test_chunk_provenance_survives_a_reload(self, corpus, tmp_path):
        add_fake_document(corpus, "aaa", "First Paper")

        reloaded = Corpus(FakeEmbedder(), corpus_dir=tmp_path / "corpus",
                          pdf_dir=tmp_path / "pdfs")
        chunk = reloaded.store.chunks[0]

        assert chunk.doc_id == "aaa"
        assert chunk.page_start == 1
        assert chunk.section == "1 Introduction"

    def test_embeddings_survive_a_reload_unchanged(self, corpus, tmp_path):
        add_fake_document(corpus, "aaa", "First Paper")
        before = corpus.store.embeddings.copy()

        reloaded = Corpus(FakeEmbedder(), corpus_dir=tmp_path / "corpus",
                          pdf_dir=tmp_path / "pdfs")

        np.testing.assert_allclose(reloaded.store.embeddings, before)

    def test_reloaded_index_is_searchable(self, corpus, tmp_path):
        add_fake_document(corpus, "aaa", "First Paper")

        reloaded = Corpus(FakeEmbedder(), corpus_dir=tmp_path / "corpus",
                          pdf_dir=tmp_path / "pdfs")
        query = reloaded.embedder.encode(["aaa chunk 1"])[0]

        # The FAISS index is rebuilt from embeddings.npy on load rather than
        # being persisted itself, so this is the test that the rebuild works.
        assert reloaded.store.search(query, top_k=1)[0].chunk.text == "aaa chunk 1"

    def test_refuses_to_load_an_incompatible_storage_format(self, corpus, tmp_path):
        add_fake_document(corpus, "aaa", "First Paper")

        blob = json.loads(corpus.documents_path.read_text())
        blob["version"] = STORAGE_FORMAT_VERSION + 1
        corpus.documents_path.write_text(json.dumps(blob))

        with pytest.raises(ValueError, match="storage format"):
            Corpus(FakeEmbedder(), corpus_dir=tmp_path / "corpus",
                   pdf_dir=tmp_path / "pdfs")

    def test_detects_a_chunk_embedding_count_mismatch(self, corpus, tmp_path):
        add_fake_document(corpus, "aaa", "First Paper")

        np.save(corpus.embeddings_path, corpus.store.embeddings[:1])

        with pytest.raises(ValueError, match="inconsistent"):
            Corpus(FakeEmbedder(), corpus_dir=tmp_path / "corpus",
                   pdf_dir=tmp_path / "pdfs")


class TestPdfArchive:
    def test_archives_indexed_pdfs_away_from_the_source_folder(self, tmp_path):
        # Writing archive copies back into the folder the user stages
        # source PDFs in made a re-index pick up those copies and index
        # every paper twice.
        staging = tmp_path / "uploads"
        staging.mkdir()
        corpus = Corpus(FakeEmbedder(), corpus_dir=tmp_path / "corpus")

        assert corpus.pdf_dir.is_relative_to(corpus.corpus_dir)
        assert not corpus.pdf_dir.is_relative_to(staging)


class TestRemoval:
    def test_removes_a_document_and_its_chunks(self, corpus):
        add_fake_document(corpus, "aaa", "First Paper")
        add_fake_document(corpus, "bbb", "Second Paper")

        assert corpus.remove_document("aaa") is True
        assert len(corpus) == 1
        assert len(corpus.store) == 3
        assert corpus.store.document_ids() == ["bbb"]

    def test_removal_persists_across_a_reload(self, corpus, tmp_path):
        add_fake_document(corpus, "aaa", "First Paper")
        add_fake_document(corpus, "bbb", "Second Paper")
        corpus.remove_document("aaa")

        reloaded = Corpus(FakeEmbedder(), corpus_dir=tmp_path / "corpus",
                          pdf_dir=tmp_path / "pdfs")

        assert [d.doc_id for d in reloaded.documents] == ["bbb"]
        assert len(reloaded.store) == 3

    def test_removing_an_unknown_document_reports_failure(self, corpus):
        assert corpus.remove_document("nope") is False


class TestLookups:
    def test_title_for_a_known_document(self, corpus):
        add_fake_document(corpus, "aaa", "First Paper")
        assert corpus.title_for("aaa") == "First Paper"

    def test_title_for_an_unknown_document_degrades_gracefully(self, corpus):
        assert corpus.title_for("nope") == "unknown document"

    def test_long_titles_are_shortened_for_display(self):
        record = DocumentRecord("id", "T" * 200, "f.pdf", 1, 1, "2026-01-01T00:00:00+00:00")
        assert len(record.short_title) == 70
        assert record.short_title.endswith("...")
