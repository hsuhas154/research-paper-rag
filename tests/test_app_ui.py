"""
Tests for UI behaviour that is easy to get subtly wrong.

app.py builds a real pipeline at import time, so the module is imported
once here with that construction stubbed out. These cover the scope
filter and the source rendering, both of which have already had bugs
that unit tests would have caught.
"""

import importlib
import sys
from types import SimpleNamespace

import pytest

from src.chunker import Chunk
from src.rag_pipeline import Answer
from src.vector_store import RetrievedChunk


def make_doc(doc_id, title):
    return SimpleNamespace(
        doc_id=doc_id, title=title,
        short_title=title if len(title) <= 70 else title[:67] + "...",
        n_pages=10, n_chunks=5, added_at="2026-01-01T00:00:00+00:00",
    )


DOCS = [
    make_doc("a", "Attention is All you Need"),
    make_doc("b", "Iron-sulfur chemistry can explain the ultraviolet absorber in the clouds of Venus"),
    # The word that identifies this paper sits past the 70-character
    # label truncation, which is the bug this fixture exists for.
    make_doc("c", "H2CO, O3, NH3, HCN, N2O, NO2, NO, and HO2 upper limits of detection in the Venus lower-mesosphere"),
]


@pytest.fixture(scope="module")
def ui(monkeypatch_module=None):
    import src.rag_pipeline as rp

    stub = SimpleNamespace(
        documents=DOCS,
        corpus=SimpleNamespace(store=[1] * 50, is_empty=lambda: False),
    )
    original = rp.RAGPipeline
    rp.RAGPipeline = lambda *a, **kw: stub
    sys.modules.pop("app", None)
    try:
        module = importlib.import_module("app")
    finally:
        rp.RAGPipeline = original
    yield module
    sys.modules.pop("app", None)


class TestScopeFilter:
    def test_empty_filter_shows_everything(self, ui):
        assert len(ui.filtered_choices("")) == len(DOCS)

    def test_filters_case_insensitively(self, ui):
        assert len(ui.filtered_choices("ATTENTION")) == 1

    def test_matches_past_the_label_truncation(self, ui):
        # Labels are cut at 70 characters. One Venus paper leads with a
        # list of molecular formulae and does not reach "Venus" until
        # character 85, so matching the visible label alone loses it.
        found = {value for _, value in ui.filtered_choices("venus")}
        assert found == {"b", "c"}

    def test_a_filter_matching_nothing_returns_nothing(self, ui):
        assert ui.filtered_choices("quantum chromodynamics") == []

    def test_filtering_keeps_selections_that_remain_visible(self, ui):
        update = ui.apply_filter("venus", ["a", "b", "c"])
        assert set(update["value"]) == {"b", "c"}

    def test_filtering_drops_only_selections_it_hides(self, ui):
        update = ui.apply_filter("attention", ["a", "b"])
        assert update["value"] == ["a"]


class TestSourceRendering:
    @staticmethod
    def answer():
        chunk = Chunk(0, "the passage text", 3, page_start=4, page_end=5,
                      section="2 Methods", doc_id="a")
        return Answer(
            text="answer", sources=[RetrievedChunk(chunk=chunk, score=1.25)],
            mode="hybrid+rerank", document_titles={"a": "A Paper"},
        )

    def test_each_excerpt_is_collapsible_with_its_citation_visible(self, ui):
        rendered = ui.format_sources(self.answer())
        assert "<details" in rendered and "<summary>" in rendered
        # The citation must sit in the summary, so a reader checking a
        # claim can find the right passage without expanding all of them.
        summary = rendered.split("</summary>")[0]
        assert "A Paper, p. 4-5, Section 2 Methods" in summary
        assert "[Excerpt 1]" in summary

    def test_passage_text_is_present(self, ui):
        assert "the passage text" in ui.format_sources(self.answer())

    def test_no_sources_renders_empty(self, ui):
        empty = Answer(text="", sources=[], mode="dense", document_titles={})
        assert ui.format_sources(empty) == ""
