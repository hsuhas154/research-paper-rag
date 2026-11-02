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


def import_app_with_share(monkeypatch, share_value):
    """
    Imports app.py fresh with SHARE set, since the flag and the layout are
    both decided at import time.
    """
    import src.rag_pipeline as rp

    removed = []
    stub = SimpleNamespace(
        documents=list(DOCS),
        corpus=SimpleNamespace(
            store=[1] * 50,
            is_empty=lambda: False,
            title_for=lambda doc_id: "First Paper",
        ),
        remove_document=lambda doc_id: removed.append(doc_id) or True,
    )
    monkeypatch.setenv("SHARE", share_value)
    monkeypatch.setattr(rp, "RAGPipeline", lambda *a, **kw: stub)
    sys.modules.pop("app", None)
    try:
        module = importlib.import_module("app")
    finally:
        sys.modules.pop("app", None)
    return module, removed


class TestSharedLinkGuards:
    """
    A public link reaches a library that took hours to build, and removal
    is the one browser action that cannot be undone.
    """

    def test_sharing_is_off_unless_asked_for(self, monkeypatch):
        module, _ = import_app_with_share(monkeypatch, "")
        assert module.SHARING is False

    def test_removal_works_when_not_sharing(self, monkeypatch):
        module, removed = import_app_with_share(monkeypatch, "")
        module.delete_document("a")
        assert removed == ["a"]

    def test_removal_is_refused_while_sharing(self, monkeypatch):
        module, removed = import_app_with_share(monkeypatch, "1")

        message = module.delete_document("a")[0]

        # Refused in the handler, not just hidden in the layout: a hidden
        # Gradio control is still a reachable endpoint for anyone holding
        # the link.
        assert removed == []
        assert "disabled" in message.lower()

    def test_uploading_stays_available_while_sharing(self, monkeypatch):
        # Showing someone their own paper answered is the point of the
        # demo, so the upload path is deliberately not guarded.
        module, _ = import_app_with_share(monkeypatch, "1")
        assert module.SHARING is True
        assert callable(module.add_document)


class TestTopKIsClamped:
    """
    Every Gradio handler is also an HTTP endpoint. The slider cannot
    produce a value outside 1 to 10, but the endpoint accepts whatever it
    is sent, and the corpus holds a few thousand chunks: an unclamped
    top_k pulls all of them into one prompt. Wasted GPU locally, and a
    real bill against a hosted endpoint.
    """

    def ask(self, module, top_k):
        """Runs one question and reports the top_k the pipeline received."""
        seen = {}

        def stream_question(question, doc_ids, mode, top_k, conversation):
            seen["top_k"] = top_k
            return Answer(text="", sources=[], mode="dense", document_titles={}), iter(["ok"])

        module.pipeline = SimpleNamespace(
            documents=list(DOCS),
            corpus=SimpleNamespace(is_empty=lambda: False),
            stream_question=stream_question,
        )
        list(module.ask_question("q", ["a"], "dense", top_k, False, []))
        return seen.get("top_k")

    def test_a_value_above_the_maximum_is_clamped(self, monkeypatch):
        module, _ = import_app_with_share(monkeypatch, "")
        assert self.ask(module, 100000) == module.MAX_TOP_K

    def test_a_value_below_the_minimum_is_clamped(self, monkeypatch):
        module, _ = import_app_with_share(monkeypatch, "")
        assert self.ask(module, -5) == module.MIN_TOP_K

    def test_a_value_in_range_is_passed_through(self, monkeypatch):
        module, _ = import_app_with_share(monkeypatch, "")
        assert self.ask(module, 3) == 3

    def test_a_non_numeric_value_does_not_crash(self, monkeypatch):
        module, _ = import_app_with_share(monkeypatch, "")
        assert module.MIN_TOP_K <= self.ask(module, "many") <= module.MAX_TOP_K

    def test_the_slider_and_the_handler_agree(self, monkeypatch):
        # Two places enforcing different bounds is how the gap reopens.
        module, _ = import_app_with_share(monkeypatch, "")
        assert (module.MIN_TOP_K, module.MAX_TOP_K) == (1, 10)


class TestTelemetryIsOff:
    def test_analytics_are_disabled(self, monkeypatch):
        # The project's premise is that nothing leaves the machine, and
        # Gradio reports usage to its own endpoint unless told not to.
        module, _ = import_app_with_share(monkeypatch, "")
        assert module.demo.analytics_enabled is False


class TestUploadIsBounded:
    """
    A per-file size cap bounds one upload, not the disk. Every indexed PDF
    is copied into the corpus directory and kept after the visitor leaves,
    so the count has to be bounded too.
    """

    def stub_with(self, module, n_documents):
        indexed = []
        module.pipeline = SimpleNamespace(
            documents=[make_doc(str(i), f"Paper {i}") for i in range(n_documents)],
            corpus=SimpleNamespace(store=[1] * 10, is_empty=lambda: n_documents == 0),
            add_document=lambda path: indexed.append(path) or SimpleNamespace(
                title="New Paper", n_pages=3, n_chunks=7
            ),
        )
        return indexed

    def test_uploading_is_allowed_below_the_cap(self, monkeypatch):
        module, _ = import_app_with_share(monkeypatch, "1")
        indexed = self.stub_with(module, module.MAX_DOCUMENTS_WHILE_SHARING - 1)

        module.add_document("/tmp/paper.pdf")

        assert indexed == ["/tmp/paper.pdf"]

    def test_uploading_is_refused_at_the_cap(self, monkeypatch):
        module, _ = import_app_with_share(monkeypatch, "1")
        indexed = self.stub_with(module, module.MAX_DOCUMENTS_WHILE_SHARING)

        message = module.add_document("/tmp/paper.pdf")[0]

        assert indexed == []
        assert str(module.MAX_DOCUMENTS_WHILE_SHARING) in message

    def test_the_cap_does_not_apply_when_not_sharing(self, monkeypatch):
        # The owner's own library is not limited to ten papers; the real
        # one holds fifty.
        module, _ = import_app_with_share(monkeypatch, "")
        indexed = self.stub_with(module, 50)

        module.add_document("/tmp/paper.pdf")

        assert indexed == ["/tmp/paper.pdf"]

    def test_the_per_file_size_cap_is_set(self, monkeypatch):
        module, _ = import_app_with_share(monkeypatch, "1")
        assert module.MAX_UPLOAD_WHILE_SHARING == "64mb"
