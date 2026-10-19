"""
Tests for streamed generation.

Streaming changes when the caller sees output, not what it is, so these
check the contract: fragments come through in order, an empty context
still yields the refusal, failures surface as a clear error rather than
a half-written answer, and the model is asked to stay resident.
"""

import pytest

from src import llm
from src.chunker import Chunk
from src.llm import KEEP_ALIVE, NO_CONTEXT_ANSWER, build_messages, stream_answer
from src.vector_store import RetrievedChunk


def sources(n=2):
    return [
        RetrievedChunk(chunk=Chunk(i, f"passage {i}", 2, doc_id="d"), score=1.0)
        for i in range(n)
    ]


def fake_stream(*fragments):
    def chat(**kwargs):
        chat.kwargs = kwargs
        return [{"message": {"content": f}} for f in fragments]
    return chat


class TestStreaming:
    def test_yields_fragments_in_order(self, monkeypatch):
        monkeypatch.setattr(llm.ollama, "chat", fake_stream("Hello", " there", "."))
        assert list(stream_answer("q", sources())) == ["Hello", " there", "."]

    def test_skips_empty_fragments(self, monkeypatch):
        # Ollama emits a final empty chunk; passing it through would make
        # the UI redraw for no reason.
        monkeypatch.setattr(llm.ollama, "chat", fake_stream("text", "", None or ""))
        assert list(stream_answer("q", sources())) == ["text"]

    def test_empty_context_yields_the_refusal_without_calling_the_model(self, monkeypatch):
        called = []
        monkeypatch.setattr(llm.ollama, "chat", lambda **kw: called.append(kw))
        assert list(stream_answer("q", [])) == [NO_CONTEXT_ANSWER]
        assert not called

    def test_requests_streaming_and_keeps_the_model_resident(self, monkeypatch):
        chat = fake_stream("x")
        monkeypatch.setattr(llm.ollama, "chat", chat)
        list(stream_answer("q", sources()))

        assert chat.kwargs["stream"] is True
        # Without this the model unloads after five minutes and the next
        # question pays a 30 to 40 second reload.
        assert chat.kwargs["keep_alive"] == KEEP_ALIVE

    def test_an_unreachable_model_raises_with_guidance(self, monkeypatch):
        def boom(**kw):
            raise ConnectionError("connection refused")

        monkeypatch.setattr(llm.ollama, "chat", boom)
        with pytest.raises(RuntimeError, match="ollama serve"):
            list(stream_answer("q", sources()))


class TestMessageAssembly:
    def test_streaming_and_blocking_send_identical_messages(self, monkeypatch):
        # The two paths must not drift: a difference here would mean the
        # streamed answer is generated from a different prompt.
        seen = {}

        def chat(**kw):
            seen.setdefault("calls", []).append(kw["messages"])
            return [{"message": {"content": "x"}}] if kw.get("stream") else \
                {"message": {"content": "x"}}

        monkeypatch.setattr(llm.ollama, "chat", chat)
        list(stream_answer("q", sources()))
        llm.generate_answer("q", sources())

        assert seen["calls"][0] == seen["calls"][1]

    def test_history_sits_between_system_prompt_and_question(self):
        messages = build_messages(
            "now", sources(), history=[{"role": "user", "content": "before"}]
        )
        assert [m["role"] for m in messages] == ["system", "user", "user"]
        assert messages[1]["content"] == "before"
        assert "now" in messages[-1]["content"]
