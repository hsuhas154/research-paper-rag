"""
Tests for streamed generation and provider selection.

Generation can go to local Ollama or to any OpenAI-compatible endpoint.
These cover the contract shared by both: fragments in order, an empty
context short-circuiting, failures surfacing as a clear error, and the
two paths sending identical messages. They also pin the default, because
"where does my paper text go" is not something that should change by
accident.
"""

import pytest

from src import llm, providers
from src.chunker import Chunk
from src.llm import NO_CONTEXT_ANSWER, build_messages, stream_answer
from src.providers import ProviderError
from src.vector_store import RetrievedChunk


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def sources(n=2):
    return [
        RetrievedChunk(chunk=Chunk(i, f"passage {i}", 2, doc_id="d"), score=1.0)
        for i in range(n)
    ]


def fake_ollama(*fragments):
    def chat(messages, model, num_ctx, stream):
        chat.seen = {"messages": messages, "model": model, "stream": stream}
        if not stream:
            return {"message": {"content": "".join(fragments)}}
        return [{"message": {"content": f}} for f in fragments]
    return chat


class TestProviderSelection:
    def test_defaults_to_local(self):
        # The project's premise is that nothing leaves the machine. That
        # must be the default, not something you remember to set.
        assert providers.provider_name() == providers.OLLAMA
        assert providers.is_local() is True

    def test_hosted_is_opt_in(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        assert providers.is_local() is False

    def test_an_unknown_provider_falls_back_to_local(self, monkeypatch):
        # Failing closed matters here: a typo must not silently start
        # shipping paper text to a remote endpoint.
        monkeypatch.setenv("LLM_PROVIDER", "opanai")
        assert providers.is_local() is True

    def test_model_defaults_differ_by_provider(self, monkeypatch):
        assert providers.model_name() == providers.DEFAULT_OLLAMA_MODEL
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        assert providers.model_name() == providers.DEFAULT_OPENAI_MODEL

    def test_description_says_where_generation_happens(self, monkeypatch):
        assert "local" in providers.describe().lower()
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com/v1")
        described = providers.describe()
        assert "hosted" in described and "api.example.com" in described


class TestHostedConfiguration:
    def test_missing_base_url_is_reported_clearly(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_API_KEY", "k")
        with pytest.raises(ProviderError, match="LLM_BASE_URL"):
            providers.complete([{"role": "user", "content": "q"}])

    def test_missing_api_key_is_reported_clearly(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com/v1")
        with pytest.raises(ProviderError, match="LLM_API_KEY"):
            providers.complete([{"role": "user", "content": "q"}])


class TestStreaming:
    def test_yields_fragments_in_order(self, monkeypatch):
        monkeypatch.setattr(providers, "_ollama_chat", fake_ollama("Hello", " there", "."))
        assert list(stream_answer("q", sources())) == ["Hello", " there", "."]

    def test_skips_empty_fragments(self, monkeypatch):
        # Ollama emits a trailing empty chunk; passing it through would
        # make the UI redraw for nothing.
        monkeypatch.setattr(providers, "_ollama_chat", fake_ollama("text", ""))
        assert list(stream_answer("q", sources())) == ["text"]

    def test_empty_context_short_circuits(self, monkeypatch):
        called = []
        monkeypatch.setattr(providers, "_ollama_chat",
                            lambda *a, **kw: called.append(1))
        assert list(stream_answer("q", [])) == [NO_CONTEXT_ANSWER]
        assert not called

    def test_a_provider_failure_surfaces_as_a_runtime_error(self, monkeypatch):
        def boom(*a, **kw):
            raise ProviderError("ollama serve is not running")

        monkeypatch.setattr(providers, "_ollama_chat", boom)
        with pytest.raises(RuntimeError, match="ollama serve"):
            list(stream_answer("q", sources()))


class TestMessageAssembly:
    def test_streaming_and_blocking_send_identical_messages(self, monkeypatch):
        # Drift here would mean the streamed answer came from a different
        # prompt than the one every measurement was taken with.
        seen = []

        def chat(messages, model, num_ctx, stream):
            seen.append(messages)
            return ([{"message": {"content": "x"}}] if stream
                    else {"message": {"content": "x"}})

        monkeypatch.setattr(providers, "_ollama_chat", chat)
        list(stream_answer("q", sources()))
        llm.generate_answer("q", sources())
        assert seen[0] == seen[1]

    def test_history_sits_between_system_prompt_and_question(self):
        messages = build_messages(
            "now", sources(), history=[{"role": "user", "content": "before"}]
        )
        assert [m["role"] for m in messages] == ["system", "user", "user"]
        assert messages[1]["content"] == "before"
        assert "now" in messages[-1]["content"]
