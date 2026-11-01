"""
Tests for how the hosted provider handles being throttled.

A local Ollama answers every request it is given. A hosted endpoint has a
quota, and the free tiers these are aimed at are small enough that a
batch of evaluation questions runs into one. A 429 means "wait and try
again", so treating it as a failure turned a forty-question run into a
crash that discarded the thirty-nine answers already paid for.

What matters, and what these pin down: a throttled request is retried, a
request the provider has refused on its merits is not, the provider's own
Retry-After is preferred to a guess, and a stream that has already
emitted text is never restarted underneath the caller.
"""

import httpx
import pytest

from src import providers
from src.providers import ProviderError

URL = "https://example.invalid/v1/chat/completions"

ANSWER = {"choices": [{"message": {"content": "an answer"}}]}


@pytest.fixture(autouse=True)
def hosted_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def slept(monkeypatch):
    """Records the waits instead of taking them, so the suite stays fast."""
    waits = []
    monkeypatch.setattr(providers.time, "sleep", waits.append)
    return waits


def reply(status, body=None, headers=None):
    return httpx.Response(
        status,
        json=body if body is not None else {"error": "too many requests"},
        headers=headers or {},
        request=httpx.Request("POST", URL),
    )


def queue_posts(monkeypatch, *items):
    """Serves the given replies in order, recording each call."""
    calls, pending = [], list(items)

    def post(**kwargs):
        calls.append(kwargs)
        item = pending.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(httpx, "post", post)
    return calls


class FakeStream:
    """Stands in for the context manager httpx.stream returns."""

    def __init__(self, status, lines=(), headers=None, error=None):
        self.status_code = status
        self.headers = httpx.Headers(headers or {})
        self.request = httpx.Request("POST", URL)
        self._lines = list(lines)
        self._error = error

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{self.status_code}", request=self.request, response=self
            )
        return self

    def read(self):
        return b""

    @property
    def text(self):
        return "provider said no"

    def iter_lines(self):
        for line in self._lines:
            yield line
        if self._error:
            raise self._error


def queue_streams(monkeypatch, *items):
    calls, pending = [], list(items)

    def stream(method, **kwargs):
        calls.append(kwargs)
        item = pending.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(httpx, "stream", stream)
    return calls


def deltas(*texts):
    """The data lines for some content fragments, without the terminator."""
    return ['data: {"choices":[{"delta":{"content":"%s"}}]}' % text for text in texts]


def fragments(*texts):
    """A complete stream: content fragments followed by the terminator."""
    return deltas(*texts) + ["data: [DONE]"]


class TestCompletionRetries:
    def test_a_throttled_request_is_retried_and_then_succeeds(self, monkeypatch):
        calls = queue_posts(monkeypatch, reply(429), reply(200, ANSWER))

        assert providers.complete([{"role": "user", "content": "q"}]) == "an answer"
        assert len(calls) == 2

    def test_a_server_error_is_retried(self, monkeypatch):
        calls = queue_posts(monkeypatch, reply(503), reply(200, ANSWER))

        assert providers.complete([{"role": "user", "content": "q"}]) == "an answer"
        assert len(calls) == 2

    def test_the_providers_own_retry_after_is_preferred(self, monkeypatch, slept):
        queue_posts(
            monkeypatch, reply(429, headers={"retry-after": "7"}), reply(200, ANSWER)
        )

        providers.complete([{"role": "user", "content": "q"}])

        # 7 came from the header; the exponential guess would have been 2.
        assert slept == [7.0]

    def test_an_absurd_retry_after_is_capped(self, monkeypatch, slept):
        queue_posts(
            monkeypatch, reply(429, headers={"retry-after": "9000"}), reply(200, ANSWER)
        )

        providers.complete([{"role": "user", "content": "q"}])

        # Waiting two and a half hours is worse than reporting a failure.
        assert slept == [providers.MAX_RETRY_DELAY_SECONDS]

    def test_backoff_grows_rather_than_hammering(self, monkeypatch, slept):
        queue_posts(monkeypatch, reply(429), reply(429), reply(429), reply(200, ANSWER))

        providers.complete([{"role": "user", "content": "q"}])

        assert slept == [2.0, 4.0, 8.0]

    def test_a_refusal_on_the_merits_is_not_retried(self, monkeypatch):
        # A retired model name answers 404 forever. Retrying it just turns
        # a clear error into a slow one.
        calls = queue_posts(monkeypatch, reply(404))

        with pytest.raises(ProviderError, match="404"):
            providers.complete([{"role": "user", "content": "q"}])
        assert len(calls) == 1

    def test_a_bad_key_is_not_retried(self, monkeypatch):
        calls = queue_posts(monkeypatch, reply(401))

        with pytest.raises(ProviderError, match="401"):
            providers.complete([{"role": "user", "content": "q"}])
        assert len(calls) == 1

    def test_giving_up_reports_the_throttling_rather_than_a_bare_failure(self, monkeypatch):
        calls = queue_posts(monkeypatch, *[reply(429)] * providers.MAX_RETRIES)

        with pytest.raises(ProviderError, match="429"):
            providers.complete([{"role": "user", "content": "q"}])
        assert len(calls) == providers.MAX_RETRIES

    def test_a_dropped_connection_is_retried(self, monkeypatch):
        calls = queue_posts(
            monkeypatch, httpx.ConnectError("reset"), reply(200, ANSWER)
        )

        assert providers.complete([{"role": "user", "content": "q"}]) == "an answer"
        assert len(calls) == 2

    def test_an_unrecognisable_reply_says_so(self, monkeypatch):
        queue_posts(monkeypatch, reply(200, {"unexpected": True}))

        with pytest.raises(ProviderError, match="unexpected shape"):
            providers.complete([{"role": "user", "content": "q"}])


class TestStreamRetries:
    def test_a_throttled_stream_is_retried_and_then_yields(self, monkeypatch):
        calls = queue_streams(
            monkeypatch,
            FakeStream(429),
            FakeStream(200, fragments("one ", "two")),
        )

        assert "".join(providers.stream([{"role": "user", "content": "q"}])) == "one two"
        # One request per attempt. An earlier version probed the endpoint
        # first and then requested again, doubling the load against the
        # very limit it was meant to respect.
        assert len(calls) == 2

    def test_a_refused_stream_is_not_retried(self, monkeypatch):
        calls = queue_streams(monkeypatch, FakeStream(404))

        with pytest.raises(ProviderError, match="404"):
            list(providers.stream([{"role": "user", "content": "q"}]))
        assert len(calls) == 1

    def test_the_error_body_reaches_the_caller(self, monkeypatch):
        # A bare status code cannot distinguish a bad URL from a retired
        # model from a key without access.
        queue_streams(monkeypatch, FakeStream(404))

        with pytest.raises(ProviderError, match="provider said no"):
            list(providers.stream([{"role": "user", "content": "q"}]))

    def test_a_failure_mid_answer_is_not_restarted(self, monkeypatch):
        # The caller has already displayed what arrived. Retrying would
        # repeat it, so the error has to propagate instead.
        calls = queue_streams(
            monkeypatch,
            # No terminator: the connection dies partway through.
            FakeStream(
                200,
                deltas("half an "),
                error=httpx.ReadError("connection lost"),
            ),
            FakeStream(200, fragments("half an answer")),
        )

        received = []
        with pytest.raises(ProviderError, match="connection lost"):
            for fragment in providers.stream([{"role": "user", "content": "q"}]):
                received.append(fragment)

        assert received == ["half an "]
        assert len(calls) == 1

    def test_a_connection_failure_before_any_text_is_retried(self, monkeypatch):
        calls = queue_streams(
            monkeypatch,
            httpx.ConnectError("refused"),
            FakeStream(200, fragments("an answer")),
        )

        assert "".join(providers.stream([{"role": "user", "content": "q"}])) == "an answer"
        assert len(calls) == 2
