"""
providers.py

Where generation happens: the local Ollama runtime, or any
OpenAI-compatible HTTP endpoint.

Local is the default and stays the default. This project was built around
the property that nothing leaves the machine, and that property is worth
keeping for the normal case: your papers, your questions and the model
all sit on one disk.

The hosted option exists for one reason. A public demo cannot ship a
5 GB model or a GPU, so a deployed instance needs generation to happen
somewhere else. That is a real trade, not a free upgrade: every request
sends the retrieved passages, roughly 800 to 1,000 words of verbatim
paper text, to whoever runs the endpoint. Opt in deliberately, and do not
point it at anything carrying unpublished work.

Only generation moves. Extraction, embedding, indexing, retrieval and
re-ranking stay local in both modes, because those are the parts that
would have to send the whole corpus rather than a slice of it.

Configuration:
    LLM_PROVIDER   "ollama" (default) or "openai"
    LLM_MODEL      model name, defaulting per provider
    LLM_BASE_URL   openai only, for example https://api.groq.com/openai/v1
    LLM_API_KEY    openai only

Any OpenAI-compatible service works, including Groq, Together, and a
self-hosted vLLM, so choosing one is a configuration change rather than a
code change.
"""

import json
import os
import time
from typing import Dict, Iterator, List, Optional

OLLAMA = "ollama"
OPENAI = "openai"

DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
# Hosted model names are not stable. llama-3.1-8b-instant was the
# obvious choice because it matches the local model, and it had already
# been retired by the time this was tested: the endpoint answers 404
# rather than saying the name is unknown. Check what a key can actually
# reach before trusting any default here:
#   curl -s $LLM_BASE_URL/models -H "Authorization: Bearer $LLM_API_KEY"
DEFAULT_OPENAI_MODEL = "openai/gpt-oss-20b"

# Ollama unloads an idle model after five minutes, and the next question
# then pays a 30 to 40 second reload.
KEEP_ALIVE = "60m"


def provider_name() -> str:
    """Which backend generation goes to."""
    name = os.environ.get("LLM_PROVIDER", "").strip().lower()
    return name if name in (OLLAMA, OPENAI) else OLLAMA


def model_name() -> str:
    """The model to generate with, defaulting per provider."""
    override = os.environ.get("LLM_MODEL", "").strip()
    if override:
        return override
    return DEFAULT_OPENAI_MODEL if provider_name() == OPENAI else DEFAULT_OLLAMA_MODEL


def is_local() -> bool:
    """Whether generation stays on this machine."""
    return provider_name() == OLLAMA


def describe() -> str:
    """One line naming where generation happens, for the UI to show."""
    if is_local():
        return f"local Ollama, {model_name()}"
    host = os.environ.get("LLM_BASE_URL", "").strip() or "an OpenAI-compatible endpoint"
    return f"hosted: {model_name()} via {host}"


class ProviderError(RuntimeError):
    """Generation could not be reached or refused the request."""


# ----------------------------------------------------------------------
# Ollama
# ----------------------------------------------------------------------

def _ollama_chat(messages, model, num_ctx, stream):
    import ollama

    try:
        return ollama.chat(
            model=model, messages=messages, stream=stream, keep_alive=KEEP_ALIVE,
            options={"num_ctx": num_ctx, "temperature": 0.2},
        )
    except Exception as error:
        raise ProviderError(
            f"Could not reach the local LLM ({model}) via Ollama: {error}\n"
            f"Check that 'ollama serve' is running and that the model has been "
            f"pulled with 'ollama pull {model}'."
        ) from error


# ----------------------------------------------------------------------
# OpenAI-compatible HTTP
# ----------------------------------------------------------------------

def _openai_request(messages, model, stream: bool):
    """Builds the request. httpx is already a dependency of the API stack."""
    import httpx

    base = os.environ.get("LLM_BASE_URL", "").strip().rstrip("/")
    key = os.environ.get("LLM_API_KEY", "").strip()
    if not base:
        raise ProviderError("LLM_PROVIDER is 'openai' but LLM_BASE_URL is not set.")
    if not key:
        raise ProviderError("LLM_PROVIDER is 'openai' but LLM_API_KEY is not set.")

    return httpx.stream if stream else httpx.post, {
        "url": f"{base}/chat/completions",
        "headers": {"Authorization": f"Bearer {key}"},
        "json": {
            "model": model,
            "messages": list(messages),
            "temperature": 0.2,
            "stream": stream,
        },
        # Generous: a cold hosted model can take a while to answer, and
        # failing early here looks identical to the service being down.
        "timeout": 120.0,
    }


# A hosted endpoint throttles; a local one does not. 429 means "wait and
# try again", not "this failed", so it is handled here rather than in
# every caller. Without it a batch of forty evaluation questions dies
# partway through and discards the answers that already succeeded.
MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 2.0

# Long enough to ride out a per-minute quota, short enough that a
# genuinely unavailable provider is reported rather than waited out.
MAX_RETRY_DELAY_SECONDS = 60.0


def _backoff(attempt: int) -> float:
    """Exponential, so a sustained limit backs off instead of hammering."""
    return BASE_BACKOFF_SECONDS * (2 ** attempt)


def _retry_delay(response, attempt: int) -> float:
    """
    How long to wait before retrying, preferring the server's own
    instruction when it sends one. Retry-After is the provider saying
    exactly when its quota resets; guessing at that is strictly worse.
    """
    try:
        return min(float(response.headers.get("retry-after", "")), MAX_RETRY_DELAY_SECONDS)
    except (TypeError, ValueError):
        return _backoff(attempt)


def _is_retryable(status: int) -> bool:
    """429 is throttling; 5xx is the provider having a bad moment."""
    return status == 429 or 500 <= status < 600


def _status_error(error) -> ProviderError:
    """
    Turns an HTTP error into one that says what went wrong.

    The body has to be read explicitly on a streaming response, and it is
    the only place the provider explains itself: a bare "404" is
    indistinguishable between a bad URL, a retired model name, and a key
    without access to it.
    """
    try:
        error.response.read()
        detail = error.response.text[:300]
    except Exception:
        detail = ""
    return ProviderError(
        f"The generation endpoint returned {error.response.status_code}"
        + (f": {detail}" if detail else "")
    )


def _openai_complete(messages, model) -> str:
    import httpx

    _, kwargs = _openai_request(messages, model, stream=False)

    for attempt in range(MAX_RETRIES):
        final = attempt == MAX_RETRIES - 1
        try:
            response = httpx.post(**kwargs)
        except httpx.RequestError as error:
            # A dropped connection is as transient as a 429 and deserves
            # the same treatment, but there is no response to read a
            # Retry-After from.
            if final:
                raise ProviderError(
                    f"Could not reach the generation endpoint: {error}"
                ) from error
            time.sleep(_backoff(attempt))
            continue

        if _is_retryable(response.status_code) and not final:
            time.sleep(_retry_delay(response, attempt))
            continue

        # Out of retries, or a status no amount of waiting will change.
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise _status_error(error) from error

        try:
            return response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as error:
            raise ProviderError(
                f"The generation endpoint returned a reply in an unexpected "
                f"shape: {response.text[:200]}"
            ) from error

    raise ProviderError("The generation endpoint could not be reached.")


def _openai_stream(messages, model) -> Iterator[str]:
    import httpx

    _, kwargs = _openai_request(messages, model, stream=True)

    # Retrying covers only establishing the stream. The status line
    # arrives before any tokens do, so a throttled attempt costs one
    # request and yields nothing. Once fragments have been yielded the
    # caller has already shown them, and restarting behind its back would
    # duplicate text on screen, so a mid-stream failure propagates.
    yielded = False

    for attempt in range(MAX_RETRIES):
        final = attempt == MAX_RETRIES - 1
        delay = None

        try:
            with httpx.stream("POST", **kwargs) as response:
                if _is_retryable(response.status_code) and not final:
                    # Worked out inside the context manager, slept outside
                    # it, so the connection is not held open while waiting.
                    delay = _retry_delay(response, attempt)
                else:
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as error:
                        raise _status_error(error) from error

                    for line in response.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            return
                        try:
                            delta = json.loads(payload)["choices"][0].get("delta", {})
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
                        fragment = delta.get("content")
                        if fragment:
                            yielded = True
                            yield fragment
                    return
        except ProviderError:
            raise
        except httpx.RequestError as error:
            # Only a failure before the first fragment can be retried. A
            # connection dropping mid-answer has already put text on the
            # caller's screen, and starting over would repeat it.
            if final or yielded:
                raise ProviderError(
                    f"Could not reach the generation endpoint: {error}"
                ) from error
            delay = _backoff(attempt)
        except Exception as error:
            raise ProviderError(
                f"Could not reach the generation endpoint: {error}"
            ) from error

        if delay is not None:
            time.sleep(delay)


# ----------------------------------------------------------------------
# One interface
# ----------------------------------------------------------------------

def complete(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    num_ctx: int = 8192,
) -> str:
    """Generates a full answer."""
    model = model or model_name()
    if provider_name() == OPENAI:
        return _openai_complete(messages, model)
    return _ollama_chat(messages, model, num_ctx, stream=False)["message"]["content"]


def stream(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    num_ctx: int = 8192,
) -> Iterator[str]:
    """Yields the answer in fragments."""
    model = model or model_name()
    if provider_name() == OPENAI:
        yield from _openai_stream(messages, model)
        return
    for part in _ollama_chat(messages, model, num_ctx, stream=True):
        fragment = part.get("message", {}).get("content", "")
        if fragment:
            yield fragment
