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
from typing import Dict, Iterator, List, Optional

OLLAMA = "ollama"
OPENAI = "openai"

DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_OPENAI_MODEL = "llama-3.1-8b-instant"

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


def _openai_complete(messages, model) -> str:
    import httpx

    _, kwargs = _openai_request(messages, model, stream=False)
    try:
        response = httpx.post(**kwargs)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as error:
        raise ProviderError(
            f"The generation endpoint returned {error.response.status_code}: "
            f"{error.response.text[:200]}"
        ) from error
    except Exception as error:
        raise ProviderError(f"Could not reach the generation endpoint: {error}") from error


def _openai_stream(messages, model) -> Iterator[str]:
    import httpx

    _, kwargs = _openai_request(messages, model, stream=True)
    try:
        with httpx.stream("POST", **kwargs) as response:
            response.raise_for_status()
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
                    yield fragment
    except httpx.HTTPStatusError as error:
        raise ProviderError(
            f"The generation endpoint returned {error.response.status_code}"
        ) from error
    except ProviderError:
        raise
    except Exception as error:
        raise ProviderError(f"Could not reach the generation endpoint: {error}") from error


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
