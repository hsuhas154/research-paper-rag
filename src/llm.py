"""
llm.py

Wraps calls to a locally running Ollama model for the "generation"
half of the RAG pipeline: given a question and a set of retrieved
context chunks, produces an answer grounded in that context.

Phase 2 changes: each excerpt in the prompt is now labeled with the
paper, page, and section it came from, and the model is told to cite
those labels. With a multi-paper corpus this stops being cosmetic - a
question can now be answered from two different papers at once, and an
answer that does not say which paper a claim came from is not
verifiable. The conversation-history parameter is also new, laying the
groundwork for the multi-turn support planned for Phase 3.

Requires 'ollama serve' to be running, with the model already pulled.
"""

from typing import Dict, Iterator, List, Optional, Sequence

from src import providers
from src.providers import ProviderError
from src.vector_store import RetrievedChunk

# Kept for callers that import it. The model actually used comes from
# providers.model_name(), which defaults to this for local Ollama and to
# a hosted equivalent when LLM_PROVIDER is set.
LLM_MODEL_NAME = providers.DEFAULT_OLLAMA_MODEL
KEEP_ALIVE = providers.KEEP_ALIVE

SYSTEM_PROMPT = (
    "You are a research assistant that answers questions about academic papers. "
    "You must answer ONLY using the provided context excerpts. If the context "
    "does not contain enough information to answer the question, say so "
    "explicitly rather than guessing or relying on outside knowledge.\n\n"
    "Every factual claim in your answer must be followed by a citation of the "
    "excerpt it came from, written as [Excerpt N]. Cite multiple excerpts as "
    "[Excerpt 1][Excerpt 3] when a claim draws on more than one. When excerpts "
    "come from different papers and disagree, say so rather than silently "
    "choosing one."
)


def format_excerpt_label(
    index: int,
    retrieved: RetrievedChunk,
    document_titles: Optional[Dict[str, str]] = None,
) -> str:
    """
    Builds the header line for one excerpt in the prompt, e.g.
    "[Excerpt 2] (Dai et al. 2024, p. 5, Section 2.2 Atmospheric chemistry)".

    The provenance goes into the prompt, not just the UI, so that the
    model can reproduce it in its citations and a reader can check a
    claim without hunting through the source panel.
    """
    titles = document_titles or {}
    title = titles.get(retrieved.chunk.doc_id)
    source = retrieved.chunk.citation(document_title=title)
    return f"[Excerpt {index}] ({source})"


def build_prompt(
    question: str,
    retrieved_chunks: Sequence[RetrievedChunk],
    document_titles: Optional[Dict[str, str]] = None,
) -> str:
    """
    Assembles the retrieved chunks and the question into a single
    prompt string, with each chunk labeled by its source so the model
    can cite where a fact came from.
    """
    context_blocks = [
        f"{format_excerpt_label(i, r, document_titles)}\n{r.chunk.text}"
        for i, r in enumerate(retrieved_chunks, start=1)
    ]
    context = "\n\n".join(context_blocks)

    return (
        f"Context excerpts:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context excerpts above, citing excerpt numbers "
        f"like [Excerpt 1] for every factual claim."
    )


def build_messages(
    question: str,
    retrieved_chunks: Sequence[RetrievedChunk],
    document_titles: Optional[Dict[str, str]] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Assembles the full message list sent to the model."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append(
        {"role": "user", "content": build_prompt(question, retrieved_chunks, document_titles)}
    )
    return messages


NO_CONTEXT_ANSWER = (
    "No relevant passages were retrieved, so there is nothing to base an "
    "answer on. Try rephrasing the question, or check that the right "
    "papers are selected."
)


def stream_answer(
    question: str,
    retrieved_chunks: Sequence[RetrievedChunk],
    document_titles: Optional[Dict[str, str]] = None,
    history: Optional[List[Dict[str, str]]] = None,
    model: Optional[str] = None,
    num_ctx: int = 8192,
) -> Iterator[str]:
    """
    Yields the answer in fragments as the model produces them.

    Generation takes 3 to 30 seconds depending on answer length, and
    during that time a non-streaming caller shows nothing at all. The
    total is the same either way; what changes is that the reader can
    start on the first sentence while the rest arrives, and can tell the
    difference between a slow answer and a hung process.
    """
    if not retrieved_chunks:
        yield NO_CONTEXT_ANSWER
        return

    messages = build_messages(question, retrieved_chunks, document_titles, history)
    try:
        yield from providers.stream(messages, model=model, num_ctx=num_ctx)
    except ProviderError as error:
        raise RuntimeError(str(error)) from error


def generate_answer(
    question: str,
    retrieved_chunks: Sequence[RetrievedChunk],
    document_titles: Optional[Dict[str, str]] = None,
    history: Optional[List[Dict[str, str]]] = None,
    model: Optional[str] = None,
    num_ctx: int = 8192,
) -> str:
    """
    Calls the local Ollama model with the assembled RAG prompt and
    returns the generated answer text.

    Args:
        question: the user's question
        retrieved_chunks: passages to ground the answer in
        document_titles: doc_id -> title, for citation labels
        history: prior {"role", "content"} turns, inserted between the
            system prompt and the current question. Unused by the Gradio
            app today; the parameter exists so Phase 3's conversational
            memory does not require reworking this signature.
        num_ctx: context window. Raised from Phase 1's 4096 because
            excerpts now carry source headers and hybrid retrieval tends
            to return longer, more varied passages - 4096 had started
            truncating prompts at top_k=5, which silently drops the last
            excerpts and makes their citations unanswerable.

    Raises:
        RuntimeError: if the Ollama server is unreachable or the model
            is not pulled, with a message saying how to fix it.
    """
    if not retrieved_chunks:
        return NO_CONTEXT_ANSWER

    messages = build_messages(question, retrieved_chunks, document_titles, history)
    try:
        return providers.complete(messages, model=model, num_ctx=num_ctx)
    except ProviderError as error:
        raise RuntimeError(str(error)) from error


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.corpus import Corpus
    from src.embedder import Embedder
    from src.retriever import HybridRetriever

    if len(sys.argv) != 2:
        print('Usage: python -m src.llm "<question>"   (answers from the saved corpus)')
        sys.exit(1)

    corpus = Corpus(Embedder())
    if corpus.is_empty():
        print("Corpus is empty. Add a paper first: python -m src.corpus add <pdf>")
        sys.exit(1)

    retriever = HybridRetriever(corpus.embedder, corpus.store)
    result = retriever.retrieve(sys.argv[1], top_k=5)
    titles = {d.doc_id: d.short_title for d in corpus.documents}

    print("Generating answer (first call is slower - model loads into VRAM)...\n")
    print("=== ANSWER ===")
    print(generate_answer(sys.argv[1], result.chunks, document_titles=titles))

    print("\n=== SOURCES ===")
    for i, r in enumerate(result.chunks, start=1):
        print(f"{format_excerpt_label(i, r, titles)}  score={r.score:.4f}")
