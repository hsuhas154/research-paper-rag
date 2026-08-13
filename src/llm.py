"""
llm.py

Wraps calls to a locally running Ollama model for the "generation"
half of the RAG pipeline: given a question and a set of retrieved
context chunks, produces an answer grounded in that context.

Requires 'ollama serve' to be running in a separate terminal, with
the model already pulled (see Stage 1 setup).
"""

from typing import List

import ollama

from src.vector_store import RetrievedChunk

LLM_MODEL_NAME = "llama3.1:8b"

SYSTEM_PROMPT = (
    "You are a research assistant that answers questions about a specific "
    "academic paper. You must answer ONLY using the provided context "
    "excerpts from the paper. If the context does not contain enough "
    "information to answer the question, say so explicitly rather than "
    "guessing or relying on outside knowledge. When you use a fact from a "
    "specific excerpt, cite it inline as [Excerpt N]."
)


def build_prompt(question: str, retrieved_chunks: List[RetrievedChunk]) -> str:
    """
    Assembles the retrieved chunks and the question into a single
    prompt string, with each chunk labeled so the model can cite
    which excerpt a fact came from.
    """
    context_blocks = []
    for i, r in enumerate(retrieved_chunks, start=1):
        context_blocks.append(f"[Excerpt {i}]\n{r.chunk.text}")
    context_str = "\n\n".join(context_blocks)

    return (
        f"Context excerpts from the paper:\n\n{context_str}\n\n"
        f"Question: {question}\n\n"
        f"Answer the question using only the context excerpts above. "
        f"Cite excerpt numbers like [Excerpt 1] when you use a specific fact."
    )


def generate_answer(
    question: str,
    retrieved_chunks: List[RetrievedChunk],
    model: str = LLM_MODEL_NAME,
) -> str:
    """
    Calls the local Ollama model with the assembled RAG prompt and
    returns the generated answer text.
    """
    user_prompt = build_prompt(question, retrieved_chunks)

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "num_ctx": 4096,   # matches Ollama's default context window; revisit if
                                # prompts grow larger in later phases (more chunks,
                                # multi-turn conversation history, etc.)
            "temperature": 0.2,  # low temperature: prioritize faithful, grounded
                                  # answers over creative variation
        },
    )
    return response["message"]["content"]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.pdf_processor import load_and_clean_pdf
    from src.chunker import chunk_text
    from src.embedder import Embedder
    from src.vector_store import VectorStore

    if len(sys.argv) != 3:
        print('Usage: python src/llm.py <path_to_pdf> "<question>"')
        sys.exit(1)

    pdf_path, question = sys.argv[1], sys.argv[2]

    print("Loading embedding model...")
    embedder = Embedder()

    print("Processing PDF...")
    cleaned = load_and_clean_pdf(pdf_path)
    chunks = chunk_text(cleaned)
    embeddings = embedder.encode([c.text for c in chunks])

    store = VectorStore(embedding_dim=embeddings.shape[1])
    store.build(chunks, embeddings)

    print(f"Retrieving top chunks for: {question}")
    query_embedding = embedder.encode([question])[0]
    retrieved = store.search(query_embedding, top_k=5)  # raised from 3, per Stage 5 findings

    print("Generating answer (first call is slower - model loads into VRAM)...")
    answer = generate_answer(question, retrieved)

    print("\n=== ANSWER ===")
    print(answer)

    print("\n=== SOURCE CHUNKS ===")
    for i, r in enumerate(retrieved, start=1):
        preview = r.chunk.text[:150].replace("\n", " ")
        print(f"\n[Excerpt {i}] score={r.score:.4f} (chunk {r.chunk.chunk_id})")
        print(f"    {preview}...")