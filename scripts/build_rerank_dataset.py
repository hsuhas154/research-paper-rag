"""
build_rerank_dataset.py

Builds an in-domain training set for re-ranker fine-tuning, from the
corpus itself.

There is no labelled query-passage data for a library of research papers,
so it is generated: the local LLM reads each chunk and writes a question
that chunk answers. That gives positives. Negatives are then mined with
the project's own retrievers - the passages BM25 and dense search rank
highest for that question, excluding the true one. Those are hard
negatives by construction: they are exactly the passages the deployed
system confuses with the right answer, including passages from other
papers, which is the cross-paper contamination the Table 2 evaluation
found.

Papers are split by document, not by chunk. Holding out whole papers is
the only way to tell whether a fine-tuned model has learned something
transferable about scientific prose rather than memorised these
particular passages, and the evaluation depends on that distinction being
real.

Usage:
    python -m scripts.build_rerank_dataset --out eval/rerank_train.jsonl
    python -m scripts.build_rerank_dataset --max-chunks 200   (quick run)
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ollama

from src.corpus import Corpus
from src.embedder import Embedder
from src.llm import LLM_MODEL_NAME
from src.retriever import HybridRetriever

# Papers held out of training entirely, chosen to span the corpus's
# subject areas so the held-out score is not a single-domain fluke.
HELD_OUT_TITLE_FRAGMENTS = [
    "attention is all you need",
    "dai et al. astronomy",
    "predicting sepsis",
    "characterization and modeling of edge",
]

QUESTION_PROMPT = (
    "Below is a passage from a research paper.\n\n"
    "Write ONE specific question that this passage answers. The question must:\n"
    "- be answerable using only this passage\n"
    "- name the specific quantity, method, or finding involved\n"
    "- not mention 'the passage', 'the excerpt', or 'this text'\n"
    "- be a single line, no preamble\n\n"
    "Passage:\n{passage}\n\n"
    "Question:"
)


def generate_question(passage: str, model: str) -> str:
    """Asks the local LLM for one question the passage answers."""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": QUESTION_PROMPT.format(passage=passage[:1500])}],
        options={"temperature": 0.3, "num_ctx": 2048, "num_predict": 60},
    )
    text = response["message"]["content"].strip()
    # Models sometimes answer with a label or quotes despite instructions.
    text = re.sub(r"^(question|q)\s*[:\-]\s*", "", text, flags=re.IGNORECASE).strip()
    return text.strip('"').split("\n")[0].strip()


def usable(question: str) -> bool:
    """Rejects degenerate generations rather than training on them."""
    if not (20 <= len(question) <= 200):
        return False
    if "?" not in question:
        return False
    banned = ("passage", "excerpt", "this text", "the above", "the following")
    return not any(b in question.lower() for b in banned)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("eval/rerank_train.jsonl"))
    ap.add_argument("--negatives", type=int, default=4, help="hard negatives per question")
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--min-words", type=int, default=60,
                    help="skip chunks too short to support a real question")
    ap.add_argument("--model", default=LLM_MODEL_NAME)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    corpus = Corpus(Embedder())
    retriever = HybridRetriever(corpus.embedder, corpus.store)
    titles = {d.doc_id: d.title.lower() for d in corpus.documents}

    held_out = {
        doc_id for doc_id, title in titles.items()
        if any(f in title for f in HELD_OUT_TITLE_FRAGMENTS)
    }
    print(f"corpus: {len(corpus)} papers, {len(corpus.store)} chunks")
    print(f"held out of training: {len(held_out)} papers")
    for doc_id in held_out:
        print(f"  {titles[doc_id][:70]}")

    train_chunks = [
        (i, c) for i, c in enumerate(corpus.store.chunks)
        if c.doc_id not in held_out and c.word_count >= args.min_words
    ]
    random.shuffle(train_chunks)
    if args.max_chunks:
        train_chunks = train_chunks[: args.max_chunks]
    print(f"generating questions for {len(train_chunks)} training chunks\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    start = time.time()

    with args.out.open("w", encoding="utf-8") as handle:
        for n, (index, chunk) in enumerate(train_chunks, 1):
            try:
                question = generate_question(chunk.text, args.model)
            except Exception as error:
                print(f"  [{n}] generation failed: {error}")
                skipped += 1
                continue

            if not usable(question):
                skipped += 1
                continue

            # Hard negatives: what the deployed retrievers actually
            # confuse with the right passage for this question.
            candidates = retriever.retrieve(question, top_k=args.negatives + 4, mode="hybrid").chunks
            negatives: List[str] = []
            for hit in candidates:
                if hit.chunk is chunk:
                    continue
                negatives.append(hit.chunk.text)
                if len(negatives) == args.negatives:
                    break

            handle.write(json.dumps({
                "question": question,
                "positive": chunk.text,
                "negatives": negatives,
                "doc_id": chunk.doc_id,
                "chunk_index": index,
            }) + "\n")
            written += 1

            if n % 25 == 0:
                rate = (time.time() - start) / n
                print(f"  [{n}/{len(train_chunks)}] written={written} skipped={skipped} "
                      f"{rate:.1f}s/chunk", flush=True)

    print(f"\nwrote {written} examples to {args.out} ({skipped} skipped) "
          f"in {time.time() - start:.0f}s")
    held_out_path = args.out.with_suffix(".heldout.json")
    held_out_path.write_text(json.dumps(sorted(held_out), indent=1), encoding="utf-8")
    print(f"held-out doc ids -> {held_out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
