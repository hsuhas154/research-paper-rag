"""
compare_retrieval.py

Measures the four retrieval strategies against a labelled query set, so
the claim that hybrid search and re-ranking improve retrieval is backed
by numbers rather than by a handful of cherry-picked examples.

Usage:
    python -m scripts.compare_retrieval                        # all modes, k=5
    python -m scripts.compare_retrieval --k 3 --verbose
    python -m scripts.compare_retrieval --queries eval/my_queries.json

Relevance is judged by content: a retrieved chunk is relevant if it
contains every phrase in the query's `must_contain` list. This keeps the
test set decoupled from chunk ids, which change whenever chunk size,
overlap, or segmentation changes - hand-labelled ids would silently rot
the first time the chunker is touched.

Metrics reported per mode:

    Hit@k       fraction of queries with at least one relevant chunk in
                the top k. "Did the pipeline find the evidence at all?" -
                the metric that matters most for RAG, since the generator
                only needs one good passage to answer from.
    MRR@k       mean reciprocal rank of the first relevant chunk. Rewards
                putting the evidence near the top, which matters because
                attention degrades over long contexts.
    Precision@k fraction of retrieved chunks that are relevant. Lower is
                tolerable; passing a little extra context to the LLM is
                far cheaper than missing the one passage that mattered.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.corpus import Corpus
from src.embedder import Embedder
from src.retriever import RETRIEVAL_MODES, HybridRetriever
from src.vector_store import RetrievedChunk

DEFAULT_QUERY_FILE = Path("eval/venus_queries.json")


def is_relevant(retrieved: RetrievedChunk, must_contain: List[str]) -> bool:
    """A chunk is relevant if it contains every required phrase."""
    text = retrieved.chunk.text.lower()
    return all(phrase.lower() in text for phrase in must_contain)


def evaluate_mode(
    retriever: HybridRetriever,
    queries: List[dict],
    mode: str,
    k: int,
) -> Dict[str, float]:
    """Runs every query through one retrieval mode and aggregates metrics."""
    hits = 0
    reciprocal_ranks = 0.0
    precision_total = 0.0
    elapsed = 0.0
    per_query: Dict[str, Optional[int]] = {}

    for query in queries:
        start = time.perf_counter()
        results = retriever.retrieve(query["question"], top_k=k, mode=mode).chunks
        elapsed += time.perf_counter() - start

        relevant_ranks = [
            rank for rank, r in enumerate(results, start=1)
            if is_relevant(r, query["must_contain"])
        ]

        if relevant_ranks:
            hits += 1
            reciprocal_ranks += 1.0 / relevant_ranks[0]
            per_query[query["id"]] = relevant_ranks[0]
        else:
            per_query[query["id"]] = None

        precision_total += len(relevant_ranks) / k if k else 0.0

    n = len(queries)
    return {
        "hit_rate": hits / n,
        "mrr": reciprocal_ranks / n,
        "precision": precision_total / n,
        "seconds_per_query": elapsed / n,
        "first_relevant_rank": per_query,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERY_FILE)
    parser.add_argument("--k", type=int, default=5, help="passages retrieved per query")
    parser.add_argument("--modes", nargs="*", default=RETRIEVAL_MODES)
    parser.add_argument("--verbose", action="store_true",
                        help="show the rank of the first relevant chunk per query")
    args = parser.parse_args()

    if not args.queries.exists():
        print(f"Query file not found: {args.queries}")
        return 1

    spec = json.loads(args.queries.read_text(encoding="utf-8"))
    queries = spec["queries"]

    print("Loading models and corpus...")
    corpus = Corpus(Embedder())
    if corpus.is_empty():
        print("Corpus is empty. Add the paper first:")
        print("  python -m src.corpus add data/uploads/<paper>.pdf")
        return 1

    retriever = HybridRetriever(corpus.embedder, corpus.store)
    print(f"Corpus: {len(corpus)} document(s), {len(corpus.store)} chunks")
    print(f"Evaluating {len(queries)} queries at k={args.k}\n")

    results = {mode: evaluate_mode(retriever, queries, mode, args.k) for mode in args.modes}

    header = f"{'mode':<16}{'Hit@k':>9}{'MRR@k':>9}{'P@k':>9}{'sec/query':>12}"
    print(header)
    print("-" * len(header))
    for mode, metrics in results.items():
        print(
            f"{mode:<16}"
            f"{metrics['hit_rate']:>9.2f}"
            f"{metrics['mrr']:>9.3f}"
            f"{metrics['precision']:>9.2f}"
            f"{metrics['seconds_per_query']:>12.3f}"
        )

    if args.verbose:
        print("\nRank of first relevant chunk per query (- = not found in top k):\n")
        id_width = max(len(q["id"]) for q in queries) + 2
        print(f"{'query':<{id_width}}" + "".join(f"{m:>16}" for m in args.modes))
        for query in queries:
            row = f"{query['id']:<{id_width}}"
            for mode in args.modes:
                rank = results[mode]["first_relevant_rank"][query["id"]]
                row += f"{(str(rank) if rank else '-'):>16}"
            print(row)

    return 0


if __name__ == "__main__":
    sys.exit(main())
