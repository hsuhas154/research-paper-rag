"""
compare_rerankers.py

Compares re-ranking models on the same first-stage candidates.

The first stage (hybrid retrieval plus Reciprocal Rank Fusion) is run once
per query and cached, so every model re-ranks an identical candidate pool.
Without that, differences between models would be entangled with run to
run variation in what the first stage proposed.

Usage:
    python -m scripts.compare_rerankers
    python -m scripts.compare_rerankers --models cross-encoder/ms-marco-MiniLM-L-6-v2 BAAI/bge-reranker-base
    python -m scripts.compare_rerankers --queries eval/corpus_queries.json --k 5
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.corpus import Corpus
from src.embedder import Embedder
from src.reranker import CrossEncoderReranker
from src.retriever import MODE_HYBRID, HybridRetriever

DEFAULT_MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "BAAI/bge-reranker-base",
]


def is_relevant(chunk, query: dict, titles: Dict[str, str]) -> bool:
    paper = query.get("paper")
    if paper and paper.lower() not in titles.get(chunk.doc_id, "").lower():
        return False
    text = chunk.text.lower()
    return all(p.lower() in text for p in query["must_contain"])


def first_stage(retriever: HybridRetriever, queries: List[dict], pool: int) -> List[list]:
    """Candidate pools, one per query, shared by every model under test."""
    return [
        retriever.retrieve(q["question"], top_k=pool, mode=MODE_HYBRID).chunks
        for q in queries
    ]


def score_order(ordered, queries, titles, k) -> Dict[str, float]:
    hits = mrr = precision = 0.0
    for chunks, query in zip(ordered, queries):
        ranks = [i for i, c in enumerate(chunks[:k], 1) if is_relevant(c, query, titles)]
        if ranks:
            hits += 1
            mrr += 1.0 / ranks[0]
        precision += len(ranks) / k
    n = len(queries)
    return {"hit": hits / n, "mrr": mrr / n, "precision": precision / n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=Path, default=Path("eval/corpus_queries.json"))
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--pool", type=int, default=30)
    ap.add_argument("--only-papers", nargs="*", default=None,
                    help="restrict to queries whose paper matches one of these substrings")
    ap.add_argument("--split", choices=["all", "heldout", "trained"], default="all",
                    help=("evaluate on all queries, only those targeting papers held out "
                          "of re-ranker training, or only those targeting trained papers"))
    args = ap.parse_args()

    spec = json.loads(args.queries.read_text(encoding="utf-8"))
    queries = spec["queries"]

    print("Loading corpus...")
    corpus = Corpus(Embedder())
    titles = {d.doc_id: d.title for d in corpus.documents}
    retriever = HybridRetriever(corpus.embedder, corpus.store, candidate_pool=args.pool)

    if args.split != "all":
        from scripts.build_rerank_dataset import HELD_OUT_TITLE_FRAGMENTS

        # Resolve each query's paper to a real corpus title before testing
        # membership. Matching on words from the shorthand label instead
        # catches "et" and "al" and marks the whole corpus held out.
        held_out_titles = [
            t for t in titles.values()
            if any(f in t.lower() for f in HELD_OUT_TITLE_FRAGMENTS)
        ]

        def targets_held_out(query: dict) -> bool:
            paper = (query.get("paper") or "").lower()
            return any(paper and paper in t.lower() for t in held_out_titles)

        want_held_out = args.split == "heldout"
        queries = [q for q in queries if targets_held_out(q) == want_held_out]
        print(f"split={args.split}: {len(queries)} queries")

    if args.only_papers:
        wanted = [p.lower() for p in args.only_papers]
        queries = [q for q in queries
                   if any(w in (q.get("paper") or "").lower() for w in wanted)]
        print(f"restricted to {len(queries)} queries")

    print(f"Running first stage for {len(queries)} queries (pool {args.pool})...")
    pools = first_stage(retriever, queries, args.pool)

    fusion_order = [[r.chunk for r in pool] for pool in pools]
    rows = [("no rerank (fusion order)", score_order(fusion_order, queries, titles, args.k), 0.0)]

    for name in args.models:
        print(f"Loading {name} ...")
        try:
            reranker = CrossEncoderReranker(model_name=name)
            # Force the lazy load before timing. Without this the first
            # query absorbs several seconds of model loading, which on a
            # small split dominates the per-query figure and makes two
            # identical architectures look minutes apart.
            reranker.rerank("warm up", pools[0][:2])
            start = time.perf_counter()
            ordered = [
                [r.chunk for r in reranker.rerank(q["question"], pool)]
                for q, pool in zip(queries, pools)
            ]
            elapsed = (time.perf_counter() - start) / len(queries)
        except Exception as error:
            print(f"  skipped: {type(error).__name__}: {str(error)[:120]}")
            continue
        rows.append((name, score_order(ordered, queries, titles, args.k), elapsed))

    width = max(len(r[0]) for r in rows) + 2
    print(f"\n{'model':<{width}}{'Hit@k':>8}{'MRR@k':>9}{'P@k':>8}{'sec/query':>11}")
    print("-" * (width + 36))
    for name, m, secs in rows:
        print(f"{name:<{width}}{m['hit']:>8.3f}{m['mrr']:>9.3f}{m['precision']:>8.3f}{secs:>11.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
