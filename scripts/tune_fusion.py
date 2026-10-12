"""
tune_fusion.py

Sweeps first-stage retrieval settings on the VALIDATION split.

The re-ranker study concluded that re-ranking was not the bottleneck:
Hit@5 was identical for every re-ranker tried, including one fine-tuned
on this corpus, which means the ceiling is whatever the first stage puts
in the candidate pool. This sweep tunes that first stage.

It reads only the validation split. The test split is deliberately not an
option here, because selecting a configuration by looking at test is how
the Phase 3 re-ranker study ended up with a biased number, and the whole
point of the three-way split is to not repeat that.

Usage:
    python -m scripts.tune_fusion
    python -m scripts.tune_fusion --k 5 --pools 30 50 --lexical-weights 1 2 3
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.corpus import Corpus
from src.embedder import Embedder
from src.reranker import CrossEncoderReranker
from src.retriever import MODE_HYBRID_RERANK, HybridRetriever


def is_relevant(chunk, query: dict, titles: Dict[str, str]) -> bool:
    paper = query.get("paper")
    if paper and paper.lower() not in titles.get(chunk.doc_id, "").lower():
        return False
    text = chunk.text.lower()
    return all(p.lower() in text for p in query["must_contain"])


def score(retriever, queries, titles, k, mode) -> Dict[str, float]:
    hits = mrr = precision = 0.0
    for query in queries:
        chunks = [r.chunk for r in retriever.retrieve(
            query["question"], top_k=k, mode=mode).chunks]
        ranks = [i for i, c in enumerate(chunks, 1) if is_relevant(c, query, titles)]
        if ranks:
            hits += 1
            mrr += 1.0 / ranks[0]
        precision += len(ranks) / k
    n = len(queries)
    return {"hit": hits / n, "mrr": mrr / n, "p": precision / n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=Path, default=Path("eval/benchmark.json"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--pools", type=int, nargs="*", default=[30, 50, 80])
    ap.add_argument("--lexical-weights", type=float, nargs="*", default=[1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--mode", default=MODE_HYBRID_RERANK)
    args = ap.parse_args()

    spec = json.loads(args.queries.read_text(encoding="utf-8"))
    queries = [q for q in spec["queries"] if q.get("split") == "val"]
    if not queries:
        print("no validation queries found; regenerate the benchmark")
        return 1

    corpus = Corpus(Embedder())
    titles = {d.doc_id: d.title for d in corpus.documents}
    reranker = CrossEncoderReranker()
    print(f"validation: {len(queries)} queries over {len(corpus)} papers, "
          f"mode={args.mode}, k={args.k}\n")

    rows: List[tuple] = []
    for pool in args.pools:
        for weight in args.lexical_weights:
            retriever = HybridRetriever(
                corpus.embedder, corpus.store, reranker=reranker,
                candidate_pool=pool, dense_weight=1.0, lexical_weight=weight,
            )
            metrics = score(retriever, queries, titles, args.k, args.mode)
            rows.append((pool, weight, metrics))
            print(f"  pool={pool:<4} lexical_weight={weight:<5} "
                  f"Hit@{args.k}={metrics['hit']:.3f} MRR={metrics['mrr']:.3f} "
                  f"P={metrics['p']:.3f}", flush=True)

    best = max(rows, key=lambda r: (r[2]["hit"], r[2]["mrr"]))
    print(f"\nbest on validation: pool={best[0]} lexical_weight={best[1]} "
          f"Hit@{args.k}={best[2]['hit']:.3f} MRR={best[2]['mrr']:.3f}")
    print("Apply by editing DEFAULT_LEXICAL_WEIGHT and the candidate_pool "
          "default in src/retriever.py, then confirm once on the test split.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
