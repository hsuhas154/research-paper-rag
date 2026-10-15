"""
tune_reranker.py

Fine-tunes several re-ranker configurations, selects one on the
VALIDATION split, and reports the winner's score on TEST once.

This exists because the first attempt at domain adaptation had no
validation split. Configurations were compared on the held-out set and
the best one reported from that same set, which makes the reported
number optimistic by construction: the choice was fitted to the data it
was then scored on. That was recorded as a flaw at the time and this
script is the fix.

The protocol here is the ordinary one:

  train    30 papers, used to build questions and mine negatives
  val      10 papers, used to choose epochs and learning rate
  test     10 papers, scored once, after the choice is locked

Any configuration that wins on validation but loses on test is reported
as such rather than quietly dropped, because that gap is the measurement
worth having.

Usage:
    python -m scripts.tune_reranker
    python -m scripts.tune_reranker --base cross-encoder/ms-marco-MiniLM-L-12-v2
"""

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.corpus import Corpus
from src.embedder import Embedder
from src.reranker import STOCK_RERANKER_MODEL, CrossEncoderReranker
from src.retriever import MODE_HYBRID, HybridRetriever

PYTHON = sys.executable


def is_relevant(chunk, query: dict, titles: Dict[str, str]) -> bool:
    paper = query.get("paper")
    if paper and paper.lower() not in titles.get(chunk.doc_id, "").lower():
        return False
    text = chunk.text.lower()
    return all(p.lower() in text for p in query["must_contain"])


def score_model(model_name, pools, queries, titles, k) -> Dict[str, float]:
    """Re-ranks pre-computed candidate pools, so only the model varies."""
    reranker = CrossEncoderReranker(model_name=model_name)
    reranker.rerank("warm up", pools[0][:2])
    hits = mrr = 0.0
    for query, pool in zip(queries, pools):
        ordered = [r.chunk for r in reranker.rerank(query["question"], pool)][:k]
        ranks = [i for i, c in enumerate(ordered, 1) if is_relevant(c, query, titles)]
        if ranks:
            hits += 1
            mrr += 1.0 / ranks[0]
    n = len(queries)
    return {"hit": hits / n, "mrr": mrr / n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=Path, default=Path("eval/benchmark.json"))
    ap.add_argument("--data", type=Path, default=Path("eval/rerank_train.jsonl"))
    ap.add_argument("--base", default=STOCK_RERANKER_MODEL)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--pool", type=int, default=30)
    ap.add_argument("--epochs", type=int, nargs="*", default=[1, 2])
    ap.add_argument("--lrs", type=float, nargs="*", default=[5e-6, 1e-5, 2e-5])
    ap.add_argument("--negatives", type=int, nargs="*", default=[3])
    ap.add_argument("--workdir", type=Path, default=Path("models/tuning"))
    args = ap.parse_args()

    spec = json.loads(args.queries.read_text(encoding="utf-8"))
    val = [q for q in spec["queries"] if q.get("split") == "val"]
    test = [q for q in spec["queries"] if q.get("split") == "test"]
    if not val or not test:
        print("benchmark has no val/test queries")
        return 1

    corpus = Corpus(Embedder())
    titles = {d.doc_id: d.title for d in corpus.documents}
    retriever = HybridRetriever(corpus.embedder, corpus.store, candidate_pool=args.pool)

    print(f"val {len(val)} queries, test {len(test)} queries, "
          f"{len(corpus)} papers\nbuilding candidate pools once...", flush=True)
    val_pools = [retriever.retrieve(q["question"], top_k=args.pool,
                                    mode=MODE_HYBRID).chunks for q in val]
    test_pools = [retriever.retrieve(q["question"], top_k=args.pool,
                                     mode=MODE_HYBRID).chunks for q in test]

    baseline_val = score_model(args.base, val_pools, val, titles, args.k)
    print(f"\nbaseline {args.base}")
    print(f"  val  Hit@{args.k}={baseline_val['hit']:.3f} MRR={baseline_val['mrr']:.3f}",
          flush=True)

    args.workdir.mkdir(parents=True, exist_ok=True)
    results: List[tuple] = []

    for epochs, lr, negatives in itertools.product(args.epochs, args.lrs, args.negatives):
        tag = f"e{epochs}_lr{lr:g}_n{negatives}"
        out = args.workdir / tag
        print(f"\ntraining {tag} ...", flush=True)
        command = [
            PYTHON, "-m", "scripts.train_reranker",
            "--data", str(args.data), "--base", args.base, "--out", str(out),
            "--epochs", str(epochs), "--lr", str(lr),
            "--negatives-per-positive", str(negatives),
        ]
        finished = subprocess.run(command, capture_output=True, text=True)
        if finished.returncode != 0:
            print(f"  training failed: {finished.stderr.strip()[-300:]}")
            continue

        metrics = score_model(str(out), val_pools, val, titles, args.k)
        results.append((tag, str(out), metrics))
        print(f"  val  Hit@{args.k}={metrics['hit']:.3f} MRR={metrics['mrr']:.3f}",
              flush=True)

    if not results:
        print("\nno configuration trained successfully")
        return 1

    print(f"\n{'config':<22}{'val Hit':>9}{'val MRR':>9}")
    print("-" * 40)
    print(f"{'baseline':<22}{baseline_val['hit']:>9.3f}{baseline_val['mrr']:>9.3f}")
    for tag, _, m in results:
        print(f"{tag:<22}{m['hit']:>9.3f}{m['mrr']:>9.3f}")

    best_tag, best_path, best_val = max(results, key=lambda r: (r[2]["hit"], r[2]["mrr"]))
    print(f"\nselected on validation: {best_tag}")

    # Test is read exactly here, once, after the choice is locked.
    baseline_test = score_model(args.base, test_pools, test, titles, args.k)
    best_test = score_model(best_path, test_pools, test, titles, args.k)

    print(f"\n{'':<22}{'test Hit':>10}{'test MRR':>10}")
    print("-" * 42)
    print(f"{'baseline':<22}{baseline_test['hit']:>10.3f}{baseline_test['mrr']:>10.3f}")
    print(f"{best_tag:<22}{best_test['hit']:>10.3f}{best_test['mrr']:>10.3f}")

    improved = (best_test["hit"], best_test["mrr"]) > (baseline_test["hit"], baseline_test["mrr"])
    print(f"\nverdict: fine-tuning {'beats' if improved else 'does NOT beat'} "
          f"the stock model on test.")
    if best_val["hit"] > baseline_val["hit"] and not improved:
        print("Won on validation and lost on test, which is the gap the "
              "three-way split exists to expose.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
