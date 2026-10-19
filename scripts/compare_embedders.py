"""
compare_embedders.py

Compares sentence embedding models for the dense retrieval arm.

The fusion sweep showed that widening the candidate pool and reweighting
the two retrievers buys nothing reliable, which leaves the obvious
remaining weakness: over 50 papers the dense arm reaches Hit@5 0.62 while
the lexical arm reaches 0.68. Fusing a weak ranker with a strong one
cannot fix the weak one. The embedding model is what sets that number.

The current model, all-MiniLM-L6-v2, has a 256 token limit and 384
dimensions. Chunks are sized to fit that limit, so the limit has shaped
the whole pipeline; a model with a longer window or a better trained
embedding space may lift the dense arm enough to change what fusion is
worth.

Each model re-embeds the corpus in memory. The stored index is never
touched, so an experiment cannot leave the deployed corpus in a strange
state.

Usage:
    python -m scripts.compare_embedders
    python -m scripts.compare_embedders --models sentence-transformers/all-mpnet-base-v2
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
from src.retriever import MODE_DENSE, MODE_HYBRID_RERANK, HybridRetriever
from src.vector_store import VectorStore

DEFAULT_MODELS = [
    "all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-base-en-v1.5",
]


# Sub- and superscript markers are a representation detail of the
# indexed text, not something a test set should depend on. Without this,
# changing how equations are rendered silently invalidates any query
# whose evidence term sits next to an exponent: "et0" stops matching once
# the chunk says "e_{t0}", and the benchmark reports a retrieval
# regression that is really a formatting change.
_SCRIPT_MARKERS = str.maketrans("", "", "^_{}")


def _normalize(text: str) -> str:
    return text.lower().translate(_SCRIPT_MARKERS)


def is_relevant(chunk, query: dict, titles: Dict[str, str]) -> bool:
    paper = query.get("paper")
    if paper and paper.lower() not in titles.get(chunk.doc_id, "").lower():
        return False
    text = _normalize(chunk.text)
    return all(_normalize(p) in text for p in query["must_contain"])


def score(retriever, queries, titles, k, mode) -> Dict[str, float]:
    hits = mrr = 0.0
    for query in queries:
        chunks = [r.chunk for r in retriever.retrieve(
            query["question"], top_k=k, mode=mode).chunks]
        ranks = [i for i, c in enumerate(chunks, 1) if is_relevant(c, query, titles)]
        if ranks:
            hits += 1
            mrr += 1.0 / ranks[0]
    n = len(queries)
    return {"hit": hits / n, "mrr": mrr / n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=Path, default=Path("eval/benchmark.json"))
    ap.add_argument("--split", default="val", choices=["val", "train"],
                    help="test is deliberately not offered; confirm once, separately")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    spec = json.loads(args.queries.read_text(encoding="utf-8"))
    queries = [q for q in spec["queries"] if q.get("split") == args.split]
    print(f"{args.split}: {len(queries)} queries")

    base = Corpus(Embedder())
    titles = {d.doc_id: d.title for d in base.documents}
    chunks = base.store.chunks
    texts = [c.text for c in chunks]
    reranker = CrossEncoderReranker()
    print(f"corpus: {len(base)} papers, {len(chunks)} chunks\n")

    rows: List[tuple] = []
    for name in args.models:
        print(f"embedding with {name} ...", flush=True)
        try:
            embedder = Embedder(model_name=name)
            start = time.perf_counter()
            vectors = embedder.encode(texts)
            index_seconds = time.perf_counter() - start

            store = VectorStore(embedding_dim=vectors.shape[1])
            store.set_contents(chunks, vectors)
            retriever = HybridRetriever(embedder, store, reranker=reranker)

            dense = score(retriever, queries, titles, args.k, MODE_DENSE)
            full = score(retriever, queries, titles, args.k, MODE_HYBRID_RERANK)
        except Exception as error:
            print(f"  skipped: {type(error).__name__}: {str(error)[:140]}")
            continue
        rows.append((name, embedder.embedding_dim, dense, full, index_seconds))
        print(f"  dense Hit@{args.k}={dense['hit']:.3f} MRR={dense['mrr']:.3f} | "
              f"full Hit@{args.k}={full['hit']:.3f} MRR={full['mrr']:.3f} | "
              f"index {index_seconds:.0f}s", flush=True)

    width = max(len(r[0]) for r in rows) + 2 if rows else 20
    print(f"\n{'model':<{width}}{'dim':>5}{'denseHit':>10}{'denseMRR':>10}"
          f"{'fullHit':>9}{'fullMRR':>9}{'index s':>9}")
    print("-" * (width + 52))
    for name, dim, dense, full, secs in rows:
        print(f"{name:<{width}}{dim:>5}{dense['hit']:>10.3f}{dense['mrr']:>10.3f}"
              f"{full['hit']:>9.3f}{full['mrr']:>9.3f}{secs:>9.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
