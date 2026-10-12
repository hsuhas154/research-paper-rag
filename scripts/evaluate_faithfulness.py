"""
evaluate_faithfulness.py

Runs benchmark questions through the full pipeline and measures whether
the answers are grounded in the passages they were given.

Retrieval metrics stop at "did the evidence arrive". This continues: of
the claims the model then made, how many does that evidence actually
support, are the citations real, and how much of the supplied context was
used at all.

Usage:
    python -m scripts.evaluate_faithfulness --split val --limit 20
    python -m scripts.evaluate_faithfulness --split test --limit 30 --out eval/faithfulness_test.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.faithfulness import aggregate, evaluate_answer
from src.llm import LLM_MODEL_NAME
from src.rag_pipeline import RAGPipeline


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=Path, default=Path("eval/benchmark.json"))
    ap.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--judge", default=LLM_MODEL_NAME)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    spec = json.loads(args.queries.read_text(encoding="utf-8"))
    queries = spec["queries"]
    if args.split != "all":
        queries = [q for q in queries if q.get("split") == args.split]
    queries = queries[: args.limit]

    pipeline = RAGPipeline(top_k=args.top_k)
    print(f"{len(queries)} questions, split={args.split}, judge={args.judge}\n", flush=True)

    reports, rows = [], []
    start = time.time()
    for i, query in enumerate(queries, 1):
        answer = pipeline.answer_question(query["question"], top_k=args.top_k)
        report = evaluate_answer(answer.text, answer.sources, model=args.judge)
        reports.append(report)
        rows.append({
            "id": query["id"],
            "question": query["question"],
            "paper": query["paper"],
            **report.as_dict(),
        })
        faith = report.faithfulness
        print(f"  [{i:>3}/{len(queries)}] "
              f"faith={'n/a' if faith is None else f'{faith:.2f}'} "
              f"claims={len(report.claims)} "
              f"cites={report.n_valid_citations}/{report.n_citations}  "
              f"{query['question'][:52]}", flush=True)

    summary = aggregate(reports)
    summary["split"] = args.split
    summary["seconds"] = round(time.time() - start)

    print("\n=== summary ===")
    for key, value in summary.items():
        print(f"  {key:<22} {value}")
    print("\nNote: the judge is the generating model, which is lenient about "
          "its own output. Treat these as a floor, not a certificate.")

    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "per_question": rows}, indent=1),
                            encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
