"""
validate_queries.py

Checks that every query in a test set is actually answerable: that at
least one chunk in the named paper contains the required phrases.

A query whose `must_contain` text appears nowhere would score zero for
every retrieval mode, making the system look broken when the fault is in
the test set. Running this before trusting a benchmark keeps the two
kinds of failure apart.

Usage:
    python -m scripts.validate_queries eval/corpus_queries.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.corpus import Corpus
from src.embedder import Embedder


def main() -> int:
    query_file = Path(sys.argv[1] if len(sys.argv) > 1 else "eval/corpus_queries.json")
    spec = json.loads(query_file.read_text(encoding="utf-8"))

    corpus = Corpus(Embedder())
    titles = {d.doc_id: d.title for d in corpus.documents}

    broken = []
    for query in spec["queries"]:
        paper = query.get("paper")
        matches = [
            c for c in corpus.store.chunks
            if (paper is None or paper.lower() in titles.get(c.doc_id, "").lower())
            and all(p.lower() in c.text.lower() for p in query["must_contain"])
        ]
        status = "ok" if matches else "UNANSWERABLE"
        if not matches:
            broken.append(query)
        print(f"{status:<13}{len(matches):>3} chunk(s)  {query['id']}")

    print(f"\n{len(spec['queries']) - len(broken)}/{len(spec['queries'])} queries answerable")
    if broken:
        print("\nFix these before trusting any benchmark run:")
        for query in broken:
            print(f"  {query['id']}: paper={query['paper']!r} "
                  f"must_contain={query['must_contain']}")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
