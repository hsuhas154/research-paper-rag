"""
reindex_corpus.py

Rebuilds the index from the archived PDFs, keeping every document's title.

Needed whenever extraction, segmentation or chunking changes, because the
stored chunks were produced by the old code and nothing re-derives them.
The embeddings are rebuilt too, so this is also the path to follow after
changing the embedding model.

Two details that matter and are easy to get wrong:

Titles are passed through explicitly rather than re-extracted. The
archived PDFs are named by document id, so a failed title extraction
would fall back to a hex filename, and eval/splits.json matches papers by
title fragment. Losing a title silently reassigns a paper's split.

The PDFs are staged outside the corpus directory first. add_pdf() copies
each file into that same directory, so reading and writing it in one pass
makes the run index its own copies.

Usage:
    python -m scripts.reindex_corpus            # rebuild in place
    python -m scripts.reindex_corpus --dry-run  # report what would change
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.corpus import Corpus
from src.embedder import Embedder


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    documents_path = args.corpus_dir / "documents.json"
    if not documents_path.exists():
        print(f"No corpus at {args.corpus_dir}")
        return 1

    existing = json.loads(documents_path.read_text(encoding="utf-8"))["documents"]
    pdf_dir = args.corpus_dir / "pdfs"

    staged = []
    missing = []
    for record in existing:
        pdf = pdf_dir / f"{record['doc_id']}.pdf"
        (staged if pdf.exists() else missing).append((pdf, record))

    print(f"{len(existing)} documents registered, {len(staged)} PDFs found")
    if missing:
        # Without its PDF a document cannot be rebuilt, and continuing would
        # drop it from the corpus silently.
        print(f"\n{len(missing)} archived PDFs are missing, so this would lose them:")
        for pdf, record in missing:
            print(f"  {record['doc_id']}  {record['title'][:60]}")
        print("\nRefusing to rebuild. Restore the PDFs first.")
        return 1

    if args.dry_run:
        print("\nDry run, nothing written.")
        return 0

    # Staged outside the corpus directory, because add_pdf copies into it.
    with tempfile.TemporaryDirectory(prefix="reindex-") as tmp:
        staging = Path(tmp)
        for pdf, record in staged:
            shutil.copyfile(pdf, staging / f"{record['doc_id']}.pdf")

        for name in ("documents.json", "chunks.json", "embeddings.npy"):
            (args.corpus_dir / name).unlink(missing_ok=True)
        shutil.rmtree(pdf_dir, ignore_errors=True)

        print("Loading the embedding model...")
        corpus = Corpus(Embedder(), corpus_dir=args.corpus_dir)

        start = time.time()
        for i, (_, record) in enumerate(staged, 1):
            source = staging / f"{record['doc_id']}.pdf"
            rebuilt = corpus.add_pdf(str(source), title=record["title"])
            print(f"  [{i:>2}/{len(staged)}] {rebuilt.n_pages:>3}p "
                  f"{rebuilt.n_chunks:>4} chunks  {record['title'][:52]}", flush=True)

    print(f"\n{len(corpus)} documents, {len(corpus.store)} chunks, "
          f"{round(time.time() - start)}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
