"""
corpus.py

The persistent library of indexed papers.

Phase 1 held one document in memory and rebuilt it from scratch on
every restart. This module makes the library durable and multi-document:
papers are added once, survive restarts, and can be removed
individually.

What is persisted, and why:

    documents.json   document registry (title, filename, page count,
                     chunk count, when it was added)
    chunks.json      every chunk's text and provenance
    embeddings.npy   the chunk embedding matrix

The FAISS index itself is deliberately *not* written to disk. It is a
derived structure - a flat index is just the embedding matrix in a
search-friendly wrapper - and rebuilding it from embeddings.npy takes
milliseconds even for tens of thousands of chunks. Persisting only the
source of truth means there is no way for a stale index file to
disagree with the chunks it supposedly indexes, and it leaves the index
type free to change later (to IVF or HNSW, if the corpus ever outgrows
brute force) without invalidating anything already stored.

Embeddings are stored rather than recomputed for the same reason: they
are the expensive artifact (a GPU forward pass per chunk), they are
small (384 floats per chunk, ~1.5 KB), and keeping them makes document
removal a pure array operation instead of a full re-index.
"""

import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

import numpy as np

from src.chunker import Chunk, chunk_segments
from src.embedder import Embedder
from src.pdf_processor import extract_pages, extract_title, segment_pages, strip_references_section
from src.vector_store import VectorStore

DEFAULT_CORPUS_DIR = Path("data/corpus")

# Indexed PDFs are archived *inside* the corpus directory, not alongside
# the user's source files. Writing them back to the staging folder made
# the corpus re-index its own archive copies the next time that folder
# was swept, silently duplicating every document.
DEFAULT_PDF_SUBDIR = "pdfs"

# Bumped whenever the on-disk layout changes in a way that older files
# cannot satisfy. A mismatch is reported clearly rather than crashing
# somewhere deep in a load.
STORAGE_FORMAT_VERSION = 2


@dataclass
class DocumentRecord:
    """Registry entry for one indexed paper."""
    doc_id: str
    title: str
    filename: str
    n_pages: int
    n_chunks: int
    added_at: str  # ISO 8601, UTC

    @property
    def short_title(self) -> str:
        """Title truncated for use in dropdowns and citation labels."""
        return self.title if len(self.title) <= 70 else self.title[:67] + "..."


class Corpus:
    """
    Owns the library of indexed papers: the document registry, the
    chunk metadata, and the vector store built over them.

    Every mutating operation writes through to disk immediately. The
    library is small and writes are rare (one per upload), so there is
    no reason to risk losing an expensive indexing run to an
    unclean shutdown for the sake of batching.
    """

    def __init__(
        self,
        embedder: Embedder,
        corpus_dir: Path = DEFAULT_CORPUS_DIR,
        pdf_dir: Optional[Path] = None,
    ):
        self.embedder = embedder
        self.corpus_dir = Path(corpus_dir)
        self.pdf_dir = Path(pdf_dir) if pdf_dir else self.corpus_dir / DEFAULT_PDF_SUBDIR
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        self.documents: List[DocumentRecord] = []
        self.store = VectorStore(embedding_dim=self.embedder.embedding_dim)
        self.load()

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def documents_path(self) -> Path:
        return self.corpus_dir / "documents.json"

    @property
    def chunks_path(self) -> Path:
        return self.corpus_dir / "chunks.json"

    @property
    def embeddings_path(self) -> Path:
        return self.corpus_dir / "embeddings.npy"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Restores the corpus from disk. A missing or incomplete corpus
        directory is treated as an empty library, not an error - that is
        simply the first-run state.
        """
        if not (self.documents_path.exists() and self.chunks_path.exists()
                and self.embeddings_path.exists()):
            return

        documents_blob = json.loads(self.documents_path.read_text(encoding="utf-8"))
        version = documents_blob.get("version")
        if version != STORAGE_FORMAT_VERSION:
            raise ValueError(
                f"Corpus at {self.corpus_dir} was written in storage format v{version}, "
                f"but this code expects v{STORAGE_FORMAT_VERSION}. Delete the directory "
                f"to re-index from the original PDFs."
            )

        self.documents = [DocumentRecord(**record) for record in documents_blob["documents"]]

        chunk_records = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        chunks = [Chunk(**record) for record in chunk_records]
        embeddings = np.load(self.embeddings_path).astype("float32")

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Corpus is inconsistent: {len(chunks)} chunks but "
                f"{embeddings.shape[0]} embeddings. Delete {self.corpus_dir} to rebuild."
            )

        # Restore in one pass. add_document() is per-document and would
        # re-stamp doc_ids the saved chunks already carry, so the whole
        # set is installed at once instead.
        dimension = embeddings.shape[1] if len(chunks) else self.embedder.embedding_dim
        self.store = VectorStore(embedding_dim=dimension)
        self.store.set_contents(chunks, embeddings)

    def save(self) -> None:
        """Writes the whole corpus to disk, replacing what was there."""
        self.corpus_dir.mkdir(parents=True, exist_ok=True)

        self.documents_path.write_text(
            json.dumps(
                {
                    "version": STORAGE_FORMAT_VERSION,
                    "embedding_model": self.embedder.model_name,
                    "documents": [asdict(d) for d in self.documents],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.chunks_path.write_text(
            json.dumps([asdict(c) for c in self.store.chunks], indent=2),
            encoding="utf-8",
        )
        np.save(self.embeddings_path, self.store.embeddings)

    # ------------------------------------------------------------------
    # Library operations
    # ------------------------------------------------------------------

    def add_pdf(self, pdf_path: str, title: Optional[str] = None) -> DocumentRecord:
        """
        Runs a PDF through extract -> segment -> chunk -> embed -> index
        and adds it to the library.

        The PDF is copied into the upload directory under its document
        id, so the library stays reproducible: every indexed paper can
        be re-processed from the exact file it was built from, even if
        the user's original is moved or deleted.

        Returns the registry entry for the new document.
        """
        source = Path(pdf_path)
        if not source.exists():
            raise FileNotFoundError(f"No such PDF: {pdf_path}")

        doc_id = uuid4().hex[:12]

        pages = extract_pages(str(source))
        if not pages:
            raise ValueError(f"No pages could be read from {source.name}")

        segments = segment_pages(strip_references_section(pages))
        chunks = chunk_segments(segments)
        if not chunks:
            raise ValueError(
                f"No text could be extracted from {source.name}. "
                f"It may be a scanned PDF with no embedded text layer."
            )

        embeddings = self.embedder.encode([c.text for c in chunks])
        self.store.add_document(doc_id, chunks, embeddings)

        shutil.copyfile(source, self.pdf_dir / f"{doc_id}.pdf")

        # Filename stem is the last resort when neither metadata nor
        # typography yields a title. Underscores and hyphens are separator
        # conventions, not part of the name, so they read better as spaces.
        from_filename = re.sub(r"[_\s]+", " ", source.stem).strip()

        record = DocumentRecord(
            doc_id=doc_id,
            title=title or extract_title(str(source)) or from_filename,
            filename=source.name,
            n_pages=len(pages),
            n_chunks=len(chunks),
            added_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self.documents.append(record)
        self.save()

        return record

    def remove_document(self, doc_id: str) -> bool:
        """
        Removes a document from the library, along with its chunks,
        embeddings, and stored PDF. Returns False if no such document.
        """
        record = self.get(doc_id)
        if record is None:
            return False

        self.store.remove_document(doc_id)
        self.documents = [d for d in self.documents if d.doc_id != doc_id]

        (self.pdf_dir / f"{doc_id}.pdf").unlink(missing_ok=True)

        self.save()
        return True

    def get(self, doc_id: str) -> Optional[DocumentRecord]:
        """Returns the registry entry for a document id, or None."""
        return next((d for d in self.documents if d.doc_id == doc_id), None)

    def title_for(self, doc_id: str) -> str:
        """Human-readable title for a document id, for use in citations."""
        record = self.get(doc_id)
        return record.short_title if record else "unknown document"

    def is_empty(self) -> bool:
        return not self.documents

    def __len__(self) -> int:
        return len(self.documents)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m src.corpus list")
        print("  python -m src.corpus add <path_to_pdf>")
        print("  python -m src.corpus remove <doc_id>")
        sys.exit(1)

    print("Loading embedding model...")
    corpus = Corpus(Embedder())
    command = sys.argv[1]

    if command == "list":
        if corpus.is_empty():
            print("Corpus is empty.")
        else:
            print(f"{len(corpus)} document(s), {len(corpus.store)} chunks total:\n")
            for doc in corpus.documents:
                print(f"  {doc.doc_id}  {doc.n_pages:>3}p  {doc.n_chunks:>4} chunks  {doc.title}")

    elif command == "add":
        new_doc = corpus.add_pdf(sys.argv[2])
        print(f"Added {new_doc.doc_id}: {new_doc.title}")
        print(f"  {new_doc.n_pages} pages -> {new_doc.n_chunks} chunks")

    elif command == "remove":
        print("Removed." if corpus.remove_document(sys.argv[2]) else "No such document.")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
