"""
chunker.py

Splits a segmented document into overlapping, sentence-respecting
chunks suitable for embedding.

Chunks are built by accumulating whole sentences (never cutting a
sentence in half) until a target word count is reached. The next
chunk then starts by re-including the last few sentences of the
previous chunk (the "overlap"), so an idea that straddles a chunk
boundary isn't retrievable only in a fragmented, context-free form.

Phase 2 change: chunks are built from Segments rather than one flat
string, and each sentence carries the page and section it came from.
A chunk therefore knows the page range it spans and which section it
belongs to, which is what lets an answer cite "page 4, Section 2.2"
instead of just naming the paper. Chunks are never allowed to span a
section boundary, so a chunk's section label is always truthful for
all of its text.

Sentence splitting uses a lightweight regex approach rather than a
heavy NLP library, with explicit protection for common academic
abbreviations (e.g. "et al.", "Fig.", "Eq.") that would otherwise be
misread as sentence endings.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from src.pdf_processor import Segment


# Abbreviations ending in a period that do NOT end a sentence.
# Academic papers are full of these; naively splitting on ". " would
# fragment text like "the model (Smith et al. 2020) shows..." badly.
_ABBREVIATIONS = [
    "et al.", "e.g.", "i.e.", "etc.", "vs.", "Fig.", "Figs.", "Eq.",
    "Eqs.", "Ref.", "Refs.", "Sect.", "cf.", "approx.", "resp.",
    "Dr.", "Mr.", "Mrs.", "No.", "pp.", "Vol.", "vol.",
]


def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences, protecting known abbreviations from
    being misread as sentence boundaries.

    Strategy: temporarily replace periods inside known abbreviations
    with a placeholder, split on '.', '!', or '?' followed by
    whitespace and a capital letter (or end of string), then restore
    the abbreviations.
    """
    protected = text
    for i, abbr in enumerate(_ABBREVIATIONS):
        protected = protected.replace(abbr, f"__ABBR{i}__")

    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z]|$)", protected)

    for i, abbr in enumerate(_ABBREVIATIONS):
        raw_sentences = [s.replace(f"__ABBR{i}__", abbr) for s in raw_sentences]

    return [s.strip() for s in raw_sentences if s.strip()]


@dataclass
class Chunk:
    """
    One embeddable passage, plus enough provenance to cite it.

    doc_id is assigned by the corpus when the chunk is indexed; it is
    empty for chunks that have just come out of the chunker and have
    not been attached to a document yet.
    """
    chunk_id: int
    text: str
    word_count: int
    page_start: int = 0
    page_end: int = 0
    section: Optional[str] = None
    doc_id: str = ""

    def citation(self, document_title: Optional[str] = None) -> str:
        """
        A short human-readable source label, e.g.
        "Dai et al. 2024, p. 4-5, Section 2.2 Atmospheric chemistry".
        """
        parts = []
        if document_title:
            parts.append(document_title)
        if self.page_start:
            if self.page_end and self.page_end != self.page_start:
                parts.append(f"p. {self.page_start}-{self.page_end}")
            else:
                parts.append(f"p. {self.page_start}")
        if self.section:
            parts.append(f"Section {self.section}")
        return ", ".join(parts) if parts else "source unknown"


@dataclass
class _TaggedSentence:
    """A sentence with the provenance of the Segment it came from."""
    text: str
    page_number: int
    section: Optional[str]
    word_count: int = field(default=0)

    def __post_init__(self):
        self.word_count = len(self.text.split())


def _flatten_to_sentences(segments: List[Segment]) -> List[_TaggedSentence]:
    """Expands Segments into individual sentences that remember their origin."""
    sentences: List[_TaggedSentence] = []
    for segment in segments:
        for sentence in split_into_sentences(segment.text):
            sentences.append(_TaggedSentence(sentence, segment.page_number, segment.section))
    return sentences


def _build_chunk(chunk_id: int, sentences: List[_TaggedSentence]) -> Chunk:
    """Assembles a Chunk from the sentences accumulated for it."""
    pages = [s.page_number for s in sentences]
    return Chunk(
        chunk_id=chunk_id,
        text=" ".join(s.text for s in sentences),
        word_count=sum(s.word_count for s in sentences),
        page_start=min(pages),
        page_end=max(pages),
        # Every sentence in a chunk shares a section by construction
        # (see chunk_segments), so the first one speaks for all of them.
        section=sentences[0].section,
    )


def chunk_segments(
    segments: List[Segment],
    target_words: int = 180,
    overlap_words: int = 40,
) -> List[Chunk]:
    """
    Groups sentences into overlapping chunks that respect both sentence
    and section boundaries.

    Args:
        segments: output of pdf_processor.segment_pages
        target_words: approximate words per chunk. Kept well under the
            ~256 token limit of all-MiniLM-L6-v2 (180 words is roughly
            230-260 tokens for typical English prose), so chunks are
            never silently truncated during embedding.
        overlap_words: approximate word count repeated at the start of
            each chunk from the end of the previous one.

    Returns:
        List of Chunk objects, numbered sequentially from 0.
    """
    sentences = _flatten_to_sentences(segments)

    chunks: List[Chunk] = []
    current: List[_TaggedSentence] = []
    current_words = 0
    chunk_id = 0

    def flush_if_meaningful(minimum_words: int) -> None:
        """Emits the accumulated sentences as a chunk, if worth keeping."""
        nonlocal chunk_id, current, current_words
        if current and current_words > minimum_words:
            chunks.append(_build_chunk(chunk_id, current))
            chunk_id += 1
        current, current_words = [], 0

    for sentence in sentences:
        # A section change closes the current chunk. Mixing two sections
        # into one chunk would make its section label wrong for part of
        # its own text, which is worse than a slightly short chunk.
        if current and sentence.section != current[0].section:
            flush_if_meaningful(minimum_words=0)

        current.append(sentence)
        current_words += sentence.word_count

        if current_words >= target_words:
            chunks.append(_build_chunk(chunk_id, current))
            chunk_id += 1

            # Build the overlap seed for the next chunk: walk backwards
            # through this chunk's sentences until we've collected
            # roughly `overlap_words` worth.
            overlap: List[_TaggedSentence] = []
            overlap_words_count = 0
            for s in reversed(current):
                overlap.insert(0, s)
                overlap_words_count += s.word_count
                if overlap_words_count >= overlap_words:
                    break

            current = overlap
            current_words = overlap_words_count

    # Flush a final chunk if meaningful content remains beyond just the
    # trailing overlap seed.
    flush_if_meaningful(minimum_words=overlap_words)

    return chunks


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.pdf_processor import load_and_segment_pdf

    if len(sys.argv) != 2:
        print("Usage: python -m src.chunker <path_to_pdf>")
        sys.exit(1)

    doc_segments = load_and_segment_pdf(sys.argv[1])
    doc_chunks = chunk_segments(doc_segments)

    print(f"Document produced {len(doc_chunks)} chunks from {len(doc_segments)} segments.")
    word_counts = [c.word_count for c in doc_chunks]
    print(f"Words per chunk: min={min(word_counts)} max={max(word_counts)} "
          f"mean={sum(word_counts) / len(word_counts):.0f}\n")

    for c in doc_chunks[:3]:
        print(f"--- Chunk {c.chunk_id} ({c.word_count} words) | {c.citation()} ---")
        print(c.text[:400])
        print()
