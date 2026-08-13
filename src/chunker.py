"""
chunker.py

Splits cleaned document text into overlapping, sentence-respecting
chunks suitable for embedding.

Chunks are built by accumulating whole sentences (never cutting a
sentence in half) until a target word count is reached. The next
chunk then starts by re-including the last few sentences of the
previous chunk (the "overlap"), so an idea that straddles a chunk
boundary isn't retrievable only in a fragmented, context-free form.

Sentence splitting uses a lightweight regex approach rather than a
heavy NLP library, with explicit protection for common academic
abbreviations (e.g. "et al.", "Fig.", "Eq.") that would otherwise be
misread as sentence endings.
"""

import re
from dataclasses import dataclass
from typing import List


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
    chunk_id: int
    text: str
    word_count: int


def chunk_text(
    text: str,
    target_words: int = 180,
    overlap_words: int = 40,
) -> List[Chunk]:
    """
    Groups sentences into overlapping chunks.

    Args:
        text: cleaned document text (output of pdf_processor.clean_text)
        target_words: approximate words per chunk. Kept well under the
            ~256 token limit of all-MiniLM-L6-v2 (180 words is roughly
            230-260 tokens for typical English prose), so chunks are
            never silently truncated during embedding.
        overlap_words: approximate word count repeated at the start of
            each chunk from the end of the previous one.

    Returns:
        List of Chunk objects.
    """
    sentences = split_into_sentences(text)

    chunks: List[Chunk] = []
    current_sentences: List[str] = []
    current_word_count = 0
    chunk_id = 0
    i = 0

    while i < len(sentences):
        sentence = sentences[i]
        current_sentences.append(sentence)
        current_word_count += len(sentence.split())
        i += 1

        if current_word_count >= target_words:
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=" ".join(current_sentences),
                word_count=current_word_count,
            ))
            chunk_id += 1

            # Build the overlap seed for the next chunk: walk backwards
            # through this chunk's sentences until we've collected
            # roughly `overlap_words` worth.
            overlap_sentences = []
            overlap_count = 0
            for s in reversed(current_sentences):
                overlap_sentences.insert(0, s)
                overlap_count += len(s.split())
                if overlap_count >= overlap_words:
                    break

            current_sentences = overlap_sentences
            current_word_count = overlap_count

    # Flush a final chunk if meaningful content remains beyond just
    # the trailing overlap seed.
    if current_sentences and current_word_count > overlap_words:
        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=" ".join(current_sentences),
            word_count=current_word_count,
        ))

    return chunks


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.pdf_processor import load_and_clean_pdf

    if len(sys.argv) != 2:
        print("Usage: python src/chunker.py <path_to_pdf>")
        sys.exit(1)

    cleaned = load_and_clean_pdf(sys.argv[1])
    chunks = chunk_text(cleaned)

    print(f"Document produced {len(chunks)} chunks.\n")
    for c in chunks[:3]:
        print(f"--- Chunk {c.chunk_id} ({c.word_count} words) ---")
        print(c.text)
        print()