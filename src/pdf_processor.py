"""
pdf_processor.py

Handles extraction, structural segmentation, and cleaning of text from
PDF research papers.

Phase 2 change: extraction is now page-aware and section-aware. Phase 1
flattened a whole paper into one string, which meant a retrieved passage
could only ever be cited as "somewhere in this paper". Answers are much
easier to verify against the original document when a citation can say
"page 7, Section 3.2 Methods", so text is now carried through the
pipeline as a list of Segments that each remember where they came from.

Uses PyMuPDF (imported as 'pymupdf', not the deprecated 'fitz' alias)
since it preserves reading order reasonably well even on multi-column
academic paper layouts.
"""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RawPage:
    """One page of PDF text, with line breaks still intact."""
    page_number: int  # 1-indexed, matching what a PDF reader displays
    text: str


@dataclass
class Segment:
    """
    A run of cleaned prose that sits on one page under one section
    heading. Segments are the unit the chunker consumes: they are the
    largest pieces of text whose page/section provenance is unambiguous.
    """
    text: str
    page_number: int
    section: Optional[str]


def extract_pages(pdf_path: str) -> List[RawPage]:
    """
    Extracts raw text from every page of a PDF, one RawPage per page.

    Line breaks are deliberately preserved here. Both reference
    stripping and section-heading detection depend on the original line
    structure, which clean_text() destroys - so cleaning happens last,
    per segment, once the structure has already been read off.
    """
    doc = pymupdf_open(pdf_path)
    pages = []

    for page_index, page in enumerate(doc):
        pages.append(RawPage(page_number=page_index + 1, text=page.get_text("text")))

    doc.close()
    return pages


def pymupdf_open(pdf_path: str):
    """
    Indirection around pymupdf.open so tests can exercise the rest of
    this module without a real PDF on disk.
    """
    import pymupdf
    return pymupdf.open(pdf_path)


_REFERENCE_HEADING_PATTERN = re.compile(
    r"^\s*(?:\d+\.?\s*)?(references|bibliography|works cited)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def strip_references_section(pages: List[RawPage]) -> List[RawPage]:
    """
    Removes the references/bibliography section, if a confident section
    heading can be found for it.

    Only considers matches in the final 30% of the document, measured in
    characters across all pages. The word "references" could appear in
    ordinary running text earlier in a paper, but a genuine bibliography
    heading is always near the end - restricting the search window
    avoids false positives that would truncate real content.

    If no confident match is found, the pages are returned unchanged: it
    is safer to leave the references section in (a minor retrieval-noise
    issue) than to risk cutting off real paper content.
    """
    total_chars = sum(len(p.text) for p in pages)
    if total_chars == 0:
        return pages

    cutoff = total_chars * 0.7
    chars_before_page = 0

    for i, page in enumerate(pages):
        for match in _REFERENCE_HEADING_PATTERN.finditer(page.text):
            absolute_position = chars_before_page + match.start()
            if absolute_position >= cutoff:
                # Truncate this page at the heading and drop every page after it.
                truncated = RawPage(page.page_number, page.text[: match.start()])
                return pages[:i] + [truncated]
        chars_before_page += len(page.text)

    return pages


# A heading line in an academic paper is short, starts with a capital or a
# section number, and does not read like a sentence. These two patterns
# cover the overwhelming majority of real layouts:
#   "3.2 Model configuration"  /  "4. RESULTS"  /  "Introduction"  /  "ABSTRACT"
_NUMBERED_HEADING = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+([A-Z][^.!?]{2,70})\s*$")
_UNNUMBERED_HEADING = re.compile(r"^\s*([A-Z][A-Za-z]+(?:\s+[A-Za-z&-]+){0,6})\s*$")

# Unnumbered headings are risky - a short line of body text or a stray
# author name can match the shape. Requiring the text to be a known
# academic section name keeps false positives near zero, at the cost of
# missing unconventional heading names (which simply leave the section
# as whatever was detected previously).
_KNOWN_SECTION_NAMES = {
    "abstract", "introduction", "background", "related work", "methods",
    "methodology", "materials and methods", "model", "model description",
    "data", "data and methods", "experiments", "experimental setup",
    "results", "results and discussion", "discussion", "analysis",
    "conclusion", "conclusions", "conclusions and future work",
    "summary", "summary and conclusions", "acknowledgements",
    "acknowledgments", "appendix", "supplementary material",
}


def detect_heading(line: str) -> Optional[str]:
    """
    Returns a normalized heading string if this line looks like a
    section heading, otherwise None.

    Deliberately conservative. A missed heading only means a chunk
    inherits the previous section label; a false positive would split
    running prose and attach a nonsense section name to real content.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return None

    numbered = _NUMBERED_HEADING.match(stripped)
    if numbered:
        number, title = numbered.groups()
        title = title.strip()
        # Guard against figure captions and numeric data rows, which share
        # the "number then capitalized text" shape.
        if title.lower().startswith(("fig", "table", "eq", "plate")):
            return None
        # Real headings end on a word, not on punctuation. Inline equations
        # picked up by the text extractor ("4 Qpn(dp) ddp,") end on a comma
        # or operator, so this cheaply rejects them.
        if not title[-1].isalnum():
            return None
        return f"{number} {title}"

    unnumbered = _UNNUMBERED_HEADING.match(stripped)
    if unnumbered:
        title = unnumbered.group(1).strip()
        if title.lower().rstrip(".") in _KNOWN_SECTION_NAMES:
            return title
        # ALL-CAPS lines are usually headings in paper layouts, but a
        # single all-caps word is far more often a chemical formula or
        # acronym sitting in a table ("COS", "HNO"). Multi-word all-caps
        # lines are safe; single words have to be a known section name,
        # which the branch above already handles.
        if title.isupper() and 2 <= len(title.split()) <= 6:
            return title.title()

    return None


def clean_text(raw_text: str) -> str:
    """
    Cleans a block of raw PDF-extracted text into continuous prose:
    rejoins words broken across a line by a hyphen, collapses single
    line breaks into spaces, and normalizes whitespace.
    """
    text = raw_text
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Lines this short are almost always page numbers, running heads, or
# stray artifacts from figure labels rather than real prose.
_MIN_PROSE_LINE_CHARS = 3


def segment_pages(pages: List[RawPage]) -> List[Segment]:
    """
    Converts raw pages into cleaned Segments, each tagged with its page
    number and the section heading in force at that point.

    A new Segment is started whenever the page changes or a heading is
    encountered, so every Segment has one unambiguous (page, section)
    provenance.
    """
    segments: List[Segment] = []
    current_section: Optional[str] = None

    for page in pages:
        buffer: List[str] = []

        def flush(section: Optional[str]) -> None:
            if not buffer:
                return
            cleaned = clean_text("\n".join(buffer))
            buffer.clear()
            if cleaned:
                segments.append(Segment(cleaned, page.page_number, section))

        for line in page.text.split("\n"):
            if len(line.strip()) < _MIN_PROSE_LINE_CHARS:
                continue

            heading = detect_heading(line)
            if heading is not None:
                # Close out the text belonging to the *previous* section
                # before switching, so prose is never mislabeled.
                flush(current_section)
                current_section = heading
                continue

            buffer.append(line)

        flush(current_section)

    return segments


def load_and_segment_pdf(pdf_path: str) -> List[Segment]:
    """Convenience wrapper: extract -> strip references -> segment, in one call."""
    pages = extract_pages(pdf_path)
    pages = strip_references_section(pages)
    return segment_pages(pages)


_TITLE_NOISE = re.compile(
    r"(doi|https?://|www\.|arxiv|preprint|\bvol\.|©|copyright|\bissn\b|"
    r"all rights reserved|general rights|downloaded from)",
    re.IGNORECASE,
)


def extract_title(pdf_path: str) -> Optional[str]:
    """
    Best-effort guess at a paper's title, used to give uploaded documents
    a human-readable name in the corpus instead of a filename.

    Works off font size rather than line position. Position-based
    heuristics ("first substantial line") break immediately on the
    repository cover pages and journal banners that many published PDFs
    carry, whereas the title is reliably the largest text in the paper's
    opening pages regardless of what precedes it.

    Falls back to the PDF's own metadata title, then to None - in which
    case callers use the filename.
    """
    doc = pymupdf_open(pdf_path)
    try:
        lines = []
        for page in list(doc)[:3]:
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    text = re.sub(r"\s+", " ", "".join(s["text"] for s in spans)).strip()
                    if not text or len(text) > 250 or _TITLE_NOISE.search(text):
                        continue
                    # Reject anything dominated by digits: page furniture,
                    # dates, and equation lines rather than a title.
                    if sum(c.isdigit() for c in text) > len(text) * 0.2:
                        continue
                    lines.append((round(max(s["size"] for s in spans), 1), text))

        # The title font is the largest one used by a line long enough to
        # actually be a title - this skips oversized drop caps, journal
        # logos, and single-letter decorative glyphs.
        title_sizes = [size for size, text in lines if len(text) >= 15]
        if title_sizes:
            title_size = max(title_sizes)
            # Titles usually wrap across several lines, all set at the same
            # size, so join every line at that size in reading order rather
            # than returning only the first one.
            parts = [text for size, text in lines if size == title_size]
            joined = " ".join(parts).strip()
            if joined:
                return joined

        metadata_title = (doc.metadata or {}).get("title", "").strip()
        return metadata_title or None
    finally:
        doc.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.pdf_processor <path_to_pdf>")
        sys.exit(1)

    raw_pages = extract_pages(sys.argv[1])
    kept_pages = strip_references_section(raw_pages)
    doc_segments = segment_pages(kept_pages)

    raw_chars = sum(len(p.text) for p in raw_pages)
    kept_chars = sum(len(p.text) for p in kept_pages)

    print(f"Pages extracted:   {len(raw_pages)}")
    print(f"Pages after refs:  {len(kept_pages)}")
    print(f"Stripped {raw_chars - kept_chars} characters of references/bibliography")
    print(f"Segments produced: {len(doc_segments)}")
    print(f"Detected title:    {extract_title(sys.argv[1])}\n")

    sections_found = []
    for seg in doc_segments:
        if seg.section and seg.section not in sections_found:
            sections_found.append(seg.section)
    print(f"Sections detected ({len(sections_found)}):")
    for name in sections_found:
        print(f"  - {name}")

    print("\n--- First 3 segments ---")
    for seg in doc_segments[:3]:
        print(f"\n[page {seg.page_number} | {seg.section or 'no section'}]")
        print(seg.text[:300])
