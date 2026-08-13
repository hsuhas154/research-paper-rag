"""
pdf_processor.py

Handles extraction and cleaning of text from PDF research papers.
Uses PyMuPDF (imported as 'pymupdf', not the deprecated 'fitz' alias)
since it preserves reading order reasonably well even on multi-column
academic paper layouts.
"""

import re
import pymupdf


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts raw text from every page of a PDF and concatenates it.
    """
    doc = pymupdf.open(pdf_path)
    pages_text = []

    for page in doc:
        text = page.get_text("text")
        pages_text.append(text)

    doc.close()
    return "\n".join(pages_text)


_REFERENCE_HEADING_PATTERN = re.compile(
    r"^\s*(references|bibliography|works cited)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def strip_references_section(raw_text: str) -> str:
    """
    Removes the references/bibliography section from raw extracted text,
    if a confident section heading can be found for it.

    Operates on RAW (pre-clean) text, where PyMuPDF still preserves line
    breaks - a references heading is reliably on its own line in the
    original document layout, but that structure disappears once
    clean_text() collapses line breaks into continuous prose.

    Only considers matches in the final 30% of the document. The word
    "references" could theoretically appear in ordinary running text
    earlier in a paper, but a genuine bibliography heading is always
    near the end - restricting the search window avoids false positives
    that would truncate real content.

    If no confident match is found, the text is returned unchanged: it
    is safer to leave the references section in (a minor retrieval-noise
    issue) than to risk cutting off real paper content.
    """
    matches = list(_REFERENCE_HEADING_PATTERN.finditer(raw_text))
    if not matches:
        return raw_text

    search_start_cutoff = len(raw_text) * 0.7
    late_matches = [m for m in matches if m.start() >= search_start_cutoff]
    if not late_matches:
        return raw_text

    cut_point = late_matches[0].start()
    return raw_text[:cut_point]


def clean_text(raw_text: str) -> str:
    """
    Cleans raw PDF-extracted text before chunking.
    """
    text = raw_text
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_and_clean_pdf(pdf_path: str) -> str:
    """Convenience wrapper: extract -> strip references -> clean, in one call."""
    raw = extract_text_from_pdf(pdf_path)
    raw_no_refs = strip_references_section(raw)
    return clean_text(raw_no_refs)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.pdf_processor <path_to_pdf>")
        sys.exit(1)

    path = sys.argv[1]

    raw = extract_text_from_pdf(path)
    raw_no_refs = strip_references_section(raw)
    removed_chars = len(raw) - len(raw_no_refs)
    cleaned_text = clean_text(raw_no_refs)

    print(f"Raw extracted: {len(raw)} characters")
    print(f"Stripped {removed_chars} characters of references/bibliography")
    print(f"Final cleaned: {len(cleaned_text)} characters\n")
    print("--- First 1000 characters ---\n")
    print(cleaned_text[:1000])