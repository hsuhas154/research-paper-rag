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

Extraction keeps each line's font size and weight, not just its text.
Purely textual heading detection works on papers that number their
sections ("3.2 Methods") or use conventional names ("Introduction"), and
fails completely on journals that mark headings by typography alone -
Nature's review format sets "Supervised learning" in 10pt against 9.4pt
body text, with no number and no recognisable name. Font size is the
only signal that such a heading exists at all, so it is carried through
and used alongside the textual rules rather than instead of them.

Uses PyMuPDF (imported as 'pymupdf', not the deprecated 'fitz' alias)
since it preserves reading order reasonably well even on multi-column
academic paper layouts.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class RawLine:
    """One extracted line of text, with the typography that framed it."""
    text: str
    size: float  # largest font size on the line, in points
    bold: bool
    y: float  # vertical position from the top of the page, in points


@dataclass
class RawPage:
    """
    One page of PDF text, with line breaks still intact.

    `lines` carries the same text with font information attached. It is
    optional so that a RawPage can still be constructed from plain text
    alone (as the tests do); when absent, heading detection falls back to
    textual rules only.
    """
    page_number: int  # 1-indexed, matching what a PDF reader displays
    text: str
    lines: Optional[List[RawLine]] = field(default=None)


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


# Unicode Private Use Area codepoints, plus the supplementary PUA planes.
# Publisher templates render UI furniture - "View Online" buttons, CrossMark
# badges, ORCID marks - with icon fonts that map glyphs into this range.
# Extracted as text they are meaningless symbols that pollute chunks and,
# worse, sit invisibly inside candidate titles where they defeat every
# textual filter.
_PRIVATE_USE_AREA = re.compile(r"[-\U000f0000-\U0010ffff]")


def _strip_icon_glyphs(text: str) -> str:
    """Removes private-use icon-font glyphs and collapses the gap they leave."""
    return re.sub(r"[ \t]+", " ", _PRIVATE_USE_AREA.sub(" ", text)).strip()


def _page_lines(page) -> List[RawLine]:
    """Reads one page's lines with their font size and weight."""
    lines: List[RawLine] = []

    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = _strip_icon_glyphs("".join(s["text"] for s in spans))
            if not text:
                continue
            lines.append(RawLine(
                text=text,
                size=round(max(s["size"] for s in spans), 1),
                # PyMuPDF packs style into a bitfield; bit 4 is the bold flag.
                bold=any(s.get("flags", 0) & 2 ** 4 for s in spans),
                y=line["bbox"][1],
            ))

    return lines


def extract_pages(pdf_path: str) -> List[RawPage]:
    """
    Extracts text from every page of a PDF, one RawPage per page, with
    per-line font information attached.

    Line breaks are deliberately preserved here. Both reference
    stripping and section-heading detection depend on the original line
    structure, which clean_text() destroys - so cleaning happens last,
    per segment, once the structure has already been read off.
    """
    doc = pymupdf_open(pdf_path)
    pages = []

    for page_index, page in enumerate(doc):
        lines = _page_lines(page)
        pages.append(RawPage(
            page_number=page_index + 1,
            # Rebuilt from the same lines the font data came from, so the
            # two views of a page can never disagree about its content.
            text="\n".join(line.text for line in lines),
            lines=lines,
        ))

    doc.close()
    return pages


def body_font_size(pages: List[RawPage]) -> Optional[float]:
    """
    The dominant font size in a document, taken as the body text size.

    Weighted by characters rather than counted by line, since body text is
    overwhelmingly the bulk of a paper's characters while headings,
    captions, and affiliations are short. Returns None when no font
    information is available.
    """
    weights: Counter = Counter()
    for page in pages:
        for line in page.lines or []:
            weights[line.size] += len(line.text)

    return weights.most_common(1)[0][0] if weights else None


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


# Things that are set in heading-like type but are not section headings:
# author lists, affiliations, contact details, and copyright furniture.
# These sit at the top of page 1 in exactly the size range a heading
# occupies, so typography alone cannot tell them apart from one.
_NOT_A_HEADING = re.compile(
    r"(@|\bhttps?://|\bdoi\b|university|universit[ée]|department|dept\.|"
    r"institute|instituto|college|faculty|school of|academy|"
    r"laborator|laboratoire|center for|centre for|"
    r"corresponding author|e-?mail|\bissn\b|©|copyright|all rights reserved|"
    r"received:|accepted:|published:|submitted:)",
    re.IGNORECASE,
)

# Journal banners and article-type labels, set large enough to outrank a
# real title. They are never section headings and never titles.
_BANNER_WORDS = {
    "review", "reviews", "review articles", "review article", "articles",
    "article", "research article", "research articles", "letter", "letters",
    "perspective", "perspectives", "editorial", "news and views", "comment",
    "brief communication", "rapid communication", "short communication",
    "original article", "technical note", "correspondence", "abstract",
    "supplementary information", "open access", "research",
}


def _looks_like_a_heading_by_shape(stripped: str) -> bool:
    """
    Whether a line has the *shape* of a heading, independent of its font:
    short, not a sentence, not page furniture.

    Used to qualify font-based detections. Typography says "this line is
    emphasized"; shape says "this line is a label, not a paragraph".
    """
    if not (2 <= len(stripped) <= 80):
        return False
    words = stripped.split()
    if not (1 <= len(words) <= 12):
        return False
    if not (stripped[0].isupper() or stripped[0].isdigit()):
        return False
    # Headings end on a word, not on sentence or clause punctuation.
    if not stripped[-1].isalnum():
        return False
    # Author lists are short, capitalized, and emphasized, but comma-heavy.
    if stripped.count(",") > 1:
        return False
    if sum(c.isdigit() for c in stripped) > len(stripped) * 0.3:
        return False
    if _NOT_A_HEADING.search(stripped):
        return False
    if stripped.lower().rstrip(".:") in _BANNER_WORDS:
        return False
    return stripped.lower().startswith(("fig", "table", "eq", "plate")) is False


def detect_heading(
    line: str,
    size: Optional[float] = None,
    bold: bool = False,
    body_size: Optional[float] = None,
) -> Optional[str]:
    """
    Returns a normalized heading string if this line looks like a
    section heading, otherwise None.

    Two independent routes to a positive result:

      1. Textual - a numbered heading ("3.2 Methods") or a known section
         name ("Introduction"). Works regardless of typography.
      2. Typographic - the line is set larger than body text, or bold at
         body size, *and* has the shape of a heading. This is the only
         way to find headings in journals that mark them by font alone,
         such as Nature's review format.

    Deliberately conservative on both routes. A missed heading only means
    a chunk inherits the previous section label; a false positive splits
    running prose and attaches a nonsense section name to real content.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return None

    if body_size and size and _looks_like_a_heading_by_shape(stripped):
        # 8% larger is comfortably outside the rounding noise of font
        # sizes while still catching the 10pt-on-9.4pt case that motivated
        # this path. Bold at body size counts too, since many layouts
        # emphasize run-in headings without enlarging them.
        if size >= body_size * 1.08 or (bold and size >= body_size * 0.98):
            return stripped

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
    body_size = body_font_size(pages)

    for page in pages:
        buffer: List[str] = []

        def flush(section: Optional[str]) -> None:
            if not buffer:
                return
            cleaned = clean_text("\n".join(buffer))
            buffer.clear()
            if cleaned:
                segments.append(Segment(cleaned, page.page_number, section))

        # Prefer the font-annotated lines when extraction supplied them,
        # and fall back to plain text otherwise.
        page_lines: List[Tuple[str, Optional[float], bool]] = (
            [(line.text, line.size, line.bold) for line in page.lines]
            if page.lines is not None
            else [(text, None, False) for text in page.text.split("\n")]
        )

        for text, size, bold in page_lines:
            if len(text.strip()) < _MIN_PROSE_LINE_CHARS:
                continue

            heading = detect_heading(text, size=size, bold=bold, body_size=body_size)
            if heading is not None:
                # Close out the text belonging to the *previous* section
                # before switching, so prose is never mislabeled.
                flush(current_section)
                current_section = heading
                # Keep the heading as the first words of the section it
                # introduces, rather than discarding it. A real heading is
                # a strong retrieval signal ("A novel mechanism to
                # eliminate O2 at 70 km" describes its section better than
                # any sentence inside it), and a misfired detection then
                # costs only a stray label instead of deleting a line of
                # real text from the document.
                buffer.append(text)
                continue

            buffer.append(text)

        flush(current_section)

    return _merge_undersized_sections(segments)


# A real section has a body. Anything shorter than this is a detection
# artifact - a bold run-in phrase, a table label, a caption fragment - and
# is folded back into the section it interrupted.
_MIN_SECTION_WORDS = 40


def _merge_undersized_sections(segments: List[Segment]) -> List[Segment]:
    """
    Relabels sections too small to be real, folding them into whatever
    section preceded them.

    Font-based heading detection is necessarily loose on papers whose
    typography is inconsistent - older scanned IEEE papers report font
    sizes that wobble mid-paragraph, producing a scatter of one-line
    "sections". Left alone these fragment the document: the chunker
    closes a chunk at every section change, so a burst of false headings
    turns one coherent passage into several context-free scraps.

    Judging a heading by what follows it, rather than by how it looks,
    catches exactly the cases typography cannot.
    """
    if not segments:
        return segments

    # Group consecutive segments that share a section label.
    groups: List[List[Segment]] = [[segments[0]]]
    for segment in segments[1:]:
        if segment.section == groups[-1][-1].section:
            groups[-1].append(segment)
        else:
            groups.append([segment])

    previous_section: Optional[str] = None
    for group in groups:
        words = sum(len(s.text.split()) for s in group)
        if words < _MIN_SECTION_WORDS:
            for segment in group:
                segment.section = previous_section
        else:
            previous_section = group[0].section

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
        # Metadata first. Where a publisher fills it in it is exact, and it
        # sidesteps every layout problem below - cover sheets, oversized
        # journal banners, titles set smaller than their own affiliations.
        # It is checked rather than trusted: some PDFs ship a placeholder.
        from_metadata = _clean_metadata_title((doc.metadata or {}).get("title", ""))
        if from_metadata:
            return from_metadata

        # Otherwise fall back to typography. Scan the opening pages rather
        # than page 1 alone: institutional repositories prepend a cover
        # sheet, so the real title is often on page 2. The size comparison
        # still finds it, because cover sheets are set in small type.
        candidates: List[Tuple[int, float, float, str]] = []
        for page_index, page in enumerate(list(doc)[:3]):
            for line in _page_lines(page):
                if len(line.text) > 250 or _TITLE_NOISE.search(line.text):
                    continue
                if _is_banner(line.text):
                    continue
                # Reject anything dominated by digits: page furniture,
                # dates, and equation lines rather than a title.
                if sum(c.isdigit() for c in line.text) > len(line.text) * 0.2:
                    continue
                candidates.append((page_index, line.size, line.y, line.text))

        return _best_title(candidates)
    finally:
        doc.close()


# Publisher toolchains sometimes leave markup in the metadata title, most
# often an inline TeX formula wrapped in XML.
_METADATA_MARKUP = re.compile(r"<[^>]+>")

# Placeholder titles written by authoring tools when no title was set.
_PLACEHOLDER_TITLES = {
    "untitled", "no title", "title", "document", "microsoft word", "paper",
    "manuscript", "untitled document", "doc", "pdf", "main", "article",
}

# Journal and publisher names, which appear in the largest type on many
# first pages and would otherwise be mistaken for the title. This list is
# necessarily incomplete; it only has to cover the banner being *larger*
# than the title, which is the case the font heuristic gets wrong.
_JOURNAL_NAMES = {
    "astronomy astrophysics", "astronomy & astrophysics", "science advances",
    "nature", "science", "icarus", "physical review", "physical review letters",
    "journal of chemical physics", "the journal of chemical physics",
    "proceedings", "communications of the acm", "online", "advertisement",
}


def _is_banner(text: str) -> bool:
    """Whether a line is a journal banner or article-type label."""
    normalized = re.sub(r"\s+", " ", text).strip().lower().rstrip(".:,")
    return normalized in _BANNER_WORDS or normalized in _JOURNAL_NAMES


def _clean_metadata_title(raw: str) -> Optional[str]:
    """
    Sanitizes a PDF metadata title, returning None if it is unusable.
    """
    text = re.sub(r"\s+", " ", _METADATA_MARKUP.sub("", raw or "")).strip()
    if len(text) < 10 or len(text) > 250:
        return None
    if text.lower().rstrip(".") in _PLACEHOLDER_TITLES:
        return None
    if _is_banner(text):
        return None
    # A metadata field holding a filename is a tool artifact, not a title.
    if re.search(r"\.(pdf|docx?|tex)$", text, re.IGNORECASE):
        return None
    return text


# Two font sizes within this ratio of each other are treated as one tier.
# Old and scanned PDFs report sizes that wobble by a few tenths of a point
# within what is visually a single style, which is enough to make a title
# (11.6pt) rank below its own author line (11.8pt) if sizes are compared
# exactly.
_SIZE_TIER_RATIO = 0.95

# A title may wrap across several lines, but consecutive lines sit close
# together. The gap is measured between *successive* lines rather than
# from the first one, so a five-line title chains correctly while a
# same-size heading further down the page does not get swept in.
_MAX_TITLE_LINE_GAP_RATIO = 2.2


def _best_title(candidates: List[Tuple[int, float, float, str]]) -> Optional[str]:
    """
    Picks the title from (page, size, y, text) candidates.

    Largest type wins, then highest on the page. Position is the
    tie-breaker rather than an equal partner: on a well-set paper the
    title is simply the biggest text, and only on papers whose title is
    no larger than its affiliations does position have to decide.

    Once a winning line is chosen, adjacent lines of the same size on the
    same page are joined onto it, because a long title wraps.
    """
    if not candidates:
        return None

    largest = max(size for _, size, _, _ in candidates)
    top_tier = [c for c in candidates if c[1] >= largest * _SIZE_TIER_RATIO]

    # Earliest page first, then highest on that page.
    top_tier.sort(key=lambda c: (c[0], c[2]))
    page, size, y, first_text = top_tier[0]

    # Walk down the page, chaining each same-size line that follows closely
    # enough on from the previous one.
    parts = [first_text]
    previous_y = y
    for p, s, line_y, text in top_tier[1:]:
        if p != page or abs(s - size) > 0.5:
            continue
        if line_y - previous_y > size * _MAX_TITLE_LINE_GAP_RATIO:
            break
        parts.append(text)
        previous_y = line_y

    joined = re.sub(r"\s+", " ", " ".join(parts)).strip()

    # Re-check the assembled string, not just its parts. Journal logos are
    # often set one word per line ("Astronomy" above "Astrophysics"), so
    # neither line is recognisable as a banner until they are joined.
    if _is_banner(joined):
        return None

    # A single short fragment is more likely a stray glyph or a logo than a
    # title; anything this short is not worth naming a document after.
    return joined if len(joined) >= 10 else None


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
