"""
Tests for PDF text cleaning, reference stripping, and structural
segmentation.

These operate on synthetic RawPages rather than real PDFs, so they test
the parsing logic without depending on any particular file.
"""

from src.pdf_processor import (
    RawPage,
    clean_text,
    detect_heading,
    segment_pages,
    strip_references_section,
)


class TestCleanText:
    def test_rejoins_words_broken_across_lines(self):
        assert clean_text("atmo-\nsphere") == "atmosphere"

    def test_does_not_rejoin_genuine_hyphenated_compounds(self):
        # A hyphen before a capital is a real compound, not a line break.
        assert clean_text("Venus-\nExpress data") == "Venus- Express data"

    def test_collapses_single_line_breaks_into_spaces(self):
        assert clean_text("first line\nsecond line") == "first line second line"

    def test_normalizes_runs_of_whitespace(self):
        assert clean_text("too    many\t\tspaces") == "too many spaces"


class TestHeadingDetection:
    def test_detects_numbered_headings(self):
        assert detect_heading("2.2 Atmospheric chemistry") == "2.2 Atmospheric chemistry"

    def test_detects_known_section_names(self):
        assert detect_heading("Introduction") == "Introduction"
        assert detect_heading("Results and discussion") == "Results and discussion"

    def test_detects_known_section_names_in_all_caps(self):
        assert detect_heading("ABSTRACT") == "ABSTRACT"

    def test_detects_multi_word_all_caps_headings(self):
        assert detect_heading("RESULTS AND DISCUSSION") is not None

    def test_ignores_figure_and_table_captions(self):
        assert detect_heading("3. Fig. of the model domain") is None
        assert detect_heading("2. Table of reaction rates") is None

    def test_ignores_single_all_caps_chemical_formulas(self):
        # These sit in tables and figures throughout a chemistry paper and
        # would otherwise be mistaken for section headings.
        assert detect_heading("COS") is None
        assert detect_heading("HNO") is None

    def test_ignores_equation_fragments(self):
        # Extracted inline equations share the "number then text" shape but
        # end on punctuation rather than a word.
        assert detect_heading("4 Qpn(dp) ddp,") is None

    def test_ignores_ordinary_prose(self):
        assert detect_heading("The model solves the continuity equation for each species.") is None

    def test_ignores_overlong_lines(self):
        assert detect_heading("Introduction " + "x" * 100) is None


class TestReferenceStripping:
    def test_removes_a_trailing_references_section(self):
        pages = [
            RawPage(1, "Body text about Venus. " * 40),
            RawPage(2, "More results here. " * 40),
            RawPage(3, "Conclusions drawn. " * 20 + "\nReferences\nSmith et al. 2020\n"),
            RawPage(4, "Jones et al. 2021\nBrown et al. 2022\n"),
        ]
        kept = strip_references_section(pages)

        assert len(kept) == 3
        assert "Smith et al. 2020" not in kept[-1].text
        assert "Conclusions drawn." in kept[-1].text

    def test_ignores_the_word_references_early_in_the_document(self):
        # A genuine bibliography heading is always near the end; matching an
        # early mention would truncate real content.
        pages = [
            RawPage(1, "\nReferences\nare discussed in this introduction.\n"),
            RawPage(2, "Body text. " * 60),
            RawPage(3, "More body text. " * 60),
        ]
        assert len(strip_references_section(pages)) == 3

    def test_leaves_the_document_alone_when_no_heading_is_found(self):
        pages = [RawPage(1, "Body text only."), RawPage(2, "More body text.")]
        assert strip_references_section(pages) == pages

    def test_handles_an_empty_document(self):
        assert strip_references_section([]) == []


class TestSegmentation:
    def test_tags_each_segment_with_its_page(self):
        segments = segment_pages([
            RawPage(3, "Some body text on page three."),
            RawPage(4, "Different text on page four."),
        ])
        assert [s.page_number for s in segments] == [3, 4]

    def test_carries_the_section_heading_onto_following_text(self):
        segments = segment_pages([
            RawPage(1, "1 Introduction\nVenus has a dense atmosphere.\n")
        ])
        assert len(segments) == 1
        assert segments[0].section == "1 Introduction"
        assert segments[0].text == "Venus has a dense atmosphere."

    def test_starts_a_new_segment_at_each_heading(self):
        segments = segment_pages([
            RawPage(1, "1 Introduction\nFirst part.\n2 Methods\nSecond part.\n")
        ])
        assert [(s.section, s.text) for s in segments] == [
            ("1 Introduction", "First part."),
            ("2 Methods", "Second part."),
        ]

    def test_text_before_the_first_heading_has_no_section(self):
        segments = segment_pages([RawPage(1, "Journal banner text.\n1 Introduction\nBody.\n")])
        assert segments[0].section is None
        assert segments[1].section == "1 Introduction"

    def test_a_section_continues_across_a_page_break(self):
        segments = segment_pages([
            RawPage(1, "2 Methods\nFirst half of the method.\n"),
            RawPage(2, "Second half of the method.\n"),
        ])
        assert [s.section for s in segments] == ["2 Methods", "2 Methods"]
        assert [s.page_number for s in segments] == [1, 2]

    def test_drops_page_furniture(self):
        # Bare page numbers and stray one-character lines are not prose.
        segments = segment_pages([RawPage(1, "7\nReal body text here.\n|\n")])
        assert [s.text for s in segments] == ["Real body text here."]
