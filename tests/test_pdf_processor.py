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


def body(lead: str, words: int = 45) -> str:
    """
    A block of prose long enough to count as a real section body.

    Sections shorter than _MIN_SECTION_WORDS are treated as detection
    artifacts and folded into the preceding section, so fixtures that
    exercise section labelling need a realistic amount of text under
    each heading.
    """
    return lead + " " + " ".join(["filler"] * words) + "."


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
            RawPage(1, f"1 Introduction\n{body('Venus has a dense atmosphere.')}\n")
        ])
        assert len(segments) == 1
        assert segments[0].section == "1 Introduction"
        assert "Venus has a dense atmosphere." in segments[0].text

    def test_keeps_the_heading_text_in_the_segment_it_introduces(self):
        # A heading describes its section better than any sentence inside
        # it, so it is retrieval signal worth indexing, not furniture to
        # discard. Dropping it would also delete real text whenever
        # detection misfires.
        segments = segment_pages([
            RawPage(1, f"2.2 Atmospheric chemistry\n{body('Reactions proceed quickly.')}\n")
        ])
        assert segments[0].text.startswith("2.2 Atmospheric chemistry")

    def test_starts_a_new_segment_at_each_heading(self):
        segments = segment_pages([
            RawPage(1, f"1 Introduction\n{body('First part.')}\n"
                       f"2 Methods\n{body('Second part.')}\n")
        ])
        assert [s.section for s in segments] == ["1 Introduction", "2 Methods"]
        assert "First part." in segments[0].text
        assert "Second part." in segments[1].text

    def test_text_before_the_first_heading_has_no_section(self):
        segments = segment_pages([
            RawPage(1, f"{body('Journal banner text.')}\n1 Introduction\n{body('Body.')}\n")
        ])
        assert segments[0].section is None
        assert segments[1].section == "1 Introduction"

    def test_a_section_continues_across_a_page_break(self):
        segments = segment_pages([
            RawPage(1, f"2 Methods\n{body('First half of the method.')}\n"),
            RawPage(2, f"{body('Second half of the method.')}\n"),
        ])
        assert [s.section for s in segments] == ["2 Methods", "2 Methods"]
        assert [s.page_number for s in segments] == [1, 2]

    def test_drops_page_furniture(self):
        # Bare page numbers and stray one-character lines are not prose.
        segments = segment_pages([RawPage(1, f"7\n{body('Real body text here.')}\n|\n")])
        assert segments[0].text.startswith("Real body text here.")
        assert "|" not in segments[0].text


class TestUndersizedSectionMerging:
    """
    Font-based heading detection is loose on papers with inconsistent
    typography, so a section is also judged by whether it has a body.
    """

    def test_folds_a_bodyless_heading_into_the_previous_section(self):
        # "2 Methods" here is a stray label, not a section: nothing follows
        # it. Merging relabels the affected segments rather than joining
        # them, which is what stops the chunker splitting at that point.
        segments = segment_pages([
            RawPage(1, f"1 Introduction\n{body('Real introduction text.')}\n"
                       f"2 Methods\nstray label\n")
        ])
        assert {s.section for s in segments} == {"1 Introduction"}
        assert "stray label" in " ".join(s.text for s in segments)

    def test_keeps_a_section_that_has_a_real_body(self):
        segments = segment_pages([
            RawPage(1, f"1 Introduction\n{body('Intro text.')}\n"
                       f"2 Methods\n{body('Method text.')}\n")
        ])
        assert [s.section for s in segments] == ["1 Introduction", "2 Methods"]

    def test_a_bodyless_heading_before_any_section_stays_unlabelled(self):
        segments = segment_pages([
            RawPage(1, f"1 Introduction\nstray\n{body('Text under introduction.')}\n")
        ])
        # The stray line is absorbed rather than deleted.
        assert "stray" in " ".join(s.text for s in segments)


class TestTitleSanity:
    """
    Guards for title extraction on the harder half of a real corpus:
    1950s to 1970s scans, publisher banners, and production filenames
    left in PDF metadata.
    """

    def test_rejects_publisher_banners_regardless_of_separator(self):
        from src.pdf_processor import _is_banner

        assert _is_banner("Science Journals — AAAS")
        assert _is_banner("Science Journals - AAAS")
        assert _is_banner("HHS Public Access")
        assert not _is_banner("Deep Residual Learning for Image Recognition")

    def test_rejects_scan_noise(self):
        from src.pdf_processor import _is_legible_text

        assert not _is_legible_text("~~tL-~;'=::::~4~=:::::==~=l")
        assert not _is_legible_text("_:: :..:- • •=")
        assert _is_legible_text("Theory of Superconductivity")

    def test_accepts_a_single_word_line_so_wrapped_titles_survive(self):
        # A five-line title legitimately puts "Machine" on a line of its
        # own; demanding two words per line truncates it.
        from src.pdf_processor import _is_legible_text

        assert _is_legible_text("Machine")
        assert _is_legible_text("Learning")

    def test_rejects_body_prose_lifted_from_a_scan(self):
        from src.pdf_processor import _is_plausible_title

        # A drop cap can make a body paragraph the largest text on the
        # page. Running prose starts mid-sentence; a title never does.
        assert not _is_plausible_title("so that the value of INd - N, I at its boundaries")
        assert _is_plausible_title("Theory of Superconductivity")

    def test_rejects_production_filenames_in_metadata(self):
        from src.pdf_processor import _clean_metadata_title

        assert _clean_metadata_title("6088 - V323.indd") is None
        assert _clean_metadata_title("manuscript_final.docx") is None
        assert _clean_metadata_title("moores paper") is None

    def test_keeps_a_genuinely_lower_case_title(self):
        from src.pdf_processor import _clean_metadata_title

        assert (_clean_metadata_title("64-channel ultrasound transducer amplifier")
                == "64-channel ultrasound transducer amplifier")


class TestSplitResolution:
    """
    The paper-level split is defined by title fragment, so it survives
    re-indexing. Fragments are checked against the corpus at load time.
    """

    @staticmethod
    def titles():
        return {
            "a": "Deep Learning",
            "b": "Deep learning in computational mechanics: a review",
            "c": "Attention is All you Need",
        }

    def test_an_exact_title_wins_over_a_longer_one_containing_it(self, tmp_path):
        import json

        from src.splits import resolve

        spec = tmp_path / "splits.json"
        spec.write_text(json.dumps({"clusters": {"dl": {
            "train": ["Deep Learning", "Deep learning in computational mechanics"],
            "val": ["Attention is All you Need"], "test": [],
        }}}))
        resolved = resolve(self.titles(), spec)
        assert set(resolved["train"]["doc_ids"]) == {"a", "b"}

    def test_an_unmatched_fragment_fails_loudly(self, tmp_path):
        import json

        import pytest

        from src.splits import SplitError, resolve

        spec = tmp_path / "splits.json"
        spec.write_text(json.dumps({"clusters": {"dl": {
            "train": ["Deep Learning", "Deep learning in computational mechanics",
                      "Attention is All you Need", "A Paper That Does Not Exist"],
            "val": [], "test": [],
        }}}))
        with pytest.raises(SplitError, match="matched 0 papers"):
            resolve(self.titles(), spec)

    def test_a_paper_in_no_split_fails_loudly(self, tmp_path):
        import json

        import pytest

        from src.splits import SplitError, resolve

        # Silently dropping a paper would shrink a split without anyone
        # noticing, and the metrics would still look fine.
        spec = tmp_path / "splits.json"
        spec.write_text(json.dumps({"clusters": {"dl": {
            "train": ["Deep Learning"], "val": [], "test": [],
        }}}))
        with pytest.raises(SplitError, match="in no split"):
            resolve(self.titles(), spec)

    def test_a_paper_in_two_splits_fails_loudly(self, tmp_path):
        import json

        import pytest

        from src.splits import SplitError, resolve

        spec = tmp_path / "splits.json"
        spec.write_text(json.dumps({"clusters": {"dl": {
            "train": ["Attention is All you Need"],
            "val": ["Attention is All you Need"], "test": [],
        }}}))
        with pytest.raises(SplitError, match="appears in both"):
            resolve(self.titles(), spec)
