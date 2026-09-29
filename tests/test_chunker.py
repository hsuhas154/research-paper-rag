"""
Tests for sentence splitting and provenance-carrying chunking.
"""

from src.chunker import Chunk, chunk_segments, split_into_sentences
from src.pdf_processor import Segment


def make_segment(n_sentences: int, page: int, section=None, word="word") -> Segment:
    """A segment of n simple sentences, each 10 words long."""
    sentence = " ".join([word] * 9).capitalize() + " end."
    return Segment(" ".join([sentence] * n_sentences), page, section)


class TestSentenceSplitting:
    def test_splits_on_sentence_boundaries(self):
        assert split_into_sentences("First one. Second one. Third one.") == [
            "First one.", "Second one.", "Third one."
        ]

    def test_does_not_split_on_academic_abbreviations(self):
        text = "The model of Smith et al. 2020 is used. It performs well."
        assert split_into_sentences(text) == [
            "The model of Smith et al. 2020 is used.",
            "It performs well.",
        ]

    def test_does_not_split_on_eg_and_ie(self):
        text = "Several species, e.g. SO2 and CO, are tracked. Others are not."
        assert len(split_into_sentences(text)) == 2

    def test_does_not_split_mid_figure_reference(self):
        text = "As shown in Fig. 3 the rate increases. This is expected."
        assert len(split_into_sentences(text)) == 2

    def test_empty_text_yields_no_sentences(self):
        assert split_into_sentences("") == []
        assert split_into_sentences("   ") == []


class TestChunking:
    def test_chunks_respect_target_word_count(self):
        chunks = chunk_segments([make_segment(60, page=1)], target_words=50, overlap_words=10)
        assert len(chunks) > 1
        # A chunk closes on the sentence that crosses the target, so it can
        # overshoot by at most one sentence's worth of words.
        assert all(c.word_count <= 50 + 10 for c in chunks)

    def test_consecutive_chunks_overlap(self):
        chunks = chunk_segments([make_segment(60, page=1)], target_words=50, overlap_words=20)
        first_sentences = set(split_into_sentences(chunks[0].text))
        second_sentences = set(split_into_sentences(chunks[1].text))
        assert first_sentences & second_sentences, "expected shared sentences between chunks"

    def test_chunk_ids_are_sequential_from_zero(self):
        chunks = chunk_segments([make_segment(60, page=1)], target_words=40)
        assert [c.chunk_id for c in chunks] == list(range(len(chunks)))

    def test_never_mixes_two_sections_in_one_chunk(self):
        segments = [
            make_segment(3, page=1, section="1 Introduction"),
            make_segment(3, page=1, section="2 Methods"),
        ]
        chunks = chunk_segments(segments, target_words=500, overlap_words=10)
        sections = [c.section for c in chunks]
        assert sections == ["1 Introduction", "2 Methods"]

    def test_records_page_range_it_spans(self):
        segments = [
            make_segment(4, page=7, section="3 Results"),
            make_segment(4, page=8, section="3 Results"),
        ]
        chunk = chunk_segments(segments, target_words=1000, overlap_words=10)[0]
        assert chunk.page_start == 7
        assert chunk.page_end == 8

    def test_empty_input_yields_no_chunks(self):
        assert chunk_segments([]) == []

    def test_trailing_overlap_alone_does_not_become_a_chunk(self):
        # After the final full chunk, only the overlap seed remains. Emitting
        # it would produce a chunk that is a strict duplicate of text already
        # indexed, inflating the index with a redundant near-copy.
        chunks = chunk_segments([make_segment(10, page=1)], target_words=50, overlap_words=30)
        assert all(c.word_count > 30 for c in chunks)


class TestCitation:
    def test_includes_title_page_and_section(self):
        chunk = Chunk(0, "text", 2, page_start=4, page_end=4,
                      section="2.2 Chemistry", doc_id="abc")
        assert chunk.citation("Dai et al. 2024") == (
            "Dai et al. 2024, p. 4, Section 2.2 Chemistry"
        )

    def test_renders_a_page_range_when_the_chunk_spans_pages(self):
        chunk = Chunk(0, "text", 2, page_start=4, page_end=6)
        assert "p. 4-6" in chunk.citation()

    def test_degrades_gracefully_without_metadata(self):
        assert Chunk(0, "text", 2).citation() == "source unknown"
