"""
Tests for groundedness measurement.

The LLM calls are stubbed: these cover the arithmetic and the citation
checking, which is where a bug would quietly produce a flattering number.
"""

import pytest

from src import faithfulness as f
from src.chunker import Chunk
from src.faithfulness import (
    FaithfulnessReport,
    aggregate,
    check_citations,
    evaluate_answer,
    extract_claims,
    judge_claim,
)
from src.vector_store import RetrievedChunk


def sources(n=3):
    return [
        RetrievedChunk(chunk=Chunk(i, f"passage {i}", 2, doc_id="d"), score=1.0)
        for i in range(n)
    ]


class TestCitationChecking:
    def test_counts_valid_markers(self):
        result = check_citations("A [Excerpt 1] and B [Excerpt 3].", n_supplied=3)
        assert result["n_citations"] == 2
        assert result["n_valid_citations"] == 2
        assert result["n_excerpts_cited"] == 2

    def test_a_marker_beyond_the_supplied_range_is_invalid(self):
        # A fabricated citation is worse than an uncited claim: the answer
        # looks sourced and defeats the reader's own check.
        result = check_citations("Claimed in [Excerpt 9].", n_supplied=3)
        assert result["n_citations"] == 1
        assert result["n_valid_citations"] == 0

    def test_repeated_citations_count_once_for_utilisation(self):
        result = check_citations("[Excerpt 1] and again [Excerpt 1].", n_supplied=4)
        assert result["n_citations"] == 2
        assert result["n_excerpts_cited"] == 1

    def test_an_uncited_answer_reports_zero(self):
        result = check_citations("No markers here.", n_supplied=3)
        assert result["n_citations"] == 0
        assert result["n_excerpts_cited"] == 0


class TestReportArithmetic:
    def test_faithfulness_is_the_supported_fraction(self):
        report = FaithfulnessReport(
            claims=["a", "b", "c", "d"],
            verdicts=["SUPPORTED", "SUPPORTED", "UNSUPPORTED", "CONTRADICTED"],
        )
        assert report.faithfulness == 0.5
        assert report.contradiction_rate == 0.25

    def test_measures_are_none_when_they_do_not_apply(self):
        # An answer with no claims is not 0% faithful; the measure simply
        # does not apply, and averaging a zero in would understate things.
        empty = FaithfulnessReport()
        assert empty.faithfulness is None
        assert empty.citation_validity is None
        assert empty.context_utilisation is None

    def test_context_utilisation_counts_distinct_excerpts(self):
        report = FaithfulnessReport(n_excerpts_supplied=5, n_excerpts_cited=2)
        assert report.context_utilisation == 0.4


class TestAggregate:
    def test_skips_inapplicable_measures_rather_than_counting_them_as_zero(self):
        reports = [
            FaithfulnessReport(claims=["a"], verdicts=["SUPPORTED"]),
            FaithfulnessReport(),  # no claims
        ]
        assert aggregate(reports)["faithfulness"] == 1.0
        assert aggregate(reports)["n_answers"] == 2

    def test_empty_input_does_not_divide_by_zero(self):
        assert aggregate([])["faithfulness"] is None


class TestJudging:
    def test_a_verdict_is_parsed_from_a_wordy_reply(self, monkeypatch):
        monkeypatch.setattr(f, "_ask", lambda *a, **kw: "I think this is SUPPORTED by it.")
        assert judge_claim("c", "ctx") == "SUPPORTED"

    def test_contradiction_wins_over_a_substring_match(self, monkeypatch):
        # "CONTRADICTED" contains neither of the others, but a naive scan
        # ordered the other way would mislabel "UNSUPPORTED".
        monkeypatch.setattr(f, "_ask", lambda *a, **kw: "CONTRADICTED")
        assert judge_claim("c", "ctx") == "CONTRADICTED"

    def test_an_unparseable_verdict_defaults_to_unsupported(self, monkeypatch):
        monkeypatch.setattr(f, "_ask", lambda *a, **kw: "hmm, hard to say")
        assert judge_claim("c", "ctx") == "UNSUPPORTED"

    def test_a_judge_failure_defaults_to_unsupported(self, monkeypatch):
        def boom(*a, **kw):
            raise RuntimeError("down")

        monkeypatch.setattr(f, "_ask", boom)
        assert judge_claim("c", "ctx") == "UNSUPPORTED"


class TestClaimExtraction:
    def test_strips_bullets_numbering_and_citations(self, monkeypatch):
        monkeypatch.setattr(f, "_ask", lambda *a, **kw:
                            "- The model uses 6 layers [Excerpt 1]\n"
                            "2. BLEU reached 28.4 on the task")
        claims = extract_claims("irrelevant")
        assert claims[0] == "The model uses 6 layers"
        assert claims[1] == "BLEU reached 28.4 on the task"

    def test_none_means_no_claims(self, monkeypatch):
        monkeypatch.setattr(f, "_ask", lambda *a, **kw: "NONE")
        assert extract_claims("I cannot answer that.") == []

    def test_an_empty_answer_is_not_sent_to_the_model(self, monkeypatch):
        called = []
        monkeypatch.setattr(f, "_ask", lambda *a, **kw: called.append(1) or "x")
        assert extract_claims("   ") == []
        assert not called


class TestEndToEnd:
    def test_evaluate_answer_combines_both_halves(self, monkeypatch):
        monkeypatch.setattr(f, "extract_claims", lambda a, m: ["claim one", "claim two"])
        monkeypatch.setattr(f, "judge_claim", lambda c, ctx, m: "SUPPORTED")
        report = evaluate_answer("An answer [Excerpt 1][Excerpt 2].", sources(3))

        assert report.faithfulness == 1.0
        assert report.n_valid_citations == 2
        assert report.context_utilisation == pytest.approx(2 / 3)
