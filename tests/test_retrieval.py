"""
Tests for BM25 scoring, rank fusion, and the multi-document vector store.

These cover the retrieval logic that Phase 2 introduced. They use
synthetic chunks and fake embeddings rather than a real PDF or a real
sentence transformer, so the suite runs in under a second and tests the
retrieval arithmetic rather than the behaviour of a downloaded model.
"""

import numpy as np
import pytest

from src.bm25 import BM25Index, tokenize
from src.chunker import Chunk
from src.retriever import diversify_by_document, looks_comparative, reciprocal_rank_fusion
from src.vector_store import VectorStore


def make_chunk(chunk_id: int, text: str, doc_id: str = "doc1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        word_count=len(text.split()),
        page_start=1,
        page_end=1,
        doc_id=doc_id,
    )


class TestTokenizer:
    def test_lowercases_and_strips_stopwords(self):
        assert tokenize("The rate of the reaction") == ["rate", "reaction"]

    def test_keeps_numeric_and_alphanumeric_terms_intact(self):
        # These are precisely the rare terms BM25 exists to match on;
        # splitting them apart would destroy the signal.
        assert tokenize("Kzz of 1e7 cm2 at 70km") == ["kzz", "1e7", "cm2", "70km"]

    def test_keeps_decimal_numbers_together(self):
        assert "3.14" in tokenize("the value 3.14 was used")


class TestBM25:
    def test_ranks_exact_term_matches_first(self):
        chunks = [
            make_chunk(0, "the atmosphere is cold and dense"),
            make_chunk(1, "the eddy diffusion coefficient Kzz is 1e7 cm2"),
            make_chunk(2, "clouds reflect sunlight efficiently"),
        ]
        index = BM25Index()
        index.build(chunks)

        results = index.search("Kzz value", top_k=3)
        assert results[0].chunk_index == 1

    def test_drops_chunks_with_no_query_terms(self):
        chunks = [make_chunk(0, "sulfur dioxide"), make_chunk(1, "carbon monoxide")]
        index = BM25Index()
        index.build(chunks)

        results = index.search("sulfur", top_k=5)
        assert len(results) == 1
        assert results[0].chunk_index == 0

    def test_rare_terms_outweigh_common_ones(self):
        # "atmosphere" appears everywhere and should carry little signal;
        # "kzz" appears once and should dominate the ranking.
        chunks = [make_chunk(i, "atmosphere model study") for i in range(9)]
        chunks.append(make_chunk(9, "atmosphere kzz"))
        index = BM25Index()
        index.build(chunks)

        results = index.search("atmosphere kzz", top_k=10)
        assert results[0].chunk_index == 9

    def test_length_normalization_does_not_favour_long_chunks(self):
        short = make_chunk(0, "kzz coefficient")
        padded = make_chunk(1, "kzz coefficient " + "filler " * 200)
        index = BM25Index()
        index.build([short, padded])

        results = index.search("kzz coefficient", top_k=2)
        assert results[0].chunk_index == 0

    def test_restricts_scoring_to_allowed_indices(self):
        chunks = [make_chunk(i, "sulfur dioxide abundance") for i in range(3)]
        index = BM25Index()
        index.build(chunks)

        results = index.search("sulfur", top_k=5, allowed_indices={2})
        assert [r.chunk_index for r in results] == [2]

    def test_empty_query_returns_nothing(self):
        index = BM25Index()
        index.build([make_chunk(0, "some text")])
        assert index.search("the of and", top_k=5) == []


class TestReciprocalRankFusion:
    def test_chunk_ranked_by_both_retrievers_wins(self):
        # 7 is second in both lists; 1 and 4 each lead only one list.
        fused = reciprocal_rank_fusion([[1, 7, 2], [4, 7, 3]])
        assert fused[0] == 7

    def test_preserves_order_when_lists_agree(self):
        assert reciprocal_rank_fusion([[1, 2, 3], [1, 2, 3]]) == [1, 2, 3]

    def test_includes_every_chunk_from_every_list(self):
        assert set(reciprocal_rank_fusion([[1, 2], [3, 4]])) == {1, 2, 3, 4}

    def test_is_deterministic_for_tied_scores(self):
        # Disjoint lists give every chunk an identical fused score, so the
        # tie-break has to be stable or results would shuffle between runs.
        first = reciprocal_rank_fusion([[5, 6], [7, 8]])
        assert all(reciprocal_rank_fusion([[5, 6], [7, 8]]) == first for _ in range(5))

    def test_handles_empty_lists(self):
        assert reciprocal_rank_fusion([[], []]) == []
        assert reciprocal_rank_fusion([[1, 2], []]) == [1, 2]


class TestVectorStore:
    @staticmethod
    def unit_vectors(rows: list) -> np.ndarray:
        matrix = np.array(rows, dtype="float32")
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    def build_store(self) -> VectorStore:
        store = VectorStore(embedding_dim=2)
        store.add_document(
            "paper_a",
            [make_chunk(0, "alpha", "x"), make_chunk(1, "beta", "x")],
            self.unit_vectors([[1, 0], [0.9, 0.1]]),
        )
        store.add_document(
            "paper_b",
            [make_chunk(0, "gamma", "y")],
            self.unit_vectors([[0, 1]]),
        )
        return store

    def test_stamps_doc_id_onto_every_chunk(self):
        store = self.build_store()
        assert [c.doc_id for c in store.chunks] == ["paper_a", "paper_a", "paper_b"]

    def test_tracks_distinct_documents(self):
        assert self.build_store().document_ids() == ["paper_a", "paper_b"]

    def test_rejects_duplicate_document_ids(self):
        store = self.build_store()
        with pytest.raises(ValueError, match="already in the store"):
            store.add_document("paper_a", [make_chunk(0, "dup")], self.unit_vectors([[1, 0]]))

    def test_rejects_mismatched_chunk_and_embedding_counts(self):
        store = VectorStore(embedding_dim=2)
        with pytest.raises(ValueError, match="Mismatch"):
            store.add_document("d", [make_chunk(0, "a")], self.unit_vectors([[1, 0], [0, 1]]))

    def test_search_returns_nearest_chunk_first(self):
        store = self.build_store()
        results = store.search(np.array([1.0, 0.0], dtype="float32"), top_k=2)
        assert results[0].chunk.text == "alpha"
        assert results[0].score > results[1].score

    def test_search_can_be_scoped_to_one_document(self):
        store = self.build_store()
        # The query points straight at paper_a's chunk, but scoping to
        # paper_b must return paper_b's chunk regardless.
        results = store.search(np.array([1.0, 0.0], dtype="float32"),
                               top_k=2, doc_ids=["paper_b"])
        assert [r.chunk.doc_id for r in results] == ["paper_b"]

    def test_scoping_to_an_unknown_document_returns_nothing(self):
        assert self.build_store().search(
            np.array([1.0, 0.0], dtype="float32"), doc_ids=["missing"]
        ) == []

    def test_removing_a_document_drops_only_its_chunks(self):
        store = self.build_store()
        removed = store.remove_document("paper_a")

        assert removed == 2
        assert len(store) == 1
        assert store.document_ids() == ["paper_b"]
        # The FAISS index must be rebuilt in step with the chunk list, or
        # a later search would return positions that no longer exist.
        assert store.index.ntotal == 1

    def test_removing_an_unknown_document_is_a_no_op(self):
        store = self.build_store()
        assert store.remove_document("nope") == 0
        assert len(store) == 3

    def test_search_after_removal_still_returns_valid_chunks(self):
        store = self.build_store()
        store.remove_document("paper_a")
        results = store.search(np.array([1.0, 0.0], dtype="float32"), top_k=5)
        assert [r.chunk.text for r in results] == ["gamma"]

    def test_empty_store_searches_return_nothing(self):
        assert VectorStore(embedding_dim=2).search(np.array([1.0, 0.0], dtype="float32")) == []


class TestDocumentQuota:
    """
    Per-document quotas, for questions that span papers.

    A single top-k budget is spent entirely on whichever paper matches
    most strongly, so a comparative question never sees the second paper's
    evidence and the generator correctly reports it cannot compare.
    """

    @staticmethod
    def hits(*doc_ids):
        """(chunk index, score) pairs with descending scores."""
        return [(i, 1.0 - i * 0.01) for i in range(len(doc_ids))]

    @staticmethod
    def chunks_for(*doc_ids):
        return [make_chunk(i, f"text {i}", doc) for i, doc in enumerate(doc_ids)]

    def test_no_quota_leaves_ranking_untouched(self):
        docs = ("a", "a", "a", "b")
        got = diversify_by_document(self.hits(*docs), self.chunks_for(*docs), 3, None)
        assert [i for i, _ in got] == [0, 1, 2]

    def test_quota_promotes_a_second_document(self):
        docs = ("a", "a", "a", "b")
        got = diversify_by_document(self.hits(*docs), self.chunks_for(*docs), 3, 2)
        assert [self.chunks_for(*docs)[i].doc_id for i, _ in got] == ["a", "a", "b"]

    def test_quota_preserves_rank_order_within_a_document(self):
        docs = ("a", "b", "a", "b")
        got = diversify_by_document(self.hits(*docs), self.chunks_for(*docs), 4, 1)
        assert [i for i, _ in got][:2] == [0, 1]

    def test_backfills_when_too_few_documents_to_fill_top_k(self):
        # Only one document exists, so the quota cannot be honoured without
        # returning fewer passages than asked for. Rank order wins.
        docs = ("a", "a", "a", "a")
        got = diversify_by_document(self.hits(*docs), self.chunks_for(*docs), 3, 1)
        assert [i for i, _ in got] == [0, 1, 2]

    def test_single_document_corpus_is_unaffected_by_a_quota(self):
        docs = ("a", "a")
        with_quota = diversify_by_document(self.hits(*docs), self.chunks_for(*docs), 2, 1)
        without = diversify_by_document(self.hits(*docs), self.chunks_for(*docs), 2, None)
        assert with_quota == without

    def test_quota_below_one_is_treated_as_no_quota(self):
        docs = ("a", "a", "b")
        got = diversify_by_document(self.hits(*docs), self.chunks_for(*docs), 2, 0)
        assert [i for i, _ in got] == [0, 1]


class TestComparativeDetection:
    def test_detects_explicit_comparison(self):
        assert looks_comparative("Compare how Adam and the Transformer paper describe settings")
        assert looks_comparative("What is X versus Y?")

    def test_detects_per_paper_phrasing(self):
        assert looks_comparative("The eddy diffusion coefficient according to each paper")
        assert looks_comparative("Which papers use a Gaussian process?")
        assert looks_comparative("Do any two papers in this library disagree?")

    def test_ignores_ordinary_single_paper_questions(self):
        # A false positive splits the passage budget on a question that
        # needed all of it on one paper, which measurably costs accuracy.
        assert not looks_comparative("What eddy diffusion coefficient Kzz was used?")
        assert not looks_comparative("How many attention heads does the base model use?")
        assert not looks_comparative("What is the die size of the ASIC?")

    def test_handles_empty_input(self):
        assert not looks_comparative("")


class TestTunedDefaultsReachThePipeline:
    def test_pipeline_does_not_shadow_the_tuned_candidate_pool(self):
        # RAGPipeline used to hardcode candidate_pool=20, so the benchmark
        # measured a pool of 30 while the app and API quietly ran 20. The
        # tuned value must live in exactly one place.
        import inspect
        from src import rag_pipeline

        signature = inspect.signature(rag_pipeline.RAGPipeline.__init__)
        assert signature.parameters["candidate_pool"].default is None
