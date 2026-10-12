"""
Tests for follow-up rewriting and conversation history.

Rewriting is the part that matters: a follow-up like "and at 60 km?" is
clear to a reader and useless to a retriever, so it is turned into a
standalone question before retrieval. These cover when that fires, when
it deliberately does not, and that a failure degrades rather than breaks.
"""

import pytest

from src import conversation as conv
from src.conversation import Conversation, looks_self_contained, rewrite_follow_up


@pytest.fixture
def chat():
    c = Conversation()
    c.add("What eddy diffusion coefficient was used at 80 km?",
          "The model uses 5.3x10^4 cm2/s at 80 km [Excerpt 1].")
    return c


class TestSelfContainedDetection:
    def test_a_full_question_needs_no_rewrite(self):
        assert looks_self_contained("What BLEU score did the big Transformer reach?")

    def test_a_dependent_opener_signals_a_follow_up(self):
        assert not looks_self_contained("And at 60 km?")
        assert not looks_self_contained("What about the lower boundary?")

    def test_a_pronoun_signals_a_follow_up(self):
        assert not looks_self_contained("How was it validated against observations?")
        assert not looks_self_contained("Why did they choose that value?")

    def test_a_very_short_question_is_treated_as_dependent(self):
        assert not looks_self_contained("Which dataset?")

    def test_detection_is_biased_towards_leaving_questions_alone(self):
        # A needless rewrite spends an LLM call and can distort a question
        # that was already fine; a missed rewrite changes nothing.
        for question in [
            "What are the four key ideas behind convolutional neural networks?",
            "How many attention heads does the base model use?",
            "Which clinical datasets were used for training and evaluation?",
        ]:
            assert looks_self_contained(question), question


class TestRewriting:
    def test_no_history_means_no_rewrite(self, monkeypatch):
        called = []
        monkeypatch.setattr(conv.ollama, "chat", lambda **kw: called.append(kw))
        assert rewrite_follow_up("And at 60 km?", Conversation()) == "And at 60 km?"
        assert not called, "should not call the model with an empty conversation"

    def test_a_self_contained_question_is_not_sent_to_the_model(self, chat, monkeypatch):
        called = []
        monkeypatch.setattr(conv.ollama, "chat", lambda **kw: called.append(kw))
        question = "What boundary conditions were set at the lower boundary?"
        assert rewrite_follow_up(question, chat) == question
        assert not called

    def test_a_follow_up_is_rewritten(self, chat, monkeypatch):
        monkeypatch.setattr(conv.ollama, "chat", lambda **kw: {
            "message": {"content": "What eddy diffusion coefficient was used at 60 km?"}})
        assert rewrite_follow_up("And at 60 km?", chat) == (
            "What eddy diffusion coefficient was used at 60 km?")

    def test_a_label_prefix_is_stripped(self, chat, monkeypatch):
        monkeypatch.setattr(conv.ollama, "chat", lambda **kw: {
            "message": {"content": 'Standalone question: "What Kzz was used at 60 km?"'}})
        assert rewrite_follow_up("And at 60 km?", chat) == "What Kzz was used at 60 km?"

    def test_an_unreachable_model_degrades_to_the_original(self, chat, monkeypatch):
        def boom(**kw):
            raise RuntimeError("ollama is down")

        monkeypatch.setattr(conv.ollama, "chat", boom)
        # A degraded follow-up beats no answer at all.
        assert rewrite_follow_up("And at 60 km?", chat) == "And at 60 km?"

    def test_an_answer_instead_of_a_rewrite_is_rejected(self, chat, monkeypatch):
        # Models sometimes answer the follow-up rather than rewriting it.
        monkeypatch.setattr(conv.ollama, "chat", lambda **kw: {
            "message": {"content": "The coefficient at 60 km is 2.1x10^4 cm2/s."}})
        assert rewrite_follow_up("And at 60 km?", chat) == "And at 60 km?"

    def test_a_rambling_rewrite_is_rejected(self, chat, monkeypatch):
        monkeypatch.setattr(conv.ollama, "chat", lambda **kw: {
            "message": {"content": "What " + "very " * 100 + "long question?"}})
        assert rewrite_follow_up("And at 60 km?", chat) == "And at 60 km?"


class TestHistoryFormatting:
    def test_messages_alternate_user_and_assistant(self, chat):
        messages = chat.as_messages()
        assert [m["role"] for m in messages] == ["user", "assistant"]

    def test_long_answers_are_truncated_for_the_generator(self):
        c = Conversation()
        c.add("q", "x" * 5000)
        # A long history crowds the retrieved passages out of the context
        # window, which is the opposite of what history is for.
        assert len(c.as_messages()[1]["content"]) <= 600

    def test_only_recent_turns_are_kept(self):
        c = Conversation()
        for i in range(10):
            c.add(f"q{i}", f"a{i}")
        assert len(c.as_messages(limit=2)) == 4
        assert "q9" in c.as_messages(limit=2)[2]["content"]

    def test_clearing_empties_the_conversation(self, chat):
        chat.clear()
        assert chat.is_empty()


class TestPipelineIntegration:
    """
    The pipeline sends a rewritten question to the retriever and the
    original to the generator. Both halves are easy to wire backwards, and
    a unit test of the rewriter alone would not notice.
    """

    def test_retriever_gets_the_rewrite_and_generator_gets_the_original(
        self, monkeypatch, chat
    ):
        from src import rag_pipeline as rp

        seen = {}

        class FakeRetrieval:
            chunks, mode = [], "hybrid+rerank"
            n_dense_candidates = n_lexical_candidates = n_fused_candidates = 0

        class FakeRetriever:
            def retrieve(self, question, **kw):
                seen["retrieved"] = question
                return FakeRetrieval()

        pipeline = rp.RAGPipeline.__new__(rp.RAGPipeline)
        pipeline.retriever = FakeRetriever()
        pipeline.top_k = 5
        pipeline.corpus = type("C", (), {"is_empty": lambda self: False,
                                         "documents": []})()
        monkeypatch.setattr(rp, "rewrite_follow_up",
                            lambda q, c: "What Kzz was used at 60 km?")
        monkeypatch.setattr(rp, "generate_answer",
                            lambda q, chunks, **kw: seen.setdefault("generated", q) or "ans")
        monkeypatch.setattr(pipeline, "document_titles", lambda: {})

        answer = pipeline.answer_question("And at 60 km?", conversation=chat)

        assert seen["retrieved"] == "What Kzz was used at 60 km?"
        assert seen["generated"] == "And at 60 km?"
        assert answer.retrieval_question == "What Kzz was used at 60 km?"

    def test_no_rewrite_reports_no_retrieval_question(self, monkeypatch):
        from src import rag_pipeline as rp

        class FakeRetrieval:
            chunks, mode = [], "hybrid+rerank"
            n_dense_candidates = n_lexical_candidates = n_fused_candidates = 0

        pipeline = rp.RAGPipeline.__new__(rp.RAGPipeline)
        pipeline.retriever = type("R", (), {"retrieve": lambda self, q, **kw: FakeRetrieval()})()
        pipeline.top_k = 5
        pipeline.corpus = type("C", (), {"is_empty": lambda self: False,
                                         "documents": []})()
        monkeypatch.setattr(rp, "generate_answer", lambda q, chunks, **kw: "ans")
        monkeypatch.setattr(pipeline, "document_titles", lambda: {})

        answer = pipeline.answer_question("A complete standalone question here?")
        assert answer.retrieval_question is None
