"""
Tests for re-ranker model selection.

The deployed system should pick up a locally fine-tuned re-ranker when
one has been trained, without a code change, and fall back cleanly on a
fresh checkout where no model has been trained yet.
"""

import pytest

from src import reranker
from src.reranker import STOCK_RERANKER_MODEL, CrossEncoderReranker, default_reranker_model


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch, tmp_path):
    """Isolate from the developer's own env var and trained model."""
    monkeypatch.delenv("RERANKER_MODEL", raising=False)
    monkeypatch.setattr(reranker, "DOMAIN_RERANKER_PATH", tmp_path / "absent")


class TestDefaultSelection:
    def test_falls_back_to_the_stock_model(self):
        assert default_reranker_model() == STOCK_RERANKER_MODEL

    def test_does_not_auto_adopt_a_locally_trained_model(self, monkeypatch, tmp_path):
        # Measured on papers held out of training, the fine-tuned model was
        # no better than stock and worse on MRR. Picking it up just because
        # someone ran the training script would silently degrade retrieval.
        trained = tmp_path / "reranker-domain"
        trained.mkdir()
        (trained / "config.json").write_text("{}")
        monkeypatch.setattr(reranker, "DOMAIN_RERANKER_PATH", trained)

        assert default_reranker_model() == STOCK_RERANKER_MODEL

    def test_a_trained_model_is_used_only_when_opted_into(self, monkeypatch, tmp_path):
        trained = tmp_path / "reranker-domain"
        trained.mkdir()
        (trained / "config.json").write_text("{}")
        monkeypatch.setattr(reranker, "DOMAIN_RERANKER_PATH", trained)
        monkeypatch.setenv("RERANKER_MODEL", str(trained))

        assert default_reranker_model() == str(trained)

    def test_environment_variable_selects_any_model(self, monkeypatch):
        monkeypatch.setenv("RERANKER_MODEL", "some/other-model")
        assert default_reranker_model() == "some/other-model"

    def test_blank_environment_variable_is_ignored(self, monkeypatch):
        monkeypatch.setenv("RERANKER_MODEL", "   ")
        assert default_reranker_model() == STOCK_RERANKER_MODEL


class TestRerankerConstruction:
    def test_uses_the_resolved_default_when_unnamed(self):
        assert CrossEncoderReranker().model_name == STOCK_RERANKER_MODEL

    def test_an_explicit_name_wins_over_the_default(self, monkeypatch):
        monkeypatch.setenv("RERANKER_MODEL", "env/model")
        assert CrossEncoderReranker("explicit/model").model_name == "explicit/model"

    def test_does_not_load_the_model_at_construction(self):
        # Construction happens at app startup; loading is deferred so a
        # session that never re-ranks never pays for it.
        assert CrossEncoderReranker()._model is None


class TestEmptyInput:
    def test_reranking_nothing_returns_nothing_without_loading(self):
        assert CrossEncoderReranker().rerank("a question", []) == []
