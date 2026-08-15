"""Tests for the retriever layer."""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluate import DATA_DIR
from src.retriever import FIELDS, BM25Retriever, field_text, load_kb, tokenize


def _kb_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [10, 20, 30],
            "question": [
                "How long should I wash my hands?",
                "How far apart should I stand?",
                "What are the symptoms of COVID-19?",
            ],
            "answer": [
                "Wash for at least 20 seconds with soap.",
                "Stay 6 feet from other people.",
                "Fever, cough and tiredness are common.",
            ],
            "source": ["WHO", "CDC_FAQ", "WHO"],
            "url": ["u1", "u2", "u3"],
            "trust": ["official", "official", "official"],
        }
    )


class TestTokenize:
    def test_keeps_digits(self) -> None:
        """A query about "20 seconds" must be able to
        match on the number, so the tokenizer cannot discard it."""
        assert "20" in tokenize("Wash for 20 seconds")
        assert "6" in tokenize("Stay 6 feet apart")
        assert "14" in tokenize("Isolate for 14 days")

    def test_lowercases(self) -> None:
        assert tokenize("COVID") == ["covid"]

    def test_splits_covid_19_into_matchable_tokens(self) -> None:
        assert tokenize("COVID-19") == ["covid", "19"]

    def test_drops_punctuation(self) -> None:
        assert tokenize("hands?  Really!") == ["hands", "really"]

    def test_empty_input(self) -> None:
        assert tokenize("") == []
        assert tokenize("???") == []


class TestFieldText:
    def test_question_field(self) -> None:
        kb = _kb_fixture()
        assert field_text(kb, "question").iloc[0] == "How long should I wash my hands?"

    def test_question_answer_field_concatenates(self) -> None:
        kb = _kb_fixture()
        combined = field_text(kb, "question_answer").iloc[0]
        assert combined.startswith("How long should I wash my hands?")
        assert "20 seconds" in combined

    def test_unknown_field_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown field"):
            field_text(_kb_fixture(), "answer")


class TestBM25Retriever:
    def test_finds_the_obvious_match(self) -> None:
        r = BM25Retriever(_kb_fixture())
        hits = r.search("what are the symptoms?", top_k=3)
        assert hits[0][0] == 30

    def test_returns_kb_ids_not_row_positions(self) -> None:
        """The KB is keyed by COUGH ids, which are not row indices after dedup.
        Confusing the two silently misattributes every result."""
        r = BM25Retriever(_kb_fixture())
        ids = {kb_id for kb_id, _ in r.search("hands", top_k=3)}
        assert ids <= {10, 20, 30}

    def test_results_are_sorted_by_descending_score(self) -> None:
        r = BM25Retriever(_kb_fixture())
        scores = [s for _, s in r.search("how should I wash", top_k=3)]
        assert scores == sorted(scores, reverse=True)

    def test_respects_top_k(self) -> None:
        r = BM25Retriever(_kb_fixture())
        assert len(r.search("how", top_k=2)) == 2

    def test_top_k_larger_than_corpus_is_safe(self) -> None:
        r = BM25Retriever(_kb_fixture())
        assert len(r.search("how", top_k=99)) == 3

    def test_digits_are_searchable(self) -> None:
        """Only reachable through the answer text, so it also proves the
        question_answer field is really being indexed."""
        r = BM25Retriever(_kb_fixture(), field="question_answer")
        assert r.search("20 seconds", top_k=1)[0][0] == 10

    def test_empty_query_returns_nothing(self) -> None:
        r = BM25Retriever(_kb_fixture())
        assert r.search("???", top_k=5) == []

    def test_search_batch_matches_individual_search(self) -> None:
        r = BM25Retriever(_kb_fixture())
        queries = ["symptoms", "how far apart"]
        assert r.search_batch(queries, top_k=2) == [r.search(q, top_k=2) for q in queries]

    def test_name_records_the_field(self) -> None:
        """Result tables identify which configuration produced them."""
        assert BM25Retriever(_kb_fixture(), field="question").name == "bm25[question]"
        assert BM25Retriever(_kb_fixture(), field="question_answer").name == "bm25[question_answer]"

    def test_unknown_field_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown field"):
            BM25Retriever(_kb_fixture(), field="nonsense")

    def test_all_declared_fields_are_constructible(self) -> None:
        for f in FIELDS:
            assert len(BM25Retriever(_kb_fixture(), field=f)) == 3


@pytest.mark.skipif(
    not (DATA_DIR / "kb.parquet").exists(),
    reason="run `python -m src.download && python -m src.prep` first",
)
class TestAgainstRealKb:
    def test_loads_and_indexes_the_real_corpus(self) -> None:
        kb = load_kb()
        r = BM25Retriever(kb)
        assert len(r) == len(kb)

        hits = r.search("how does covid-19 spread?", top_k=10)
        assert len(hits) == 10
        assert all(kb_id in set(kb["id"]) for kb_id, _ in hits)
        assert hits[0][1] > 0
