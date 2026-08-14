"""Tests for the metric functions (PRD §19).

Metrics are checked against hand-computed values rather than a golden file. A
metric that is subtly wrong still produces plausible numbers, which is precisely
how the previous implementation reported an F1 of 0.9576 that meant nothing.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.evaluate import (
    DATA_DIR,
    EvalResult,
    evaluate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    select_split,
)
from src.retriever import Retriever


class TestReciprocalRank:
    def test_first_hit_at_rank_one(self) -> None:
        assert reciprocal_rank([5, 1, 2], {5}) == 1.0

    def test_first_hit_at_rank_three(self) -> None:
        assert reciprocal_rank([1, 2, 5], {5}) == pytest.approx(1 / 3)

    def test_only_the_first_hit_counts(self) -> None:
        """MRR rewards the rank of the first relevant result, nothing after."""
        assert reciprocal_rank([1, 5, 6], {5, 6}) == pytest.approx(1 / 2)

    def test_no_hit_scores_zero(self) -> None:
        assert reciprocal_rank([1, 2, 3], {9}) == 0.0

    def test_hit_beyond_cutoff_does_not_count(self) -> None:
        assert reciprocal_rank([1, 2, 3, 4, 5], {5}, k=4) == 0.0
        assert reciprocal_rank([1, 2, 3, 4, 5], {5}, k=5) == pytest.approx(1 / 5)

    def test_empty_ranking(self) -> None:
        assert reciprocal_rank([], {1}) == 0.0


class TestPrecisionAtK:
    def test_counts_hits_in_top_k(self) -> None:
        assert precision_at_k([1, 2, 3, 4], {1, 3}, 4) == 0.5

    def test_p_at_1(self) -> None:
        assert precision_at_k([7, 1], {7}, 1) == 1.0
        assert precision_at_k([1, 7], {7}, 1) == 0.0

    def test_divides_by_k_not_by_results_returned(self) -> None:
        """A retriever that returns 2 results for P@5 has failed to fill the
        other three slots and must be scored as such."""
        assert precision_at_k([1, 2], {1, 2}, 5) == pytest.approx(2 / 5)

    def test_zero_k(self) -> None:
        assert precision_at_k([1], {1}, 0) == 0.0


class TestRecallAtK:
    def test_finds_half_the_positives(self) -> None:
        assert recall_at_k([1, 2], {1, 2, 3, 4}, k=10) == 0.5

    def test_divides_by_total_relevant(self) -> None:
        assert recall_at_k([1, 2, 3], {1}, k=10) == 1.0

    def test_no_relevant_documents(self) -> None:
        assert recall_at_k([1, 2], set(), k=10) == 0.0

    def test_capped_by_cutoff(self) -> None:
        """With more positives than slots, recall cannot reach 1.0 -- the reason
        PRD 8.2 leads with MRR/P@k instead."""
        ranked = list(range(20))
        relevant = set(range(20))
        assert recall_at_k(ranked, relevant, k=10) == 0.5


class TestNdcgAtK:
    def test_perfect_ranking_scores_one(self) -> None:
        assert ndcg_at_k([1, 2, 3], {1, 2, 3}, k=10) == pytest.approx(1.0)

    def test_ideal_accounts_for_more_positives_than_slots(self) -> None:
        """Unlike recall, nDCG's ideal is capped at k, so a query with 20
        positives can still score 1.0 when the top 10 are all relevant."""
        assert ndcg_at_k(list(range(10)), set(range(20)), k=10) == pytest.approx(1.0)

    def test_known_value(self) -> None:
        # One hit at rank 2: DCG = 1/log2(3); ideal (1 positive) = 1/log2(2) = 1.
        assert ndcg_at_k([9, 1], {1}, k=10) == pytest.approx(1 / math.log2(3))

    def test_order_matters(self) -> None:
        good = ndcg_at_k([1, 2, 9, 9], {1, 2}, k=10)
        bad = ndcg_at_k([9, 9, 1, 2], {1, 2}, k=10)
        assert good > bad

    def test_no_relevant_documents(self) -> None:
        assert ndcg_at_k([1, 2], set(), k=10) == 0.0


class _StubRetriever(Retriever):
    """Returns a fixed ranking, so evaluate() is tested without BM25."""

    name = "stub"

    def __init__(self, ranking: list[int]) -> None:
        self._ranking = ranking

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        return [(kb_id, 1.0 / (i + 1)) for i, kb_id in enumerate(self._ranking[:top_k])]


class TestEvaluate:
    QUERIES = pd.DataFrame({"query_id": [1, 2], "query": ["a", "b"], "split": ["test", "test"]})

    def test_averages_across_queries(self) -> None:
        # Query 1 hits at rank 1 (RR 1.0), query 2 hits at rank 2 (RR 0.5).
        result = evaluate(
            _StubRetriever([10, 20, 30]),
            self.QUERIES,
            {1: {10}, 2: {20}},
            split="test",
            corpus_size=3,
        )
        assert result.mrr_at_10 == pytest.approx(0.75)
        assert result.p_at_1 == pytest.approx(0.5)
        assert result.n_queries == 2

    def test_records_mean_positives(self) -> None:
        result = evaluate(
            _StubRetriever([10]),
            self.QUERIES,
            {1: {10}, 2: {10, 20, 30}},
            split="test",
            corpus_size=3,
        )
        assert result.mean_positives_per_query == pytest.approx(2.0)

    def test_query_with_no_judgments_scores_zero_not_crash(self) -> None:
        result = evaluate(
            _StubRetriever([10]), self.QUERIES, {1: {10}}, split="test", corpus_size=3
        )
        assert result.mrr_at_10 == pytest.approx(0.5)

    def test_empty_split_raises(self) -> None:
        with pytest.raises(ValueError, match="no queries"):
            evaluate(_StubRetriever([1]), self.QUERIES.iloc[:0], {}, split="test")

    def test_per_query_detail_is_opt_in(self) -> None:
        without = evaluate(_StubRetriever([10]), self.QUERIES, {1: {10}}, split="test")
        assert without.per_query == []
        with_detail = evaluate(
            _StubRetriever([10]), self.QUERIES, {1: {10}}, split="test", keep_per_query=True
        )
        assert len(with_detail.per_query) == 2


class TestSelectSplit:
    QUERIES = pd.DataFrame(
        {"query_id": [1, 2, 3], "query": ["a", "b", "c"], "split": ["train", "dev", "test"]}
    )

    def test_selects_one_split(self) -> None:
        assert select_split(self.QUERIES, "test")["query_id"].tolist() == [3]

    def test_all_returns_everything(self) -> None:
        assert len(select_split(self.QUERIES, "all")) == 3

    def test_unknown_split_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown split"):
            select_split(self.QUERIES, "validation")


class TestEvalResult:
    def test_headline_metric_order(self) -> None:
        r = EvalResult(retriever="x", split="test", n_queries=1, corpus_size=1)
        assert list(r.headline()) == ["MRR@10", "P@1", "P@3", "P@5", "Recall@10", "nDCG@10"]

    def test_row_excludes_per_query_detail(self) -> None:
        r = EvalResult(retriever="x", split="test", n_queries=1, corpus_size=1)
        assert "per_query" not in r.to_row()
        assert "per_query" not in r.to_dict()


# --------------------------------------------------------------------------
# Regression guard on the real benchmark (PRD §19)
# --------------------------------------------------------------------------
needs_artifacts = pytest.mark.skipif(
    not (DATA_DIR / "kb.parquet").exists(),
    reason="run `python -m src.download && python -m src.prep` first",
)


@needs_artifacts
class TestBm25Regression:
    """Committed baseline metrics. CI fails if retrieval quality drops.

    Tolerances are wide enough to absorb library differences but tight enough
    to catch a real regression, e.g. a tokenizer change that drops digits.
    """

    def test_reproduces_published_baseline_on_all_queries(self) -> None:
        from src.evaluate import evaluate_bm25

        result = evaluate_bm25(split="all", field="question")
        # PRD 8.3 reports MRR@10 0.530 / P@1 0.421 on the pre-dedup corpus.
        assert result.mrr_at_10 == pytest.approx(0.524, abs=0.015)
        assert result.p_at_1 == pytest.approx(0.421, abs=0.015)

    def test_question_only_beats_question_plus_answer(self) -> None:
        """The PRD 8.3 finding that drives the default field choice."""
        from src.evaluate import evaluate_bm25

        q = evaluate_bm25(split="all", field="question")
        qa = evaluate_bm25(split="all", field="question_answer")
        assert q.mrr_at_10 > qa.mrr_at_10
        assert q.p_at_1 > qa.p_at_1

    def test_test_split_baseline_is_stable(self) -> None:
        """The number Phases 3 and 4 must beat."""
        from src.evaluate import evaluate_bm25

        result = evaluate_bm25(split="test", field="question")
        assert result.mrr_at_10 == pytest.approx(0.5127, abs=0.02)
        assert result.p_at_1 == pytest.approx(0.4033, abs=0.02)
