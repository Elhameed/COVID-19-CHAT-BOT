"""Tests for the fine-tuning data pipeline (PRD §19).

Training itself is not exercised here -- it needs a GPU and minutes. What is
tested is everything that decides whether the training run is *valid*: which
queries it sees, which pairs it builds, and that test never leaks in. Those are
the failures that would produce a real-looking but meaningless result.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluate import DATA_DIR
from src.train import BASE_MODEL, TrainReport, _assert_no_leakage, build_training_pairs


def _kb() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [10, 20, 30],
            "question": ["how does it spread?", "should I mask?", "how long to isolate?"],
            "answer": ["a1", "a2", "a3"],
            "source": ["WHO", "WHO", "CDC_FAQ"],
            "url": ["u1", "u2", "u3"],
            "trust": ["official", "official", "official"],
        }
    )


def _queries() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": [1, 2, 3],
            "query": ["how is covid transmitted?", "do masks help?", "isolation period?"],
            "split": ["train", "dev", "test"],
        }
    )


def _qrels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": [1, 1, 1, 2, 3],
            "kb_id": [10, 20, 30, 20, 30],
            "relevant": [True, True, False, True, True],
        }
    )


class TestBuildTrainingPairs:
    def test_one_row_per_positive(self) -> None:
        pairs = build_training_pairs(_kb(), _queries(), _qrels(), split="train")
        # Query 1 has two positives (10, 20); kb 30 is judged negative.
        assert len(pairs) == 2
        assert set(pairs["positive"]) == {"how does it spread?", "should I mask?"}

    def test_negatives_are_excluded(self) -> None:
        pairs = build_training_pairs(_kb(), _queries(), _qrels(), split="train")
        assert "how long to isolate?" not in set(pairs["positive"])

    def test_anchor_is_the_user_query(self) -> None:
        pairs = build_training_pairs(_kb(), _queries(), _qrels(), split="train")
        assert set(pairs["anchor"]) == {"how is covid transmitted?"}

    def test_positive_is_the_kb_question_not_the_answer(self) -> None:
        """Training must use the same field the index is built on, or training
        and inference see different text."""
        pairs = build_training_pairs(_kb(), _queries(), _qrels(), split="train")
        assert not set(pairs["positive"]) & {"a1", "a2", "a3"}

    def test_only_the_requested_split_is_used(self) -> None:
        """The guard that matters most: a training set built from dev or test
        queries would invalidate every downstream number."""
        pairs = build_training_pairs(_kb(), _queries(), _qrels(), split="train")
        assert set(pairs["query_id"]) == {1}

        dev_pairs = build_training_pairs(_kb(), _queries(), _qrels(), split="dev")
        assert set(dev_pairs["query_id"]) == {2}

    def test_empty_split_raises(self) -> None:
        queries = _queries()
        queries["split"] = "train"
        with pytest.raises(ValueError, match="no queries in split"):
            build_training_pairs(_kb(), queries, _qrels(), split="dev")

    def test_qrels_pointing_outside_the_kb_are_dropped(self) -> None:
        qrels = pd.concat(
            [_qrels(), pd.DataFrame({"query_id": [1], "kb_id": [999], "relevant": [True]})]
        )
        pairs = build_training_pairs(_kb(), _queries(), qrels, split="train")
        assert len(pairs) == 2  # the dangling id contributes nothing
        assert pairs["positive"].notna().all()

    def test_columns_match_what_the_trainer_expects(self) -> None:
        pairs = build_training_pairs(_kb(), _queries(), _qrels(), split="train")
        assert list(pairs.columns) == ["query_id", "anchor", "positive"]


class TestLeakageGuard:
    def test_disjoint_splits_pass(self) -> None:
        _assert_no_leakage(_queries())

    @pytest.mark.parametrize(("a", "b"), [("train", "dev"), ("train", "test"), ("dev", "test")])
    def test_any_overlap_raises(self, a: str, b: str) -> None:
        queries = _queries()
        # Duplicate one query id into the other split.
        dup = queries.loc[queries["split"] == a].copy()
        dup["split"] = b
        with pytest.raises(AssertionError, match="overlap"):
            _assert_no_leakage(pd.concat([queries, dup], ignore_index=True))


class TestTrainReport:
    def test_serializes_for_the_artifact(self) -> None:
        report = TrainReport(train_pairs=10, train_queries=3)
        d = report.to_dict()
        assert d["train_pairs"] == 10
        assert d["base_model"] == BASE_MODEL

    def test_defaults_record_the_settled_base_model(self) -> None:
        """PRD §22 settled on MiniLM; the default should not drift silently."""
        assert "MiniLM" in BASE_MODEL


@pytest.mark.skipif(
    not (DATA_DIR / "queries.parquet").exists(),
    reason="run `python -m src.download && python -m src.prep` first",
)
class TestAgainstRealSplits:
    def test_real_splits_are_disjoint(self) -> None:
        _assert_no_leakage(pd.read_parquet(DATA_DIR / "queries.parquet"))

    def test_training_pairs_come_only_from_train_queries(self) -> None:
        from src.retriever import load_kb

        queries = pd.read_parquet(DATA_DIR / "queries.parquet")
        qrels = pd.read_parquet(DATA_DIR / "qrels.parquet")
        pairs = build_training_pairs(load_kb(), queries, qrels, split="train")

        train_ids = set(queries.loc[queries["split"] == "train", "query_id"])
        held_out = set(queries.loc[queries["split"].isin(["dev", "test"]), "query_id"])
        assert set(pairs["query_id"]) <= train_ids
        assert not set(pairs["query_id"]) & held_out
