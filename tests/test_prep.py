"""Tests for the data pipeline (PRD §19).

Focus is on the properties whose failure would be silent: preprocessing that
destroys content, splits that aren't reproducible, and qrels that lose
judgments during dedup. Each of those would still produce plausible-looking
metrics, which is exactly what went wrong in the previous implementation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from src.prep import (
    DATA_DIR,
    SPLIT_RATIOS,
    PrepReport,
    build_kb,
    build_qrels,
    classify_trust,
    make_splits,
    normalize_text,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# Text normalization — PRD hard constraint #5
# --------------------------------------------------------------------------
class TestNormalizeText:
    @pytest.mark.parametrize(
        "text",
        [
            "Wash your hands for 20 seconds",
            "Stay 6 feet apart",
            "Isolate for 14 days",
            "COVID-19 spreads person-to-person",
            "A 20 by 20 inches (51 cm) square",
            "Temperature above 100.4°F",
        ],
    )
    def test_preserves_numbers_and_units(self, text: str) -> None:
        """The old pipeline ran re.sub(r"[^a-zA-Z\\s]", "", text), turning
        "20 seconds" into "seconds" and "COVID-19" into "COVID". Health content
        cannot survive that.

        Numbers are compared as whole quantities ("100.4", not "1004") so the
        decimal point counts as content too.
        """
        result = normalize_text(text)
        quantities = re.findall(r"\d+(?:\.\d+)?", text)
        assert quantities, "fixture should contain a number"
        for quantity in quantities:
            assert quantity in result, f"quantity {quantity!r} lost from {text!r}"

    def test_preserves_case_and_hyphenation(self) -> None:
        assert normalize_text("COVID-19") == "COVID-19"

    def test_collapses_non_breaking_space(self) -> None:
        # COUGH inherits \xa0 from scraped HTML, e.g. "51\xa0cm".
        assert normalize_text("51\xa0cm") == "51 cm"

    def test_strips_residual_html(self) -> None:
        assert normalize_text("<p>Wear a <b>mask</b></p>") == "Wear a mask"

    def test_collapses_whitespace(self) -> None:
        assert normalize_text("  a \n\n b\t c  ") == "a b c"

    def test_handles_missing_values(self) -> None:
        assert normalize_text(None) == ""
        assert normalize_text(float("nan")) == ""


# --------------------------------------------------------------------------
# Trust tiering
# --------------------------------------------------------------------------
class TestClassifyTrust:
    def test_official_and_community(self) -> None:
        assert classify_trust("WHO") == "official"
        assert classify_trust("CDC_FAQ") == "official"
        assert classify_trust("WikiHow") == "community"

    def test_unknown_source_raises_rather_than_guessing(self) -> None:
        """Defaulting an unknown source to `official` would present community
        content to a user as authoritative."""
        with pytest.raises(ValueError, match="unknown source"):
            classify_trust("Some New Blog")


# --------------------------------------------------------------------------
# Splits — PRD §8.5 anti-leakage
# --------------------------------------------------------------------------
class TestMakeSplits:
    IDS: ClassVar[list[int]] = list(range(1000))

    def test_deterministic_for_a_given_seed(self) -> None:
        assert make_splits(self.IDS, seed=42) == make_splits(self.IDS, seed=42)

    def test_seed_actually_changes_the_partition(self) -> None:
        assert make_splits(self.IDS, seed=42) != make_splits(self.IDS, seed=7)

    def test_independent_of_input_order(self) -> None:
        """Sorting before shuffling means the split depends on the id set and
        the seed only — never on the order rows arrived in."""
        shuffled = list(reversed(self.IDS))
        assert make_splits(shuffled, seed=42) == make_splits(self.IDS, seed=42)

    def test_partitions_exactly_with_no_overlap(self) -> None:
        splits = make_splits(self.IDS, seed=42)
        train, dev, test = set(splits["train"]), set(splits["dev"]), set(splits["test"])
        assert train | dev | test == set(self.IDS)
        assert not (train & dev) and not (train & test) and not (dev & test)

    def test_respects_ratios(self) -> None:
        splits = make_splits(self.IDS, seed=42)
        assert len(splits["train"]) == 700
        assert len(splits["dev"]) == 150
        assert len(splits["test"]) == 150

    def test_rejects_ratios_that_do_not_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match=r"sum to 1\.0"):
            make_splits(self.IDS, ratios={"train": 0.5, "dev": 0.2, "test": 0.2})


# --------------------------------------------------------------------------
# KB construction and the dedup/qrels contract
# --------------------------------------------------------------------------
def _faq_fixture() -> pd.DataFrame:
    """Five rows: two duplicates differing only in case/spacing, one unusable."""
    return pd.DataFrame(
        {
            "index": [0, 1, 2, 3, 4],
            "question": [
                "How does COVID-19 spread?",
                "how does covid-19   spread?",  # dup of 0, different case/spacing
                "Should I wear a mask?",
                "",  # unusable: no question
                "How long to isolate?",
            ],
            "answer": [
                "Mainly person to person.",
                "Mainly person to person.",
                "Yes, in public settings.",
                "orphaned answer",
                "For 14 days.",
            ],
            "source": ["WikiHow", "WHO", "CDC_FAQ", "WHO", "WHO"],
            "url": ["u0", "u1", "u2", "u3", "u4"],
            "language": ["en"] * 5,
        }
    )


class TestBuildKb:
    def test_collapses_near_duplicates_and_drops_unusable(self) -> None:
        report = PrepReport()
        kb, _ = build_kb(_faq_fixture(), report)

        assert report.dropped_missing_question == [3]
        assert report.duplicate_groups == 1
        assert kb["id"].tolist() == [1, 2, 4]  # 0 merged into 1, 3 dropped

    def test_duplicate_group_keeps_the_more_authoritative_source(self) -> None:
        """Row 0 is WikiHow, row 1 is the identical WHO entry. The official one
        must survive, even though it has the higher id."""
        report = PrepReport()
        kb, remap = build_kb(_faq_fixture(), report)

        assert remap[0] == 1
        assert kb.loc[kb["id"] == 1, "source"].item() == "WHO"
        assert kb.loc[kb["id"] == 1, "trust"].item() == "official"

    def test_remap_covers_survivors_and_merged_rows_but_not_dropped(self) -> None:
        report = PrepReport()
        _, remap = build_kb(_faq_fixture(), report)

        assert remap[0] == 1 and remap[1] == 1  # merged + survivor
        assert remap[2] == 2 and remap[4] == 4  # identity
        assert 3 not in remap  # unusable rows are unrecoverable

    def test_rejects_non_english_corpus(self) -> None:
        faq = _faq_fixture()
        faq.loc[0, "language"] = "es"
        with pytest.raises(ValueError, match="all English"):
            build_kb(faq, PrepReport())


class TestBuildQrels:
    """The critical property: dedup must not silently delete positives."""

    def test_judgments_are_remapped_onto_the_surviving_duplicate(self) -> None:
        report = PrepReport()
        _, remap = build_kb(_faq_fixture(), report)

        qrels = pd.DataFrame(
            {
                "Query_index": [100, 101],
                "FAQ_index": [0, 0],  # both point at the merged-away row
                "Label (mean score > 3 as positive)": ["positive", "negative"],
            }
        )
        out = build_qrels(qrels, remap, {100, 101}, report)

        assert out["kb_id"].tolist() == [1, 1], "judgments should follow the survivor"
        assert report.qrels_remapped == 2
        assert report.qrels_remapped_positive == 1
        assert report.positives_lost == 0 or report.qrels_dropped_dangling_faq_positive == 0

    def test_orphan_query_ids_are_dropped(self) -> None:
        report = PrepReport()
        _, remap = build_kb(_faq_fixture(), report)
        qrels = pd.DataFrame(
            {
                "Query_index": [100, 999],
                "FAQ_index": [2, 2],
                "Label (mean score > 3 as positive)": ["positive", "positive"],
            }
        )
        out = build_qrels(qrels, remap, {100}, report)

        assert out["query_id"].tolist() == [100]
        assert report.qrels_dropped_orphan_query == 1

    def test_duplicate_judgments_keep_the_positive_verdict(self) -> None:
        """After remapping, one query may judge the same surviving entry twice.
        Collapsing must never downgrade a positive to a negative."""
        report = PrepReport()
        _, remap = build_kb(_faq_fixture(), report)
        qrels = pd.DataFrame(
            {
                "Query_index": [100, 100],
                "FAQ_index": [0, 1],  # both resolve to id 1
                "Label (mean score > 3 as positive)": ["negative", "positive"],
            }
        )
        out = build_qrels(qrels, remap, {100}, report)

        assert len(out) == 1
        assert bool(out["relevant"].iloc[0]) is True


# --------------------------------------------------------------------------
# Built artifacts — skipped until `python -m src.prep` has run
# --------------------------------------------------------------------------
needs_artifacts = pytest.mark.skipif(
    not (DATA_DIR / "kb.parquet").exists(),
    reason="run `python -m src.download && python -m src.prep` first",
)


@pytest.fixture(scope="module")
def artifacts() -> dict:
    """The artifacts written by `python -m src.prep`, loaded once."""
    return {
        "kb": pd.read_parquet(DATA_DIR / "kb.parquet"),
        "queries": pd.read_parquet(DATA_DIR / "queries.parquet"),
        "qrels": pd.read_parquet(DATA_DIR / "qrels.parquet"),
        "report": json.loads((DATA_DIR / "prep_report.json").read_text(encoding="utf-8")),
    }


@needs_artifacts
class TestBuiltArtifacts:
    def test_kb_matches_prd_schema(self, artifacts: dict) -> None:
        assert list(artifacts["kb"].columns) == [
            "id",
            "question",
            "answer",
            "source",
            "url",
            "trust",
        ]

    def test_kb_has_no_empty_or_duplicate_entries(self, artifacts: dict) -> None:
        kb = artifacts["kb"]
        assert kb["id"].is_unique
        assert (kb["question"].str.len() > 0).all()
        assert (kb["answer"].str.len() > 0).all()
        assert set(kb["trust"]) <= {"official", "community"}

    def test_every_qrel_points_at_a_real_kb_entry(self, artifacts: dict) -> None:
        """A dangling qrel is a silently lost positive."""
        kb_ids = set(artifacts["kb"]["id"])
        assert set(artifacts["qrels"]["kb_id"]) <= kb_ids

    def test_every_qrel_points_at_a_real_query(self, artifacts: dict) -> None:
        query_ids = set(artifacts["queries"]["query_id"])
        assert set(artifacts["qrels"]["query_id"]) <= query_ids

    def test_positive_accounting_reconciles(self, artifacts: dict) -> None:
        r = artifacts["report"]
        assert (
            r["positive_rows"] + r["positives_lost"] + r["positives_merged"]
            == r["raw_positive_rows"]
        )

    def test_every_query_has_at_least_one_positive(self, artifacts: dict) -> None:
        qrels = artifacts["qrels"]
        with_pos = set(qrels.loc[qrels["relevant"], "query_id"])
        assert set(artifacts["queries"]["query_id"]) == with_pos

    def test_splits_partition_the_queries_without_overlap(self, artifacts: dict) -> None:
        queries = artifacts["queries"]
        splits = {
            name: set(json.loads((DATA_DIR / "splits" / f"{name}.json").read_text())["query_ids"])
            for name in ("train", "dev", "test")
        }
        assert splits["train"] | splits["dev"] | splits["test"] == set(queries["query_id"])
        assert not (splits["train"] & splits["dev"])
        assert not (splits["train"] & splits["test"])
        assert not (splits["dev"] & splits["test"])

    def test_split_column_agrees_with_split_files(self, artifacts: dict) -> None:
        queries = artifacts["queries"]
        for name in ("train", "dev", "test"):
            ids = set(json.loads((DATA_DIR / "splits" / f"{name}.json").read_text())["query_ids"])
            assert set(queries.loc[queries["split"] == name, "query_id"]) == ids

    def test_split_proportions_are_roughly_as_configured(self, artifacts: dict) -> None:
        sizes = artifacts["report"]["split_sizes"]
        total = sum(sizes.values())
        for name, expected in SPLIT_RATIOS.items():
            assert abs(sizes[name] / total - expected) < 0.01

    def test_corpus_is_not_split(self, artifacts: dict) -> None:
        """All KB entries stay retrievable at every stage (PRD §7.4)."""
        assert "split" not in artifacts["kb"].columns

    def test_numbers_survived_the_real_pipeline(self, artifacts: dict) -> None:
        """Spot-check hard constraint #5 against the built KB, not a fixture."""
        kb = artifacts["kb"]
        assert kb["answer"].str.contains(r"\d", regex=True).any()
        assert kb["question"].str.contains("COVID-19", regex=False).any()
