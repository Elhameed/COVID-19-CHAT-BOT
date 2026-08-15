"""Tests for embedding index construction, caching and search (PRD §19).

The cache-staleness guard gets the most attention here. Embeddings that no
longer match the KB would misalign every id while still producing plausible
metrics -- the same class of silent failure the rebuild exists to eliminate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluate import DATA_DIR
from src.index import (
    EmbeddingIndex,
    cache_path,
    kb_fingerprint,
    load_or_build_index,
    prefixes_for,
)
from src.retriever import field_text


def _kb_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [10, 20, 30],
            "question": ["how does it spread?", "should I wear a mask?", "what are symptoms?"],
            "answer": ["person to person", "yes indoors", "fever and cough"],
            "source": ["WHO", "CDC_FAQ", "WHO"],
            "url": ["u1", "u2", "u3"],
            "trust": ["official", "official", "official"],
        }
    )


def _index_fixture(dim: int = 4) -> EmbeddingIndex:
    """Three orthogonal-ish unit vectors, so nearest neighbours are obvious."""
    embeddings = np.eye(3, dim, dtype=np.float32)
    return EmbeddingIndex(
        model_name="test-model",
        field="question",
        ids=np.array([10, 20, 30]),
        embeddings=embeddings,
        fingerprint="deadbeef",
    )


class TestPrefixes:
    def test_bge_prefixes_the_query_only(self) -> None:
        """bge was trained with an instruction on the query side. Omitting it
        is a silent quality loss, not an error, so it is worth asserting."""
        q, d = prefixes_for("BAAI/bge-base-en-v1.5")
        assert q.startswith("Represent this sentence")
        assert d == ""

    def test_e5_prefixes_both_sides(self) -> None:
        assert prefixes_for("intfloat/e5-base-v2") == ("query: ", "passage: ")

    def test_minilm_needs_no_prefix(self) -> None:
        assert prefixes_for("sentence-transformers/all-MiniLM-L6-v2") == ("", "")

    def test_matching_is_case_insensitive(self) -> None:
        assert prefixes_for("BAAI/BGE-Large")[0].startswith("Represent")


class TestFingerprint:
    def test_same_kb_same_fingerprint(self) -> None:
        kb = _kb_fixture()
        a = kb_fingerprint(kb, field_text(kb, "question"))
        b = kb_fingerprint(kb, field_text(kb, "question"))
        assert a == b

    def test_changed_text_changes_fingerprint(self) -> None:
        kb = _kb_fixture()
        before = kb_fingerprint(kb, field_text(kb, "question"))
        kb.loc[0, "question"] = "something else"
        assert kb_fingerprint(kb, field_text(kb, "question")) != before

    def test_changed_ids_change_fingerprint(self) -> None:
        kb = _kb_fixture()
        before = kb_fingerprint(kb, field_text(kb, "question"))
        kb.loc[0, "id"] = 999
        assert kb_fingerprint(kb, field_text(kb, "question")) != before

    def test_different_field_changes_fingerprint(self) -> None:
        kb = _kb_fixture()
        assert kb_fingerprint(kb, field_text(kb, "question")) != kb_fingerprint(
            kb, field_text(kb, "question_answer")
        )


class TestEmbeddingIndexSearch:
    def test_finds_the_nearest_vector(self) -> None:
        index = _index_fixture()
        query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        assert index.search(query, top_k=1)[0][0] == (10, pytest.approx(1.0))

    def test_returns_kb_ids(self) -> None:
        index = _index_fixture()
        query = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        assert index.search(query, top_k=3)[0][0][0] == 20

    def test_results_sorted_descending(self) -> None:
        index = _index_fixture()
        query = np.array([[0.9, 0.4, 0.1, 0.0]], dtype=np.float32)
        scores = [s for _, s in index.search(query, top_k=3)[0]]
        assert scores == sorted(scores, reverse=True)

    def test_accepts_a_one_dimensional_query(self) -> None:
        index = _index_fixture()
        assert len(index.search(np.array([1.0, 0, 0, 0], dtype=np.float32), top_k=2)[0]) == 2

    def test_batch_search_returns_one_list_per_query(self) -> None:
        index = _index_fixture()
        queries = np.eye(2, 4, dtype=np.float32)
        results = index.search(queries, top_k=2)
        assert len(results) == 2
        assert results[0][0][0] == 10
        assert results[1][0][0] == 20

    def test_top_k_larger_than_corpus_is_safe(self) -> None:
        index = _index_fixture()
        assert len(index.search(np.eye(1, 4, dtype=np.float32), top_k=99)[0]) == 3

    def test_dim_reports_vector_width(self) -> None:
        assert _index_fixture(dim=4).dim == 4


class TestCaching:
    def test_round_trips_through_disk(self, tmp_path: Path) -> None:
        index = _index_fixture()
        path = index.save(tmp_path / "idx.npz")
        loaded = EmbeddingIndex.load(path)

        assert loaded.model_name == index.model_name
        assert loaded.field == index.field
        assert loaded.fingerprint == index.fingerprint
        np.testing.assert_array_equal(loaded.ids, index.ids)
        np.testing.assert_allclose(loaded.embeddings, index.embeddings)

    def test_load_rejects_a_stale_cache(self, tmp_path: Path) -> None:
        """The guard that matters: a cache built for a different KB must not
        load, because misaligned ids produce wrong-but-plausible results."""
        path = _index_fixture().save(tmp_path / "idx.npz")
        with pytest.raises(ValueError, match="different knowledge base"):
            EmbeddingIndex.load(path, expected_fingerprint="not-the-same")

    def test_load_accepts_a_matching_fingerprint(self, tmp_path: Path) -> None:
        path = _index_fixture().save(tmp_path / "idx.npz")
        assert EmbeddingIndex.load(path, expected_fingerprint="deadbeef").fingerprint == "deadbeef"

    def test_cache_path_separates_models_and_fields(self) -> None:
        a = cache_path("BAAI/bge-base-en-v1.5", "question")
        b = cache_path("BAAI/bge-base-en-v1.5", "question_answer")
        c = cache_path("sentence-transformers/all-MiniLM-L6-v2", "question")
        assert len({a, b, c}) == 3
        assert "/" not in a.name  # model slug is filesystem-safe

    def test_stale_cache_is_rebuilt_rather_than_raising(self, tmp_path: Path) -> None:
        """load_or_build_index recovers from staleness; only direct load() raises."""
        kb = _kb_fixture()

        class _StubEncoder:
            def encode(self, texts, **kwargs):
                # Deterministic unit vectors, one per text.
                return np.eye(len(texts), 4, dtype=np.float32)

        first = load_or_build_index(
            kb,
            model_name="stub",
            field="question",
            embeddings_dir=tmp_path,
            encoder=_StubEncoder(),
        )
        assert len(first) == 3

        kb.loc[0, "question"] = "changed text invalidates the cache"
        second = load_or_build_index(
            kb,
            model_name="stub",
            field="question",
            embeddings_dir=tmp_path,
            encoder=_StubEncoder(),
        )
        assert second.fingerprint != first.fingerprint


@pytest.mark.skipif(
    not (DATA_DIR / "kb.parquet").exists(),
    reason="run `python -m src.download && python -m src.prep` first",
)
class TestAgainstRealKb:
    def test_cached_index_matches_the_corpus(self) -> None:
        from src.retriever import load_kb

        kb = load_kb()
        index = load_or_build_index(kb)
        assert len(index) == len(kb)
        np.testing.assert_array_equal(index.ids, kb["id"].to_numpy())

    def test_embeddings_are_unit_normalized(self) -> None:
        """Cosine similarity via inner product only holds for unit vectors, and
        the abstention threshold assumes scores live in [-1, 1]."""
        from src.retriever import load_kb

        index = load_or_build_index(load_kb())
        norms = np.linalg.norm(index.embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)
