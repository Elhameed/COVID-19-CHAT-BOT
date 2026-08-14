"""Retrievers behind one interface: BM25 now, dense encoders in Phases 3-4.

Every retriever answers the same question -- given a query, which KB entries are
most relevant -- so `src.evaluate` and `src.api` never care which one they hold.
That shared interface is what lets Phase 3 swap a bi-encoder in and compare it
against BM25 on identical machinery (PRD §7.2).

    from src.retriever import BM25Retriever, load_kb

    kb = load_kb()
    retriever = BM25Retriever(kb)
    hits = retriever.search("how does covid spread?", top_k=10)

Tokenization deliberately keeps digits (PRD hard constraint #5): "20 seconds"
and "6 feet" are content, and a query asking how long to wash your hands should
be able to match on the number.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Text fields a retriever can index. PRD §8.3 measured question-only as the
# stronger choice; `question_answer` exists so that finding stays reproducible
# rather than being taken on trust.
FIELDS = ("question", "question_answer")

# Keeps alphanumeric runs, so "COVID-19" -> ["covid", "19"] and "20" survives as
# a token. Splitting the hyphen is fine for BM25: both halves still match.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric tokens, digits included."""
    return _TOKEN_RE.findall(text.lower())


def load_kb(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the knowledge base built by `python -m src.prep`."""
    path = data_dir / "kb.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.download && python -m src.prep` first."
        )
    return pd.read_parquet(path)


def field_text(kb: pd.DataFrame, field: str) -> pd.Series:
    """The text a retriever indexes for each KB row."""
    if field == "question":
        return kb["question"]
    if field == "question_answer":
        return kb["question"] + " " + kb["answer"]
    raise ValueError(f"unknown field {field!r}, expected one of {FIELDS}")


class Retriever(ABC):
    """Common interface for every retrieval strategy.

    Implementations return `(kb_id, score)` pairs sorted by descending score.
    Scores are only meaningful within a retriever -- BM25 scores are unbounded
    sums, cosine similarities are bounded -- so the abstention threshold in
    PRD §7.5 is tuned per retriever, never shared across them.
    """

    name: str = "retriever"

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return the top_k `(kb_id, score)` pairs for one query."""

    def search_batch(self, queries: list[str], top_k: int = 10) -> list[list[tuple[int, float]]]:
        """Return results for many queries.

        The default loops; dense retrievers override this to embed the whole
        batch in one pass, which is where their speed comes from.
        """
        return [self.search(q, top_k) for q in queries]


class BM25Retriever(Retriever):
    """Okapi BM25 over a KB text field -- the lexical baseline (PRD §7.2).

    This is the number every semantic retriever has to beat. It trains on
    nothing, so there is no leakage to reason about: the same index serves
    every split.
    """

    def __init__(self, kb: pd.DataFrame, field: str = "question") -> None:
        if field not in FIELDS:
            raise ValueError(f"unknown field {field!r}, expected one of {FIELDS}")

        self.kb = kb.reset_index(drop=True)
        self.field = field
        self.name = f"bm25[{field}]"
        # Positional row -> KB id. BM25 works on positions; callers speak ids.
        self._ids = self.kb["id"].to_numpy()
        corpus = [tokenize(t) for t in field_text(self.kb, field)]
        self._bm25 = BM25Okapi(corpus)

    def __len__(self) -> int:
        return len(self.kb)

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        # argpartition finds the top_k without sorting all 7k scores, then only
        # those k are sorted. Meaningful across 1,201 queries.
        k = min(top_k, len(scores))
        top = scores.argpartition(-k)[-k:]
        top = top[scores[top].argsort()[::-1]]
        return [(int(self._ids[i]), float(scores[i])) for i in top]
