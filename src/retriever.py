"""Retrievers behind one interface: lexical BM25 and a dense bi-encoder.

Every retriever answers the same question -- given a query, which KB entries are
most relevant -- so `src.evaluate` and `src.api` never care which one they hold.
That shared interface is what lets the two be compared on identical machinery
(PRD §7.2).

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


class BiEncoderRetriever(Retriever):
    """Dense retrieval over precomputed KB embeddings (PRD §7.2).

    Where BM25 matches words, this matches meaning: query and KB entry are
    embedded separately and compared by cosine similarity. That is what lets it
    reach the 27% of test queries BM25 misses entirely, several of which share
    no vocabulary at all with their relevant entry.

    Both sides are L2-normalized, so scores are cosine similarities in [-1, 1]
    and are directly comparable across queries -- the property the abstention
    threshold τ depends on (PRD §7.5). BM25 scores have no such scale, which is
    why τ is tuned per retriever rather than shared.
    """

    def __init__(self, index, encoder=None, batch_size: int = 64) -> None:
        from src.index import load_encoder, prefixes_for

        self.index = index
        self.field = index.field
        self.model_name = index.model_name
        self.batch_size = batch_size
        self.name = f"biencoder[{index.model_name.split('/')[-1]}]"
        self._query_prefix, _ = prefixes_for(index.model_name)
        self._encoder = encoder or load_encoder(index.model_name)

    def __len__(self) -> int:
        return len(self.index)

    def _embed(self, queries: list[str]):
        from src.index import embed_texts

        return embed_texts(
            self._encoder, queries, prefix=self._query_prefix, batch_size=self.batch_size
        )

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        if not query.strip():
            return []
        return self.index.search(self._embed([query]), top_k=top_k)[0]

    def search_batch(self, queries: list[str], top_k: int = 10) -> list[list[tuple[int, float]]]:
        """Embed the whole batch in one pass.

        This is the point of a bi-encoder: 1,201 queries become a single encode
        call and one matrix product, instead of 1,201 round trips.
        """
        if not queries:
            return []
        blank = [i for i, q in enumerate(queries) if not q.strip()]
        if blank:
            non_blank = [q for q in queries if q.strip()]
            hits = self.index.search(self._embed(non_blank), top_k=top_k) if non_blank else []
            out, it = [], iter(hits)
            for q in queries:
                out.append([] if not q.strip() else next(it))
            return out
        return self.index.search(self._embed(queries), top_k=top_k)
