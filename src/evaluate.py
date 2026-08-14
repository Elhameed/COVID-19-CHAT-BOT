"""Retrieval metrics on the COUGH benchmark (PRD §8).

Single source of truth for every number this project reports. The notebook and
the API import the same module, so a figure quoted in the README is one command
away from being re-derived:

    python -m src.evaluate                      # BM25, test split
    python -m src.evaluate --split all          # reproduce PRD §8.3
    python -m src.evaluate --field question_answer

Metric choice (PRD §8.2) follows from the data. Queries average ~6.5 relevant
entries, so Recall@10 is capped by annotation density rather than retrieval
quality -- a perfect retriever showing 10 results cannot recall 25 positives.
MRR@10 and P@1 lead instead, because the chat UI shows exactly one answer.

Anti-leakage (PRD §8.5): `--split test` is the default and the only split whose
numbers are ever reported as results. BM25 trains on nothing, so its "all"
figure is safe to quote for comparison against the published benchmark, but the
acceptance chain in §8.4 is judged on test alone.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.retriever import FIELDS, BM25Retriever, Retriever, load_kb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

SPLITS = ("train", "dev", "test", "all")
DEFAULT_K = 10


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def reciprocal_rank(ranked_ids: list[int], relevant: set[int], k: int = DEFAULT_K) -> float:
    """1/rank of the first relevant hit within the top k, else 0."""
    for rank, kb_id in enumerate(ranked_ids[:k], start=1):
        if kb_id in relevant:
            return 1.0 / rank
    return 0.0


def precision_at_k(ranked_ids: list[int], relevant: set[int], k: int) -> float:
    """Fraction of the top k that are relevant.

    Divides by k, not by len(ranked_ids). A retriever returning 3 results for a
    P@5 query is being penalised correctly -- it failed to fill the slots.
    """
    if k <= 0:
        return 0.0
    return sum(1 for kb_id in ranked_ids[:k] if kb_id in relevant) / k


def recall_at_k(ranked_ids: list[int], relevant: set[int], k: int = DEFAULT_K) -> float:
    """Fraction of all relevant entries found in the top k."""
    if not relevant:
        return 0.0
    return sum(1 for kb_id in ranked_ids[:k] if kb_id in relevant) / len(relevant)


def ndcg_at_k(ranked_ids: list[int], relevant: set[int], k: int = DEFAULT_K) -> float:
    """Binary-gain nDCG@k.

    The ideal ranking puts min(len(relevant), k) hits at the top, so nDCG is not
    unfairly capped for queries with more positives than slots -- unlike recall,
    which is why both are reported.
    """
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, kb_id in enumerate(ranked_ids[:k], start=1)
        if kb_id in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------
@dataclass
class EvalResult:
    """Metrics for one retriever on one split."""

    retriever: str
    split: str
    n_queries: int
    corpus_size: int
    mrr_at_10: float = 0.0
    p_at_1: float = 0.0
    p_at_3: float = 0.0
    p_at_5: float = 0.0
    recall_at_10: float = 0.0
    ndcg_at_10: float = 0.0
    mean_positives_per_query: float = 0.0
    per_query: list[dict] = field(default_factory=list, repr=False)

    def headline(self) -> dict[str, float]:
        return {
            "MRR@10": self.mrr_at_10,
            "P@1": self.p_at_1,
            "P@3": self.p_at_3,
            "P@5": self.p_at_5,
            "Recall@10": self.recall_at_10,
            "nDCG@10": self.ndcg_at_10,
        }

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "per_query"}
        return d

    def to_row(self) -> dict:
        """One row for a comparison table."""
        return {
            "retriever": self.retriever,
            "split": self.split,
            "n": self.n_queries,
            **{k: round(v, 4) for k, v in self.headline().items()},
        }


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_eval_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, dict[int, set[int]]]:
    """Load queries (with split labels) and positive-only qrels."""
    queries_path = data_dir / "queries.parquet"
    qrels_path = data_dir / "qrels.parquet"
    for path in (queries_path, qrels_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `python -m src.prep` first.")

    queries = pd.read_parquet(queries_path)
    qrels = pd.read_parquet(qrels_path)

    positives = qrels.loc[qrels["relevant"]]
    relevant_by_query = {
        int(qid): set(group.astype(int)) for qid, group in positives.groupby("query_id")["kb_id"]
    }
    return queries, relevant_by_query


def select_split(queries: pd.DataFrame, split: str) -> pd.DataFrame:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
    if split == "all":
        return queries
    return queries.loc[queries["split"] == split]


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(
    retriever: Retriever,
    queries: pd.DataFrame,
    relevant_by_query: dict[int, set[int]],
    split: str = "test",
    k: int = DEFAULT_K,
    corpus_size: int = 0,
    keep_per_query: bool = False,
) -> EvalResult:
    """Score `retriever` over `queries`, averaging metrics across queries."""
    rows = queries.to_dict("records")
    if not rows:
        raise ValueError(f"no queries in split {split!r}")

    result = EvalResult(
        retriever=retriever.name, split=split, n_queries=len(rows), corpus_size=corpus_size
    )

    ranked_batch = retriever.search_batch([r["query"] for r in rows], top_k=k)

    totals = dict(rr=0.0, p1=0.0, p3=0.0, p5=0.0, rec=0.0, ndcg=0.0, pos=0)
    for row, hits in zip(rows, ranked_batch, strict=True):
        qid = int(row["query_id"])
        relevant = relevant_by_query.get(qid, set())
        ranked_ids = [kb_id for kb_id, _ in hits]

        rr = reciprocal_rank(ranked_ids, relevant, k)
        p1 = precision_at_k(ranked_ids, relevant, 1)
        p3 = precision_at_k(ranked_ids, relevant, 3)
        p5 = precision_at_k(ranked_ids, relevant, 5)
        rec = recall_at_k(ranked_ids, relevant, k)
        ndcg = ndcg_at_k(ranked_ids, relevant, k)

        totals["rr"] += rr
        totals["p1"] += p1
        totals["p3"] += p3
        totals["p5"] += p5
        totals["rec"] += rec
        totals["ndcg"] += ndcg
        totals["pos"] += len(relevant)

        if keep_per_query:
            result.per_query.append(
                {
                    "query_id": qid,
                    "query": row["query"],
                    "rr": rr,
                    "p_at_1": p1,
                    "n_relevant": len(relevant),
                    "top_score": hits[0][1] if hits else 0.0,
                    "top_kb_id": ranked_ids[0] if ranked_ids else None,
                }
            )

    n = len(rows)
    result.mrr_at_10 = totals["rr"] / n
    result.p_at_1 = totals["p1"] / n
    result.p_at_3 = totals["p3"] / n
    result.p_at_5 = totals["p5"] / n
    result.recall_at_10 = totals["rec"] / n
    result.ndcg_at_10 = totals["ndcg"] / n
    result.mean_positives_per_query = totals["pos"] / n
    return result


def evaluate_bm25(
    split: str = "test",
    field: str = "question",
    k: int = DEFAULT_K,
    data_dir: Path = DATA_DIR,
    keep_per_query: bool = False,
) -> EvalResult:
    """Convenience path used by the notebook and the CLI."""
    kb = load_kb(data_dir)
    queries, relevant = load_eval_data(data_dir)
    retriever = BM25Retriever(kb, field=field)
    return evaluate(
        retriever,
        select_split(queries, split),
        relevant,
        split=split,
        k=k,
        corpus_size=len(kb),
        keep_per_query=keep_per_query,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def format_table(results: list[EvalResult]) -> str:
    header = f"{'retriever':<24} {'split':<6} {'n':>5}  " + "  ".join(
        f"{m:>9}" for m in ("MRR@10", "P@1", "P@3", "P@5", "Recall@10", "nDCG@10")
    )
    lines = [header, "-" * len(header)]
    for r in results:
        vals = "  ".join(f"{v:>9.4f}" for v in r.headline().values())
        lines.append(f"{r.retriever:<24} {r.split:<6} {r.n_queries:>5}  {vals}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on the COUGH benchmark.")
    parser.add_argument(
        "--split",
        default="test",
        choices=SPLITS,
        help="query split (default: test -- the only one reported as a result)",
    )
    parser.add_argument(
        "--field",
        default="question",
        choices=FIELDS,
        help="KB text field to index (default: question)",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="cutoff (default: 10)")
    parser.add_argument(
        "--all-fields",
        action="store_true",
        help="evaluate every field, reproducing the PRD 8.3 comparison",
    )
    parser.add_argument("--json", type=Path, help="also write results to this JSON file")
    args = parser.parse_args(argv)

    kb = load_kb()
    queries, relevant = load_eval_data()
    subset = select_split(queries, args.split)

    fields = list(FIELDS) if args.all_fields else [args.field]
    results = []
    for f in fields:
        retriever = BM25Retriever(kb, field=f)
        results.append(
            evaluate(retriever, subset, relevant, split=args.split, k=args.k, corpus_size=len(kb))
        )

    print(
        f"corpus {len(kb):,} entries | split {args.split!r} | {len(subset):,} queries "
        f"| {results[0].mean_positives_per_query:.2f} positives per query\n"
    )
    print(format_table(results))

    if args.split == "test":
        print("\nTest split only -- this is the number Phases 3 and 4 must beat (PRD 8.4).")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps([r.to_dict() for r in results], indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
