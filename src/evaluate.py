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

# Fixed so a reported confidence interval is reproducible, not resampled anew.
SEED_BOOTSTRAP = 42


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


def evaluate_biencoder(
    split: str = "test",
    model_name: str | None = None,
    field: str = "question",
    k: int = DEFAULT_K,
    data_dir: Path = DATA_DIR,
    keep_per_query: bool = False,
) -> EvalResult:
    """Same, for a dense retriever. Reuses the cached embeddings if valid."""
    from src.index import DEFAULT_MODEL, load_or_build_index
    from src.retriever import BiEncoderRetriever

    kb = load_kb(data_dir)
    queries, relevant = load_eval_data(data_dir)
    index = load_or_build_index(kb, model_name=model_name or DEFAULT_MODEL, field=field)
    retriever = BiEncoderRetriever(index)
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
# Abstention threshold (PRD §7.5)
# --------------------------------------------------------------------------
# COUGH gives every query at least one positive, so the benchmark contains no
# unanswerable questions and cannot, on its own, validate the behaviour PRD §4
# actually asks for ("What's the weather?" -> abstain). Tuning τ for F1 on it
# degenerates to "never abstain", because recall is maximised by answering
# everything.
#
# This hand-written probe covers the gap. It is a smoke test, not a benchmark:
# 12 queries, written by us, with no relevance judgments. It answers one narrow
# question -- does an obviously off-topic query score below an in-scope one --
# and nothing more. Treat any number derived from it as indicative.
OFF_TOPIC_PROBE: tuple[str, ...] = (
    "What's the weather like today?",
    "How do I bake sourdough bread?",
    "Who won the football match last night?",
    "What is the capital of France?",
    "How do I change a car tyre?",
    "Recommend a good science fiction novel",
    "What is the derivative of x squared?",
    "How do I reset my email password?",
    "Best restaurants near me",
    "How tall is Mount Everest?",
    "Write me a poem about the sea",
    "What time does the train leave?",
)


@dataclass
class AbstentionSummary:
    """How a threshold behaves on in-scope vs off-topic queries."""

    tau: float
    in_scope_answered: float
    off_topic_answered: float
    n_in_scope: int
    n_off_topic: int
    leaked: list[tuple[str, float]] = field(default_factory=list, repr=False)


def summarize_abstention(
    tau: float, in_scope_scores: list[float], off_topic_scores: list[tuple[str, float]]
) -> AbstentionSummary:
    """Share of each population that would be answered at this τ."""
    n_in, n_off = len(in_scope_scores), len(off_topic_scores)
    leaked = sorted(((q, s) for q, s in off_topic_scores if s >= tau), key=lambda x: -x[1])
    return AbstentionSummary(
        tau=tau,
        in_scope_answered=sum(1 for s in in_scope_scores if s >= tau) / n_in if n_in else 0.0,
        off_topic_answered=len(leaked) / n_off if n_off else 0.0,
        n_in_scope=n_in,
        n_off_topic=n_off,
        leaked=leaked,
    )


@dataclass
class ThresholdChoice:
    """A tuned abstention threshold and the trade-off it represents."""

    tau: float
    objective: str
    answered: int
    n_queries: int
    precision: float  # of answers given, how many had a relevant top-1
    recall: float  # of answerable queries, how many we answered
    f1: float
    curve: list[dict] = field(default_factory=list, repr=False)

    @property
    def answer_rate(self) -> float:
        return self.answered / self.n_queries if self.n_queries else 0.0


def tune_threshold(
    top_scores: list[float],
    top1_correct: list[bool],
    objective: str = "f1",
    off_topic_scores: list[float] | None = None,
) -> ThresholdChoice:
    """Choose τ on the dev split (PRD §7.5).

    Frames abstention as a decision problem. Answering when the top-1 is wrong
    is the failure mode PRD §21 calls out -- a confident wrong answer in a
    health context -- so `objective="precision"` targets a conservative
    operating point; `"f1"` balances that against staying useful.

    Returns the full curve as well as the choice, because the shape of the
    trade-off matters more than any single number.
    """
    if len(top_scores) != len(top1_correct):
        raise ValueError("scores and labels must be the same length")
    if not top_scores:
        raise ValueError("no dev queries to tune on")

    n = len(top_scores)
    answerable = sum(top1_correct)
    candidates = sorted({round(s, 4) for s in top_scores})

    curve = []
    for tau in candidates:
        answered = [c for s, c in zip(top_scores, top1_correct, strict=True) if s >= tau]
        n_answered = len(answered)
        n_right = sum(answered)
        precision = n_right / n_answered if n_answered else 1.0
        recall = n_right / answerable if answerable else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        curve.append(
            {
                "tau": tau,
                "answered": n_answered,
                "answer_rate": n_answered / n,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    if objective == "f1":
        # Kept for the record rather than for use: with no unanswerable queries
        # in the benchmark, recall is maximised by answering everything, so F1
        # reliably selects the bottom of the score range. The notebook shows
        # this explicitly -- it is why `off_topic` is the real objective.
        best = max(curve, key=lambda r: (r["f1"], r["tau"]))
    elif objective == "off_topic":
        # Lowest τ that rejects the entire off-topic probe, which keeps as much
        # in-scope coverage as possible while still refusing nonsense. Requires
        # off_topic_scores.
        if off_topic_scores is None:
            raise ValueError("objective='off_topic' needs off_topic_scores")
        clean = [r for r in curve if all(s < r["tau"] for s in off_topic_scores)]
        if not clean:
            raise ValueError(
                "no threshold rejects every off-topic query; the score "
                "distributions overlap completely"
            )
        best = min(clean, key=lambda r: r["tau"])
    elif objective == "precision":
        # Highest precision while still answering at least half the queries;
        # a threshold that abstains on everything is trivially precise.
        viable = [r for r in curve if r["answer_rate"] >= 0.5] or curve
        best = max(viable, key=lambda r: (r["precision"], r["tau"]))
    else:
        raise ValueError(
            f"unknown objective {objective!r}, expected 'f1', 'off_topic' or 'precision'"
        )

    return ThresholdChoice(
        tau=best["tau"],
        objective=objective,
        answered=best["answered"],
        n_queries=n,
        precision=best["precision"],
        recall=best["recall"],
        f1=best["f1"],
        curve=curve,
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


# --------------------------------------------------------------------------
# Comparing two retrievers (PRD §8.4)
# --------------------------------------------------------------------------
@dataclass
class PairedComparison:
    """Whether one retriever genuinely beats another, with error bars.

    A raw delta between two systems on 181 queries is not evidence on its own.
    On 181 test queries the resolution limit is roughly +/-0.03 MRR@10, so a
    delta below that is not evidence of anything. Reporting one without an
    interval would be exactly the sort of claim this project was rebuilt to
    avoid.
    """

    metric: str
    baseline_name: str
    candidate_name: str
    n_queries: int
    baseline_mean: float
    candidate_mean: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    n_better: int
    n_worse: int
    n_tied: int

    @property
    def significant(self) -> bool:
        """True when the 95% interval excludes zero."""
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def to_dict(self) -> dict:
        return {**self.__dict__, "significant": self.significant}


def paired_bootstrap(
    baseline: EvalResult,
    candidate: EvalResult,
    metric: str = "rr",
    n_resamples: int = 10_000,
    seed: int = SEED_BOOTSTRAP,
) -> PairedComparison:
    """Paired bootstrap over per-query scores.

    Paired because both systems answer the same queries: differencing per query
    removes query difficulty, which is the dominant variance source here and
    would otherwise swamp a small effect.

    Both results must come from `evaluate(..., keep_per_query=True)` on the same
    split.
    """
    import numpy as np

    if not baseline.per_query or not candidate.per_query:
        raise ValueError("both results need keep_per_query=True")

    base_by_id = {q["query_id"]: q for q in baseline.per_query}
    cand_by_id = {q["query_id"]: q for q in candidate.per_query}
    shared = sorted(set(base_by_id) & set(cand_by_id))
    if len(shared) != len(base_by_id) or len(shared) != len(cand_by_id):
        raise ValueError("results cover different queries; compare on one split")

    a = np.array([base_by_id[i][metric] for i in shared], dtype=float)
    b = np.array([cand_by_id[i][metric] for i in shared], dtype=float)
    diff = b - a

    rng = np.random.default_rng(seed)
    means = rng.choice(diff, size=(n_resamples, len(diff)), replace=True).mean(axis=1)
    ci_low, ci_high = (float(x) for x in np.percentile(means, [2.5, 97.5]))
    p_value = 2 * min(float((means <= 0).mean()), float((means >= 0).mean()))

    return PairedComparison(
        metric=metric,
        baseline_name=baseline.retriever,
        candidate_name=candidate.retriever,
        n_queries=len(shared),
        baseline_mean=float(a.mean()),
        candidate_mean=float(b.mean()),
        delta=float(diff.mean()),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=min(p_value, 1.0),
        n_better=int((b > a).sum()),
        n_worse=int((b < a).sum()),
        n_tied=int((b == a).sum()),
    )


def tune_and_write_config(
    model_name: str, field: str = "question", data_dir: Path = DATA_DIR
) -> dict:
    """Tune τ on dev, probe it against off-topic queries, and persist the config.

    The result is what `src/api.py` loads at startup: encoder, field, and the
    threshold below which the bot abstains. Tuned on **dev** only -- test is
    never used to pick a hyperparameter (PRD §8.5).
    """
    from src.index import ARTIFACTS_DIR, load_or_build_index
    from src.retriever import BiEncoderRetriever

    kb = load_kb(data_dir)
    queries, relevant = load_eval_data(data_dir)
    index = load_or_build_index(kb, model_name=model_name, field=field)
    retriever = BiEncoderRetriever(index)

    dev = evaluate(
        retriever,
        select_split(queries, "dev"),
        relevant,
        split="dev",
        corpus_size=len(kb),
        keep_per_query=True,
    )
    in_scope = [q["top_score"] for q in dev.per_query]
    correct = [q["p_at_1"] == 1.0 for q in dev.per_query]

    probe = list(OFF_TOPIC_PROBE)
    off_hits = retriever.search_batch(probe, top_k=1)
    off_scores = [h[0][1] if h else 0.0 for h in off_hits]

    choice = tune_threshold(in_scope, correct, objective="off_topic", off_topic_scores=off_scores)
    summary = summarize_abstention(choice.tau, in_scope, list(zip(probe, off_scores, strict=True)))

    config = {
        "encoder": model_name,
        "field": field,
        "tau": choice.tau,
        "tuned_on": "dev",
        "objective": "off_topic",
        "dev_in_scope_answered": round(summary.in_scope_answered, 4),
        "probe_off_topic_answered": round(summary.off_topic_answered, 4),
        "probe_size": len(probe),
        "note": (
            "tau rejects the whole off-topic probe while keeping as much in-scope "
            "coverage as possible. The probe is 12 hand-written queries, not a "
            "benchmark -- see src.evaluate.OFF_TOPIC_PROBE."
        ),
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / "retriever_config.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


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
    parser.add_argument(
        "--retriever",
        default="bm25",
        choices=("bm25", "biencoder", "both"),
        help="which retriever to score (default: bm25)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="encoder for --retriever biencoder (default: MiniLM)",
    )
    parser.add_argument("--json", type=Path, help="also write results to this JSON file")
    parser.add_argument(
        "--tune-threshold",
        action="store_true",
        help="tune the abstention threshold on dev and write artifacts/retriever_config.json",
    )
    args = parser.parse_args(argv)

    if args.tune_threshold:
        from src.index import DEFAULT_MODEL

        config = tune_and_write_config(args.model or DEFAULT_MODEL, field=args.field)
        print("Abstention threshold (PRD 7.5), tuned on dev:\n")
        for key in ("encoder", "field", "tau", "objective"):
            print(f"  {key:<26} {config[key]}")
        print(f"  {'dev in-scope answered':<26} {config['dev_in_scope_answered']:.1%}")
        print(
            f"  {'off-topic probe answered':<26} {config['probe_off_topic_answered']:.1%} "
            f"of {config['probe_size']}"
        )
        print("\nwrote artifacts/retriever_config.json")
        return 0

    kb = load_kb()
    queries, relevant = load_eval_data()
    subset = select_split(queries, args.split)

    fields = list(FIELDS) if args.all_fields else [args.field]
    results = []

    if args.retriever in ("bm25", "both"):
        for f in fields:
            retriever = BM25Retriever(kb, field=f)
            results.append(
                evaluate(
                    retriever, subset, relevant, split=args.split, k=args.k, corpus_size=len(kb)
                )
            )

    if args.retriever in ("biencoder", "both"):
        results.append(
            evaluate_biencoder(split=args.split, model_name=args.model, field=args.field, k=args.k)
        )

    print(
        f"corpus {len(kb):,} entries | split {args.split!r} | {len(subset):,} queries "
        f"| {results[0].mean_positives_per_query:.2f} positives per query\n"
    )
    print(format_table(results))

    if args.split == "test":
        print("\nTest split only -- the split every reported result comes from (PRD 8.5).")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps([r.to_dict() for r in results], indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
