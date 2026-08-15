"""Fine-tune the bi-encoder retriever on COUGH (PRD §7.4, Option B).

    python -m src.train                     # train, evaluate on dev, export
    python -m src.train --epochs 5 --batch-size 256
    python -m src.train --dry-run           # build the data, train nothing

Produces ``artifacts/encoder/``.

.. note::
   **The result of this run was not adopted.** Phase 4 measured +0.021 MRR@10 over the
   off-the-shelf encoder on test, with a 95% CI of [-0.009, +0.051] -- indistinguishable
   from noise at n=181. The API serves the off-the-shelf encoder instead. This module is
   retained because the experiment is worth reproducing and the decision is worth
   revisiting with a properly powered evaluation. See ``artifacts/finetune_experiment.json``
   and PRD §22.

Method
------
Training pairs are ``(user query, KB question)`` taken from the qrels positives
of the **train** split only. Loss is ``MultipleNegativesRankingLoss``: for each
anchor, the other positives in the batch act as negatives, so no negative
mining is needed and batch size is a direct quality lever -- a bigger batch
means more negatives per step.

That in-batch trick has one sharp edge. Two different queries can be positive
for the *same* KB entry, and if both land in one batch the loss punishes the
model for a correct match. ``BatchSamplers.NO_DUPLICATES`` prevents it.

Anti-leakage (PRD §8.5)
-----------------------
Train uses train queries. Early stopping and model selection use dev. Test is
never read here -- not for stopping, not for tuning, not for reporting. The
split assertions below fail loudly rather than trusting that.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ENCODER_DIR = ARTIFACTS_DIR / "encoder"

# Resolved in PRD §22: MiniLM is the better abstainer, small enough to vendor
# in git, and its size allows the large batches MNRL wants.
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SEED = 42
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 2e-5

# The evaluator emits `dev_cosine_mrr@10`; the trainer prefixes it with `eval_`.
BEST_METRIC = "eval_dev_cosine_mrr@10"


@dataclass
class TrainReport:
    """What the run did, for the notebook and the commit message."""

    base_model: str = BASE_MODEL
    train_pairs: int = 0
    train_queries: int = 0
    dev_queries: int = 0
    corpus_size: int = 0
    epochs: int = 0
    batch_size: int = 0
    learning_rate: float = 0.0
    seed: int = SEED
    dev_before: dict = field(default_factory=dict)
    dev_after: dict = field(default_factory=dict)
    output_dir: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def build_training_pairs(
    kb: pd.DataFrame, queries: pd.DataFrame, qrels: pd.DataFrame, split: str = "train"
) -> pd.DataFrame:
    """Expand each train query into one row per positive KB entry.

    Anchors are user queries; positives are KB *questions*, matching the field
    the index is built on so training and inference see the same text.
    """
    split_queries = queries.loc[queries["split"] == split]
    if split_queries.empty:
        raise ValueError(f"no queries in split {split!r}")

    positives = qrels.loc[qrels["relevant"]]
    question_by_id = dict(zip(kb["id"], kb["question"], strict=True))

    pairs = (
        split_queries[["query_id", "query"]]
        .merge(positives[["query_id", "kb_id"]], on="query_id", how="inner")
        .assign(positive=lambda d: d["kb_id"].map(question_by_id))
        .dropna(subset=["positive"])
        .rename(columns={"query": "anchor"})
    )
    return pairs[["query_id", "anchor", "positive"]].reset_index(drop=True)


def build_ir_evaluator(
    kb: pd.DataFrame,
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
    split: str = "dev",
    name: str = "dev",
):
    """Retrieval evaluator over the **full** corpus for one query split.

    The corpus is never subset -- the model is scored against all 7k entries at
    every evaluation, exactly as it will be at inference (PRD §7.4).
    """
    from sentence_transformers.sentence_transformer.evaluation import (
        InformationRetrievalEvaluator,
    )

    split_queries = queries.loc[queries["split"] == split]
    positives = qrels.loc[qrels["relevant"]]

    corpus = {int(i): q for i, q in zip(kb["id"], kb["question"], strict=True)}
    eval_queries = {
        int(qid): text
        for qid, text in zip(split_queries["query_id"], split_queries["query"], strict=True)
    }
    relevant: dict[int, set[int]] = {}
    for qid, kb_id in zip(positives["query_id"], positives["kb_id"], strict=True):
        if int(qid) in eval_queries:
            relevant.setdefault(int(qid), set()).add(int(kb_id))

    return InformationRetrievalEvaluator(
        queries=eval_queries,
        corpus=corpus,
        relevant_docs=relevant,
        name=name,
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1],
        precision_recall_at_k=[1, 10],
        map_at_k=[10],
        show_progress_bar=False,
        batch_size=256,
    )


def _assert_no_leakage(queries: pd.DataFrame) -> None:
    """Train, dev and test query ids must be disjoint."""
    groups = {
        s: set(queries.loc[queries["split"] == s, "query_id"]) for s in ("train", "dev", "test")
    }
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = groups[a] & groups[b]
        if overlap:
            raise AssertionError(f"{a}/{b} splits overlap on {len(overlap)} query ids")


def finetune(
    base_model: str = BASE_MODEL,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LR,
    seed: int = SEED,
    output_dir: Path = ENCODER_DIR,
    data_dir: Path | None = None,
    dry_run: bool = False,
) -> TrainReport:
    """Fine-tune, select on dev, and export to ``artifacts/encoder/``."""
    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
    from sentence_transformers.sentence_transformer.losses import (
        MultipleNegativesRankingLoss,
    )
    from sentence_transformers.sentence_transformer.training_args import (
        BatchSamplers,
        SentenceTransformerTrainingArguments,
    )

    from src.evaluate import DATA_DIR, load_eval_data
    from src.retriever import load_kb

    data_dir = data_dir or DATA_DIR
    kb = load_kb(data_dir)
    queries = pd.read_parquet(data_dir / "queries.parquet")
    qrels = pd.read_parquet(data_dir / "qrels.parquet")
    _assert_no_leakage(queries)
    load_eval_data(data_dir)  # validates the artifacts exist and parse

    pairs = build_training_pairs(kb, queries, qrels, split="train")
    report = TrainReport(
        base_model=base_model,
        train_pairs=len(pairs),
        train_queries=pairs["query_id"].nunique(),
        dev_queries=int((queries["split"] == "dev").sum()),
        corpus_size=len(kb),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        output_dir=str(output_dir),
    )
    if dry_run:
        return report

    model = SentenceTransformer(base_model, device="cuda" if torch.cuda.is_available() else "cpu")
    dev_evaluator = build_ir_evaluator(kb, queries, qrels, split="dev", name="dev")

    # Baseline before any training, so the notebook can show the delta the
    # fine-tune actually produced rather than an absolute number in isolation.
    report.dev_before = {k: round(float(v), 4) for k, v in dev_evaluator(model).items()}

    train_dataset = Dataset.from_pandas(pairs[["anchor", "positive"]], preserve_index=False)
    loss = MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir.parent / "_train_checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        # Two queries can share a positive; without this the loss would treat a
        # correct match as a negative.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=BEST_METRIC,
        greater_is_better=True,
        logging_steps=10,
        report_to=[],
        seed=seed,
    )

    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=train_dataset, loss=loss, evaluator=dev_evaluator
    )
    trainer.train()

    report.dev_after = {k: round(float(v), 4) for k, v in dev_evaluator(model).items()}

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))
    (ARTIFACTS_DIR / "train_report.json").write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true", help="build data, train nothing")
    args = parser.parse_args(argv)

    report = finetune(
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    print(f"base model     {report.base_model}")
    print(f"train pairs    {report.train_pairs:,} from {report.train_queries} queries")
    print(f"dev queries    {report.dev_queries}")
    print(f"corpus         {report.corpus_size:,} (never subset)")
    print(f"epochs         {report.epochs}   batch {report.batch_size}   lr {report.learning_rate}")

    if args.dry_run:
        print("\ndry run: nothing trained")
        return 0

    key = "dev_cosine_mrr@10"
    before, after = report.dev_before.get(key, 0.0), report.dev_after.get(key, 0.0)
    print(f"\ndev MRR@10     {before:.4f} -> {after:.4f}  ({after - before:+.4f})")
    print(f"exported       {report.output_dir}")
    print(
        "\nDev only. Run `python -m src.evaluate --split test --retriever biencoder "
        "--model artifacts/encoder` for the test number."
    )
    print(
        "NOTE: the Phase 4 run of this experiment was NOT adopted -- see "
        "artifacts/finetune_experiment.json."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
