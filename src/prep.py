"""Build the knowledge base, qrels, and seeded query splits from raw COUGH.

Deterministic and reproducible: same inputs and seed always yield byte-identical
artifacts. Run after :mod:`src.download`::

    python -m src.prep

Outputs under ``data/``:

    kb.parquet          knowledge base
    queries.parquet     query id -> text + split assignment
    qrels.parquet       relevance judgments, remapped onto surviving KB ids
    splits/*.json       train / dev / test query ids
    prep_report.json    what was dropped, merged and remapped, and why

Three properties this module exists to guarantee
------------------------------------------------
1. **The corpus is never split.** All surviving FAQ entries stay retrievable at
   every stage; only the *queries* are partitioned.
2. **No judgment is silently lost.** Dropping a duplicate FAQ row without
   remapping the qrels that point at it would delete real positives and quietly
   understate every metric. 62 positive judgments depend on this.
3. **Numbers and units survive.** "20 seconds", "6 feet", "14 days" are
   load-bearing in health content; the previous pipeline stripped digits
   entirely.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR = PROJECT_ROOT / "data"

SEED = 42
SPLIT_RATIOS = {"train": 0.70, "dev": 0.15, "test": 0.15}

# The qrels label column, named for its own threshold rule.
LABEL_COLUMN = "Label (mean score > 3 as positive)"

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------
# Trust tiering
# --------------------------------------------------------------------------
# Two tiers: `official` (WHO/CDC/government) and `community` (everything else).
# Every one of COUGH's 56 sources is listed explicitly rather than matched by a
# regex — a pattern like "contains 'health'"
# would silently mis-tier new sources, and mis-tiering is a safety issue when
# the UI presents the label as an authority signal.
#
# Note the judgement call: Harvard, JHU, Penn Med, AMA and Children's Hospital
# LA are reputable but are not public-health authorities, so they sit in
# `community` under that definition. See the notebook for the argument
# that a third `academic` tier would be a reasonable v2 refinement.
OFFICIAL_SOURCES: frozenset[str] = frozenset(
    {
        # International / national bodies
        "WHO",
        "WHOMyth",
        "CDC_FAQ",
        "NIH_FAQ",
        "USDA",
        "US Labor",
        "Veterans Affairs",
        "Tricare",
        "EU",
        "UN",
        "GOV_UK",
        "Canada Gov",
        # US state / county / city government
        "AHCCCS",
        "Alabama",
        "CITY_FAQ",
        "CLEVERLAND_FAQ",
        "California DoH",
        "Delaware DoH",
        "Florida DoH",
        "Georgia DoL",
        "Georgia Explore",
        "Illinois DoPH",
        "King County",
        "Massachusetts",
        "Michigan",
        "Minnesota DoH Facecover",
        "Minnesota DoH School_Scraper",
        "Minnesota Employees",
        "Minnesota Testing",
        "NY Health Insurance",
        "NY Life Annuity Credit",
        "NYC",
        "NYSEQ",
        "New Jersey",
        "North_Carolina_Phase2",
        "Pennsylvania Businesses Operating",
        "Pennsylvania Unemployment Compensation",
        "San Mateo County Health",
        "Santa Clara DoH",
        "Texas Health Services",
        "VDH_FAQ",
        "Washington State",
        "colorado",
    }
)

COMMUNITY_SOURCES: frozenset[str] = frozenset(
    {
        "AMA",
        "CNN QA",
        # RUF001 flags the U+2019 apostrophe as ambiguous, but this string must
        # match COUGH's `source` value byte for byte or classify_trust() raises.
        "Children’s Hospital LA",  # noqa: RUF001
        "Drug_FAQ",
        "Harvard_FAQ",
        "JHU HUB",
        "JHU Medicine",
        "Kids Health from Nemours",
        "Medical Myth",
        "New York Times",
        "Penn Med",
        "WikiHow",
        "WikiHow_QA",
    }
)


def classify_trust(source: str) -> str:
    """Map a COUGH source label to a trust tier.

    Raises on an unknown source rather than defaulting. A silent default is how
    a community source ends up presented to a user as official.
    """
    source = str(source).strip()
    if source in OFFICIAL_SOURCES:
        return "official"
    if source in COMMUNITY_SOURCES:
        return "community"
    raise ValueError(
        f"unknown source {source!r}: add it to OFFICIAL_SOURCES or "
        f"COMMUNITY_SOURCES in src/prep.py. Refusing to guess a trust tier."
    )


# Ranking used when duplicate entries disagree on provenance: keep the most
# authoritative copy.
_TRUST_RANK = {"official": 0, "community": 1}


# --------------------------------------------------------------------------
# Text normalization
# --------------------------------------------------------------------------
def normalize_text(value: object) -> str:
    """Normalize whitespace, unicode and stray markup — but never content.

    Deliberately does NOT lowercase, strip punctuation, or remove digits. The
    old pipeline's ``re.sub(r"[^a-zA-Z\\s]", "", text)`` turned "COVID-19" into
    "covid" and "20 seconds" into "seconds"; both matter here.

    NFKC folds the non-breaking spaces COUGH inherited from scraped HTML (e.g.
    "51\\xa0cm") into ordinary spaces while leaving real characters intact.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    text = unicodedata.normalize("NFKC", text)
    text = _TAG_RE.sub(" ", text)  # 9 rows still carry HTML in `answer`
    text = _WS_RE.sub(" ", text)
    return text.strip()


def _dedup_key(question: str, answer: str) -> tuple[str, str]:
    """Key for near-duplicate detection: case- and whitespace-insensitive."""
    return (question.casefold(), answer.casefold())


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
@dataclass
class PrepReport:
    """Everything that changed between raw COUGH and the built artifacts."""

    raw_faq_rows: int = 0
    raw_query_rows: int = 0
    raw_qrel_rows: int = 0

    dropped_missing_question: list[int] = field(default_factory=list)
    dropped_missing_answer: list[int] = field(default_factory=list)
    duplicate_groups: int = 0
    dropped_duplicates: list[int] = field(default_factory=list)
    duplicate_groups_spanning_sources: int = 0

    kb_rows: int = 0
    trust_counts: dict[str, int] = field(default_factory=dict)

    qrels_remapped: int = 0
    qrels_remapped_positive: int = 0
    qrels_dropped_orphan_query: int = 0
    qrels_dropped_orphan_query_positive: int = 0
    qrels_dropped_dangling_faq: int = 0
    qrels_dropped_dangling_faq_positive: int = 0
    qrels_deduped_after_remap: int = 0
    # Accounting that must reconcile exactly:
    #   raw positives == final + lost + merged
    raw_positive_rows: int = 0
    positives_lost: int = 0
    positives_merged: int = 0

    qrel_rows: int = 0
    positive_rows: int = 0
    queries_with_positives: int = 0
    mean_positives_per_query: float = 0.0

    seed: int = SEED
    split_sizes: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 2


def load_raw(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three COUGH files this pipeline uses."""
    _raise_csv_field_limit()
    missing = [
        name
        for name in (
            "FAQ_Bank_eval.csv",
            "User_Query_Bank.csv",
            "Annotated_Relevance_Set.csv",
        )
        if not (raw_dir / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"missing {', '.join(missing)} in {raw_dir}. Run `python -m src.download` first."
        )

    # engine="python" tolerates the embedded newlines in COUGH's answer fields.
    faq = pd.read_csv(raw_dir / "FAQ_Bank_eval.csv", engine="python")
    queries = pd.read_csv(raw_dir / "User_Query_Bank.csv", engine="python")
    qrels = pd.read_csv(raw_dir / "Annotated_Relevance_Set.csv", engine="python")
    return faq, queries, qrels


# --------------------------------------------------------------------------
# Knowledge base
# --------------------------------------------------------------------------
def build_kb(faq: pd.DataFrame, report: PrepReport) -> tuple[pd.DataFrame, dict[int, int]]:
    """Build the knowledge base and the old-id -> surviving-id remap.

    Returns ``(kb, remap)`` where ``remap`` covers *every* original id: identity
    for survivors, the survivor's id for merged duplicates, and absent for rows
    dropped as unusable.
    """
    report.raw_faq_rows = len(faq)

    # FAQ_Bank_eval is documented as all-English; assert rather
    # than filter, because a filter that silently removed rows here would break
    # the qrel index alignment the whole benchmark rests on.
    languages = set(faq["language"].dropna().unique())
    if languages != {"en"}:
        raise ValueError(
            f"expected FAQ_Bank_eval to be all English, found languages={sorted(languages)}"
        )

    work = pd.DataFrame(
        {
            "id": faq["index"].astype(int),
            "question": faq["question"].map(normalize_text),
            "answer": faq["answer"].map(normalize_text),
            "source": faq["source"].astype(str).str.strip(),
            "url": faq["url"].map(normalize_text),
        }
    )
    work["trust"] = work["source"].map(classify_trust)

    # Rows with no question cannot be retrieved by question matching; rows with
    # no answer cannot be served. Either way they are unusable, so they go --
    # and the report records the judgments that pointed at them.
    no_question = work["question"].str.len() == 0
    no_answer = work["answer"].str.len() == 0
    report.dropped_missing_question = sorted(work.loc[no_question, "id"].tolist())
    report.dropped_missing_answer = sorted(work.loc[no_answer, "id"].tolist())
    work = work.loc[~(no_question | no_answer)].copy()

    # Near-duplicate collapse, case- and whitespace-insensitive. Within a group
    # keep the most authoritative source, tie-broken by lowest id so the choice
    # is deterministic across runs.
    work["_key"] = [_dedup_key(q, a) for q, a in zip(work["question"], work["answer"], strict=True)]
    work["_rank"] = work["trust"].map(_TRUST_RANK)
    work = work.sort_values(["_key", "_rank", "id"], kind="mergesort")

    grouped = work.groupby("_key", sort=False)
    report.duplicate_groups = int((grouped.size() > 1).sum())
    report.duplicate_groups_spanning_sources = int((grouped["source"].nunique() > 1).sum())

    survivor_of: dict[int, int] = {}
    for _, group in grouped:
        ids = group["id"].tolist()
        keeper = ids[0]
        for old_id in ids:
            survivor_of[old_id] = keeper

    keep_mask = work["id"].map(lambda i: survivor_of[i] == i)
    report.dropped_duplicates = sorted(work.loc[~keep_mask, "id"].tolist())

    kb = (
        work.loc[keep_mask, ["id", "question", "answer", "source", "url", "trust"]]
        .sort_values("id", kind="mergesort")
        .reset_index(drop=True)
    )

    report.kb_rows = len(kb)
    report.trust_counts = kb["trust"].value_counts().to_dict()
    return kb, survivor_of


# --------------------------------------------------------------------------
# Qrels
# --------------------------------------------------------------------------
def build_qrels(
    qrels: pd.DataFrame,
    remap: dict[int, int],
    valid_query_ids: set[int],
    report: PrepReport,
) -> pd.DataFrame:
    """Align relevance judgments to surviving KB ids and valid query ids."""
    report.raw_qrel_rows = len(qrels)

    frame = pd.DataFrame(
        {
            "query_id": qrels["Query_index"].astype(int),
            "kb_id": qrels["FAQ_index"].astype(int),
            "relevant": qrels[LABEL_COLUMN].astype(str).str.strip().str.lower() == "positive",
        }
    )
    report.raw_positive_rows = int(frame["relevant"].sum())

    # COUGH ships judgments for 35 query ids that are absent from the query bank
    # (they line up exactly with its index gaps). All are negative, so this
    # removes no signal -- but the report proves that rather than assuming it.
    orphan = ~frame["query_id"].isin(valid_query_ids)
    report.qrels_dropped_orphan_query = int(orphan.sum())
    report.qrels_dropped_orphan_query_positive = int((orphan & frame["relevant"]).sum())
    frame = frame.loc[~orphan]

    # Judgments pointing at rows dropped as unusable cannot be rescued.
    dangling = ~frame["kb_id"].isin(remap.keys())
    report.qrels_dropped_dangling_faq = int(dangling.sum())
    report.qrels_dropped_dangling_faq_positive = int((dangling & frame["relevant"]).sum())
    frame = frame.loc[~dangling].copy()

    # The load-bearing step: point judgments at the surviving duplicate rather
    # than dropping them. Skipping this would delete real positives.
    remapped = frame["kb_id"].map(remap).astype(int)
    changed = remapped != frame["kb_id"]
    report.qrels_remapped = int(changed.sum())
    report.qrels_remapped_positive = int((changed & frame["relevant"]).sum())
    frame["kb_id"] = remapped

    # Remapping can make a query judge the same surviving entry twice; keep the
    # most favourable verdict so a merge never downgrades a positive.
    before = len(frame)
    before_positive = int(frame["relevant"].sum())
    frame = (
        frame.sort_values(["query_id", "kb_id", "relevant"], ascending=[True, True, False])
        .drop_duplicates(subset=["query_id", "kb_id"], keep="first")
        .reset_index(drop=True)
    )
    report.qrels_deduped_after_remap = before - len(frame)
    # Two positives collapsing onto one surviving entry is a merge, not a loss:
    # the query still has that entry marked relevant. Tracked separately so the
    # positive accounting reconciles exactly.
    report.positives_merged = before_positive - int(frame["relevant"].sum())

    report.qrel_rows = len(frame)
    positives = frame.loc[frame["relevant"]]
    report.positive_rows = len(positives)
    report.queries_with_positives = int(positives["query_id"].nunique())
    report.mean_positives_per_query = (
        round(float(positives.groupby("query_id").size().mean()), 3) if len(positives) else 0.0
    )
    return frame


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------
def make_splits(
    query_ids: list[int], seed: int = SEED, ratios: dict[str, float] | None = None
) -> dict[str, list[int]]:
    """Partition query ids into train/dev/test deterministically.

    Sorts before shuffling so the result depends only on the id *set* and the
    seed, never on the order rows happened to arrive in.
    """
    ratios = ratios or SPLIT_RATIOS
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {sum(ratios.values())}")

    ordered = sorted(query_ids)
    rng = random.Random(seed)
    rng.shuffle(ordered)

    total = len(ordered)
    n_train = int(total * ratios["train"])
    n_dev = int(total * ratios["dev"])

    return {
        "train": sorted(ordered[:n_train]),
        "dev": sorted(ordered[n_train : n_train + n_dev]),
        "test": sorted(ordered[n_train + n_dev :]),
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run(raw_dir: Path = RAW_DIR, out_dir: Path = DATA_DIR, seed: int = SEED) -> PrepReport:
    """Build every artifact. Returns the report describing what happened."""
    report = PrepReport(seed=seed)
    faq, queries_raw, qrels_raw = load_raw(raw_dir)

    kb, remap = build_kb(faq, report)

    report.raw_query_rows = len(queries_raw)
    queries = pd.DataFrame(
        {
            "query_id": queries_raw["index"].astype(int),
            "query": queries_raw["query"].map(normalize_text),
        }
    )
    valid_query_ids = set(queries["query_id"])

    qrels = build_qrels(qrels_raw, remap, valid_query_ids, report)

    # Every query must retain at least one positive, or it contributes nothing
    # but noise to MRR/P@k.
    with_positives = set(qrels.loc[qrels["relevant"], "query_id"])
    unusable = sorted(valid_query_ids - with_positives)
    if unusable:
        queries = queries.loc[queries["query_id"].isin(with_positives)].copy()
        valid_query_ids = with_positives
        qrels = qrels.loc[qrels["query_id"].isin(with_positives)].reset_index(drop=True)

    splits = make_splits(sorted(valid_query_ids), seed=seed)
    split_of = {qid: name for name, ids in splits.items() for qid in ids}
    queries["split"] = queries["query_id"].map(split_of)
    queries = queries.sort_values("query_id").reset_index(drop=True)
    report.split_sizes = {name: len(ids) for name, ids in splits.items()}

    # Write artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)

    kb.to_parquet(out_dir / "kb.parquet", index=False)
    queries.to_parquet(out_dir / "queries.parquet", index=False)
    qrels.to_parquet(out_dir / "qrels.parquet", index=False)

    for name, ids in splits.items():
        (out_dir / "splits" / f"{name}.json").write_text(
            json.dumps({"seed": seed, "n": len(ids), "query_ids": ids}, indent=2) + "\n",
            encoding="utf-8",
        )

    report.positives_lost = (
        report.qrels_dropped_dangling_faq_positive + report.qrels_dropped_orphan_query_positive
    )

    # Hard constraint #4 is honest evaluation, so prove the positives balance
    # instead of trusting it. Any drift means judgments vanished unnoticed.
    accounted = report.positive_rows + report.positives_lost + report.positives_merged
    if accounted != report.raw_positive_rows:
        raise AssertionError(
            f"positive accounting does not reconcile: raw={report.raw_positive_rows:,} "
            f"but final={report.positive_rows:,} + lost={report.positives_lost:,} "
            f"+ merged={report.positives_merged:,} = {accounted:,}"
        )

    (out_dir / "prep_report.json").write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return report


def _print_report(report: PrepReport) -> None:
    r = report
    print("\nKnowledge base")
    print(f"  raw FAQ rows                     {r.raw_faq_rows:>7,}")
    print(
        f"  dropped, no question             {len(r.dropped_missing_question):>7,}"
        f"   {r.dropped_missing_question}"
    )
    print(
        f"  dropped, no answer               {len(r.dropped_missing_answer):>7,}"
        f"   {r.dropped_missing_answer}"
    )
    print(
        f"  duplicate groups collapsed       {r.duplicate_groups:>7,}"
        f"   ({r.duplicate_groups_spanning_sources} spanned >1 source)"
    )
    print(f"  dropped as duplicates            {len(r.dropped_duplicates):>7,}")
    print(f"  KB rows                          {r.kb_rows:>7,}")
    for tier, count in sorted(r.trust_counts.items()):
        print(f"    {tier:<28} {count:>7,}")

    print("\nRelevance judgments")
    print(f"  raw qrel rows                    {r.raw_qrel_rows:>7,}")
    print(
        f"  dropped, query not in bank       {r.qrels_dropped_orphan_query:>7,}"
        f"   ({r.qrels_dropped_orphan_query_positive} positive)"
    )
    print(
        f"  dropped, FAQ row unusable        {r.qrels_dropped_dangling_faq:>7,}"
        f"   ({r.qrels_dropped_dangling_faq_positive} positive)"
    )
    print(
        f"  REMAPPED onto surviving dup      {r.qrels_remapped:>7,}"
        f"   ({r.qrels_remapped_positive} positive, rescued not dropped)"
    )
    print(f"  merged after remap               {r.qrels_deduped_after_remap:>7,}")
    print(f"  final qrel rows                  {r.qrel_rows:>7,}")
    print(f"  queries with >=1 positive        {r.queries_with_positives:>7,}")
    print(f"  mean positives per query         {r.mean_positives_per_query:>7.2f}")

    print("\n  Positive accounting (must balance)")
    print(f"    raw positives                  {r.raw_positive_rows:>7,}")
    print(f"    - lost (unusable FAQ row)      {r.positives_lost:>7,}")
    print(f"    - merged onto surviving dup    {r.positives_merged:>7,}")
    print(f"    = final positives              {r.positive_rows:>7,}   balanced")

    print(f"\nQuery splits (seed {r.seed})")
    for name in ("train", "dev", "test"):
        n = r.split_sizes.get(name, 0)
        total = sum(r.split_sizes.values())
        print(f"  {name:<8} {n:>5,}  ({n / total:.1%})")
    print("\n  Reminder: metrics are reported on `test` only.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build KB, qrels and query splits from COUGH.")
    parser.add_argument("--seed", type=int, default=SEED, help=f"split seed (default {SEED})")
    args = parser.parse_args(argv)

    report = run(seed=args.seed)
    _print_report(report)
    rel = DATA_DIR.relative_to(PROJECT_ROOT)
    print(f"\nWrote {rel}/kb.parquet, queries.parquet, qrels.parquet, splits/, prep_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
