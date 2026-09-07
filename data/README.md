# data/

This directory is **generated**, not committed.

## Why nothing here is in git

The knowledge base is derived from **COUGH** ([sunlab-osu/covid-faq](https://github.com/sunlab-osu/covid-faq)),
which is licensed **CC BY-NC-SA 4.0** — research and education use only, attribution
required to the source sites listed in the dataset's `List_of_websites.txt`.

Two consequences, both encoded in `.gitignore`:

1. **Raw dumps are never committed.** `python -m src.download` fetches them.
2. **`kb.parquet` is a derivative work** of a ShareAlike corpus, so it is regenerated
   by `python -m src.prep` rather than redistributed here.

Embeddings are likewise rebuilt by `python -m src.index`. No binaries are committed
anywhere in this repository.

## Reproducing the data

```bash
python -m src.download   # fetch COUGH into data/raw/
python -m src.prep       # -> kb.parquet + query splits + qrels
python -m src.index      # -> embeddings
```

## Expected layout after a full run

```
data/
├── raw/                        # COUGH working copy (gitignored)
│   ├── FAQ_Bank.csv                    15,919 rows
│   ├── FAQ_Bank_eval.csv                7,117 rows  <- retrieval corpus
│   ├── User_Query_Bank.csv              1,201 rows  <- queries, split below
│   └── Annotated_Relevance_Set.csv     39,760 judgments
├── kb.parquet                  # normalized knowledge base, 7,077 entries
├── queries.parquet             # the 1,201 queries
├── qrels.parquet               # relevance judgments aligned to KB ids
├── splits/                     # seeded train/dev/test query ids (committed —
│   │                           #   tiny, and the split must be reproducible)
│   ├── train.json                      840 queries   reserved, unused by the served system
│   ├── dev.json                        180 queries   threshold tuning
│   └── test.json                       181 queries   the ONLY split metrics come from
├── embeddings/                 # precomputed KB vectors (gitignored)
└── prep_report.json            # what preprocessing changed (committed)
```

## Two invariants worth stating loudly

**1. The served corpus and the evaluated corpus are the same set.**
`FAQ_Bank_eval.csv` is the corpus the relevance labels reference, so it is also the
corpus we serve. Changing one without the other silently invalidates every metric.

**2. The corpus is never split — only the queries are.**
All 7,077 knowledge-base entries stay retrievable at every stage. The train/dev/test
partition applies to the 1,201 *queries*: τ is tuned on dev, results are reported on test
only, and the seed is fixed so the split is reproducible. Nothing is trained, so the train
split is held in reserve and currently unused.

## Dedup and qrels alignment

`src.prep` drops exact and near-duplicate FAQ entries, but the qrels reference original
`FAQ_Bank_eval` row indices (0–7116). Dropped rows are therefore **remapped onto their
surviving twin**, not discarded — a dangling qrel is a silently lost positive, which would
understate every metric reported. Where duplicates span sources, the **official** copy is
the one kept.

`prep_report.json` records the full accounting: entries removed, judgments remapped, and a
balance assertion that `raw positives == final + lost + merged`. Prep refuses to write
artifacts if it does not balance.

## Attribution

> Zhang, Xinliang Frederick et al. *COUGH: A Challenge Dataset and Models for COVID-19
> FAQ Retrieval.* Licensed CC BY-NC-SA 4.0. Content aggregated from WHO, CDC, government
> health departments, and community sources — see the dataset's `List_of_websites.txt`.
