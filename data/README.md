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

Embeddings are likewise rebuilt by `python -m src.index`. The one binary this project
*does* commit is the trained encoder — see [`artifacts/README.md`](../artifacts/README.md)
for why that exception exists.

## Reproducing the data

```bash
python -m src.download    # fetch COUGH into data/raw/
python -m src.prep         # -> kb.parquet + query splits + qrels
python -m src.index        # -> embeddings
```

## Expected layout after a full run

```
data/
├── download.py                 # fetch script (committed)
├── raw/                        # COUGH working copy (gitignored)
│   ├── FAQ_Bank.csv                    15,919 rows
│   ├── FAQ_Bank_eval.csv                7,117 rows  <- retrieval corpus
│   ├── User_Query_Bank.csv              1,201 rows  <- queries, split below
│   └── Annotated_Relevance_Set.csv     39,760 judgments
├── kb.parquet                  # normalized knowledge base (PRD §6.5)
├── splits/                     # seeded train/dev/test query ids (committed —
│   │                           #   tiny, and the split must be reproducible)
│   ├── train.json                      ~70%
│   ├── dev.json                        ~15%   threshold tuning, early stopping
│   └── test.json                       ~15%   the ONLY split metrics come from
├── qrels.parquet               # relevance judgments aligned to KB ids
└── embeddings/                 # precomputed KB vectors (gitignored)
```

## Two invariants worth stating loudly

**1. The served corpus and the evaluated corpus are the same set.**
`FAQ_Bank_eval.csv` is the corpus the relevance labels reference, so it is also the
corpus we serve (PRD §6.2). Changing one without the other silently invalidates every
metric.

**2. The corpus is never split — only the queries are.**
All 7,117 FAQ entries stay retrievable at every stage. The train/dev/test partition
applies to the 1,201 *queries* (PRD §7.4). Train on train, tune τ and early-stop on
dev, report on test only, with a fixed seed so the split is reproducible.

## Dedup and qrels alignment

`src.prep` drops exact/near-duplicate FAQ entries, but the qrels reference original
`FAQ_Bank_eval` row indices (0–7116). Dropped rows must therefore be **remapped onto
their surviving twin**, not discarded — a dangling qrel is a silently lost positive,
which would understate every metric we report. The prep step records how many entries
were removed and how many judgments were remapped (PRD §6.4 step 4).

## Attribution

> Zhang, Xinliang Frederick et al. *COUGH: A Challenge Dataset and Models for COVID-19
> FAQ Retrieval.* Licensed CC BY-NC-SA 4.0. Content aggregated from WHO, CDC, government
> health departments, and community sources — see the dataset's `List_of_websites.txt`.
