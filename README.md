# Covicare — COVID-19 FAQ Chatbot

A **retrieval-only** COVID-19 information chatbot: it matches a user's plain-language
question to the best vetted FAQ answer, returns it with its source and a medical
disclaimer, and **abstains when it isn't confident**. It never generates medical text.

The retriever is a bi-encoder **fine-tuned locally** on the COUGH benchmark and exported
as a model artifact the API loads at startup.

> **Status: rebuild in progress.** This repository is being rebuilt from scratch against
> [`PRD.md`](PRD.md). Phase 0 (hygiene and scaffold) is complete; the retrieval pipeline
> lands in Phases 1–5. **No accuracy numbers are published here yet** — the ones that
> appear will be reproducible by `python -m src.evaluate` on the held-out test split, and
> nothing else.

## Why retrieval, not generation

The previous implementation fine-tuned generative and extractive models (DialoGPT,
BERT-QA) on ~110 unique QA pairs duplicated into a 24k-row file. That produced metrics
that looked excellent and meant nothing, and a backend that could not start because the
weights were git-ignored.

The rebuild changes the framing of the problem. For a health-information bot over a few
thousand curated FAQ pairs, retrieval is the better answer on every axis that matters:

| | Generation | **Retrieval (chosen)** |
|---|---|---|
| Hallucinated medical claims | possible | **impossible — answers are stored text** |
| Attribution | hard | **every answer carries its source** |
| Honest evaluation | awkward at this scale | **standard IR metrics on a real benchmark** |
| Data needed | far more than we have | **fits the corpus exactly** |

A confident wrong match is the remaining failure mode, which is what the confidence
threshold and abstention path exist to control.

Retrieval-only does **not** mean nothing is trained. The bi-encoder is fine-tuned on the
COUGH training queries with `MultipleNegativesRankingLoss` — training the model that
*ranks* vetted answers, never one that writes them.

## How quality is established

Three retrievers, measured on the same held-out queries, each required to beat the last:

```
BM25 (lexical baseline)  <  off-the-shelf bi-encoder  <  fine-tuned bi-encoder
```

The 1,201 COUGH queries are split train/dev/test with a fixed seed. Training uses train,
threshold tuning and early stopping use dev, and **every reported number comes from test**.
The 7,117-entry FAQ corpus is never split — all of it stays retrievable at every stage.

If the fine-tune fails to beat the off-the-shelf encoder, it is not adopted, and that
result gets recorded rather than buried.

## Data

**[COUGH](https://github.com/sunlab-osu/covid-faq)** — a COVID-19 FAQ retrieval benchmark
that ships with relevance judgments, which is what makes honest evaluation possible.

| File | Rows | Role |
|---|---|---|
| `FAQ_Bank_eval.csv` | 7,117 | Retrieval corpus **and** served knowledge base |
| `User_Query_Bank.csv` | 1,201 | Queries, split train/dev/test |
| `Annotated_Relevance_Set.csv` | 39,760 | Relevance judgments (qrels) |

Licensed **CC BY-NC-SA 4.0**, research and education use only. Raw data is fetched, never
committed — see [`data/README.md`](data/README.md).

The content dates from ~2020–2021 and mixes WHO/CDC with community sources, so answers
carry a source, a trust tier, and a disclaimer.

## Architecture

```
Flutter app ──POST /predict──► FastAPI
                                 ├─ retriever      BM25 │ fine-tuned bi-encoder (+ optional re-ranker)
                                 ├─ knowledge base data/kb.parquet + precomputed embeddings
                                 ├─ encoder        artifacts/encoder/  (the trained artifact)
                                 └─ threshold      → answer + source + disclaimer, or abstention
```

### Notebook and `src/` share one implementation

`src/` holds every reusable piece — prep, index, retriever, train, evaluate, api.
[`notebooks/development.ipynb`](notebooks/) **imports** them; it never keeps its own copy
of preprocessing, retrieval, training, or metric code. The API imports exactly the same
modules.

One implementation, two consumers. This is what guarantees the numbers reported during
development are the numbers the app actually serves — the failure the old project made
unavoidable by having its notebook train BERT-QA while its API served an unrelated
DialoGPT.

## Getting started

Development is fully local and uses the GPU for the fine-tune (PRD §16.1):

```bash
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt   # Windows

python -c "import torch; print(torch.cuda.is_available())"      # must print True
```

Once the pipeline lands (Phases 1–6):

```bash
python -m src.download         # fetch COUGH
python -m src.prep              # -> data/kb.parquet + seeded query splits
python -m src.train             # fine-tune -> artifacts/encoder/
python -m src.evaluate          # MRR@10 / P@k / Recall / nDCG on the test split
uvicorn src.api:app --reload    # serve on :8000
pytest                          # tests
```

Serving needs no GPU — CPU inference is fine at this corpus size.

The Flutter client lives in [`app/`](app/) — see [`app/README.md`](app/README.md).

## Roadmap

Phases and their acceptance criteria are defined in [`PRD.md`](PRD.md) §18.

| Phase | | Status |
|---|---|---|
| 0 | Hygiene & scaffold | ✅ done |
| 1 | Data pipeline (`download.py`, `prep.py`, splits, EDA) | next |
| 2 | BM25 baseline + evaluation harness | |
| 3 | Off-the-shelf semantic retriever + threshold/abstention | |
| 4 | **Fine-tune the retriever** → `artifacts/encoder/` | |
| 5 | Cross-encoder re-ranker (optional) | |
| 6 | FastAPI service | |
| 7 | Flutter integration | |
| 8 | Tests + CI | |
| 9 | Docker, deploy, docs | |

## Not medical advice

General information only, drawn from a dated static snapshot. It does not diagnose,
triage, or personalize. For personal or urgent concerns consult a healthcare professional
or [WHO](https://www.who.int/emergencies/diseases/novel-coronavirus-2019) /
[CDC](https://www.cdc.gov/covid/).
