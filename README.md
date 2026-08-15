# Covicare — COVID-19 FAQ Chatbot

A **retrieval-only** COVID-19 information chatbot: it matches a user's plain-language
question to the best vetted FAQ answer, returns it with its source and a medical
disclaimer, and **abstains when it isn't confident**. It never generates medical text.

The retriever is a locally-run bi-encoder over a vetted COVID-19 FAQ corpus, evaluated on
the COUGH retrieval benchmark with confidence intervals on every claim.

> **Status: rebuild in progress.** Built against [`PRD.md`](PRD.md). The retrieval
> pipeline is complete through Phase 4; the API, app, CI and deployment are Phases 5–9.
> Every number below is reproducible by `python -m src.evaluate` on the held-out test
> split, and nothing is quoted that isn't.

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

Retrieval-only does not rule out training. A fine-tune of the *ranker* was built and
evaluated (see below); what is never trained is a model that writes medical text.

## Results

Measured on 181 held-out test queries against the full 7,077-entry corpus. Reproduce with
`python -m src.evaluate --split test`.

| System | MRR@10 | P@1 | Recall@10 | nDCG@10 |
|---|---|---|---|---|
| BM25 (lexical baseline) | 0.5127 | 0.4033 | 0.2401 | 0.2670 |
| **MiniLM-L6 bi-encoder** ← shipped | **0.6128** | **0.4972** | **0.3432** | **0.3570** |

The improvement over BM25 is **+0.100 MRR@10, 95% CI [+0.044, +0.157], p = 0.0004** by
paired bootstrap over per-query scores.

### The fine-tune was built, measured, and declined

PRD §7.4 planned to fine-tune the retriever with `MultipleNegativesRankingLoss` and ship the
result. That was done — `src/train.py` works and reproduces in ~4 minutes — and the trained
encoder scored **+0.021 MRR@10** over off-the-shelf on test.

It isn't shipped, because the same significance test tells two different stories:

| Comparison | Δ MRR@10 | 95% CI | p | verdict |
|---|---|---|---|---|
| BM25 → MiniLM | +0.100 | [+0.044, +0.157] | 0.0004 | adopted |
| MiniLM → fine-tuned | +0.021 | [−0.009, +0.051] | 0.161 | **declined** |

The second interval spans zero. Per query the fine-tune wins 32 and loses 25 — a coin flip.
A health-information product shouldn't ship a model whose benefit can't be measured, and
declining costs nothing here since the retrieval upgrade is already significant.

The result is **inconclusive, not negative**: resolving a +0.021 effect needs ~375 test
queries and we allocated 181. Cross-validation over all 1,201 would settle it, and is
recorded as deferred work rather than quietly dropped. Full record:
[`artifacts/finetune_experiment.json`](artifacts/finetune_experiment.json).

### Method

The 1,201 COUGH queries are split train/dev/test with a fixed seed. Training uses train,
threshold tuning and model selection use dev, and **every reported number comes from test**.
The FAQ corpus is never split — all 7,077 entries stay retrievable at every stage.

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
                                 ├─ retriever      MiniLM-L6 bi-encoder (+ optional re-ranker)
                                 ├─ knowledge base data/kb.parquet, 7,077 vetted FAQ entries
                                 ├─ embeddings     data/embeddings/, rebuilt by src.index
                                 └─ threshold      τ = 0.693 from artifacts/retriever_config.json
                                                   → answer + source + trust + disclaimer
                                                   → or abstention + WHO/CDC pointer
```

At τ = 0.693 the bot answers 77% of in-scope queries and 0% of an off-topic probe. Tuned on
dev, never on test. COUGH contains no unanswerable queries, so that probe is 12
hand-written out-of-scope questions — a smoke test, not a benchmark, and labelled as such
in the notebook.

### Notebook and `src/` share one implementation

`src/` holds every reusable piece — prep, index, retriever, train, evaluate, api.
[`notebooks/development.ipynb`](notebooks/) **imports** them; it never keeps its own copy
of preprocessing, retrieval, training, or metric code. The API imports exactly the same
modules.

One implementation, two consumers. This is what guarantees the numbers reported during
development are the numbers the app actually serves — the failure the old project made
unavoidable by having its notebook train BERT-QA while its API served an unrelated
DialoGPT.

The notebook is developed against a **live kernel** through the
[Jupyter MCP server](docs/jupyter-mcp.md), so every cell is written, executed and checked
in place rather than authored blind and hoped over.

## Getting started

Development is fully local and uses the GPU for the fine-tune (PRD §16.1):

```bash
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt   # Windows

python -c "import torch; print(torch.cuda.is_available())"      # must print True
```

For notebook work, start the JupyterLab instance that Claude Code's MCP server attaches
to (see [docs/jupyter-mcp.md](docs/jupyter-mcp.md)):

```bash
python scripts/start_jupyter.py
```

Pipeline:

```bash
python -m src.download         # fetch COUGH
python -m src.prep              # -> data/kb.parquet + seeded query splits
python -m src.train             # reproduce the fine-tune experiment (not shipped)
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
| 0 | Hygiene & scaffold | ✅ |
| 1 | Data pipeline (`download.py`, `prep.py`, splits, EDA) | ✅ |
| 2 | BM25 baseline + evaluation harness | ✅ |
| 3 | Semantic retriever + threshold/abstention | ✅ |
| 4 | Fine-tune the retriever | ✅ measured, not adopted |
| 5 | Cross-encoder re-ranker (optional) | next |
| 6 | FastAPI service | |
| 7 | Flutter integration | |
| 8 | Tests + CI | |
| 9 | Docker, deploy, docs | |

## Not medical advice

General information only, drawn from a dated static snapshot. It does not diagnose,
triage, or personalize. For personal or urgent concerns consult a healthcare professional
or [WHO](https://www.who.int/emergencies/diseases/novel-coronavirus-2019) /
[CDC](https://www.cdc.gov/covid/).
