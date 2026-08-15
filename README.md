# Covicare — COVID-19 FAQ Chatbot

A **retrieval-only** COVID-19 information chatbot: it matches a user's plain-language
question to the best vetted FAQ answer, returns it with its source and a medical
disclaimer, and **abstains when it isn't confident**. It never generates medical text.

The retriever is a locally-run bi-encoder over a vetted COVID-19 FAQ corpus, evaluated on
the COUGH retrieval benchmark with confidence intervals on every claim.

> **Status: rebuild in progress.** Built against [`PRD.md`](PRD.md). The retrieval
> pipeline, the API and the Flutter client are complete (Phases 0–3, 5, 6); CI and
> deployment remain.
> Every number below is reproducible by `python -m src.evaluate` on the held-out test
> split, and nothing is quoted that isn't.

## Why retrieval, not generation

The previous implementation trained generative and extractive models (DialoGPT,
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

The retriever is a stock open-source encoder used as-is. The engineering here is the
pipeline around it — corpus preparation, indexing, calibration and evaluation — not the
model weights.

## Results

Measured on 181 held-out test queries against the full 7,077-entry corpus. Reproduce with
`python -m src.evaluate --split test`.

| System | MRR@10 | P@1 | Recall@10 | nDCG@10 |
|---|---|---|---|---|
| BM25 (lexical baseline) | 0.5127 | 0.4033 | 0.2401 | 0.2670 |
| **MiniLM-L6 bi-encoder** ← shipped | **0.6128** | **0.4972** | **0.3432** | **0.3570** |

The improvement over BM25 is **+0.100 MRR@10, 95% CI [+0.044, +0.157], p = 0.0004** by
paired bootstrap over per-query scores.

The biggest movement is **Recall@10: 0.240 → 0.343 (+43%)**, and that is the point. BM25's
failures were bimodal — 40% of test queries answered at rank 1, but **27% with nothing
relevant retrieved at all**, several sharing no vocabulary whatsoever with their correct
answer. Those are unreachable by better ranking; only matching on meaning finds them.

### Method

The 1,201 COUGH queries are split train/dev/test with a fixed seed. The threshold τ is tuned
on **dev**, and **every reported number comes from test** — no metric is quoted from the split
that selected a hyperparameter. The FAQ corpus is never split: all 7,077 entries stay
retrievable for every query, at every stage.

Comparisons carry a paired bootstrap interval rather than a bare delta. On 181 queries the
resolution limit is roughly ±0.03 MRR@10, so a smaller difference would not be evidence of
anything.

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

### API

`POST /predict` returns a stored answer with its attribution, or an honest abstention:

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json'      -d '{"question":"How long should I isolate after testing positive?"}'
```
```json
{
  "answer": "If you have confirmed or suspected COVID-19 and have symptoms, you can end home isolation when: ...",
  "matched_question": "How long do I need to isolate if I test positive for COVID-19?",
  "source": "Washington State", "trust": "official",
  "url": "https://www.doh.wa.gov/...", "score": 0.7777, "abstained": false,
  "disclaimer": "This is general information, not medical advice. ..."
}
```

Below τ it declines, and returns no source rather than attributing a safe message to a real
FAQ entry. `GET /health` reports the loaded encoder, corpus size and threshold.

Measured on the running service: **median 25 ms** per request after warm start, against a
300 ms budget. The model and embeddings load once at startup; `/predict` is a sync handler
so blocking encode work runs in Starlette's threadpool instead of stalling the event loop.
CORS is restricted to explicit origins, `/predict` is rate limited per client IP, and
internal errors return a generic 500 — the detail goes to the log, never the response.

Health questions are not written to logs. Each request records a salted hash of the query
plus its length, enough to correlate a bug report without storing what someone asked.

### Notebook and `src/` share one implementation

`src/` holds every reusable piece — download, prep, index, retriever, evaluate, api.
[`notebooks/development.ipynb`](notebooks/) **imports** them; it never keeps its own copy
of preprocessing, retrieval, or metric code. The API imports exactly the same modules.

One implementation, two consumers. This is what guarantees the numbers reported during
development are the numbers the app actually serves — the failure the old project made
unavoidable by having its notebook train BERT-QA while its API served an unrelated
DialoGPT.

The notebook is developed against a **live kernel** through the
[Jupyter MCP server](docs/jupyter-mcp.md), so every cell is written, executed and checked
in place rather than authored blind and hoped over.

## Getting started

Development is fully local. The GPU embeds the corpus in about five seconds; serving needs
no GPU at all.

```bash
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt   # Windows
```

For notebook work, start the JupyterLab instance that Claude Code's MCP server attaches
to (see [docs/jupyter-mcp.md](docs/jupyter-mcp.md)):

```bash
python scripts/start_jupyter.py
```

Pipeline:

```bash
python -m src.download          # fetch COUGH
python -m src.prep              # -> data/kb.parquet + seeded query splits + qrels
python -m src.index             # -> data/embeddings/
python -m src.evaluate          # MRR@10 / P@k / Recall / nDCG on the test split
uvicorn src.api:app --reload    # serve on :8000
pytest                          # tests
```

The Flutter client lives in [`app/`](app/):

```bash
cd app
flutter run --dart-define=API_BASE_URL=http://10.0.2.2:8000   # Android emulator
flutter run                                                    # desktop/web default
```

It renders each answer with its source, a trust badge (`official` vs `community`), and the
disclaimer — attribution the user can see, not just a field in the payload. Abstentions are
visually distinct and carry no source. See [`app/README.md`](app/README.md).

## Roadmap

Phases and their acceptance criteria are defined in [`PRD.md`](PRD.md) §18.

| Phase | | Status |
|---|---|---|
| 0 | Hygiene & scaffold | ✅ |
| 1 | Data pipeline (`download.py`, `prep.py`, splits, EDA) | ✅ |
| 2 | BM25 baseline + evaluation harness | ✅ |
| 3 | Semantic retriever + threshold/abstention | ✅ |
| 4 | Cross-encoder re-ranker (optional) | next |
| 5 | FastAPI service | ✅ |
| 6 | Flutter integration | ✅ |
| 7 | Tests + CI | ✅ |
| 8 | Docker, deploy, docs | next |

## Tests and CI

```bash
pytest                       # 163 Python tests, ~90s
cd app && flutter test       # 37 Flutter tests
```

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push:

| Job | What it guards |
|---|---|
| Python lint | `ruff check` + `format --check` |
| Python tests | 163 tests, plus a regression check on the test-split metrics |
| Flutter analyze and test | 37 tests against a mocked client |
| **Android release build** | builds a release APK and asserts `INTERNET` and the network-security config survive into the manifest |
| Repository integrity | no corrupt PNGs, no committed data or model artifacts |

The release build is not redundant. Two real defects — a missing `INTERNET` permission and
35 CRLF-corrupted PNGs — reached the repository because only debug builds were ever run;
debug skips PNG crunching and uses a different manifest. Both would fail this job.

## Not medical advice

General information only, drawn from a dated static snapshot. It does not diagnose,
triage, or personalize. For personal or urgent concerns consult a healthcare professional
or [WHO](https://www.who.int/emergencies/diseases/novel-coronavirus-2019) /
[CDC](https://www.cdc.gov/covid/).
