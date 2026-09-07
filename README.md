# Covicare — COVID-19 FAQ Chatbot

[![CI](https://github.com/Elhameed/COVID-19-CHAT-BOT/actions/workflows/ci.yml/badge.svg)](https://github.com/Elhameed/COVID-19-CHAT-BOT/actions/workflows/ci.yml)

A **retrieval-only** COVID-19 information chatbot. It matches a user's plain-language
question to the best vetted FAQ answer, returns it with its source and a medical
disclaimer, and **abstains when it isn't confident**. It never generates medical text.

A locally-run bi-encoder searches a 7,077-entry vetted FAQ corpus, evaluated on the COUGH
retrieval benchmark with a confidence interval on every claim. A Flutter app is the client;
a FastAPI service is the backend.

```
Flutter app ──POST /predict──► FastAPI
                                 ├─ retriever      MiniLM-L6 bi-encoder, cosine similarity
                                 ├─ knowledge base data/kb.parquet, 7,077 vetted FAQ entries
                                 ├─ embeddings     data/embeddings/, rebuilt by src.index
                                 └─ threshold      τ = 0.693 from artifacts/retriever_config.json
                                                   → answer + source + trust + disclaimer
                                                   → or abstention + WHO/CDC pointer
```

## Why retrieval, not generation

For a health-information bot over a few thousand curated FAQ pairs, retrieval beats
generation on every axis that matters:

| | Generation | **Retrieval (chosen)** |
|---|---|---|
| Hallucinated medical claims | possible | **impossible — answers are stored text** |
| Attribution | hard | **every answer carries its source** |
| Honest evaluation | awkward at this scale | **standard IR metrics on a real benchmark** |
| Data needed | far more than is available | **fits the corpus exactly** |

A confident wrong match is the remaining failure mode, which is what the confidence
threshold and the abstention path exist to control.

The retriever is a stock open-source encoder used as-is. The engineering here is the
pipeline around it — corpus preparation, indexing, calibration and evaluation — not the
model weights.

## Results

Measured on 181 held-out test queries against the full 7,077-entry corpus. Reproduce with
`python -m src.evaluate --split test --retriever both`.

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

### Encoder choice

Two pre-trained encoders were measured on the test split:

| | MRR@10 | P@1 | τ | in-scope answered at τ | size |
|---|---|---|---|---|---|
| **MiniLM-L6** ← shipped | 0.6128 | 0.4972 | 0.693 | **77%** | ~90 MB |
| bge-base-en-v1.5 | **0.6294** | **0.5304** | 0.782 | 58% | ~440 MB |

bge-base ranks marginally better, but MiniLM is the better *abstainer*: at equal off-topic
rejection it answers 19 points more in-scope queries, because bge's in-scope and off-topic
score distributions sit closer together. A good ranker is not automatically a good
confidence estimator, and for a bot whose worst failure is a confident wrong answer,
calibration outweighs 0.017 MRR@10. It is also 5× smaller.

## Data

**[COUGH](https://github.com/sunlab-osu/covid-faq)** — a COVID-19 FAQ retrieval benchmark
that ships with relevance judgments, which is what makes honest evaluation possible.

| File | Rows | Role |
|---|---|---|
| `FAQ_Bank_eval.csv` | 7,117 | Retrieval corpus **and** served knowledge base |
| `User_Query_Bank.csv` | 1,201 | Queries, split train/dev/test |
| `Annotated_Relevance_Set.csv` | 39,760 | Relevance judgments (qrels) |

Preparation deduplicates the bank to **7,077 entries** and remaps the affected relevance
judgments onto the surviving copies, so no positive is silently lost. Licensed
**CC BY-NC-SA 4.0**, research and education use only. Raw data is fetched, never committed
— see [`data/README.md`](data/README.md).

The content dates from ~2020–2021 and mixes WHO/CDC with community sources, so every answer
carries a source, a trust tier and a disclaimer.

## API

`POST /predict` returns a stored answer with its attribution, or an honest abstention:

```bash
curl -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"question":"How long should I isolate after testing positive?"}'
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

Below τ it declines, and returns a null source rather than attributing a safe fallback
message to a real FAQ entry. `GET /health` reports the loaded encoder, corpus size and
threshold. Interactive docs are at `/docs`.

At τ = 0.693 the bot answers 77% of in-scope queries and 0% of an off-topic probe. COUGH
contains no unanswerable queries, so that probe is 12 hand-written out-of-scope questions —
a smoke test, not a benchmark, and labelled as such wherever it is reported.

Measured on the running service: **median 25 ms** per request after warm start, against a
300 ms budget. The model and embeddings load once at startup; `/predict` is a sync handler
so blocking encode work runs in Starlette's threadpool instead of stalling the event loop.
CORS is restricted to explicit origins, `/predict` is rate limited per client IP, and
internal errors return a generic 500 — the detail goes to the log, never the response.

Health questions are not written to logs. Each request records a salted hash of the query
plus its length, enough to correlate a bug report without storing what someone asked. See
[`.env.example`](.env.example) for the runtime settings.

## Tech stack

| | |
|---|---|
| **Retrieval** | `sentence-transformers` (MiniLM-L6-v2), PyTorch, `rank-bm25`, NumPy |
| **Data** | pandas, PyArrow / Parquet, SciPy |
| **API** | Python 3.11, FastAPI, Uvicorn, Pydantic, SlowAPI |
| **Client** | Flutter / Dart, Material 3 |
| **Tooling** | pytest, ruff, JupyterLab, GitHub Actions |

## Repository layout

```
src/            all reusable logic — download, prep, index, retriever, evaluate, api
tests/          163 Python tests
app/            Flutter client (see app/README.md)
notebooks/      development.ipynb — how the retriever was built and measured
data/           generated corpus, splits and embeddings (see data/README.md)
artifacts/      retriever_config.json — encoder, field, threshold τ
```

`src/` holds every reusable piece. [`notebooks/development.ipynb`](notebooks/development.ipynb)
**imports** them; it never keeps its own copy of preprocessing, retrieval or metric code.
`src/api.py` imports exactly the same modules. One implementation, two consumers — which is
what guarantees the numbers reported during development are the numbers the app serves.

## Getting started

Development is fully local and needs no API keys. A GPU embeds the corpus in about five
seconds; serving needs no GPU at all.

```bash
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt      # Windows, CUDA torch
# python3.11 -m venv .venv && pip install -r requirements-cpu.txt  # CPU-only
```

Then build the pipeline and serve it:

```bash
python -m src.download          # fetch COUGH into data/raw/
python -m src.prep              # -> data/kb.parquet + seeded query splits + qrels
python -m src.index             # -> data/embeddings/
python -m src.evaluate          # MRR@10 / P@k / Recall / nDCG on the test split
uvicorn src.api:app --reload    # serve on :8000
```

The Flutter client lives in [`app/`](app/):

```bash
cd app
flutter run --dart-define=API_BASE_URL=http://10.0.2.2:8000   # Android emulator
flutter run                                                    # desktop/web default
```

It renders each answer with its source, a trust badge (`official` vs `community`) and the
disclaimer — attribution the user can see, not just a field in the payload. Abstentions are
visually distinct and carry no source. See [`app/README.md`](app/README.md) for the
per-platform addresses.

To open the analysis notebook:

```bash
jupyter lab notebooks/development.ipynb
```

## Tests and CI

```bash
pytest                       # 163 Python tests
cd app && flutter test       # 37 Flutter tests
```

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push:

| Job | What it guards |
|---|---|
| Python lint | `ruff check` + `format --check` |
| Python tests | 163 tests, plus a regression check on the test-split metrics |
| Flutter analyze and test | 31 widget tests against a mocked client, plus a live-API integration test that skips when no server is running |
| Android release build | builds a release APK and asserts `INTERNET` and the network-security config survive into the manifest |
| Repository integrity | no corrupt PNGs, no committed data or model artifacts |

The release build is deliberate rather than redundant: debug builds skip PNG crunching and
use a different manifest, so a release job is the only thing that catches packaging defects
before a user does.

## Limitations

- The knowledge base is a static ~2020–2021 snapshot. Guidance on variants, vaccines and
  long COVID has moved on since.
- ~44% of the corpus is community content (WikiHow and similar). It is surfaced as
  `community` rather than presented as authoritative, but it is still in the pool.
- Abstention is calibrated against a 12-query hand-written probe, not a labelled
  unanswerable-query benchmark. COUGH contains none, so this is the honest gap in the
  evaluation.
- English only.

## Not medical advice

General information only, drawn from a dated static snapshot. It does not diagnose,
triage, or personalize. For personal or urgent concerns consult a healthcare professional
or [WHO](https://www.who.int/emergencies/diseases/novel-coronavirus-2019) /
[CDC](https://www.cdc.gov/covid/).

## Attribution

> Zhang, Xinliang Frederick et al. *COUGH: A Challenge Dataset and Models for COVID-19 FAQ
> Retrieval.* Licensed CC BY-NC-SA 4.0 — research and education use only. Content
> aggregated from WHO, CDC, government health departments and community sources.
