# COVID-19 Chatbot — Product Requirements Document (PRD)

| | |
|---|---|
| **Project** | COVID-19 Chatbot (clean rebuild) |
| **Status** | Approved direction — pre-implementation |
| **Document type** | Source of truth for implementation (read before making changes) |
| **Last updated** | 2026-08-14 |

---

## 0. How to use this document (read first)

This PRD defines a **ground-up rebuild** of the COVID-19 Chatbot. It supersedes the old
notebook, the old `app.py`/`api.py`, and the old model artifacts. Those exist in git history
as **reference only** — do **not** revive, patch, or extend them.

**Hard constraints for all work on this repo. Do not violate without an explicit decision recorded in §22:**

1. **Retrieval-only.** The chatbot answers by retrieving vetted FAQ answers from a fixed
   knowledge base. It does **not** generate free-form text and does **not** fine-tune a
   generative/chat model. **Fine-tuning the _retriever_ (bi-encoder, optionally a cross-encoder)
   is in scope and is the chosen approach — see §7.4.** No RAG in v1.
2. **Fully local / open-source.** All development, training, and inference run locally on
   open-source models. No external LLM APIs, no paid services, no network calls at inference time.
   (Internet is used at dev time only, to download base models and the dataset once.)
3. **Grounded + attributed.** Every answer is a stored answer from the knowledge base, returned
   with its source and a medical disclaimer. The bot never invents medical facts.
4. **Honest evaluation.** Metrics come from the held-out COUGH query/relevance benchmark, on a
   fixed test split. No train/test leakage. Never report a metric that can't be reproduced by
   `src/evaluate.py`.
5. **Preserve numbers and units in text preprocessing** ("20 seconds", "6 feet", "14 days" are
   load-bearing in health content — the old pipeline stripped digits; do not repeat that).
 
---

## 1. Project overview

### 1.1 Purpose
Build a reliable COVID-19 information chatbot that answers users' plain-language questions with
accurate, source-attributed answers drawn from trusted public-health FAQ content, and integrate
it into the existing Flutter mobile app.

### 1.2 Objective
Replace the previous, non-functional implementation with a clean, maintainable, technically
defensible retrieval system whose quality is measured on a real benchmark — suitable as a
portfolio-grade project that stands up to technical review.

### 1.3 Background / what is changing
The original project attempted to fine-tune generative/extractive models (DialoGPT, BERT-QA) on
~110 unique QA pairs duplicated into a ~24k-row file, producing leaked, meaningless metrics and a
backend that no longer runs (weights were git-ignored). The rebuild uses **semantic FAQ
retrieval** and **fine-tunes the retriever** on a proper held-out benchmark, which fits the data
scale, the factual/health domain, and the need for honest evaluation.

---

## 2. Problem statement
People need quick, trustworthy answers to common COVID-19 questions (symptoms, transmission,
prevention, testing, vaccines, isolation) without wading through long documents or unreliable
sources. A generative model risks hallucinating medical guidance. The problem is therefore framed
as: **given a user's question, return the most relevant vetted FAQ answer, or abstain if no good
match exists.**

---

## 3. Goals and non-goals

### Goals
- Accurately match a user's free-text question to the best FAQ entry in the knowledge base.
- Return the stored answer with its source and a disclaimer; abstain when confidence is low.
- Fine-tune a retriever that beats both a BM25 lexical baseline **and** the off-the-shelf encoder
  on a held-out test split.
- Ship a clean FastAPI backend and integrate it into the existing Flutter app.
- Be fully reproducible and fully local: one command each to prep data, train, evaluate, serve.

### Non-goals (v1)
- No answer generation / summarization / paraphrasing of answers.
- No multi-turn dialogue memory or conversational state.
- No medical diagnosis, triage, or personalized advice.
- No multilingual support (English only in v1).
- No live data feeds; the knowledge base is a static, dated snapshot.

---

## 4. Target users and use cases
**Primary users:** general public seeking quick COVID-19 information via the mobile app.

**Representative use cases:**
- "How does COVID-19 spread?" → returns transmission FAQ answer + source.
- "How long should I isolate after testing positive?" → returns isolation guidance + source.
- "Do vaccines stop transmission?" → returns best-matched vetted answer + source.
- Off-topic or unanswerable ("What's the weather?") → bot abstains and points to official sources.

---

## 5. Product functionality
1. Accept a free-text question from the app.
2. Retrieve the top-k relevant FAQ entries from the knowledge base (semantic similarity).
3. (Optional) Re-rank the top-k for precision.
4. If the top score ≥ threshold τ: return the stored answer, its source, and the matched question.
5. If below τ: return a safe abstention message directing users to WHO/CDC.
6. Always attach a "not medical advice" disclaimer and the source/trust label.
7. Expose a health endpoint for readiness checks.

---

## 6. Data

### 6.1 Requirements
- Curated question→answer pairs about COVID-19 from credible sources.
- An evaluation set of realistic user queries with relevance judgments (for honest metrics).
- English text. Redistribution/licensing terms understood and respected.

### 6.2 Primary source — COUGH (anchor dataset)
Repo: `github.com/sunlab-osu/covid-faq` (clone in the data-prep step; do not commit raw dumps).
COUGH is purpose-built for COVID-19 FAQ **retrieval** and ships with an evaluation set.

| File | Rows | Role |
|---|---|---|
| `FAQ_Bank.csv` | 15,919 (multilingual; ~9,151 English) | Full FAQ pool |
| `FAQ_Bank_eval.csv` | 7,117 (all English) | **Retrieval corpus** the relevance labels reference |
| `User_Query_Bank.csv` | 1,201 | Paraphrased user queries (split into train/dev/test) |
| `Annotated_Relevance_Set.csv` | 39,760 judgments (7,856 positive) | **Qrels** (query → relevant FAQ indices) |

Sources include WHO, CDC, and state health departments (higher trust) alongside
WikiHow / Inspire / MedHelp (lower trust). Retain source authority as a **trust signal**.

**Corpus/eval coupling (important):** relevance labels reference the row `index` of
`FAQ_Bank_eval.csv` (0–7116). The **served corpus and the evaluated corpus must be the same set**
in v1 → use `FAQ_Bank_eval.csv` (7,117 English items) as the knowledge base.

### 6.3 Supplementary sources (optional, evaluate before adding)
- **deepset COVID-QA** (`github.com/deepset-ai/COVID-QA`): 2,019 expert pairs, SQuAD-style
  extractive over scientific papers — format mismatch; consider only if useful.
- **CShorten/CDC-COVID-FAQ** (Hugging Face): clean CDC FAQ Q→A; tidy supplement.
Do not add supplements in v1 if they compromise the clean COUGH eval mapping.

### 6.4 Preprocessing and preparation (`src/prep.py`)
Deterministic, scripted, reproducible:
1. Load `FAQ_Bank_eval.csv`; keep `index, question, answer, source, url, language`.
2. Filter to `language == en`.
3. Normalize whitespace/encoding; strip HTML if present. **Keep digits, units, and meaningful
   punctuation.**
4. Drop exact/near-duplicate FAQ entries (record how many); retain an `index` mapping so qrels
   remain valid after dedup.
5. Add a `trust` tier from `source` (`official` = WHO/CDC/gov, `community` = other).
6. Emit `data/kb.parquet` (normalized knowledge base) and the query split + qrels artifacts.

### 6.5 Knowledge base schema
```
id: int            # stable KB id (maps to COUGH FAQ_Bank_eval index)
question: str
answer: str
source: str        # e.g. "World Health Organization"
url: str
trust: str         # "official" | "community"
```

---

## 7. ML / NLP approach

### 7.1 Chosen approach and rationale
**Semantic FAQ retrieval** (dense bi-encoder nearest-neighbour search over the KB), with a BM25
lexical baseline and an optional cross-encoder re-ranker, and with the **bi-encoder fine-tuned**
on COUGH (Option B). Rationale:
- **Data scale:** thousands of curated pairs suit retrieval; too few to fine-tune a reliable
  generator (the original failure mode).
- **Safety:** retrieval returns vetted text → no hallucinated medical claims.
- **Evaluability:** retrieval has clean, leak-resistant metrics on the COUGH benchmark.
- **Locality:** small open-source encoders train and run locally with no API dependency.

### 7.2 Components
- **Baseline — BM25** (`rank_bm25`): lexical retrieval; first number to beat.
- **Semantic retriever — bi-encoder** (`sentence-transformers`): embed KB questions once; embed
  the user query at inference; cosine/inner-product nearest neighbour.
- **Re-ranker — cross-encoder (optional):** re-score the bi-encoder's top-k for precision. Still
  "retrieval" (no generation). Recommended **after** the fine-tuned bi-encoder lands.

### 7.3 Model choices (local, open-source)
- Bi-encoder default: `sentence-transformers/all-MiniLM-L6-v2` (~22M params, ~80 MB, 384-dim).
- Bi-encoder accuracy option: `BAAI/bge-base-en-v1.5` (~110M) or `intfloat/e5-base-v2`
  (note: e5/bge require `query:` / `passage:` prefixes — implement correctly).
- Cross-encoder (optional): `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Index: exact search over ~7k vectors is trivial (NumPy / `sentence-transformers` util);
  `faiss-cpu` optional and not required at this scale.

### 7.4 Training / fine-tuning strategy (Option B — chosen)
The **bi-encoder retriever is fine-tuned** on COUGH and exported as a real model artifact.
- **Data:** pair each training query with its positive FAQ entries (from qrels). Use in-batch
  negatives via `MultipleNegativesRankingLoss` (no manual negative mining needed).
- **Split:** partition the 1,201 queries into **train / dev / test** (e.g. 70 / 15 / 15), fixed
  and seeded. Train on train, tune τ and early-stop on dev, report final metrics on test only.
  The FAQ corpus stays full (7,117) for retrieval at all times.
- **Export:** `model.save("artifacts/encoder")` — this fine-tuned encoder folder **is** the
  trained-model artifact loaded by the API (this is the notebook-trains-and-exports flow).
- **Optional second step:** fine-tune the cross-encoder re-ranker on the same splits.
- **Gate:** the fine-tuned encoder must beat the off-the-shelf encoder on the test split, and both
  must beat BM25, or the fine-tune is not adopted (record the decision).
- **Do not** fine-tune a generative model.

### 7.5 Confidence threshold and abstention
- Tune similarity threshold τ on the **dev** split.
- Top score ≥ τ → answer; below τ → abstain with a fixed safe message + WHO/CDC pointer.
- Log score and abstention decisions for evaluation and debugging.

### 7.6 Notebook ↔ `src/` relationship (development workflow)
This is the workflow contract; follow it exactly.
- **The notebook (`notebooks/`) is where ML development is done and presented:** EDA, preprocessing
  trials, BM25 vs off-the-shelf vs fine-tuned comparisons, **running the fine-tune**, threshold
  tuning, and the final evaluation tables/plots.
- **Reusable logic lives in `src/`**, not in the notebook. The notebook **imports** it
  (`from src.prep import build_kb`, `from src.train import finetune`, `from src.evaluate import score`)
  and orchestrates/experiments. It must not hold its own private copy of preprocessing, retrieval,
  training, or metric code.
- **The API imports the same `src/` modules.** One shared implementation, two consumers (notebook
  for experimentation, API for serving) — this is what guarantees the reported numbers match what
  the app actually does.
- Prototype messily in the notebook first; once a piece stabilizes, lift it into a `src/` function
  and have the notebook import it.

---

## 8. Evaluation methodology

### 8.1 Benchmark
COUGH: corpus = `FAQ_Bank_eval.csv` (7,117), queries = `User_Query_Bank.csv` (1,201, split
train/dev/test), qrels = `Annotated_Relevance_Set.csv`. Implemented in `src/evaluate.py`.

### 8.2 Metrics
Primary: **MRR@10** and **P@1**. Secondary: P@3, P@5, Recall@10, nDCG@10.
(Recall is naturally low because queries average ~6.5 positives; lead with MRR/P@k.)

### 8.3 Measured BM25 baseline (reference — computed on all 1,201 queries)
| Retriever | MRR@10 | P@1 | P@3 | P@5 |
|---|---|---|---|---|
| **BM25 over question** | **0.530** | **0.421** | 0.267 | 0.211 |
| BM25 over question+answer | 0.469 | 0.344 | 0.244 | 0.199 |

Finding: indexing the **question field alone** outperforms question+answer — use question-only as
the default text field unless evaluation shows otherwise.

> Note: these numbers are over all 1,201 queries. Once the train/dev/test split is fixed,
> **recompute BM25 and the off-the-shelf encoder on the test split** so all methods are compared
> apples-to-apples on the same held-out queries.

### 8.4 Acceptance targets
- Off-the-shelf bi-encoder must **beat BM25** on MRR@10 and P@1 (test split).
- Fine-tuned bi-encoder must **beat the off-the-shelf** encoder on MRR@10 and P@1 (test split),
  else it is not adopted.
- Every reported number must be reproducible via `src/evaluate.py` from committed artifacts.

### 8.5 Anti-leakage rules
- Corpus, queries, and qrels are separate.
- Queries are split train/dev/test with a fixed seed; **evaluate only on test**; never tune or
  train on test queries.
- No FAQ entry may be duplicated across the corpus in a way that inflates matches.

---

## 9. System architecture and components
```
Flutter app ──HTTP POST /predict──► FastAPI backend
                                      │
                                      ├─ Retriever (BM25 | fine-tuned bi-encoder + optional re-ranker)
                                      ├─ Knowledge base (kb.parquet) + precomputed embeddings
                                      ├─ Fine-tuned encoder (artifacts/encoder/)
                                      └─ Threshold/abstention + disclaimer + source attribution
                                      ▼
                              JSON response ──► rendered in chat UI
```
Major components: data-prep pipeline, training routine, knowledge base + embedding index,
retriever module, evaluation harness, FastAPI service, Flutter client.

---

## 10. Backend / API requirements

### 10.1 Endpoints
- `GET /health` → `{ "status": "ok" }`.
- `POST /predict` → main query endpoint.
- `GET /` (optional) → basic service/info page.

### 10.2 `/predict` contract
Request:
```json
{ "question": "how does covid spread?" }
```
Response:
```json
{
  "answer": "…stored FAQ answer…",
  "matched_question": "How is COVID-19 transmitted?",
  "source": "World Health Organization",
  "trust": "official",
  "url": "https://…",
  "score": 0.71,
  "abstained": false,
  "disclaimer": "This is general information, not medical advice. For personal or urgent concerns, consult a healthcare professional or official sources (WHO/CDC)."
}
```
Abstention returns `abstained: true`, a fixed safe `answer`, and no misleading source.

### 10.3 Behavior / non-functional
- Load fine-tuned encoder + embeddings once at startup; keep in memory.
- Input validation: non-empty, length cap, reject malformed input.
- Proper CORS (explicit allowed origins — not `*` with credentials).
- Basic rate limiting.
- Structured logging (query, top score, abstained). No PII.
- Target added latency < ~300 ms after warm start on CPU for this corpus size.

---

## 11. Frontend / application requirements
Reuse the existing Flutter app, fixing the known defects:
- Make the API base URL **configurable** (`--dart-define` / env), not hardcoded `127.0.0.1`.
  Document emulator (`10.0.2.2`) vs physical-device (LAN IP) vs deployed URL.
- Configure cleartext/HTTPS correctly (Android `usesCleartextTraffic` / network security config;
  iOS ATS) — prefer HTTPS in deployed builds.
- Parse the new response schema; render answer + source + disclaimer; show abstention clearly.
- UX: loading indicator, auto-scroll to newest message, error/empty states, and wire up or remove
  the non-functional menu button.
- Rebrand: remove leftover `agri_chatbot` identifiers (pubspec name, Android label, package
  `com.example.agri_chatbot`).
- Replace the default counter `widget_test.dart` with a real test.

---

## 12. Integration with existing application
- Contract-first: the Flutter `api_service` targets `POST /predict` and the schema in §10.2.
- Local dev: run FastAPI on `:8000`; point the app at the correct host per platform.
- The disclaimer and source must be visible in the UI, not just in the payload.
- Keep the app and API decoupled; the app holds no model logic.

---

## 13. Expected user flow
1. User opens app → sees chat with a one-line scope/disclaimer.
2. User types a question → app shows it + a loading state.
3. App POSTs to `/predict`.
4. Backend retrieves (→ optional re-rank → threshold check).
5. Response returns: answer + source + disclaimer, **or** an abstention message.
6. UI renders it, attributes the source, auto-scrolls. User can ask again.

---

## 14. Functional requirements
- FR1: Prep pipeline produces a deterministic, deduplicated English KB from COUGH.
- FR2: Query split (train/dev/test) is fixed and seeded; evaluation runs only on test.
- FR3: Evaluation harness computes MRR/P@k/Recall/nDCG from committed artifacts.
- FR4: BM25 baseline reproduces the numbers in §8.3.
- FR5: Off-the-shelf bi-encoder retriever returns top-k with scores and beats BM25 on test.
- FR6: Fine-tuned bi-encoder beats the off-the-shelf encoder on test and is exported to
  `artifacts/encoder/`.
- FR7: Threshold/abstention behaves per §7.5.
- FR8: API implements §10 contract, including `/health`.
- FR9: App sends questions and renders answer + source + disclaimer, handling abstention/errors.
- FR10: Every answer includes source attribution and disclaimer.

## 15. Non-functional requirements
- NFR1 Reproducibility: `prep → train → evaluate → serve` each run by a single documented command.
- NFR2 Maintainability: typed, modular `src/`; notebook imports from `src/`; no duplicated logic;
  no committed model binaries/caches.
- NFR3 Performance: in-memory index; sub-second responses on CPU for ~7k docs.
- NFR4 Reliability: graceful handling of empty/garbage input and no-match cases.
- NFR5 Security: input validation, scoped CORS, rate limiting, no secrets in repo.
- NFR6 Portability: runs locally and in Docker with pinned dependencies.
- NFR7 Transparency: sources and confidence surfaced; abstention is honest.

---

## 16. Technology stack

### 16.1 Development environment
- **Fully local.** Dev machine: Dell Precision 5550, NVIDIA Quadro T2000 (4 GB VRAM), CUDA-capable.
- Install the **CUDA-enabled PyTorch** build (not CPU-only); verify `torch.cuda.is_available()`.
- Training is lightweight (small encoders, ~1.2k queries): expect a few minutes per fine-tune on
  the GPU; CPU fallback works but is slower. Mixed precision (fp16) optional for `bge-base`.
- Internet needed only at dev time to download base models + dataset; inference is offline.

### 16.2 Stack
- **Backend:** Python 3.11+, FastAPI, Uvicorn.
- **Retrieval/NLP:** PyTorch (CUDA), `sentence-transformers`, `rank_bm25`, `scikit-learn`, NumPy,
  pandas; `faiss-cpu` optional.
- **Frontend:** Flutter / Dart, `http`.
- **Tooling:** `pytest`, `ruff`/`black`, pinned `requirements.txt`, Docker, GitHub Actions CI.

---

## 17. Project structure
```
covid-chatbot/
├── data/
│   ├── kb.parquet          # generated knowledge base artifact
│   ├── splits/             # seeded train/dev/test query ids (committed; tiny)
│   └── raw/                # COUGH working copy (never committed)
├── artifacts/
│   └── encoder/            # exported fine-tuned bi-encoder (the trained-model artifact)
├── src/
│   ├── download.py         # fetch COUGH from GitHub (no raw dumps committed)
│   ├── prep.py             # clean/filter/dedupe COUGH -> kb.parquet + query splits/qrels
│   ├── train.py            # fine-tune the bi-encoder (Option B) -> artifacts/encoder/
│   ├── retriever.py        # BM25 + bi-encoder (+ optional cross-encoder) behind one interface
│   ├── index.py            # build/load KB embeddings
│   ├── evaluate.py         # MRR / P@k / Recall / nDCG on the COUGH test split
│   └── api.py              # FastAPI app: /predict, /health, threshold, disclaimer
├── notebooks/
│   └── development.ipynb   # EDA + experiments + training run + eval; imports from src/
│                           # driven via the Jupyter MCP server (docs/jupyter-mcp.md)
├── scripts/
│   └── start_jupyter.py    # JupyterLab the MCP server attaches to
├── docs/
│   └── jupyter-mcp.md      # live-kernel notebook workflow
├── .mcp.json               # project-scoped MCP config (committed; no secrets)
├── tests/
│   ├── test_retriever.py
│   ├── test_evaluate.py
│   └── test_api.py
├── app/                    # existing Flutter app (renamed, fixed)
├── requirements.txt
├── Dockerfile
├── .gitignore              # ignore caches, raw data, large embeddings/model files as needed
├── CLAUDE.md
├── PRD.md
└── README.md               # describes what actually exists, with real numbers
```

---

## 18. Development phases and roadmap
Each phase has explicit acceptance criteria; do not advance until met.

- **Phase 0 — Hygiene & scaffold.** New structure, pinned `requirements.txt` (CUDA torch),
  `.gitignore`, remove old cruft (`__pycache__`, tfevents, orphaned `.pyc`), rebrand `agri_chatbot`.
  *Done when:* repo builds clean; structure matches §17; `torch.cuda.is_available()` is `True`.
- **Phase 1 — Data pipeline.** `download.py` + `prep.py` → `kb.parquet`, seeded query splits, qrels; EDA.
  *Done when:* KB is English/deduped/schema §6.5; splits fixed; dedup/leakage documented.
- **Phase 2 — Baseline + eval harness.** BM25 + `evaluate.py`.
  *Done when:* reproduces §8.3 on all queries and reports BM25 on the test split.
- **Phase 3 — Off-the-shelf semantic retriever.** Bi-encoder + index + threshold/abstention.
  *Done when:* beats BM25 on MRR@10 and P@1 (test); abstention works.
- **Phase 4 — Fine-tune the retriever (Option B).** `src/train.py`, run from the notebook; export
  `artifacts/encoder/`. *Done when:* beats the off-the-shelf encoder on test, or is dropped with rationale.
- **Phase 5 — Re-ranker (optional).** Cross-encoder over top-k.
  *Done when:* measurable P@1 gain, or explicitly dropped.
- **Phase 6 — API.** FastAPI per §10, loading `artifacts/encoder/` + embeddings.
  *Done when:* `/predict` + `/health` meet the contract; validation/CORS/rate-limit in place.
- **Phase 7 — Flutter integration.** Configurable URL, cleartext/ATS, new schema, UX fixes, rebrand.
  *Done when:* app talks to API on emulator + device; renders answer/source/disclaimer/abstention.
- **Phase 8 — Testing + CI.** Python + Flutter tests; GitHub Actions.
  *Done when:* CI green on lint + tests for both.
- **Phase 9 — Deploy + docs.** Dockerize; deploy API; rewrite README with real numbers + demo.
  *Done when:* reproducible container; honest README; working demo path.

---

## 19. Testing requirements
- Unit: retriever ranking, metric functions (tiny fixtures), prep/dedup logic, split determinism.
- Integration: `/predict` happy path, abstention path, malformed input; `/health`.
- Regression: committed test-split metrics; CI fails if MRR/P@1 drop beyond a tolerance.
- Flutter: a real widget test (send question → mocked response rendered); remove default counter test.
- CI: GitHub Actions runs Python lint+tests and Flutter analyze+test.

---

## 20. Deployment considerations
- The fine-tuned encoder (`artifacts/encoder/`) and precomputed KB embeddings are produced locally
  and loaded at startup; runtime needs no internet.
- Containerize the API with pinned deps; expose `/health`.
- Decide whether to vendor `artifacts/encoder/` in git (small) or fetch it as a build artifact.
- Hosting: any container host (Render / Railway / Fly / self-host); keep it free-tier friendly.
  Note: hosted CPU inference is fine at this corpus size — no GPU needed to serve.
- Flutter: document build/run for Android and iOS; point at the deployed HTTPS API.
- Keep raw datasets and large caches out of git (download step / artifacts).

---

## 21. Limitations, risks, and ethical considerations
- **Data recency:** COUGH content is ~2020–2021; guidance on variants, vaccines, and long COVID
  may be outdated. Answers must be dated/sourced and carry a disclaimer.
- **Source quality is mixed:** WHO/CDC alongside community sources. Surface trust tier; prefer
  official; never present community content as authoritative.
- **Not medical advice:** general information only; no diagnosis/triage/personalization. Disclaimer
  on every answer; abstain when unsure.
- **Retrieval errors:** a confident wrong match is the main failure mode → threshold + abstention
  exist to reduce it; tune conservatively.
- **Licensing:** COUGH aggregates third-party content; review its license file and source terms and
  comply with attribution/redistribution before publishing.
- **Scope creep to generation/RAG:** explicitly deferred; would reintroduce hallucination risk.

---

## 22. Assumptions and settled/open decisions
**Assumptions:** English-only v1; static dated KB acceptable; existing Flutter app is the target
client; fully local development on the Quadro T2000; CPU-only inference is sufficient to serve.

**Settled decisions:**
- **Approach = Option B:** fine-tune the retriever and export the encoder artifact (§7.4).
- **Environment = fully local** on the Dell Precision 5550 / Quadro T2000 (§16.1).
- **KB scope = COUGH eval bank only** (7,117) in v1, to keep the eval mapping clean.
- **`download.py` lives in `src/`, not `data/`** (2026-08-14). Every module is then
  reachable as `python -m src.*`, and `data/` stays purely generated output. §17 updated.
- **Milestone commits per phase** (2026-08-14) on branch `rebuild/retrieval-v1`, so each
  phase boundary is a rollback point.
- **Notebook development runs against a live kernel via the Jupyter MCP server**
  (2026-08-14). Claude Code edits, executes and inspects `notebooks/development.ipynb`
  through `datalayer/jupyter-mcp-server`, rather than rewriting the `.ipynb` JSON blind.
  Consequences, detailed in `docs/jupyter-mcp.md`:
  - **The MCP server owns the notebook.** Generating it from a script and running
    `jupyter execute --inplace` (the Phase 1 method) is retired: with RTC enabled the
    notebook also lives in a YDoc CRDT, and file-level writes can be overwritten or
    diverge. `src/` modules are still edited as ordinary files.
  - **Config is project-scoped and committed** (`.mcp.json`), holding no secret — it
    expands `${JUPYTER_TOKEN}` from the environment. The token lives only in the
    gitignored `.claude/settings.local.json`.
  - **`jupyter-collaboration==5.0.0` is in `requirements.txt`** (RTC is server-side); the
    MCP server itself is deliberately *not*, since its unpinned `fastapi`/`uvicorn` would
    break the §16 pins. `uvx` runs it in an isolated environment.

- **Bi-encoder = `all-MiniLM-L6-v2`** (2026-08-15, resolves open decision #2). Both encoders
  were measured off-the-shelf on the test split in Phase 3:

  | | MRR@10 | P@1 | τ | in-scope answered at τ | size |
  |---|---|---|---|---|---|
  | MiniLM-L6 | 0.6128 | 0.4972 | 0.693 | **77%** | ~90 MB |
  | bge-base-en-v1.5 | **0.6294** | **0.5304** | 0.782 | 58% | ~440 MB |

  bge-base ranks better by +0.017 MRR@10, but MiniLM wins on the three things that
  decide it: (a) it is the better *abstainer* — at equal off-topic rejection it answers
  19 points more in-scope queries, because bge's score distributions sit closer together;
  (b) at ~90 MB the exported encoder can be vendored in git, which keeps the repo
  clone-and-run and is the exact failure this rebuild exists to eliminate; (c) 384-dim /
  22M params allows far larger fine-tuning batches on 4 GB VRAM, and
  `MultipleNegativesRankingLoss` draws its negatives in-batch, so batch size is a direct
  quality lever in Phase 4. Revisit if the fine-tune fails to beat 0.6128.
- **Vendor `artifacts/encoder/` in git** (2026-08-15, resolves #5), which follows from the
  MiniLM choice: ~90 MB is small enough, and `.gitignore` already carries the exception.
- **Index = exact NumPy** (resolves #3), implemented in `src/index.py`. At ~7k vectors a
  brute-force product is sub-millisecond and exact; FAISS not adopted.

**Open decisions (record the choice here before implementing the affected phase):**
1. **Cross-encoder re-ranker** now vs. after the fine-tuned bi-encoder. *Recommendation: after Phase 4.*
2. **Supplementary data** (deepset / CDC-FAQ): include or not. *Recommendation: not in v1.*
3. **Deployment target** for the public demo. *Recommendation: decide at Phase 9.*
4. **A labelled unanswerable-query set** to validate abstention properly. Phase 3 found that
   COUGH contains no unanswerable queries, so τ is currently tuned against a 12-query
   hand-written probe (`src.evaluate.OFF_TOPIC_PROBE`) rather than a benchmark. *Known gap;
   out of scope for v1.*

---

## 23. Appendix
- **Old artifacts (reference only):** previous `app.py` (DialoGPT), `api.py` (Keras intents),
  `notebook/` (BERT-QA), old `model/`. Do not revive.
- **Datasets:** COUGH `github.com/sunlab-osu/covid-faq`; deepset COVID-QA
  `github.com/deepset-ai/COVID-QA`; CDC-COVID-FAQ (Hugging Face `CShorten/CDC-COVID-FAQ`).
- **Glossary:** *bi-encoder* (embeds query and doc separately for fast search); *cross-encoder*
  (scores a query–doc pair jointly, more accurate, slower); *MultipleNegativesRankingLoss*
  (contrastive loss using other in-batch examples as negatives); *MRR* (mean reciprocal rank of
  first relevant hit); *P@k* (precision in top k); *qrels* (query relevance judgments);
  *abstention* (declining to answer below a confidence threshold).
