# CLAUDE.md

Guidance for Claude Code working in this repository. Read this first, every session.

## What this project is
A **COVID-19 information chatbot** being **rebuilt from scratch** as a **retrieval-only,
fully local** system: match a user's question to the best vetted FAQ answer, attribute the
source, and abstain when unsure. The retriever (bi-encoder) is **fine-tuned** on the COUGH
benchmark (Option B) and exported as a model artifact. It integrates into an existing Flutter app.

## Source of truth
**`PRD.md` is the authoritative spec.** Read it before making or suggesting changes. If this
file and the PRD ever conflict, the PRD wins. If the PRD is silent, ask before assuming.

## Hard constraints (do not violate without a recorded decision in PRD §22)
1. **Retrieval-only.** No text generation, no RAG, no fine-tuning of generative/chat models.
   Fine-tuning the **retriever** (bi-encoder, optionally cross-encoder) is the chosen approach.
2. **Fully local / open-source.** No external LLM APIs or paid services; no network calls at
   inference. Internet is used only at dev time to download base models + dataset.
3. **Grounded + attributed.** Every answer is a stored KB answer returned with its source and a
   medical disclaimer. Never invent medical facts.
4. **Honest evaluation.** Metrics come only from the COUGH **test split** via `src/evaluate.py`.
   No leakage. Never report a number you can't reproduce.
5. **Preserve numbers/units in text preprocessing** ("20 seconds", "6 feet", "14 days"). Do not
   strip digits.

## Notebook vs `src/` (workflow contract — see PRD §7.6)
- **Reusable logic lives in `src/`** (prep, train, retriever, index, evaluate, api).
- **The notebook imports from `src/`** and is where experimentation, the training run, tuning, and
  the final eval tables live. It must **not** keep its own copy of that logic.
- **The API imports the same `src/` modules.** One implementation, two consumers — this keeps the
  notebook's reported numbers identical to what the app serves.

## Do NOT
- Revive or extend the old `app.py` / `api.py` / notebook / `model/` (reference only).
- Introduce generation, RAG, or external APIs "to improve answers."
- Duplicate `src/` logic inside the notebook.
- Commit raw datasets, `__pycache__`, tfevents, or large caches. (Model artifact: see PRD §22 #5.)
- Report metrics that `src/evaluate.py` can't reproduce, or evaluate on train/dev queries.
- Change the served corpus without keeping it aligned to the evaluated corpus.

## Key facts to keep in mind
- **Baseline to beat:** BM25 over the question field → **MRR@10 0.530, P@1 0.421** (over all 1,201
  queries; recompute on the test split for fair comparison). Question-only beats question+answer.
- **Acceptance chain:** off-the-shelf bi-encoder must beat BM25; the fine-tuned encoder must beat
  off-the-shelf; all on the **test split**, else the fine-tune isn't adopted.
- **Corpus/eval coupling:** relevance labels reference `FAQ_Bank_eval.csv` (7,117 English items),
  so the **served KB and evaluated KB must be the same set** in v1.
- **Data is dated (~2020–2021)** and mixed-authority (WHO/CDC + community). Surface source/trust;
  disclaimer required; abstain below the confidence threshold.
- **API contract** (`POST /predict`, `/health`) and response schema are in PRD §10 — follow exactly.
- **Trained-model artifact** = `artifacts/encoder/` (exported fine-tuned bi-encoder) + `data/kb.parquet`
  + precomputed embeddings + config (encoder name, threshold τ).

## Environment
- **Fully local** on a Dell Precision 5550 / **Quadro T2000 (4 GB VRAM)**, CUDA-capable.
- Install **CUDA-enabled PyTorch** (not CPU-only); confirm `torch.cuda.is_available()` is `True`.
- Fine-tuning is lightweight (~minutes on GPU). Mixed precision optional for `bge-base`.

## Working conventions
- Code in `src/`; data artifacts in `data/`; model in `artifacts/`; tests in `tests/`; app in `app/`.
- Typical commands (align to PRD §17 as files are created):
  - `python -m src.prep` — build the KB + query splits from COUGH
  - `python -m src.train` — fine-tune the bi-encoder → `artifacts/encoder/`
  - `python -m src.evaluate` — MRR / P@k / Recall / nDCG on the test split
  - `uvicorn src.api:app --reload` — serve the API
  - `pytest` — run tests
- **Phase gating:** follow PRD §18 in order; don't advance until acceptance criteria are met.
  State which phase you're in.
- When unsure between options, follow the **recommendations in PRD §22** unless told otherwise.

## Current status
Pre-implementation. Next: **Phase 0 (hygiene & scaffold)** → **Phase 1 (data pipeline)** →
**Phase 2 (BM25 baseline + eval harness)**. Do not jump ahead to training, API, or app work.

## Before finishing any change
- Does it respect the five hard constraints?
- Is any new metric reproducible via `src/evaluate.py` on the test split?
- Are source attribution and the disclaimer preserved in responses?
- Did the notebook import from `src/` rather than duplicate logic?
- Did you avoid committing data/cache artifacts?
