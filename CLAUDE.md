# CLAUDE.md

Guidance for Claude Code working in this repository. Read this first, every session.

## What this project is
A **COVID-19 information chatbot** being **rebuilt from scratch** as a **retrieval-only,
fully local** system: match a user's question to the best vetted FAQ answer, attribute the
source, and abstain when unsure. It serves a pre-trained **MiniLM bi-encoder** over a
7,077-entry vetted FAQ corpus, and integrates into an existing Flutter app.

## Source of truth
**`PRD.md` is the authoritative spec.** Read it before making or suggesting changes. If this
file and the PRD ever conflict, the PRD wins. If the PRD is silent, ask before assuming.

## Hard constraints (do not violate without a recorded decision in PRD §22)
1. **Retrieval-only.** No text generation, no RAG, no training of generative/chat models. The
   retriever is a pre-trained encoder used as-is; the engineering is the pipeline around it.
2. **Fully local / open-source.** No external LLM APIs or paid services; no network calls at
   inference. Internet is used only at dev time to download base models + dataset.
3. **Grounded + attributed.** Every answer is a stored KB answer returned with its source and a
   medical disclaimer. Never invent medical facts.
4. **Honest evaluation.** Metrics come only from the COUGH **test split** via `src/evaluate.py`.
   No leakage. Never report a number you can't reproduce.
5. **Preserve numbers/units in text preprocessing** ("20 seconds", "6 feet", "14 days"). Do not
   strip digits.

## Notebook vs `src/` (workflow contract — see PRD §7.6)
- **Reusable logic lives in `src/`** (download, prep, index, retriever, evaluate, api).
- **The notebook imports from `src/`** and is where EDA, experimentation, threshold tuning and
  the final eval tables live. It must **not** keep its own copy of that logic.
- **The API imports the same `src/` modules.** One implementation, two consumers — this keeps the
  notebook's reported numbers identical to what the app serves.

## Working the notebook (live kernel — see `docs/jupyter-mcp.md`)
`notebooks/development.ipynb` is driven through the **Jupyter MCP server**, not by editing its
JSON. Prerequisite: `python scripts/start_jupyter.py` must be running.
- **Session bootstrap:** the MCP server starts with an empty token (`.mcp.json`'s
  `${JUPYTER_TOKEN}` expands against Claude Code's own environment, not `settings.local.json`),
  so the first call 403s. Read `.jupyter_token` and call `connect_to_jupyter` once, then
  `use_notebook`.
- Use the MCP tools: `use_notebook` → `read_notebook` / `insert_cell` / `edit_cell_source` /
  `execute_cell`, and `execute_code` to probe kernel state without touching the notebook.
- **The MCP server owns that file.** Do not regenerate it from a script, do not
  `jupyter execute --inplace` it, and do not edit it as JSON — RTC keeps notebook state in a
  YDoc as well as on disk, so file-level writes can be silently overwritten.
- Plots return as PNG (`ALLOW_IMG_OUTPUT=true`). Never call `matplotlib.use("Agg")` in a cell;
  it suppresses the inline backend and no image comes back.
- `use_notebook`'s `kernel_id` wants a running kernel's UUID from `list_kernels`, not a
  kernelspec name. Omit it to auto-start one.

## Do NOT
- Revive or extend the old `app.py` / `api.py` / notebook / `model/` (reference only).
- Introduce generation, RAG, or external APIs "to improve answers."
- Duplicate `src/` logic inside the notebook.
- Commit raw datasets, `__pycache__`, embeddings, or model weights.
- Widen scope into fine-tuning, generation or RAG (see PRD §23 for what is deliberately
  out of scope).
- Report metrics that `src/evaluate.py` can't reproduce, or evaluate on train/dev queries.
- Change the served corpus without keeping it aligned to the evaluated corpus.

## Key facts to keep in mind
- **Baseline to beat:** BM25 over the question field → **MRR@10 0.530, P@1 0.421** (over all 1,201
  queries; recompute on the test split for fair comparison). Question-only beats question+answer.
- **Results (test split, n=181):** BM25 0.5127 MRR@10 / 0.4033 P@1 → MiniLM **0.6128 / 0.4972**
  (+0.100 MRR@10, 95% CI [+0.044, +0.157], p=0.0004). τ = 0.693, tuned on dev.
- **Report deltas with intervals**, not bare point estimates — `src.evaluate.paired_bootstrap`.
  On 181 queries the resolution limit is roughly ±0.03 MRR@10; anything smaller is noise.
- **Corpus/eval coupling:** relevance labels reference `FAQ_Bank_eval.csv` (7,117 English items),
  so the **served KB and evaluated KB must be the same set** in v1.
- **Data is dated (~2020–2021)** and mixed-authority (WHO/CDC + community). Surface source/trust;
  disclaimer required; abstain below the confidence threshold.
- **API contract** (`POST /predict`, `/health`) and response schema are in PRD §10 — follow exactly.
- **Served artifact** = `artifacts/retriever_config.json` (encoder name + threshold τ) +
  `data/kb.parquet` + embeddings rebuilt by `src.index`. No model weights are committed; the
  encoder is fetched from Hugging Face on first use.

## Environment
- **Fully local** on a Dell Precision 5550 / **Quadro T2000 (4 GB VRAM)**, CUDA-capable.
- Install **CUDA-enabled PyTorch** (not CPU-only); confirm `torch.cuda.is_available()` is `True`.
- The GPU is used to embed the corpus (~5s for 7k entries). Serving needs no GPU.

## Working conventions
- Code in `src/`; data artifacts in `data/`; model in `artifacts/`; tests in `tests/`; app in `app/`.
- Typical commands (align to PRD §17 as files are created):
  - `python -m src.prep` — build the KB + query splits from COUGH
  - `python -m src.evaluate` — MRR / P@k / Recall / nDCG on the test split
  - `uvicorn src.api:app --reload` — serve the API
  - `pytest` — run tests
- **Phase gating:** follow PRD §18 in order; don't advance until acceptance criteria are met.
  State which phase you're in.
- When unsure between options, follow the **recommendations in PRD §22** unless told otherwise.

## Current status
Phases 0–3 complete on branch `rebuild/retrieval-v1`, one commit per phase. The retrieval
pipeline is finished and measured; **next is Phase 4 (optional cross-encoder re-ranker),
then Phase 5 (FastAPI)**. `src/api.py` and the Flutter rewiring do not exist yet.
Follow PRD §18 in order and state which phase you're in.

## Before finishing any change
- Does it respect the five hard constraints?
- Is any new metric reproducible via `src/evaluate.py` on the test split?
- Are source attribution and the disclaimer preserved in responses?
- Did the notebook import from `src/` rather than duplicate logic?
- Did you avoid committing data/cache artifacts?
