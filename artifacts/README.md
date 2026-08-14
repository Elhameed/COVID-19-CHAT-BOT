# artifacts/

The **trained-model artifact**. Produced by Phase 4, consumed by the API.

```
artifacts/
└── encoder/          # fine-tuned bi-encoder, exported via model.save(...)
```

## What this is

`artifacts/encoder/` is the bi-encoder fine-tuned on the COUGH training split
(PRD §7.4, Option B). It is written by:

```bash
python -m src.train        # or run from notebooks/development.ipynb
```

and loaded at startup by `src/api.py`. Together with `data/kb.parquet`, the
precomputed embeddings, and the config (base encoder name, threshold τ), it is
what the deployed service actually runs.

## Why it *is* committed, unlike everything else binary

Almost every other binary in this project is gitignored. This one is deliberately
not — see the exception block in [`.gitignore`](../.gitignore).

The previous implementation's entire `.gitignore` was one line, `*.safetensors`.
The fine-tuned weights therefore never reached the repository, and
`AutoModelForCausalLM.from_pretrained("./model")` raised `OSError` on every fresh
clone. The backend could not start for anyone, including its author.

So: the exported encoder is a **deliverable**, not a cache. Training scratch
(`checkpoint-*/`, `optimizer.pt`, `scheduler.pt`) *is* a cache and stays ignored.

Per PRD §22 #5, vendoring holds while the artifact is small — MiniLM exports at
~90 MB. If we adopt `bge-base` (~440 MB), switch to a release asset or Git LFS
and update the ignore rules together with this note.

## Regenerating

Nothing here is hand-edited. To rebuild from scratch:

```bash
python -m src.download    # fetch COUGH
python -m src.prep         # -> data/kb.parquet + seeded query splits + qrels
python -m src.train        # -> artifacts/encoder/
python -m src.evaluate     # metrics on the test split
```

The fine-tune is only adopted if it beats the off-the-shelf encoder on the test
split, which must in turn beat BM25 (PRD §8.4). If it doesn't, the decision and
the numbers get recorded rather than buried.
