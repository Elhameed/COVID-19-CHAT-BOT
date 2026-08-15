"""COVID-19 Chatbot — retrieval-only FAQ answering over the COUGH corpus.

Modules (PRD §17). Reusable logic lives here and nowhere else: the notebook in
``notebooks/`` and the API in :mod:`src.api` both import these, so the numbers
reported during development are by construction the numbers the app serves
(PRD §7.6).

    src.prep       clean/filter/dedupe COUGH -> data/kb.parquet, seeded
                   train/dev/test query splits, and aligned qrels
    src.index      build/load KB embeddings
    src.retriever  BM25 + bi-encoder (+ optional cross-encoder) behind one
                   interface
    src.evaluate   MRR / P@k / Recall / nDCG on the COUGH *test* split
    src.api        FastAPI service: POST /predict, GET /health

The raw corpus is fetched by :mod:`src.download`.

The bot never generates text. Every answer is a stored KB answer returned with
its source and a medical disclaimer, or an explicit abstention below the
confidence threshold.
"""

__version__ = "0.1.0"
