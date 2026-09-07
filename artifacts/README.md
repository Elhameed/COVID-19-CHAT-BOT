# artifacts/

Configuration for the served retriever.

```
artifacts/
└── retriever_config.json    # encoder, indexed field, abstention threshold τ
```

## What this is

`src/api.py` reads this file at startup to know which encoder to load, which KB field it was
indexed on, and the similarity threshold below which the bot abstains rather than answering.

```json
{
  "encoder": "sentence-transformers/all-MiniLM-L6-v2",
  "field": "question",
  "tau": 0.6932,
  "tuned_on": "dev"
}
```

The file also records how τ was chosen: the tuning objective, the in-scope coverage and
off-topic rejection it achieves, and the size of the probe those came from.

τ is tuned on the **dev** split and never on test. At this value the retriever answers 77% of
in-scope dev queries and none of the off-topic probe in `src.evaluate.OFF_TOPIC_PROBE` — 12
hand-written out-of-scope questions, a smoke test rather than a benchmark.

## No model weights live here

The encoder is a stock Hugging Face model, downloaded on first use and cached by
`sentence-transformers`. Nothing about it is modified, so there is nothing to vendor — the
model id in the config is a complete description of what runs.

The KB embeddings are derived data and are rebuilt rather than committed; see
[`data/README.md`](../data/README.md).

## Regenerating

Nothing here is hand-edited:

```bash
python -m src.download                  # fetch COUGH
python -m src.prep                      # -> data/kb.parquet + splits + qrels
python -m src.index                     # -> data/embeddings/
python -m src.evaluate --tune-threshold # -> retriever_config.json
```
