# artifacts/

Configuration and experiment records for the served retriever.

```
artifacts/
├── retriever_config.json        # what the API loads: encoder, field, threshold τ
└── finetune_experiment.json     # the Phase 4 fine-tune, and why it was not adopted
```

## There is no `encoder/` directory, and that is the result

PRD §7.4 planned for `artifacts/encoder/` to hold a fine-tuned bi-encoder, exported by
`python -m src.train`. Phase 4 ran that fine-tune and **rejected it**.

The short version: it improved the point estimate on both held-out splits, but on the
181-query test split the gain was **+0.021 MRR@10 with a 95% CI of [-0.009, +0.051]**
(p = 0.16). The interval spans zero. Per query it was better on 32 and worse on 25.

PRD §8.4 would have permitted adoption on point estimates alone. We declined, because a
health-information product should not ship a model whose claimed benefit cannot be
measured — and because the off-the-shelf encoder already beats BM25 by **+0.100 MRR@10**
with an interval nowhere near zero, so nothing is lost by saying no.

Full numbers, the training config, and the reasoning are in
[`finetune_experiment.json`](finetune_experiment.json). `src/train.py` is retained and
working: the experiment reproduces in about four minutes on the dev machine.

```bash
python -m src.train
python -m src.evaluate --split test --retriever biencoder --model artifacts/encoder
```

The `.gitignore` exception that would let a trained encoder be committed is deliberately
kept, so the decision can be revisited without re-deriving how to ship the weights.

## What *is* served

`retriever_config.json` records the off-the-shelf encoder, the indexed field, and the
abstention threshold τ tuned on dev. `src/api.py` loads it at startup, together with
`data/kb.parquet` and the embeddings rebuilt by `python -m src.index`.

Nothing here is hand-edited. To regenerate:

```bash
python -m src.download                  # fetch COUGH
python -m src.prep                      # -> data/kb.parquet + splits + qrels
python -m src.index                     # -> data/embeddings/
python -m src.evaluate --tune-threshold # -> retriever_config.json
```
