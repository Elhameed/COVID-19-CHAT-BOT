"""Build, cache and search dense embeddings over the knowledge base (PRD §7.2).

Embedding 7k KB questions takes seconds on the GPU and a minute or two on CPU,
but it happens on every API start and every notebook run, so the vectors are
cached to ``data/embeddings/``.

    python -m src.index                                  # default encoder
    python -m src.index --model BAAI/bge-base-en-v1.5
    python -m src.index --rebuild

Search is exact (PRD §22 #3). At 7k vectors a brute-force matrix product is
sub-millisecond and returns the true nearest neighbours, so an ANN index would
add a dependency, an approximation, and a tuning surface to solve a problem we
do not have.

Cache correctness
-----------------
A stale cache is the dangerous failure here: embeddings that no longer match the
KB would silently misalign every id, and the metrics would still look plausible.
Each cache records the model, the field, and a fingerprint of the KB ids and
text, and refuses to load if any of them moved.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Some encoders were trained with an instruction prefix and lose accuracy
# without it. bge asks for an instruction on the query side only; e5 prefixes
# both sides. MiniLM was trained without either.
#
# Getting this wrong is a silent quality bug, not an error -- the model still
# returns vectors, just worse ones.
MODEL_PREFIXES: dict[str, tuple[str, str]] = {
    # model substring -> (query prefix, document prefix)
    "bge": ("Represent this sentence for searching relevant passages: ", ""),
    "e5": ("query: ", "passage: "),
}


def prefixes_for(model_name: str) -> tuple[str, str]:
    """Return the (query, document) prefixes this model expects."""
    lowered = model_name.lower()
    for key, pair in MODEL_PREFIXES.items():
        if key in lowered:
            return pair
    return ("", "")


def kb_fingerprint(kb: pd.DataFrame, field_values: pd.Series) -> str:
    """Stable digest of the exact ids and text that were embedded."""
    hasher = hashlib.sha256()
    for kb_id, text in zip(kb["id"], field_values, strict=True):
        hasher.update(str(kb_id).encode())
        hasher.update(b"\x00")
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\x01")
    return hasher.hexdigest()


def cache_path(model_name: str, field: str, embeddings_dir: Path = EMBEDDINGS_DIR) -> Path:
    """Filesystem-safe cache filename for a model/field pair.

    `model_name` may be a Hub id ("BAAI/bge-base-en-v1.5") or a local directory
    ("artifacts/encoder", or a Windows absolute path), so every separator and
    drive colon has to be folded out.
    """
    slug = model_name
    for ch in ("/", "\\", ":"):
        slug = slug.replace(ch, "__")
    return embeddings_dir / f"{slug.strip('_')}__{field}.npz"


@dataclass
class EmbeddingIndex:
    """Normalized KB embeddings plus the metadata needed to trust them."""

    model_name: str
    field: str
    ids: np.ndarray
    embeddings: np.ndarray
    fingerprint: str

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    def search(
        self, query_embeddings: np.ndarray, top_k: int = 10
    ) -> list[list[tuple[int, float]]]:
        """Exact nearest-neighbour search for a batch of query vectors.

        Both sides are L2-normalized, so the inner product *is* cosine
        similarity and scores land in [-1, 1] -- which is what makes a single
        threshold τ meaningful across queries (PRD §7.5).
        """
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings[None, :]

        scores = query_embeddings @ self.embeddings.T
        k = min(top_k, scores.shape[1])

        # Partition to the top k, then sort only those.
        top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        results = []
        for row, cols in enumerate(top):
            ordered = cols[np.argsort(-scores[row, cols])]
            results.append([(int(self.ids[c]), float(scores[row, c])) for c in ordered])
        return results

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            ids=self.ids,
            embeddings=self.embeddings,
            model_name=np.array(self.model_name),
            field=np.array(self.field),
            fingerprint=np.array(self.fingerprint),
        )
        return path

    @classmethod
    def load(cls, path: Path, expected_fingerprint: str | None = None) -> EmbeddingIndex:
        with np.load(path, allow_pickle=False) as data:
            index = cls(
                model_name=str(data["model_name"]),
                field=str(data["field"]),
                ids=data["ids"],
                embeddings=data["embeddings"],
                fingerprint=str(data["fingerprint"]),
            )
        if expected_fingerprint is not None and index.fingerprint != expected_fingerprint:
            raise ValueError(
                f"embedding cache {path.name} was built for a different knowledge base "
                f"(fingerprint {index.fingerprint[:12]} != {expected_fingerprint[:12]}). "
                f"Rebuild with `python -m src.index --rebuild`."
            )
        return index


def load_encoder(model_name: str = DEFAULT_MODEL, device: str | None = None):
    """Load a sentence-transformers encoder, preferring the GPU when present."""
    from sentence_transformers import SentenceTransformer

    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


def embed_texts(
    encoder, texts: list[str], prefix: str = "", batch_size: int = 64, show_progress: bool = False
) -> np.ndarray:
    """Encode texts to L2-normalized float32 vectors."""
    if prefix:
        texts = [prefix + t for t in texts]
    vectors = encoder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return np.asarray(vectors, dtype=np.float32)


def build_index(
    kb: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
    field: str = "question",
    encoder=None,
    batch_size: int = 64,
    show_progress: bool = False,
) -> EmbeddingIndex:
    """Embed the KB and return an in-memory index."""
    from src.retriever import field_text

    values = field_text(kb, field)
    _, doc_prefix = prefixes_for(model_name)
    encoder = encoder or load_encoder(model_name)

    embeddings = embed_texts(
        encoder,
        values.tolist(),
        prefix=doc_prefix,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    return EmbeddingIndex(
        model_name=model_name,
        field=field,
        ids=kb["id"].to_numpy(),
        embeddings=embeddings,
        fingerprint=kb_fingerprint(kb, values),
    )


def load_or_build_index(
    kb: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
    field: str = "question",
    embeddings_dir: Path = EMBEDDINGS_DIR,
    rebuild: bool = False,
    encoder=None,
    show_progress: bool = False,
) -> EmbeddingIndex:
    """Return a cached index when it still matches the KB, else rebuild it."""
    from src.retriever import field_text

    path = cache_path(model_name, field, embeddings_dir)
    fingerprint = kb_fingerprint(kb, field_text(kb, field))

    if path.exists() and not rebuild:
        try:
            return EmbeddingIndex.load(path, expected_fingerprint=fingerprint)
        except ValueError:
            # Stale cache: the KB changed under it. Rebuilding is always
            # correct, so this recovers rather than failing.
            pass

    index = build_index(
        kb, model_name=model_name, field=field, encoder=encoder, show_progress=show_progress
    )
    index.save(path)
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build KB embeddings.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--field", default="question")
    parser.add_argument("--rebuild", action="store_true", help="ignore any cached vectors")
    args = parser.parse_args(argv)

    from src.retriever import load_kb

    kb = load_kb()
    q_prefix, d_prefix = prefixes_for(args.model)
    print(f"model   {args.model}")
    print(f"field   {args.field}")
    print(f"prefix  query={q_prefix!r} document={d_prefix!r}")
    print(f"corpus  {len(kb):,} entries\n")

    index = load_or_build_index(
        kb, model_name=args.model, field=args.field, rebuild=args.rebuild, show_progress=True
    )
    path = cache_path(args.model, args.field)
    print(f"\n{len(index):,} vectors x {index.dim} dims")
    print(f"cached at {path.relative_to(PROJECT_ROOT)} ({path.stat().st_size / 1024**2:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
