"""Embedding backends.

- `hashed`: deterministic feature-hashed n-grams over the token stream.
  Zero dependencies, zero API, fully reproducible. The baseline.
- `local`: voyage-4-nano running locally (open weights, Apache 2.0) via
  sentence-transformers. Shares an embedding space with the larger Voyage 4
  API models, so a local index can later be queried with API embeddings.
- `voyage`: voyage-4-nano via the Voyage API (needs VOYAGE_API_KEY).

All backends take a `progress(done, total)` callback for TUI reporting.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Callable

HASHED_DIM = 512
LOCAL_DIM = 512  # MRL truncation; voyage-4 supports 2048/1024/512/256
LOCAL_MODEL = "voyageai/voyage-4-nano"

Progress = Callable[[int, int], None]


def _noop(done: int, total: int) -> None:
    pass


def _hash_idx(feature: str, dim: int) -> tuple[int, float]:
    h = hashlib.blake2b(feature.encode(), digest_size=8).digest()
    idx = int.from_bytes(h[:4], "little") % dim
    sign = 1.0 if h[4] & 1 else -1.0
    return idx, sign


def embed_hashed(token_docs: list[str], progress: Progress = _noop,
                 dim: int = HASHED_DIM) -> list[list[float]]:
    """Feature-hash 1/2/3-grams of instruction tokens, sublinear TF, L2 norm."""
    out = []
    for i, doc in enumerate(token_docs):
        lines = doc.splitlines()
        toks = lines[-1].split() if lines else []
        vec = [0.0] * dim
        counts: dict[str, int] = {}
        for n in (1, 2, 3):
            for j in range(len(toks) - n + 1):
                g = " ".join(toks[j : j + n])
                counts[g] = counts.get(g, 0) + 1
        for g, c in counts.items():
            idx, sign = _hash_idx(g, dim)
            vec[idx] += sign * (1.0 + math.log(c))
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out.append([v / norm for v in vec])
        if i % 500 == 0:
            progress(i, len(token_docs))
    progress(len(token_docs), len(token_docs))
    return out


_local_model = None


def embed_local(token_docs: list[str], progress: Progress = _noop,
                batch_size: int = 64) -> list[list[float]]:
    """voyage-4-nano, self-hosted via sentence-transformers (MPS/CUDA/CPU)."""
    global _local_model
    from sentence_transformers import SentenceTransformer  # lazy

    if _local_model is None:
        _local_model = SentenceTransformer(
            LOCAL_MODEL, trust_remote_code=True, truncate_dim=LOCAL_DIM)
    vecs: list[list[float]] = []
    for i in range(0, len(token_docs), batch_size):
        batch = [d[:16000] for d in token_docs[i : i + batch_size]]
        emb = _local_model.encode_document(
            batch, batch_size=batch_size, normalize_embeddings=True)
        vecs.extend(e.tolist() for e in emb)
        progress(min(i + batch_size, len(token_docs)), len(token_docs))
    return vecs


def embed_voyage(token_docs: list[str], progress: Progress = _noop,
                 model: str = "voyage-4-nano",
                 batch_size: int = 128) -> list[list[float]]:
    import voyageai  # lazy: only needed for this backend

    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    vecs: list[list[float]] = []
    for i in range(0, len(token_docs), batch_size):
        batch = [d[:16000] for d in token_docs[i : i + batch_size]]
        res = client.embed(batch, model=model, input_type="document",
                           output_dimension=LOCAL_DIM)
        vecs.extend(res.embeddings)
        progress(min(i + batch_size, len(token_docs)), len(token_docs))
    return vecs


BACKENDS = {
    "hashed": embed_hashed,
    "local": embed_local,
    "voyage": embed_voyage,
}


def embed(token_docs: list[str], backend: str = "hashed",
          progress: Progress = _noop) -> list[list[float]]:
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; have {sorted(BACKENDS)}")
    return BACKENDS[backend](token_docs, progress)


def default_backend() -> str:
    return "hashed"
