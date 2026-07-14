"""Embedding backends.

- `hashed`: deterministic feature-hashed n-grams over the token stream.
  Zero dependencies, zero API, fully reproducible. The workhorse.
- `voyage`: voyage-4-nano via the Voyage API (needs VOYAGE_API_KEY).
  Semantic robustness on top of the lexical baseline.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Iterable

HASHED_DIM = 512


def _hash_idx(feature: str, dim: int) -> tuple[int, float]:
    h = hashlib.blake2b(feature.encode(), digest_size=8).digest()
    idx = int.from_bytes(h[:4], "little") % dim
    sign = 1.0 if h[4] & 1 else -1.0
    return idx, sign


def embed_hashed(token_docs: list[str], dim: int = HASHED_DIM) -> list[list[float]]:
    """Feature-hash 1/2/3-grams of instruction tokens, sublinear TF, L2 norm."""
    out = []
    for doc in token_docs:
        lines = doc.splitlines()
        toks = lines[-1].split() if lines else []
        vec = [0.0] * dim
        counts: dict[str, int] = {}
        for n in (1, 2, 3):
            for i in range(len(toks) - n + 1):
                g = " ".join(toks[i : i + n])
                counts[g] = counts.get(g, 0) + 1
        for g, c in counts.items():
            idx, sign = _hash_idx(g, dim)
            vec[idx] += sign * (1.0 + math.log(c))
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out.append([v / norm for v in vec])
    return out


def embed_voyage(token_docs: list[str], model: str = "voyage-4-nano",
                 batch_size: int = 128) -> list[list[float]]:
    import voyageai  # lazy: only needed for this backend

    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    vecs: list[list[float]] = []
    for i in range(0, len(token_docs), batch_size):
        batch = [d[:16000] for d in token_docs[i : i + batch_size]]
        res = client.embed(batch, model=model, input_type="document")
        vecs.extend(res.embeddings)
    return vecs


BACKENDS = {
    "hashed": embed_hashed,
    "voyage": embed_voyage,
}


def embed(token_docs: list[str], backend: str = "hashed") -> list[list[float]]:
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; have {sorted(BACKENDS)}")
    return BACKENDS[backend](token_docs)


def default_backend() -> str:
    return "voyage" if os.environ.get("VOYAGE_API_KEY") else "hashed"
