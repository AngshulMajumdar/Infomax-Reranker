"""End-to-end reranker: scores passages with two frozen sentence-transformers models, then selects.

Requires the optional dependency:  pip install "infomax-reranker[neural]"
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .core import TUNED, fused, infomax_select, rank

CE_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
METHODS = ("infomax+f", "infomax", "ce", "dense", "fusion", "mmr")


class InfoMaxReranker:
    """Rerank passages for a query.

    >>> rr = InfoMaxReranker()                       # downloads both models on first use
    >>> rr.rerank("what causes tides?", passages, k=5)  # list of dicts, best first
    """

    def __init__(self, ce_model: str = CE_MODEL, dense_model: str = DENSE_MODEL, device: str | None = None,
                 max_length: int = 256, batch_size: int = 32):
        self.ce_model, self.dense_model = ce_model, dense_model
        self.device, self.max_length, self.batch_size = device, max_length, batch_size
        self._ce = self._enc = None

    def _models(self):
        if self._ce is None:
            try:
                from sentence_transformers import CrossEncoder, SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise ImportError('Install the neural extra: pip install "infomax-reranker[neural]"') from exc
            self._ce = CrossEncoder(self.ce_model, max_length=self.max_length, device=self.device)
            self._enc = SentenceTransformer(self.dense_model, device=self.device)
        return self._ce, self._enc

    def score(self, query: str, passages: Sequence[str]):
        """Return (cross-encoder logits (n,), passage embeddings (n, d), query embedding (d,))."""
        if not passages:
            raise ValueError("at least one passage is required")
        ce, enc = self._models()
        r = np.asarray(ce.predict([(query, p) for p in passages], batch_size=self.batch_size,
                                  show_progress_bar=False), dtype=np.float32).reshape(-1)
        X = enc.encode([query] + list(passages), batch_size=self.batch_size, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        return r, X[1:], X[0]

    def rerank(self, query: str, passages: Sequence[str], k: int = 10, method: str = "infomax+f",
               tau: float | None = None, **overrides) -> list[dict]:
        """Rerank and return the top items as dicts: index, passage, ce_score, dense_score, gain (InfoMax only).

        method : one of "infomax+f" (default), "infomax", "ce", "dense", "fusion", "mmr".
        tau    : InfoMax only; stop early once the information gain of a pick drops below tau (nats).
        overrides : InfoMax/MMR/fusion knobs (rank, gamma, rho, sigma2, beta, delta, alpha, lam).
        """
        method = method.lower()
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        passages = list(passages)
        r, E, q = self.score(query, passages)
        return self.rerank_scores(r, E, q, passages, k=k, method=method, tau=tau, **overrides)

    @staticmethod
    def rerank_scores(r, E, q, passages: Sequence[str] | None = None, k: int = 10, method: str = "infomax+f",
                      tau: float | None = None, **overrides) -> list[dict]:
        """Same as rerank() but from precomputed scores and embeddings (no model needed)."""
        method = method.lower()
        r, E, q = np.asarray(r), np.asarray(E), np.asarray(q)
        dense = E @ q
        gains: dict[int, float] = {}
        if method in ("infomax", "infomax+f"):
            p = dict(TUNED[method], **overrides)
            base = r if method == "infomax" else fused(r, E, q, p.pop("alpha"))
            order, g = infomax_select(base, E, q, k=k, tau=tau, **p)
            gains = dict(zip(order, g))
        else:
            order = rank(method, r, E, q, k=k, **overrides)[:k]
        out = []
        for j in order:
            item = {"index": int(j), "ce_score": float(r[j]), "dense_score": float(dense[j])}
            if passages is not None:
                item["passage"] = passages[j]
            if j in gains:
                item["gain"] = gains[j]
            out.append(item)
        return out
