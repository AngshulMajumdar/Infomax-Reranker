"""Use the selector inside an existing pipeline: you already have reranker scores and embeddings.

Only numpy and scipy are needed (pip install git+https://github.com/AngshulMajumdar/infomax-reranker).
"""
import numpy as np

from infomax_reranker import infomax_select, rank

rng = np.random.default_rng(0)
n, d = 50, 384
E = rng.standard_normal((n, d)).astype(np.float32)
E /= np.linalg.norm(E, axis=1, keepdims=True)          # unit-norm passage embeddings from your retriever
q = E[:5].mean(0)
q /= np.linalg.norm(q)                                  # unit-norm query embedding
r = (E @ q * 8 + rng.standard_normal(n)).astype(np.float32)  # your reranker's scores (any scale)

# Full ranking of the pool (first 10 chosen by InfoMax, rest in score order).
order = rank("infomax", r, E, q, k=10)
print("InfoMax top-10:", order[:10])

# Selection with per-step information gain and adaptive stopping.
selected, gains = infomax_select(r, E, q, k=10, tau=0.05)
print("selected:", selected)
print("gain (nats):", np.round(gains, 3).tolist())
