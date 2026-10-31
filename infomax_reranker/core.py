"""Model-free selection rules over a fixed candidate pool.

Every function takes, for one query and n candidates:
  r : (n,) relevance scores (e.g. cross-encoder logits), higher = more relevant
  E : (n, d) unit-norm passage embeddings
  q : (d,) unit-norm query embedding
and returns indices into the pool. Only numpy and scipy are required.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

# Settings tuned once on dev splits (SciFact train, NFCorpus dev, FiQA dev), used unchanged on every test set.
TUNED = {
    "infomax": dict(rank=1, gamma=4.0, rho=0.3, sigma2=0.03, beta=0.0, delta=0.0),
    "infomax+f": dict(rank=4, gamma=8.0, rho=1.0, sigma2=0.03, beta=3.0, delta=0.0, alpha=0.5),
    "mmr": dict(lam=0.95),
    "fusion": dict(alpha=0.5),
}


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo + 1e-9 else np.zeros_like(x)


def by_score(s: np.ndarray) -> list[int]:
    """Indices sorted by descending score; ties keep the original order."""
    return [int(i) for i in np.argsort(-np.asarray(s), kind="stable")]


def complete(order: list[int], base: list[int]) -> list[int]:
    """Append the items of `base` that are not in `order`, in base order."""
    seen = set(order)
    return list(order) + [j for j in base if j not in seen]


def fused(r, E, q, alpha: float = 0.5) -> np.ndarray:
    """z-score fusion of relevance r and dense cosine E @ q; alpha is the dense weight."""
    r = np.asarray(r)
    d = np.asarray(E) @ np.asarray(q)
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    return (1 - alpha) * z(r) + alpha * z(d)


def mmr(r, E, lam: float = 0.95, k: int = 10) -> list[int]:
    """Maximal Marginal Relevance: lam * minmax(r) - (1 - lam) * max similarity to the selected set."""
    rel = minmax(r)
    E = np.asarray(E)
    sim = E @ E.T
    n = len(rel)
    S: list[int] = []
    rem = np.ones(n, bool)
    for _ in range(min(k, n)):
        if not S:
            j = int(np.argmax(rel))
        else:
            v = lam * rel - (1 - lam) * sim[:, S].max(1)
            v[~rem] = -np.inf
            j = int(np.argmax(v))
        S.append(j)
        rem[j] = False
    return complete(S, by_score(r))


def _condvar(M: np.ndarray, S: list[int]) -> np.ndarray:
    """Conditional variance of every coordinate given the coordinates in S (Gaussian, covariance M)."""
    if not S:
        return np.diag(M).copy()
    c = cho_factor(M[np.ix_(S, S)], lower=True)
    A = cho_solve(c, M[S, :])
    return np.diag(M) - np.einsum("ij,ij->j", M[S, :], A)


def infomax_select(r, E, q, k: int = 10, tau: float | None = None, rank: int = 4, gamma: float = 8.0,
                   rho: float = 1.0, sigma2: float = 0.03, beta: float = 3.0, delta: float = 0.0
                   ) -> tuple[list[int], list[float]]:
    """Greedy information-maximising selection.

    Model:  y_j = sqrt(w_j) <U^T e_j, z> + <g_j, u> + eps_j
            z ~ N(0, Lambda)   query-aspect latent (U = [query direction, top relevance-weighted directions])
            u ~ N(0, beta I)   shared off-query content -> correlated noise between similar passages
            eps ~ N(0, sigma2), w = minmax(r)^gamma, g_j = (I - U U^T) e_j, embeddings centred on the pool mean.
    Step:   add argmax_j I(z; y_j | y_S) = 1/2 log Var(y_j | y_S) / Var(y_j | y_S, z), computed exactly.
    Anchor: the first pick is the top of r; afterwards the next item b of the r-ranking is replaced by the
            max-gain item only if gain > (1 + delta) * gain(b). delta = inf reproduces the r-ranking.
    Stop:   after k picks, or earlier when the gain of the chosen item drops below tau (adaptive K).

    Returns (selected indices, information gain in nats of each pick).
    """
    r = np.asarray(r)
    E = np.asarray(E)
    q = np.asarray(q)
    n = len(r)
    if E.shape[0] != n:
        raise ValueError("E must have one row per score")
    if n == 0:
        return [], []
    base = by_score(r)
    w = minmax(r) ** gamma
    mu = E.mean(0)
    Ec = E - mu
    Ec /= np.linalg.norm(Ec, axis=1, keepdims=True) + 1e-9
    qc = q - mu
    qc /= np.linalg.norm(qc) + 1e-9
    U, lam = [qc], [1.0]
    if rank > 1:
        Wr = np.sqrt(w)[:, None] * (Ec - np.outer(Ec @ qc, qc))
        _, sv, Vt = np.linalg.svd(Wr, full_matrices=False)
        m = min(rank - 1, len(sv))
        U += list(Vt[:m])
        lam += list(rho * (sv[:m] / (sv[0] + 1e-9)) ** 2)
    U = np.array(U).T
    P = Ec @ U
    H = np.sqrt(w)[:, None] * P
    G = Ec - P @ U.T
    R = beta * (G @ G.T) + sigma2 * np.eye(n)
    C = H @ np.diag(lam) @ H.T + R
    S: list[int] = []
    gains: list[float] = []
    rem = np.ones(n, bool)
    for t in range(min(k, n)):
        g = 0.5 * np.log(np.maximum(_condvar(C, S), 1e-12) / np.maximum(_condvar(R, S), 1e-12))
        b = next(j for j in base if rem[j])
        if t == 0:
            j = b
        else:
            gm = np.where(rem, g, -np.inf)
            js = int(np.argmax(gm))
            j = js if gm[js] > (1 + delta) * g[b] + 1e-12 else b
        if tau is not None and t > 0 and g[j] < tau:
            break
        S.append(int(j))
        gains.append(float(g[j]))
        rem[j] = False
    return S, gains


def infomax(r, E, q, k: int = 10, **params) -> list[int]:
    """Full ranking: InfoMax picks the first k, the rest follow in r order."""
    S, _ = infomax_select(r, E, q, k=k, **params)
    return complete(S, by_score(r))


def rank(method: str, r, E, q, k: int = 10, **overrides) -> list[int]:
    """Rank a pool with one of: 'ce', 'dense', 'mmr', 'fusion', 'infomax', 'infomax+f'.

    Unspecified knobs default to the dev-tuned settings in TUNED.
    """
    m = method.lower()
    p = dict(TUNED.get(m, {}), **overrides)
    if m == "ce":
        return by_score(r)
    if m == "dense":
        return by_score(np.asarray(E) @ np.asarray(q))
    if m == "mmr":
        return mmr(r, E, k=k, **p)
    if m == "fusion":
        return by_score(fused(r, E, q, **p))
    if m == "infomax":
        return infomax(r, E, q, k=k, **p)
    if m == "infomax+f":
        a = p.pop("alpha")
        return infomax(fused(r, E, q, a), E, q, k=k, **p)
    raise ValueError(f"unknown method {method!r}")
