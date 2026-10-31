"""Rerankers over a fixed candidate pool. Each returns a full ordering (indices into the pool).

Inputs per query: r = cross-encoder logits (n,), E = unit doc embeddings (n, d), q = unit query embedding (d,).
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def minmax(x):
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo + 1e-9 else np.zeros_like(x)


def by_score(s):
    return list(np.argsort(-s, kind="stable"))


def complete(order, base):
    seen = set(order)
    return list(order) + [j for j in base if j not in seen]


# ---------------------------------------------------------------- baselines
def mmr(r, E, lam, K=10):
    rel = minmax(r)
    sim = E @ E.T
    S, rem = [], set(range(len(r)))
    for _ in range(min(K, len(r))):
        if not S:
            j = int(np.argmax(rel))
        else:
            red = sim[:, S].max(1)
            cand = np.array(sorted(rem))
            v = lam * rel[cand] - (1 - lam) * red[cand]
            j = int(cand[np.argmax(v)])
        S.append(j)
        rem.discard(j)
    return complete(S, by_score(r))


def fused(r, E, q, alpha):
    """z-score fusion of cross-encoder logits and dense cosine; alpha = weight on dense."""
    d = E @ q
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    return (1 - alpha) * z(r) + alpha * z(d)


def zfuse(r, E, q, alpha):
    return by_score(fused(r, E, q, alpha))


# ---------------------------------------------------------------- information-max selector
def _condvar(M, S):
    if not S:
        return np.diag(M).copy()
    c = cho_factor(M[np.ix_(S, S)], lower=True)
    A = cho_solve(c, M[S, :])
    return np.diag(M) - np.einsum("ij,ij->j", M[S, :], A)


def infomax(r, E, q, rank=4, gamma=2.0, rho=1.0, sigma2=0.1, beta=0.0, delta=np.inf, K=10,
            return_gains=False):
    """Greedy I(z; y_j | y_S) with z = query-aspect latent and correlated (nuisance) noise.

    y_j = s_j <U^T e_j, z> + <g_j, u> + eps_j,  z ~ N(0, Lambda), u ~ N(0, beta I), eps ~ N(0, sigma2)
      e_j  : pool-centred unit embedding; U: [query direction, top relevance-weighted pool directions]
      s_j^2: minmax(cross-encoder)^gamma ; g_j = (I - U U^T) e_j  (off-aspect content -> shared noise)
    Anchored to the cross-encoder order: the CE top remaining item b is replaced by the max-gain
    item only if gain(j*) > (1 + delta) * gain(b).  delta = inf reproduces the cross-encoder.
    """
    n = len(r)
    base = by_score(r)
    w = minmax(r) ** gamma
    mu = E.mean(0)
    Ec = E - mu
    Ec /= np.linalg.norm(Ec, axis=1, keepdims=True) + 1e-9
    qc = q - mu
    qc /= np.linalg.norm(qc) + 1e-9
    U = [qc]
    lam = [1.0]
    if rank > 1:
        Wr = np.sqrt(w)[:, None] * (Ec - np.outer(Ec @ qc, qc))
        _, sv, Vt = np.linalg.svd(Wr, full_matrices=False)
        m = min(rank - 1, len(sv))
        U += list(Vt[:m])
        lam += list(rho * (sv[:m] / (sv[0] + 1e-9)) ** 2)
    U = np.array(U).T                      # d x r
    Lam = np.diag(lam)
    P = Ec @ U                             # n x r aspect coordinates
    H = np.sqrt(w)[:, None] * P
    G = Ec - P @ U.T                       # off-aspect residual
    R = beta * (G @ G.T) + sigma2 * np.eye(n)
    C = H @ Lam @ H.T + R
    S, rem, gains_log = [], np.ones(n, bool), []
    for t in range(min(K, n)):
        g = 0.5 * np.log(np.maximum(_condvar(C, S), 1e-12) / np.maximum(_condvar(R, S), 1e-12))
        b = next(j for j in base if rem[j])
        if t == 0:
            j = b
        else:
            gm = np.where(rem, g, -np.inf)
            js = int(np.argmax(gm))
            j = js if gm[js] > (1 + delta) * g[b] + 1e-12 else b
        gains_log.append(float(g[j]))
        S.append(j)
        rem[j] = False
    order = complete(S, base)
    return (order, gains_log) if return_gains else order
