import numpy as np
import pytest

from infomax_reranker import InfoMaxReranker, by_score, fused, infomax, infomax_select, mmr, rank


def pool(n=30, d=32, seed=0):
    rng = np.random.default_rng(seed)
    E = rng.standard_normal((n, d))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    q = rng.standard_normal(d)
    q /= np.linalg.norm(q)
    r = E @ q * 5 + rng.standard_normal(n)
    return r.astype(np.float32), E.astype(np.float32), q.astype(np.float32)


def test_rankings_are_permutations():
    r, E, q = pool()
    for m in ("ce", "dense", "mmr", "fusion", "infomax", "infomax+f"):
        o = rank(m, r, E, q, k=10)
        assert sorted(o) == list(range(len(r))), m


def test_first_pick_is_base_top1():
    r, E, q = pool()
    assert infomax(r, E, q, k=5)[0] == int(np.argmax(r))
    assert rank("infomax+f", r, E, q, k=5)[0] == by_score(fused(r, E, q, 0.5))[0]


def test_infinite_delta_reproduces_base_ranking():
    r, E, q = pool()
    assert infomax(r, E, q, k=10, delta=np.inf) == by_score(r)
    assert rank("infomax+f", r, E, q, k=10, delta=np.inf) == by_score(fused(r, E, q, 0.5))


def test_gains_nonnegative_and_k_respected():
    r, E, q = pool()
    S, g = infomax_select(r, E, q, k=7)
    assert len(S) == 7 == len(set(S))
    assert all(x >= -1e-9 for x in g)


def test_tau_stops_early():
    r, E, q = pool()
    S_all, g = infomax_select(r, E, q, k=20)
    tau = float(np.median(g))
    S, g2 = infomax_select(r, E, q, k=20, tau=tau)
    assert 1 <= len(S) < len(S_all)
    assert all(x >= tau for x in g2[1:])


def test_exact_duplicate_is_not_picked_twice_early():
    # passage 1 is an exact copy of passage 0 (the most relevant one); with shared-noise beta > 0
    # the duplicate carries no new information and must not be the second pick.
    r, E, q = pool()
    top = int(np.argmax(r))
    E2 = np.vstack([E, E[top]])
    r2 = np.append(r, r[top] - 1e-3)
    S, _ = infomax_select(r2, E2, q, k=2, beta=3.0, delta=0.0)
    assert S[0] == top and S[1] != len(r2) - 1


def test_mmr_lambda_one_is_relevance_order():
    r, E, q = pool()
    assert mmr(r, E, lam=1.0, k=10) == by_score(r)


def test_rerank_scores_without_models():
    r, E, q = pool(10)
    out = InfoMaxReranker.rerank_scores(r, E, q, [f"p{i}" for i in range(10)], k=4)
    assert len(out) == 4 and all("gain" in x and "passage" in x for x in out)
    out = InfoMaxReranker.rerank_scores(r, E, q, None, k=3, method="ce")
    assert [x["index"] for x in out] == by_score(r)[:3]


def test_bad_method():
    r, E, q = pool(5)
    with pytest.raises(ValueError):
        rank("nope", r, E, q)
