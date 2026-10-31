"""Tune every method's knobs on dev splits only (scifact.train, nfcorpus.dev, fiqa.dev; one global
setting per method), then evaluate on the five BEIR test splits.

Usage: python run_eval.py [--quick]
"""
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import OUT, evaluate, load_qrels, order_to_run  # noqa: E402
from infomax_reranker.core import by_score, rank  # noqa: E402

DEV = ["scifact.train", "nfcorpus.dev", "fiqa.dev"]
TEST = ["scifact.test", "nfcorpus.test", "fiqa.test", "scidocs.test", "trec-covid.test"]


def load(tag):
    name, split = tag.split(".")
    z = np.load(os.path.join(OUT, f"{tag}.scores.npz"))
    qs = []
    for q in z["qids"]:
        qs.append({k: z[f"{q}|{k}"] for k in ("ids", "bm25", "ce", "emb", "qemb")} | {"qid": str(q)})
    return qs, load_qrels(name, split)


def score(Q, qrels, orders):
    run = {x["qid"]: order_to_run([x["ids"][j] for j in o]) for x, o in zip(Q, orders)}
    m, per = evaluate(run, qrels)
    return m, {q: v["ndcg_cut_10"] for q, v in per.items()}


def method_orders(Q, method, p):
    if method == "BM25":
        return [by_score(x["bm25"]) for x in Q]
    m = {"Dense": "dense", "CE": "ce", "MMR": "mmr", "Fusion": "fusion",
         "InfoMax": "infomax", "InfoMax+F": "infomax+f"}[method]
    return [rank(m, x["ce"], x["emb"], x["qemb"], k=10, **p) for x in Q]


GRIDS = {
    "MMR": [{"lam": l} for l in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)],
    "Fusion": [{"alpha": a} for a in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)],
    "InfoMax": [dict(rank=r, gamma=g, rho=rh, sigma2=s, beta=b, delta=d)
                for r, g, rh, s, b, d in itertools.product(
                    (1, 4, 16), (1.0, 2.0, 4.0), (0.3, 1.0), (0.03, 0.3), (0.0, 0.3, 3.0),
                    (0.0, 0.5, 2.0, np.inf))],
    "InfoMax+F": [dict(rank=r, gamma=g, rho=1.0, sigma2=s, beta=b, delta=d)
                  for r, g, s, b, d in itertools.product(
                      (1, 4, 16), (1.0, 2.0, 4.0, 8.0), (0.03, 0.3), (0.0, 0.3, 3.0), (0.0, 0.5, 2.0, np.inf))],
}


def boot_ci(a, b, n=10000, seed=0):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    quick = "--quick" in sys.argv
    best = {}
    bp = os.path.join(OUT, "best_params.json")
    if "--test" in sys.argv:
        best = {k: {kk: (np.inf if vv is None else vv) for kk, vv in v.items()}
                for k, v in json.load(open(bp)).items()}
        grids = {}
    else:
        dev = {t: load(t) for t in DEV}
        grids = GRIDS
    for meth, grid in grids.items():
        if meth == "InfoMax+F":
            grid = [p | {"alpha": best["Fusion"]["alpha"]} for p in grid]
        if quick:
            grid = grid[:4]
        t0 = time.time()
        res = []
        for p in grid:
            v = np.mean([score(Q, qr, method_orders(Q, meth, p))[0]["nDCG@10"] for Q, qr in dev.values()])
            res.append((v, p))
        res.sort(key=lambda t: -t[0])
        best[meth] = res[0][1]
        ce_dev = np.mean([score(Q, qr, method_orders(Q, "CE", {}))[0]["nDCG@10"] for Q, qr in dev.values()])
        print(f"[tune] {meth}: best dev nDCG@10 {res[0][0]:.4f} (CE {ce_dev:.4f}) with {res[0][1]}  "
              f"[{len(grid)} settings, {time.time()-t0:.0f}s]", flush=True)
        if meth.startswith("InfoMax"):
            for v, p in res[:5]:
                print(f"        {v:.4f} {p}")
    if "--test" not in sys.argv:
        json.dump({k: {kk: (None if vv == np.inf else vv) for kk, vv in v.items()} for k, v in best.items()},
                  open(bp, "w"), indent=1)
        return

    methods = ["BM25", "Dense", "CE", "MMR", "Fusion", "InfoMax", "InfoMax+F"]
    table, perq = {}, {}
    for tag in TEST:
        Q, qr = load(tag)
        for meth in methods:
            o = method_orders(Q, meth, best.get(meth, {}))
            m, pq = score(Q, qr, o)
            table[(tag, meth)] = m
            perq[(tag, meth)] = pq
    print("\nnDCG@10 on test (BM25 top-50 pools; knobs tuned on dev only)")
    print(f"{'dataset':16s}" + "".join(f"{m:>9s}" for m in methods))
    for tag in TEST:
        print(f"{tag:16s}" + "".join(f"{table[(tag, m)]['nDCG@10']:9.4f}" for m in methods))
    print("\nInfoMax+F minus X, nDCG@10, paired bootstrap 95% CI")
    rows = {}
    for tag in TEST:
        qids = sorted(perq[(tag, "CE")])
        for other in ("CE", "MMR", "Fusion"):
            a = [perq[(tag, "InfoMax+F")][q] for q in qids]
            b = [perq[(tag, other)][q] for q in qids]
            rows[(tag, other)] = boot_ci(a, b)
        print(f"{tag:16s}" + "  ".join(f"vs {o}: {rows[(tag, o)][0]:+.4f} [{rows[(tag, o)][1]:+.4f},{rows[(tag, o)][2]:+.4f}]"
                                       for o in ("CE", "MMR", "Fusion")))
    json.dump({"best": {k: {kk: (None if vv == np.inf else vv) for kk, vv in v.items()} for k, v in best.items()},
               "test": {f"{t}|{m}": v for (t, m), v in table.items()},
               "ci_infomax_minus": {f"{t}|{o}": v for (t, o), v in rows.items()}},
              open(os.path.join(OUT, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
