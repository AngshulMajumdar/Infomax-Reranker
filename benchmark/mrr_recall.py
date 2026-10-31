"""MRR@10 and Recall@{3,5,10} on the test splits for every method, using the dev-tuned settings in best_params.json.
Relevance = qrel grade > 0. Paired bootstrap 95% CIs for InfoMax+F minus CE and minus Fusion."""
import json
import os

import numpy as np

from common import OUT
from run_eval import TEST, boot_ci, load, method_orders

METHODS = ["BM25", "Dense", "CE", "MMR", "Fusion", "InfoMax", "InfoMax+F"]
best = {k: {kk: (np.inf if vv is None else vv) for kk, vv in v.items()}
        for k, v in json.load(open(os.path.join(OUT, "best_params.json"))).items()}


def per_query(Q, qrels, orders):
    out = {"MRR@10": [], "R@3": [], "R@5": [], "R@10": []}
    for x, o in zip(Q, orders):
        rel = {d for d, g in qrels.get(x["qid"], {}).items() if g > 0}
        if not rel:
            continue
        ranked = [x["ids"][j] for j in o]
        hit = [d in rel for d in ranked[:10]]
        out["MRR@10"].append(1.0 / (hit.index(True) + 1) if any(hit) else 0.0)
        for k in (3, 5, 10):
            out[f"R@{k}"].append(sum(hit[:k]) / len(rel))
    return {k: np.array(v) for k, v in out.items()}


res = {}
for tag in TEST:
    Q, qr = load(tag)
    for m in METHODS:
        o = method_orders(Q, m, best.get(m, {}))
        res[(tag, m)] = per_query(Q, qr, o)

for metric in ("MRR@10", "R@3", "R@5", "R@10"):
    print(f"\n{metric}")
    print(f"{'dataset':16s}" + "".join(f"{m:>10s}" for m in METHODS))
    for tag in TEST:
        print(f"{tag:16s}" + "".join(f"{res[(tag, m)][metric].mean():10.4f}" for m in METHODS))

print("\nInfoMax+F minus CE / minus Fusion, paired bootstrap 95% CI")
for metric in ("MRR@10", "R@10"):
    for tag in TEST:
        a = res[(tag, "InfoMax+F")][metric]
        s = []
        for o in ("CE", "Fusion"):
            d, lo, hi = boot_ci(a, res[(tag, o)][metric])
            s.append(f"vs {o}: {d:+.4f} [{lo:+.4f},{hi:+.4f}]")
        print(f"{metric:7s}{tag:16s}" + "   ".join(s))
json.dump({f"{t}|{m}|{k}": float(v.mean()) for (t, m), d in res.items() for k, v in d.items()},
          open(os.path.join(OUT, "mrr_recall.json"), "w"), indent=1)
