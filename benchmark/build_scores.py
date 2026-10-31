"""Merge pools + dense embeddings + cross-encoder logits into out/<tag>.scores.npz (format read by run_eval.py)."""
import json
import os
import sys

import numpy as np

from common import OUT

TAGS = ["scifact.test", "nfcorpus.test", "fiqa.test", "scidocs.test", "trec-covid.test",
        "scifact.train", "nfcorpus.dev", "fiqa.dev"]


def main(depth, tags):
    for tag in tags:
        name = tag.split(".")[0]
        pools = json.load(open(os.path.join(OUT, f"{tag}.pools.json")))
        z = np.load(os.path.join(OUT, f"{name}.emb.npz"))
        emb = dict(zip(z["ids"].tolist(), z["emb"]))
        zq = np.load(os.path.join(OUT, f"{name}.qemb.npz"))
        qemb = dict(zip(zq["ids"].tolist(), zq["emb"]))
        logits = np.load(os.path.join(OUT, f"{tag}.ce.npz"))["logits"]
        qids = sorted(pools)
        assert len(logits) == sum(len(pools[q]["ids"][:depth]) for q in qids), tag
        out, p = {"qids": np.array(qids)}, 0
        for q in qids:
            ds = pools[q]["ids"][:depth]
            out[f"{q}|ids"] = np.array(ds)
            out[f"{q}|bm25"] = np.asarray(pools[q]["bm25"][:depth], np.float32)
            out[f"{q}|ce"] = logits[p:p + len(ds)]
            out[f"{q}|emb"] = np.stack([emb[d] for d in ds])
            out[f"{q}|qemb"] = qemb[q]
            p += len(ds)
        np.savez_compressed(os.path.join(OUT, f"{tag}.scores.npz"), **out)
        print(tag, len(qids), "queries")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2:] or TAGS)
