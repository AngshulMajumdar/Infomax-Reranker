"""Cross-encoder stage: MS MARCO MiniLM-L6-v2 logits (max_length 256) for every (query, pool doc), top-DEPTH.
Resumable: out/<tag>.ce.npz holds logits computed so far in pool order.
Usage: python ce_stage.py CE_DIR DEPTH
"""
import json
import os
import sys
import time

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from common import OUT, load_corpus, load_queries

torch.set_num_threads(os.cpu_count())
TAGS = ["scifact.test", "nfcorpus.test", "trec-covid.test", "scifact.train", "nfcorpus.dev", "fiqa.dev",
        "fiqa.test", "scidocs.test"]


def main():
    cdir, depth = sys.argv[1], int(sys.argv[2])
    ce = CrossEncoder(cdir, max_length=256, device="cpu")
    cache = {}
    for tag in TAGS:
        name = tag.split(".")[0]
        if name not in cache:
            cache.clear()
            ids, titles, texts = load_corpus(name)
            idx = {d: i for i, d in enumerate(ids)}
            cache[name] = (idx, titles, texts, load_queries(name))
        idx, titles, texts, queries = cache[name]
        pools = json.load(open(os.path.join(OUT, f"{tag}.pools.json")))
        pairs = [(queries[q], (titles[idx[d]] + ". " if titles[idx[d]] else "") + texts[idx[d]])
                 for q in sorted(pools) for d in pools[q]["ids"][:depth]]
        path = os.path.join(OUT, f"{tag}.ce.npz")
        done = np.load(path)["logits"] if os.path.exists(path) else np.zeros(0, np.float32)
        if len(done) == len(pairs):
            print(f"{tag}: complete", flush=True)
            continue
        t0, start = time.time(), len(done)
        for s in range(start, len(pairs), 2048):
            chunk = pairs[s:s + 2048]
            o = np.argsort([len(a) + len(b) for a, b in chunk], kind="stable")  # length-sort: less padding
            out = np.asarray(ce.predict([chunk[i] for i in o], batch_size=32, show_progress_bar=False),
                             np.float32).reshape(-1)
            res = np.empty_like(out)
            res[o] = out
            done = np.concatenate([done, res])
            np.savez(path, logits=done)
            rate = (len(done) - start) / (time.time() - t0)
            print(f"{tag}: {len(done)}/{len(pairs)} pairs, {rate:.0f} pairs/s", flush=True)


if __name__ == "__main__":
    main()
