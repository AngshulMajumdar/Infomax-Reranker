"""Dense stage: all-MiniLM-L6-v2 embeddings for every doc in the top-DEPTH pools and every query.
Resumable: per-dataset store out/<name>.emb.npz updated after each chunk.
Usage: python dense_stage.py MINILM_DIR DEPTH
"""
import json
import os
import sys
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from common import OUT, load_corpus, load_queries

torch.set_num_threads(os.cpu_count())
TAGS = ["scifact.test", "scifact.train", "nfcorpus.test", "nfcorpus.dev", "trec-covid.test",
        "fiqa.test", "fiqa.dev", "scidocs.test"]


def main():
    mdir, depth = sys.argv[1], int(sys.argv[2])
    enc = SentenceTransformer(mdir, device="cpu")
    by_ds = {}
    for t in TAGS:
        by_ds.setdefault(t.split(".")[0], []).append(t)
    for name, tags in by_ds.items():
        store = os.path.join(OUT, f"{name}.emb.npz")
        have = {}
        if os.path.exists(store):
            z = np.load(store)
            have = dict(zip(z["ids"].tolist(), z["emb"]))
        need, qneed = set(), set()
        for t in tags:
            pools = json.load(open(os.path.join(OUT, f"{t}.pools.json")))
            for q, p in pools.items():
                need.update(p["ids"][:depth])
                qneed.add(q)
        ids, titles, texts = load_corpus(name)
        idx = {d: i for i, d in enumerate(ids)}
        queries = load_queries(name)
        todo = sorted(d for d in need if d not in have)
        t0 = time.time()
        for s in range(0, len(todo), 2048):
            chunk = todo[s:s + 2048]
            E = enc.encode([(titles[idx[d]] + ". " if titles[idx[d]] else "") + texts[idx[d]] for d in chunk],
                           batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            have.update(zip(chunk, E.astype(np.float32)))
            k = sorted(have)
            np.savez(store, ids=np.array(k), emb=np.stack([have[d] for d in k]))
            print(f"{name}: {min(s + 2048, len(todo))}/{len(todo)} docs, {time.time()-t0:.0f}s", flush=True)
        qk = sorted(qneed)
        QE = enc.encode([queries[q] for q in qk], batch_size=64, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)
        np.savez(os.path.join(OUT, f"{name}.qemb.npz"), ids=np.array(qk), emb=QE.astype(np.float32))
        print(f"{name}: done ({len(have)} docs, {len(qk)} queries)", flush=True)


if __name__ == "__main__":
    main()
