"""First stage: Lucene-style BM25 (k1=0.9, b=0.4, Porter stemming) top-100 pools per test query."""
import json
import os
import re
import time

import numpy as np
import scipy.sparse as sp
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from common import DATASETS, OUT, evaluate, load_corpus, load_qrels, load_queries

TOK = re.compile(r"[a-z0-9]+")
_stem = PorterStemmer().stem
_cache = {}


def analyze(text):
    out = []
    for t in TOK.findall(text.lower()):
        if t in ENGLISH_STOP_WORDS:
            continue
        s = _cache.get(t)
        if s is None:
            s = _cache[t] = _stem(t)
        out.append(s)
    return out


def run(name, split="test", depth=100, k1=0.9, b=0.4, max_q=None, seed=0):
    t0 = time.time()
    ids, titles, texts = load_corpus(name)
    docs = [(ti + " " + tx).strip() for ti, tx in zip(titles, texts)]
    queries = load_queries(name)
    qrels = load_qrels(name, split)
    qids = sorted(q for q in qrels if q in queries)
    if max_q and len(qids) > max_q:
        qids = sorted(np.random.default_rng(seed).choice(qids, max_q, replace=False).tolist())
    cv = CountVectorizer(analyzer=analyze)
    X = cv.fit_transform(docs).tocsc().astype(np.float32)  # docs x terms
    N = X.shape[0]
    df = np.diff(X.indptr)
    idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
    X = X.tocsr()
    dl = np.asarray(X.sum(1)).ravel()
    norm = k1 * (1 - b + b * dl / dl.mean())
    X = X.tocoo()
    w = X.data * (k1 + 1) / (X.data + norm[X.row]) * idf[X.col]
    W = sp.csc_matrix((w, (X.row, X.col)), shape=X.shape)
    vocab = cv.vocabulary_
    pools, run_ = {}, {}
    for qid in qids:
        terms = [vocab[t] for t in analyze(queries[qid]) if t in vocab]
        if not terms:
            s = np.zeros(N, dtype=np.float32)
        else:
            u, c = np.unique(terms, return_counts=True)
            s = np.asarray(W[:, u] @ c.astype(np.float32)).ravel()
        top = np.argpartition(-s, depth)[:depth]
        top = top[np.argsort(-s[top], kind="stable")]
        pools[qid] = {"ids": [ids[i] for i in top], "bm25": s[top].tolist()}
        run_[qid] = {ids[i]: float(s[i]) for i in top}
    metrics, _ = evaluate(run_, qrels)
    os.makedirs(OUT, exist_ok=True)
    json.dump(pools, open(os.path.join(OUT, f"{name}.{split}.pools.json"), "w"))
    print(f"{name}/{split}: {N} docs, {len(qids)} queries, {time.time()-t0:.0f}s  " +
          "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)


if __name__ == "__main__":
    for n in DATASETS:
        run(n, "test")
    for n, sp_ in [("scifact", "train"), ("nfcorpus", "dev"), ("fiqa", "dev")]:
        run(n, sp_, max_q=200)
