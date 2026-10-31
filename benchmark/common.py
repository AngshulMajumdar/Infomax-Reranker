"""Shared I/O and evaluation for the BEIR reranking experiments."""
import csv
import json
import os

import numpy as np
import pytrec_eval

DATA = os.environ.get("BEIR_DATA", "beir")
OUT = os.environ.get("EXP_OUT", "out")
DATASETS = ["scifact", "nfcorpus", "fiqa", "scidocs", "trec-covid"]


def load_corpus(name):
    ids, texts, titles = [], [], []
    with open(os.path.join(DATA, name, "corpus.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            ids.append(d["_id"])
            titles.append(d.get("title") or "")
            texts.append(d.get("text") or "")
    return ids, titles, texts


def load_queries(name):
    q = {}
    with open(os.path.join(DATA, name, "queries.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            q[d["_id"]] = d["text"]
    return q


def load_qrels(name, split="test"):
    qrels = {}
    with open(os.path.join(DATA, name, "qrels", f"{split}.tsv")) as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for qid, did, score in r:
            qrels.setdefault(qid, {})[did] = int(score)
    return qrels


def evaluate(run, qrels, ks=(3, 5, 10)):
    """run: {qid: {docid: score}}. Returns mean nDCG@k and Recall@k (BEIR/pytrec_eval conventions)."""
    measures = {f"ndcg_cut.{','.join(map(str, ks))}", f"recall.{','.join(map(str, ks))}"}
    q = {k: v for k, v in qrels.items() if k in run}
    ev = pytrec_eval.RelevanceEvaluator(q, measures)
    res = ev.evaluate({k: run[k] for k in q})
    out = {}
    for k in ks:
        out[f"nDCG@{k}"] = float(np.mean([r[f"ndcg_cut_{k}"] for r in res.values()]))
        out[f"R@{k}"] = float(np.mean([r[f"recall_{k}"] for r in res.values()]))
    return out, res


def order_to_run(order_ids):
    """Selected ordered list -> descending pseudo-scores so pytrec_eval keeps the order."""
    n = len(order_ids)
    return {d: float(n - i) for i, d in enumerate(order_ids)}
