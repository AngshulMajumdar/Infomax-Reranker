# Reproducing the BEIR results

Install: `pip install -e "..[bench]"` from this folder, or `pip install -e ".[bench]"` from the repository root.

1. Download the five BEIR zips and unzip them into one folder:
   https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{scifact,nfcorpus,fiqa,scidocs,trec-covid}.zip
2. Set the paths and run the stages in order:

```bash
cd benchmark
export BEIR_DATA=/path/to/beir   # contains scifact/, nfcorpus/, fiqa/, scidocs/, trec-covid/
export EXP_OUT=/path/to/out

python bm25_pools.py                                                    # BM25 top-100 pools: test + 200-query dev samples
python dense_stage.py sentence-transformers/all-MiniLM-L6-v2 50         # embeddings of every pooled doc and query
python ce_stage.py cross-encoder/ms-marco-MiniLM-L6-v2 50               # cross-encoder logits, top-50 per query
python build_scores.py 50                                               # merge into <split>.scores.npz
python run_eval.py                                                      # tune each method on dev only -> best_params.json
python run_eval.py --test                                               # nDCG@k / Recall@k on test + bootstrap CIs
python mrr_recall.py                                                    # MRR@10 and Recall@3/5/10
```

The dense and cross-encoder stages are resumable. On two CPU threads they take about 30 and 80 minutes; on a GPU they
take a few minutes.

## Checks

- BM25 (Lucene-style k1 = 0.9, b = 0.4, Porter stemming) nDCG@10 is within about 0.01 of the published BEIR BM25
  numbers on all five datasets.
- all-MiniLM-L6-v2 over the full SciFact corpus gives nDCG@10 0.648.
- The cross-encoder reproduces its model-card example scores (8.607, −4.320).
- Running `run_eval.py --test` with the package's `infomax_reranker.core` reproduces the published test numbers and
  bootstrap intervals exactly.

## Files

- `common.py`: BEIR I/O and evaluation (pytrec_eval).
- `bm25_pools.py`, `dense_stage.py`, `ce_stage.py`, `build_scores.py`: the candidate pools and neural scores.
- `run_eval.py`: tuning grids, test evaluation and paired bootstrap.
- `mrr_recall.py`: MRR and recall tables.
