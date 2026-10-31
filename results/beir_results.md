## BEIR results (test splits)

Setup: BM25 (k1=0.9, b=0.4, Porter) top-50 candidates per query; MS MARCO MiniLM-L6-v2 cross-encoder (max length 256);
all-MiniLM-L6-v2 embeddings. All knobs tuned once on dev splits only (SciFact train, NFCorpus dev, FiQA dev; 200 queries each)
and applied unchanged to all five test sets.

- **CE**: cross-encoder ranking (the standard reranker).
- **InfoMax**: information-maximising greedy selection, relevance = CE.
- **InfoMax+F**: same selector, relevance = z-score fusion of CE and dense cosine.
- **BM25 / Dense**: first-stage scores of the same pool (no reranker). **MMR**: CE relevance, dense redundancy, λ=0.95.
  **Fusion**: z-score average of CE and dense cosine (α=0.5), no set selection.

### nDCG@10

| Dataset | BM25 | Dense | CE | MMR | Fusion | InfoMax | InfoMax+F |
|---|---|---|---|---|---|---|---|
| SciFact | 0.6790 | 0.6776 | 0.6763 | 0.6737 | 0.7121 | 0.6948 | **0.7155** |
| NFCorpus | 0.3261 | 0.3297 | 0.3459 | 0.3450 | 0.3513 | 0.3459 | **0.3522** |
| FiQA | 0.2431 | 0.3455 | 0.3224 | 0.3208 | **0.3585** | 0.3369 | 0.3577 |
| SciDocs | 0.1490 | **0.1987** | 0.1639 | 0.1627 | 0.1937 | 0.1778 | 0.1944 |
| TREC-COVID | 0.6451 | 0.7502 | 0.7443 | 0.7379 | **0.7971** | 0.7131 | 0.7896 |

### MRR@10

| Dataset | BM25 | Dense | CE | MMR | Fusion | InfoMax | InfoMax+F |
|---|---|---|---|---|---|---|---|
| SciFact | 0.6437 | 0.6298 | 0.6488 | 0.6475 | 0.6795 | 0.6590 | **0.6816** |
| NFCorpus | 0.5235 | 0.5333 | 0.5770 | 0.5773 | 0.5721 | **0.5800** | 0.5688 |
| FiQA | 0.3025 | 0.4357 | 0.4015 | 0.4008 | **0.4505** | 0.4151 | 0.4490 |
| SciDocs | 0.2617 | **0.3503** | 0.2934 | 0.2926 | 0.3368 | 0.3089 | 0.3380 |
| TREC-COVID | 0.8967 | 0.9067 | 0.8880 | 0.8973 | 0.9500 | 0.8983 | **0.9533** |

### Recall@3

| Dataset | BM25 | Dense | CE | MMR | Fusion | InfoMax | InfoMax+F |
|---|---|---|---|---|---|---|---|
| SciFact | 0.7109 | 0.6764 | 0.6833 | 0.6916 | 0.7160 | 0.7107 | **0.7196** |
| NFCorpus | 0.1038 | 0.1054 | 0.1126 | 0.1122 | **0.1174** | 0.1172 | 0.1158 |
| FiQA | 0.1991 | 0.2963 | 0.2815 | 0.2795 | **0.3067** | 0.2997 | 0.3067 |
| SciDocs | 0.0785 | **0.1106** | 0.0890 | 0.0876 | 0.1056 | 0.1026 | 0.1052 |
| TREC-COVID | 0.0061 | 0.0068 | 0.0065 | 0.0064 | 0.0070 | 0.0063 | **0.0071** |

### Recall@5

| Dataset | BM25 | Dense | CE | MMR | Fusion | InfoMax | InfoMax+F |
|---|---|---|---|---|---|---|---|
| SciFact | 0.7526 | 0.7553 | 0.7502 | 0.7502 | 0.7628 | 0.7646 | **0.7728** |
| NFCorpus | 0.1233 | 0.1299 | 0.1331 | 0.1330 | **0.1364** | 0.1352 | 0.1360 |
| FiQA | 0.2419 | 0.3457 | 0.3263 | 0.3258 | 0.3484 | **0.3503** | 0.3445 |
| SciDocs | 0.1091 | **0.1507** | 0.1207 | 0.1184 | 0.1428 | 0.1334 | 0.1447 |
| TREC-COVID | 0.0095 | 0.0110 | 0.0109 | 0.0107 | 0.0113 | 0.0100 | **0.0115** |

### Recall@10

| Dataset | BM25 | Dense | CE | MMR | Fusion | InfoMax | InfoMax+F |
|---|---|---|---|---|---|---|---|
| SciFact | 0.8077 | **0.8396** | 0.7884 | 0.7818 | 0.8312 | 0.8252 | 0.8369 |
| NFCorpus | 0.1564 | 0.1574 | 0.1624 | 0.1610 | **0.1642** | 0.1602 | 0.1637 |
| FiQA | 0.2968 | 0.3935 | 0.3740 | 0.3725 | **0.4036** | 0.3929 | 0.4017 |
| SciDocs | 0.1565 | **0.2025** | 0.1685 | 0.1677 | 0.1999 | 0.1830 | 0.2001 |
| TREC-COVID | 0.0176 | 0.0216 | 0.0206 | 0.0204 | **0.0221** | 0.0193 | 0.0220 |

### InfoMax+F minus baseline, nDCG@10, paired bootstrap 95% CI

| Dataset | vs CE | vs MMR | vs Fusion |
|---|---|---|---|
| SciFact | +0.0392 [+0.0229, +0.0560] | +0.0418 [+0.0252, +0.0587] | +0.0034 [-0.0009, +0.0080] |
| NFCorpus | +0.0063 [+0.0001, +0.0127] | +0.0072 [+0.0010, +0.0137] | +0.0009 [-0.0021, +0.0042] |
| FiQA | +0.0353 [+0.0262, +0.0447] | +0.0369 [+0.0278, +0.0461] | -0.0007 [-0.0034, +0.0019] |
| SciDocs | +0.0305 [+0.0254, +0.0357] | +0.0317 [+0.0266, +0.0369] | +0.0007 [-0.0012, +0.0027] |
| TREC-COVID | +0.0453 [+0.0082, +0.0853] | +0.0517 [+0.0143, +0.0917] | -0.0075 [-0.0216, +0.0066] |

**Reading.** InfoMax+F beats the cross-encoder on nDCG@10 on all five datasets (intervals above zero). It ties Fusion everywhere:
on these single-hop benchmarks the gain comes from adding dense relevance, not from set selection. nDCG does not penalise
redundant relevant passages. TREC-COVID recall is near zero by construction (~494 relevant documents per query).
