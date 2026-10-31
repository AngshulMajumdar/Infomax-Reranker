---
title: InfoMax Reranker
emoji: 🔎
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
short_description: Entropy-minimising passage reranker vs MS MARCO CE
---

# InfoMax Reranker

A greedy, information-maximising selector over a retrieved candidate pool, compared with the standard
`cross-encoder/ms-marco-MiniLM-L6-v2` reranker. Neural scoring runs on ZeroGPU; selection runs on CPU.

- `app.py`: Gradio app. Paste a query and passages separated by blank lines.
- `infomax_rerank.py`: the selectors: `infomax`, plus the baselines `mmr`, `fused`/`zfuse` and `by_score`.
- `METHOD.md`: the model and the selection rule.
- `results/`: BEIR test results (`beir_results.md`, `beir_results.json`).

Code, benchmark scripts and the write-up: https://github.com/AngshulMajumdar/infomax-reranker

## Hardware

Create the Space with the Gradio SDK and select **ZeroGPU** as hardware. Hosting a ZeroGPU Space needs a PRO or Enterprise
account. The `@spaces.GPU` decorator has no effect elsewhere, so the app also runs on CPU hardware and locally:

```
pip install -r requirements.txt gradio
python app.py
```
