"""InfoMax reranker demo (Gradio, ZeroGPU).

Neural scoring (MS MARCO MiniLM-L6 cross-encoder + all-MiniLM-L6-v2 embeddings) runs inside the
@spaces.GPU function; every selector then runs on CPU from those cached scores.
"""
import spaces  # import first on ZeroGPU

import os

import gradio as gr
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from infomax_rerank import by_score, fused, infomax, mmr, zfuse

CE_ID = os.environ.get("CE_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
DENSE_ID = os.environ.get("DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
ce = CrossEncoder(CE_ID, max_length=256, device="cpu")
enc = SentenceTransformer(DENSE_ID, device="cpu")

# Settings tuned on dev splits only (SciFact train, NFCorpus dev, FiQA dev); one setting for all datasets.
TUNED = {
    "InfoMax": dict(rank=1, gamma=4.0, rho=0.3, sigma2=0.03, beta=0.0, delta=0.0),
    "InfoMax+F": dict(rank=4, gamma=8.0, rho=1.0, sigma2=0.03, beta=3.0, delta=0.0, alpha=0.5),
    "MMR": dict(lam=0.95),
    "Fusion": dict(alpha=0.5),
}
METHODS = ["CE", "InfoMax", "InfoMax+F", "Dense", "MMR", "Fusion"]


@spaces.GPU(duration=60)
def neural_scores(query, passages):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ce.model.to(dev)
    enc.to(dev)
    logits = np.asarray(ce.predict([(query, p) for p in passages], batch_size=64,
                                   show_progress_bar=False), np.float32).reshape(-1)
    E = enc.encode([query] + passages, batch_size=64, normalize_embeddings=True,
                   convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    return logits, E[1:], E[0]


def split_passages(text):
    blocks = [b.strip() for b in text.replace("\r\n", "\n").split("\n\n")]
    return [b for b in blocks if b]


def rank_all(query, passages_text, k, rank, gamma, sigma2, beta, delta, alpha):
    query = (query or "").strip()
    passages = split_passages(passages_text or "")
    if not query:
        raise gr.Error("Enter a query.")
    if len(passages) < 2:
        raise gr.Error("Enter at least two passages, separated by a blank line.")
    if len(passages) > 200:
        raise gr.Error("At most 200 passages per request.")
    k = int(min(k, len(passages)))
    r, E, q = neural_scores(query, passages)
    user_f = dict(rank=int(rank), gamma=gamma, rho=1.0, sigma2=sigma2, beta=beta,
                  delta=np.inf if delta >= 10 else delta)
    orders, gains = {}, {}
    orders["CE"] = by_score(r)
    orders["Dense"] = by_score(E @ q)
    orders["MMR"] = mmr(r, E, K=k, **TUNED["MMR"])
    orders["Fusion"] = zfuse(r, E, q, **TUNED["Fusion"])
    orders["InfoMax"], gains["InfoMax"] = infomax(r, E, q, K=k, return_gains=True, **TUNED["InfoMax"])
    orders["InfoMax+F"], gains["InfoMax+F"] = infomax(fused(r, E, q, alpha), E, q, K=k, return_gains=True,
                                                       **user_f)
    short = lambda i: f"[{i + 1}] " + (passages[i][:90] + ("…" if len(passages[i]) > 90 else ""))
    table = pd.DataFrame({m: [short(i) for i in orders[m][:k]] for m in METHODS})
    table.insert(0, "Rank", list(range(1, k + 1)))
    detail = pd.DataFrame({
        "Passage": [f"[{i + 1}]" for i in range(len(passages))],
        "CE logit": np.round(r, 3),
        "Dense cosine": np.round(E @ q, 3),
        "InfoMax+F step": [orders["InfoMax+F"][:k].index(i) + 1 if i in orders["InfoMax+F"][:k] else "" for i in
                           range(len(passages))],
    })
    g = gains["InfoMax+F"]
    note = ("**InfoMax+F information gain per step (nats):** " +
            ", ".join(f"{t + 1}: {v:.3f}" for t, v in enumerate(g)))
    return table, detail, note


with open(os.path.join(os.path.dirname(__file__), "results", "beir_results.md")) as f:
    RESULTS_MD = f.read()

EX_PASSAGES = """Metformin commonly causes gastrointestinal side effects such as nausea, diarrhoea and abdominal discomfort, especially when treatment starts.

Gastrointestinal upset, including nausea and diarrhoea, is the most frequent side effect reported by patients starting metformin.

Long-term metformin use is associated with reduced vitamin B12 absorption, so periodic B12 monitoring is recommended.

Lactic acidosis is a rare but serious adverse effect of metformin, mainly in patients with severe kidney impairment.

Metformin lowers hepatic glucose production and improves insulin sensitivity in type 2 diabetes.

Taking metformin with meals or using an extended-release formulation can reduce stomach upset.

Sulfonylureas such as glipizide can cause hypoglycaemia and weight gain."""

with gr.Blocks(title="InfoMax reranker") as demo:
    gr.Markdown("# InfoMax reranker\nGreedy information-maximising selection over a candidate pool, compared with "
                "the standard MS MARCO MiniLM-L6 cross-encoder. Paste a query and passages separated by blank lines.")
    with gr.Tab("Rerank"):
        with gr.Row():
            with gr.Column(scale=1):
                query = gr.Textbox(label="Query", value="What are the side effects of metformin?")
                passages = gr.Textbox(label="Passages (blank line between passages)", lines=16, value=EX_PASSAGES)
                k = gr.Slider(1, 20, value=5, step=1, label="K (passages to select)")
                with gr.Accordion("InfoMax+F settings (defaults = dev-tuned)", open=False):
                    rank = gr.Slider(1, 16, value=4, step=1, label="Aspect rank r")
                    gamma = gr.Slider(0.5, 16, value=8.0, step=0.5, label="Relevance exponent γ")
                    sigma2 = gr.Slider(0.01, 1.0, value=0.03, step=0.01, label="Observation noise σ²")
                    beta = gr.Slider(0.0, 10.0, value=3.0, step=0.1, label="Shared nuisance variance β")
                    delta = gr.Slider(0.0, 10.0, value=0.0, step=0.1,
                                      label="Anchor margin δ (10 = never leave the fused ranking)")
                    alpha = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Dense weight α in relevance fusion")
                btn = gr.Button("Rerank", variant="primary")
            with gr.Column(scale=2):
                out = gr.Dataframe(label="Top-K per method", wrap=True)
                note = gr.Markdown()
                detail = gr.Dataframe(label="Per-passage scores", wrap=True)
        btn.click(rank_all, [query, passages, k, rank, gamma, sigma2, beta, delta, alpha], [out, detail, note])
    with gr.Tab("BEIR results"):
        gr.Markdown(RESULTS_MD)
    with gr.Tab("Method"):
        gr.Markdown(open(os.path.join(os.path.dirname(__file__), "METHOD.md")).read())

if __name__ == "__main__":
    demo.queue().launch()
