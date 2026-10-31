"""Rerank a handful of passages with InfoMax+F and compare with the plain cross-encoder.

pip install "infomax-reranker[neural] @ git+https://github.com/AngshulMajumdar/infomax-reranker"
python examples/quickstart.py
"""
from infomax_reranker import InfoMaxReranker

query = "What are the side effects of metformin?"
passages = [
    "Metformin commonly causes gastrointestinal side effects such as nausea, diarrhoea and abdominal "
    "discomfort, especially when treatment starts.",
    "Gastrointestinal upset, including nausea and diarrhoea, is the most frequent side effect reported "
    "by patients starting metformin.",
    "Long-term metformin use is associated with reduced vitamin B12 absorption, so periodic B12 "
    "monitoring is recommended.",
    "Lactic acidosis is a rare but serious adverse effect of metformin, mainly in patients with severe "
    "kidney impairment.",
    "Metformin lowers hepatic glucose production and improves insulin sensitivity in type 2 diabetes.",
    "Taking metformin with meals or using an extended-release formulation can reduce stomach upset.",
    "Sulfonylureas such as glipizide can cause hypoglycaemia and weight gain.",
]

rr = InfoMaxReranker()  # downloads the two MiniLM models on first use (about 180 MB)
for method in ("ce", "infomax+f"):
    print(f"\n== {method}")
    for i, x in enumerate(rr.rerank(query, passages, k=4, method=method), 1):
        gain = f"  gain={x['gain']:.3f}" if "gain" in x else ""
        print(f"{i}. [{x['index']}] ce={x['ce_score']:.2f}{gain}  {x['passage'][:70]}...")

# Adaptive K: stop once a pick adds less than 0.05 nats of information.
picked = rr.rerank(query, passages, k=7, method="infomax+f", tau=0.05)
print(f"\nAdaptive K with tau=0.05 selected {len(picked)} passages:", [x["index"] for x in picked])
