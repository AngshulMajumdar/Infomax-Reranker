"""Command line: rerank passages from a file.

  infomax-rerank --query "what causes tides?" --passages passages.txt --k 5
  infomax-rerank --query "..." --passages passages.jsonl --method ce --json

passages.txt  : passages separated by blank lines
passages.jsonl: one JSON object per line with a "text" field (or a plain JSON string per line)
"""
from __future__ import annotations

import argparse
import json
import sys

from .neural import METHODS, InfoMaxReranker


def read_passages(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    if path.endswith(".jsonl"):
        out = []
        for line in raw.splitlines():
            if line.strip():
                v = json.loads(line)
                out.append(v["text"] if isinstance(v, dict) else str(v))
        return out
    return [b.strip() for b in raw.replace("\r\n", "\n").split("\n\n") if b.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="infomax-rerank", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", required=True)
    ap.add_argument("--passages", required=True, help=".txt (blank-line separated) or .jsonl")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--method", default="infomax+f", choices=METHODS)
    ap.add_argument("--tau", type=float, default=None, help="InfoMax early-stop threshold on gain (nats)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    a = ap.parse_args(argv)
    passages = read_passages(a.passages)
    if not passages:
        ap.error("no passages found")
    res = InfoMaxReranker(device=a.device).rerank(a.query, passages, k=a.k, method=a.method, tau=a.tau)
    if a.json:
        json.dump(res, sys.stdout, indent=1)
        print()
    else:
        for i, x in enumerate(res, 1):
            g = f"  gain={x['gain']:.3f}" if "gain" in x else ""
            print(f"{i:>2}. [{x['index'] + 1}] ce={x['ce_score']:.2f} dense={x['dense_score']:.3f}{g}")
            print("    " + x["passage"][:160].replace("\n", " "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
