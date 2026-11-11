#!/usr/bin/env python3
"""Rebuild battles.jsonl (the reward models' input) from the Arena 100k file and the judge-run sample.
Not shipped in this repository because it contains model answers, which fall under each provider's terms."""
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent.parent
S = pd.read_csv(HERE.parent / "arena_judge" / "sample.csv")
d = pd.read_parquet(ROOT / "public_corpora/arena100k/arena-explorer-preference-100k.parquet")
d = d[(d.language == "English") & (d.turn == 1) & d.winner.isin(["model_a", "model_b"])].reset_index(drop=True)
with open(HERE / "battles.jsonl", "w") as f:
    for r in S.itertuples():
        b = d.iloc[r.bidx]
        f.write(json.dumps({"row": int(r.row), "prompt": b.conversation_a[0]["content"],
                            "a": b.conversation_a[1]["content"], "b": b.conversation_b[1]["content"]}) + "\n")
print(len(S), "battles ->", HERE / "battles.jsonl")
