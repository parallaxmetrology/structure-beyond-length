#!/usr/bin/env python3
"""Read the judged answers of English, single-turn, non-tie Arena 100k battles with v19 + frame r1.
Writes data/battles.csv (one row per battle) and data/answers.csv.gz (one row per answer)."""
import re, sys, time
from multiprocessing import Pool
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "reading_frame"))
WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def rep4(t):
    w = WORD.findall(t.lower()); g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 0.0 if not g else 1 - len(set(g)) / len(g)


def last_answer(conv):
    msgs = [m["content"] for m in conv if m["role"] == "assistant"]
    return msgs[-1] if msgs else ""


def work(chunk):
    import reading_frame as RF
    reader = RF.Reader()
    out = []
    for key, text in chunk:
        row = RF.flat_row(reader.read(text)); row.update(key=key, rep4=rep4(text)); out.append(row)
    return out


if __name__ == "__main__":
    d = pd.read_parquet(ROOT / "public_corpora/arena100k/arena-explorer-preference-100k.parquet")
    n0 = len(d)
    d = d[(d.language == "English") & (d.turn == 1) & d.winner.isin(["model_a", "model_b"])].reset_index(drop=True)
    d["bid"] = range(len(d))
    items = []
    for r in d.itertuples():
        items.append((f"{r.bid}|a", last_answer(r.conversation_a)))
        items.append((f"{r.bid}|b", last_answer(r.conversation_b)))
    chunks = [items[i:i + 2000] for i in range(0, len(items), 2000)]
    t0 = time.time(); rows = []
    with Pool(8) as p:
        for k, res in enumerate(p.imap_unordered(work, chunks)):
            rows += res
            if k % 10 == 0:
                print(k, len(chunks), f"{time.time()-t0:.0f}s", flush=True)
    (HERE / "data").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(HERE / "data" / "answers.csv.gz", index=False)
    d[["bid", "question_id", "model_a", "model_b", "winner", "is_code", "is_refusal", "tstamp", "anony"]].to_csv(HERE / "data" / "battles.csv", index=False)
    print("done", n0, "battles total;", len(d), "English single-turn non-tie;", len(rows), "answers", f"{time.time()-t0:.0f}s")
