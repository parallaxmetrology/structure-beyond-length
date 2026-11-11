#!/usr/bin/env python3
"""Read every AlpacaEval model answer with v19 through reading frame r1 (+ repetition).
Writes data/readings.parquet-like CSV (data/readings.csv.gz): one row per (model folder, instruction)."""
import glob, hashlib, json, re, sys, time
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


def work(path):
    import reading_frame as RF
    reader = RF.Reader()
    folder = Path(path).parent.name
    rows = []
    for i, r in enumerate(json.load(open(path))):
        out = r.get("output") or ""
        if not isinstance(out, str):
            out = str(out)
        row = RF.flat_row(reader.read(out))
        row.update(folder=folder, generator=r.get("generator"), dataset=r.get("dataset"), idx=i,
                   iid=hashlib.sha1(r["instruction"].encode()).hexdigest()[:12], rep4=rep4(out), chars=len(out))
        rows.append(row)
    return rows


if __name__ == "__main__":
    files = sorted(glob.glob(str(ROOT / "public_corpora" / "alpaca_eval" / "results" / "*" / "model_outputs.json")))
    t0 = time.time(); allrows = []
    with Pool(8) as p:
        for k, rows in enumerate(p.imap_unordered(work, files)):
            allrows += rows
            if k % 25 == 0:
                print(k, len(files), f"{time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(allrows).to_csv(HERE / "data" / "readings.csv.gz", index=False)
    print("done", len(files), "models", len(allrows), "answers", f"{time.time()-t0:.0f}s")
