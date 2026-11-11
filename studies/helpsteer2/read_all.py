#!/usr/bin/env python3
"""Read every HelpSteer2 response (train + validation) with v19 through reading frame r1.
Writes data/readings.csv: one row per response with scores, prompt id, raw + framed readings."""
import gzip, hashlib, json, sys, time
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "reading_frame"))
import reading_frame as RF

SRC = ROOT / "public_corpora" / "helpsteer2"
rows = []
for split in ("train", "validation"):
    for line in gzip.open(SRC / f"{split}.jsonl.gz", "rt", encoding="utf-8"):
        r = json.loads(line); r["split"] = split; rows.append(r)

reader = RF.Reader()
out, t0 = [], time.time()
for i, r in enumerate(rows):
    row = RF.flat_row(reader.read(r["response"]))
    row.update({k: r[k] for k in ("helpfulness", "correctness", "coherence", "complexity", "verbosity", "split")})
    row["pid"] = hashlib.sha1(r["prompt"].encode("utf-8")).hexdigest()[:12]
    row["multi_turn"] = "<extra_id_1>" in r["prompt"]
    row["rid"] = i
    out.append(row)
    if i % 2000 == 0:
        print(i, f"{time.time()-t0:.0f}s", flush=True)

(HERE / "data").mkdir(exist_ok=True)
pd.DataFrame(out).to_csv(HERE / "data" / "readings.csv", index=False)
print("done", len(out), f"{time.time()-t0:.0f}s", "kernel", reader.kernel_sha256[:16])
