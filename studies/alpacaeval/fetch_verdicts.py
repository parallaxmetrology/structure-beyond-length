#!/usr/bin/env python3
"""Stream each model's GPT-4-turbo judge verdicts from the AlpacaEval repo and keep only
instruction / dataset / generator_1 / generator_2 / preference. ~1 GB transferred, a few MB kept."""
import json, urllib.request, concurrent.futures as cf
from pathlib import Path
HERE = Path(__file__).resolve().parent
tree = json.load(urllib.request.urlopen("https://api.github.com/repos/tatsu-lab/alpaca_eval/git/trees/main?recursive=1"))
paths = [x["path"] for x in tree["tree"] if x["path"].endswith("/weighted_alpaca_eval_gpt4_turbo/annotations.json")
         and x["path"].count("/") == 3]
BASE = "https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/"
def get(p):
    for attempt in range(3):
        try:
            d = json.load(urllib.request.urlopen(BASE + p, timeout=120))
            return p, [{k: r.get(k) for k in ("instruction", "dataset", "generator_1", "generator_2", "preference")} for r in d], None
        except Exception as e:
            err = repr(e)
    return p, [], err
rows, fails = [], []
with cf.ThreadPoolExecutor(4) as ex:
    for i, (p, rs, err) in enumerate(ex.map(get, paths)):
        for r in rs: r["folder"] = p.split("/")[1]
        rows += rs
        if err: fails.append((p, err))
        if i % 20 == 0: print(i, len(paths), flush=True)
(HERE / "data" / "verdicts.json").write_text(json.dumps(rows))
print("done", len(paths), "files", len(rows), "rows", "fails", fails)
