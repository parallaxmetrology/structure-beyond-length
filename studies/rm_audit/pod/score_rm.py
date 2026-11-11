#!/usr/bin/env python3
"""Score the Arena battles with one reward model. Writes results/<rm>.csv (one row per battle).

  python3 score_rm.py armorm [--limit N]   # ArmoRM-Llama3-8B-v0.1: scalar score per answer
  python3 score_rm.py gpm    [--limit N]   # GPM-Gemma-2B: pairwise, both orders
  python3 score_rm.py pairrm [--limit N]   # PairRM: pairwise, both orders

Columns: row, then armorm: score_a, score_b; gpm/pairrm: pref_ab (a over b), pref_ba (b over a).
Positive pref_ab / negative pref_ba favour answer a."""
import argparse, csv, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
MAXLEN = 4096

ap = argparse.ArgumentParser(); ap.add_argument("rm"); ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
battles = [json.loads(l) for l in open(HERE / "battles.jsonl")]
if args.limit:
    battles = battles[:args.limit]
suffix = "_smoke" if args.limit else ""
out_path = OUT / f"{args.rm}{suffix}.csv"
done = set()
if out_path.exists() and not args.limit:  # resume
    done = {int(r["row"]) for r in csv.DictReader(open(out_path))}
f = open(out_path, "a", newline=""); w = csv.writer(f)
if not done:
    f.seek(0); f.truncate()
t0 = time.time()

import torch
dev = "cuda" if torch.cuda.is_available() else "cpu"


def log(i, n):
    if i % 250 == 0:
        el = time.time() - t0
        print(f"{args.rm}: {i}/{n}  {el/60:.1f} min elapsed", flush=True)


if args.rm == "armorm":
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
    model = AutoModelForSequenceClassification.from_pretrained(path, device_map=dev, trust_remote_code=True, torch_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    if not done: w.writerow(["row", "score_a", "score_b"])
    msgs = lambda p, r: [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
    n_short = [0]
    def score(p, r):
        ids = tok.apply_chat_template(msgs(p, r), return_tensors="pt")
        if ids.shape[1] > MAXLEN:
            # shorten the texts, not the token ids, so the chat template (which ArmoRM's gating
            # looks for) stays intact: answer keeps up to half the budget, prompt the rest
            pt = tok.encode(p, add_special_tokens=False); rt = tok.encode(r, add_special_tokens=False)
            budget = MAXLEN - 64
            keep_r = min(len(rt), max(budget // 2, budget - len(pt)))
            keep_p = min(len(pt), budget - keep_r)
            ids = tok.apply_chat_template(msgs(tok.decode(pt[:keep_p]), tok.decode(rt[:keep_r])), return_tensors="pt")
            n_short[0] += 1
        with torch.no_grad():
            return model(ids.to(dev)).score.float().item()
    for i, b in enumerate(battles):
        if b["row"] in done: continue
        w.writerow([b["row"], score(b["prompt"], b["a"]), score(b["prompt"], b["b"])]); f.flush(); log(i, len(battles))
    print(f"armorm: {n_short[0]} answers shortened to fit {MAXLEN} tokens", flush=True)

elif args.rm == "gpm":
    sys.path.insert(0, str(HERE))
    from gpm_pipeline import GPMPipeline, generate_high_dim_result_with_prompt
    rm = GPMPipeline("general-preference/GPM-Gemma-2B", device=torch.device(dev), max_length=MAXLEN)
    if not done: w.writerow(["row", "pref_ab", "pref_ba"])
    def pref(p, x, y):
        c1 = [{"role": "user", "content": p}, {"role": "assistant", "content": x}]
        c2 = [{"role": "user", "content": p}, {"role": "assistant", "content": y}]
        r1, ph = rm([c1], return_prompt=True); r2 = rm([c2])
        return generate_high_dim_result_with_prompt(rm.model, rm.value_head_dim, r1, r2, ph).float().item()
    for i, b in enumerate(battles):
        if b["row"] in done: continue
        w.writerow([b["row"], pref(b["prompt"], b["a"], b["b"]), pref(b["prompt"], b["b"], b["a"])]); f.flush(); log(i, len(battles))

elif args.rm == "pairrm":
    import llm_blender
    blender = llm_blender.Blender(); blender.loadranker("llm-blender/PairRM")
    if not done: w.writerow(["row", "pref_ab", "pref_ba"])
    todo = [b for b in battles if b["row"] not in done]
    B = 16
    for s in range(0, len(todo), B):
        chunk = todo[s:s + B]
        P = [b["prompt"] for b in chunk]; A = [b["a"] for b in chunk]; Bb = [b["b"] for b in chunk]
        try:
            ab = blender.compare(P, A, Bb, return_logits=True, batch_size=B)
            ba = blender.compare(P, Bb, A, return_logits=True, batch_size=B)
            ab, ba = [float(x) for x in ab], [float(x) for x in ba]
        except TypeError:  # older llm-blender without return_logits: booleans as +1/-1
            ab = [1.0 if x else -1.0 for x in blender.compare(P, A, Bb, batch_size=B)]
            ba = [1.0 if x else -1.0 for x in blender.compare(P, Bb, A, batch_size=B)]
        for b, x, y in zip(chunk, ab, ba):
            w.writerow([b["row"], x, y])
        f.flush(); log(s, len(todo))
else:
    sys.exit("rm must be armorm, gpm or pairrm")
f.close()
print(f"{args.rm}: done, {len(battles)} battles, {(time.time()-t0)/60:.1f} min -> {out_path}", flush=True)
