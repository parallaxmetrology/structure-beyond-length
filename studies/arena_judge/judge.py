#!/usr/bin/env python3
"""Arena 2024 judge run (PLAN.md). Uses the Message Batches API (50% price).

  python3 judge.py build                 # sample battles, write requests.jsonl (no API calls)
  python3 judge.py estimate              # token and cost estimate from the built requests (no API calls)
  python3 judge.py pilot --model M       # 20 requests sent directly, to check parsing and token use (cents)
  python3 judge.py submit --model M      # submit the batch (spends money; needs ANTHROPIC_API_KEY)
  python3 judge.py collect               # download results when the batch has ended (safe to repeat)

Model is chosen at submit time and recorded in results/meta.json."""
import argparse, json, runpy, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RES = HERE / "results"; RES.mkdir(exist_ok=True)
N_PER_COHORT = 2897
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0)}  # $/MTok standard; batch is half

SYSTEM = ("Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants "
          "to the user question displayed below. Choose the assistant that follows the user's instructions and answers "
          "the user's question better, considering helpfulness, relevance, accuracy, depth and level of detail. "
          "Do not let the order of the responses, their length or the assistants' names influence you. "
          "Give your verdict in the required JSON field: \"A\" if Assistant A is better, \"B\" if Assistant B is better, "
          "\"tie\" if they are equally good.")
SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["A", "B", "tie"]}},
          "required": ["verdict"], "additionalProperties": False}


def build():
    g = runpy.run_path(str(ROOT / "studies/arena100k/analyze_core.py"))
    AR, ids = g["AR"].copy(), list(g["ids"])
    AR["row"] = range(len(AR)); AR["bidx"] = ids
    AR["cohort"] = np.where(AR.both_style, "both", np.where(AR.neither_style, "neither", "mixed"))
    rng = np.random.default_rng(0)
    AR = AR[~AR.model_a.str.contains("claude", case=False) & ~AR.model_b.str.contains("claude", case=False)]
    pick = []
    for c in ["both", "neither"]:
        sub = AR[AR.cohort == c]
        pick.append(sub if len(sub) <= N_PER_COHORT else sub.iloc[rng.choice(len(sub), N_PER_COHORT, replace=False)])
    S = pd.concat(pick)
    d = pd.read_parquet(ROOT / "public_corpora/arena100k/arena-explorer-preference-100k.parquet")
    d = d[(d.language == "English") & (d.turn == 1) & d.winner.isin(["model_a", "model_b"])].reset_index(drop=True)
    with open(HERE / "requests.jsonl", "w") as f:
        for r in S.itertuples():
            b = d.iloc[r.bidx]
            q, a1, a2 = b.conversation_a[0]["content"], b.conversation_a[1]["content"], b.conversation_b[1]["content"]
            for order, (x, y) in (("ab", (a1, a2)), ("ba", (a2, a1))):
                user = (f"[User Question]\n{q}\n\n[The Start of Assistant A's Answer]\n{x}\n[The End of Assistant A's Answer]\n\n"
                        f"[The Start of Assistant B's Answer]\n{y}\n[The End of Assistant B's Answer]")
                f.write(json.dumps({"custom_id": f"{r.row}-{order}", "system": SYSTEM, "user": user}) + "\n")
    S[["row", "bidx", "cohort", "cluster", "y", "dlen", "drep4", "struct", "model_a", "model_b"]].to_csv(HERE / "sample.csv", index=False)
    print("battles", len(S), S.cohort.value_counts().to_dict(), "requests", 2 * len(S))


def estimate():
    n, chars = 0, 0
    for line in open(HERE / "requests.jsonl"):
        r = json.loads(line); n += 1; chars += len(r["system"]) + len(r["user"])
    tin = chars / 3.8  # rough chars-per-token for English; confirm with count_tokens on a sample before submitting
    for m, (pi, po) in PRICES.items():
        for out_tok in (20, 300):
            cost = (tin * pi + n * out_tok * po) / 1e6 / 2
            print(f"{m}: {n:,} requests, ~{tin/1e6:.1f}M input tokens, {out_tok} output tokens each -> ~${cost:,.0f} (batch)")


def _params(model, r):
    return {"model": model, "max_tokens": 4096, "system": SYSTEM,
            "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            "messages": [{"role": "user", "content": r["user"]}]}


def _text(msg):
    return "".join(bl.text for bl in msg.content if bl.type == "text").strip()


def _verdict(msg):
    try:
        v = json.loads(_text(msg)).get("verdict")
    except (ValueError, AttributeError):
        return "unparsed"
    return v if v in ("A", "B", "tie") else "unparsed"


def pilot(model):
    import anthropic
    client = anthropic.Anthropic()
    reqs = [json.loads(l) for l in open(HERE / "requests.jsonl")][:20]
    tin = tout = 0; out = []
    for r in reqs:
        msg = client.messages.create(**_params(model, r))
        v = _verdict(msg); tin += msg.usage.input_tokens; tout += msg.usage.output_tokens
        out.append((r["custom_id"], v, msg.stop_reason, msg.usage.input_tokens, msg.usage.output_tokens, _text(msg)[:300]))
        print(r["custom_id"], v, msg.stop_reason, msg.usage.input_tokens, msg.usage.output_tokens, flush=True)
    n = sum(1 for _ in open(HERE / "requests.jsonl"))
    pi, po = PRICES.get(model, (4.0, 20.0))
    full = (tin / len(reqs) * pi + tout / len(reqs) * po) * n / 1e6 / 2
    pairs = {}
    for cid, v, *_ in out:
        row, order = cid.split("-"); pairs.setdefault(row, {})[order] = v
    consistent = sum(1 for p in pairs.values() if {"ab", "ba"} <= p.keys() and
                     ((p["ab"] == "A" and p["ba"] == "B") or (p["ab"] == "B" and p["ba"] == "A") or (p["ab"] == p["ba"] == "tie")))
    pilot_cost = (tin * pi + tout * po) / 1e6
    (RES / f"pilot_{model}.json").write_text(json.dumps({"model": model, "rows": out, "pilot_cost_usd": pilot_cost,
                                                "batch_estimate_usd": full}, indent=1))
    print(f"unparsed {sum(v == 'unparsed' for _, v, *_ in out)}/{len(out)}; position-consistent {consistent}/{len(pairs)} battles")
    print(f"pilot cost ~${pilot_cost:.2f}; full batch estimate ~${full:,.0f} for {n:,} requests")


def submit(model):
    import anthropic
    client = anthropic.Anthropic()
    reqs = []
    for line in open(HERE / "requests.jsonl"):
        r = json.loads(line)
        reqs.append({"custom_id": r["custom_id"], "params": _params(model, r)})
    batch = client.messages.batches.create(requests=reqs)
    (RES / "meta.json").write_text(json.dumps({"model": model, "effort": "low", "batch_id": batch.id,
                                               "submitted": time.strftime("%Y-%m-%d %H:%M:%S"), "requests": len(reqs)}, indent=1))
    print("submitted", batch.id, len(reqs))


def collect():
    import anthropic
    client = anthropic.Anthropic()
    meta = json.loads((RES / "meta.json").read_text())
    b = client.messages.batches.retrieve(meta["batch_id"])
    print("status", b.processing_status, b.request_counts)
    if b.processing_status != "ended":
        return
    out = []
    for res in client.messages.batches.results(meta["batch_id"]):
        row = {"custom_id": res.custom_id, "type": res.result.type}
        if res.result.type == "succeeded":
            msg = res.result.message
            row.update(stop=msg.stop_reason, verdict=_verdict(msg),
                       in_tok=msg.usage.input_tokens, out_tok=msg.usage.output_tokens)
        out.append(row)
    pd.DataFrame(out).to_csv(RES / "verdicts.csv", index=False)
    print("collected", len(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--model", default="claude-opus-5-5")
    a = ap.parse_args()
    if a.cmd in ("pilot", "submit"):
        if (RES / "meta.json").exists() and a.cmd == "submit":
            sys.exit("a batch was already submitted (results/meta.json); run collect")
        {"pilot": pilot, "submit": submit}[a.cmd](a.model)
    else:
        {"build": build, "estimate": estimate, "collect": collect}[a.cmd]()
