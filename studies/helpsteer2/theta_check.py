#!/usr/bin/env python3
"""Added analysis G: what θ carries within pairs.

θ = 1 - mean word-cosine between adjacent sentences. Hypothesis (formed after F):
low θ marks repetition, which annotators score down. Second bearing: share of
repeated word 4-grams, computed here from the text, not by the kernel. Same text,
different measurement, so not fully independent (stated in the lab book).
Writes data/theta_check.json and data/theta_examples.json."""
import gzip, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "public_corpora" / "helpsteer2"
texts = []
for split in ("train", "validation"):
    for line in gzip.open(SRC / f"{split}.jsonl.gz", "rt", encoding="utf-8"):
        r = json.loads(line); texts.append((r["prompt"], r["response"]))

WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def rep4(t):
    w = WORD.findall(t.lower())
    g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 0.0 if not g else 1 - len(set(g)) / len(g)


m = pd.read_csv(HERE / "data" / "readings.csv")
m = m[m.eligible].copy()
m["loglen"] = np.log1p(m.prose_tokens)
m["rep4"] = [rep4(texts[i][1]) for i in m.rid]
pairs = m.groupby("pid").filter(lambda g: len(g) == 2).sort_values(["pid", "rid"])
a, b = pairs.iloc[0::2].reset_index(drop=True), pairs.iloc[1::2].reset_index(drop=True)


def resid(y, X):
    X = np.column_stack([np.ones(len(y))] + list(X))
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))


rng = np.random.default_rng(0)
idx = [rng.integers(0, len(a), len(a)) for _ in range(500)]
ci = lambda f: [float(np.percentile([f(ix) for ix in idx], q)) for q in (2.5, 97.5)]

dL = rankdata(a.loglen.values - b.loglen.values)
dT = rankdata(a["framed_θ"].values - b["framed_θ"].values)
dR = rankdata(a.rep4.values - b.rep4.values)
out = {"pooled_spearman_theta_rep4": corr(rankdata(m["framed_θ"]), rankdata(m.rep4)),
       "pair_spearman_dtheta_drep4": corr(dT, dR)}
for s in ["helpfulness", "correctness", "coherence"]:
    dS = rankdata(a[s].values - b[s].values)
    xT, yS = resid(dT, [dL]), resid(dS, [dL])
    xT2, yS2 = resid(dT, [dL, dR]), resid(dS, [dL, dR])
    xR, yR = resid(dR, [dL]), resid(dS, [dL])
    out[s] = {
        "theta_given_len": corr(xT, yS), "theta_given_len_ci": ci(lambda ix: corr(xT[ix], yS[ix])),
        "theta_given_len_rep4": corr(xT2, yS2), "theta_given_len_rep4_ci": ci(lambda ix: corr(xT2[ix], yS2[ix])),
        "rep4_given_len": corr(xR, yR), "rep4_given_len_ci": ci(lambda ix: corr(xR[ix], yR[ix])),
    }
    print(s, {k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in out[s].items()})
print("theta vs rep4", round(out["pooled_spearman_theta_rep4"], 3), "pair", round(out["pair_spearman_dtheta_drep4"], 3))

# θ deciles vs scores and repetition (pooled)
m["theta_dec"] = pd.qcut(m["framed_θ"], 10, labels=False, duplicates="drop")
dec = m.groupby("theta_dec").agg(theta=("framed_θ", "median"), rep4=("rep4", "median"),
                                 coherence=("coherence", "mean"), helpfulness=("helpfulness", "mean"),
                                 tokens=("prose_tokens", "median"), n=("rid", "size")).round(3)
print(dec.to_string())
out["theta_deciles"] = dec.reset_index().to_dict("records")

# examples: pairs with the largest θ gap where the lower-θ answer scored lower on coherence
d = pd.DataFrame({"ra": a.rid, "rb": b.rid, "dth": a["framed_θ"] - b["framed_θ"], "dco": a.coherence - b.coherence,
                  "dlen": a.loglen - b.loglen})
d = d[(d.dth * d.dco > 0) & (d.dlen.abs() < 0.2)].assign(g=lambda x: x.dth.abs()).sort_values("g", ascending=False).head(6)
ex = []
for _, r in d.iterrows():
    lo, hi = (r.ra, r.rb) if r.dth < 0 else (r.rb, r.ra)
    row = lambda i: {"rid": int(i), "theta": float(m.set_index("rid").loc[i, "framed_θ"]),
                     "coherence": int(m.set_index("rid").loc[i, "coherence"]),
                     "rep4": float(m.set_index("rid").loc[i, "rep4"]), "text": texts[int(i)][1][:700]}
    ex.append({"prompt": texts[int(lo)][0][:300], "low_theta": row(lo), "high_theta": row(hi)})
(HERE / "data" / "theta_examples.json").write_text(json.dumps(ex, indent=1, ensure_ascii=False), encoding="utf-8")
(HERE / "data" / "theta_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
for e in ex[:4]:
    print("\nPROMPT:", e["prompt"][:150].replace("\n", " "))
    for k in ("low_theta", "high_theta"):
        print(f"  {k} θ={e[k]['theta']:.2f} coh={e[k]['coherence']} rep4={e[k]['rep4']:.2f}: {e[k]['text'][:260]!r}")
