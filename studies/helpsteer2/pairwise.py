#!/usr/bin/env python3
"""Added analysis F (not in the frozen plan): a model trained on pair differences.

Analysis C fits one pooled model across all responses, so its weights mix
between-prompt and within-prompt patterns. Here each example is one prompt's
pair: features are differences (A minus B), target is which response scored
higher. Pairs with equal scores are dropped. 5-fold CV, so every pair (= prompt)
is held out once. Writes data/pairwise.json."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
DIMS = ["Δx", "rᵥ", "ρᵣ", "μ", "Δy", "xₚ", "θ", "dₛ"]
SCORES = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
rng = np.random.default_rng(0)

m = pd.read_csv(HERE / "data" / "readings.csv")
m = m[m.eligible].copy()
m["loglen"] = np.log1p(m.prose_tokens)
pairs = m.groupby("pid").filter(lambda g: len(g) == 2).sort_values(["pid", "rid"])
a, b = pairs.iloc[0::2].reset_index(drop=True), pairs.iloc[1::2].reset_index(drop=True)


def diffs(side):
    D = pd.DataFrame({"dlen": a.loglen - b.loglen})
    for d in DIMS:
        D[d] = a[f"{side}_{d}"] - b[f"{side}_{d}"]
    return D


def cv_prob(X, y):
    p = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        p[te] = LogisticRegression(max_iter=2000).fit((X[tr] - mu) / sd, y[tr]).predict_proba((X[te] - mu) / sd)[:, 1]
    return p


def boot(y, p0, p1, n=1000):
    acc_g, auc_g = [], []
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        acc_g.append(((p1[ix] > .5) == y[ix]).mean() - ((p0[ix] > .5) == y[ix]).mean())
        auc_g.append(roc_auc_score(y[ix], p1[ix]) - roc_auc_score(y[ix], p0[ix]))
    return [float(np.percentile(acc_g, q)) for q in (2.5, 97.5)], [float(np.percentile(auc_g, q)) for q in (2.5, 97.5)]


out = {}
Df, Dr = diffs("framed"), diffs("raw")
for s in SCORES:
    keep = (a[s] != b[s]).values
    y = (a[s] > b[s]).values[keep].astype(int)
    p0 = cv_prob(Df[["dlen"]].values[keep], y)
    p1 = cv_prob(Df.values[keep], y)
    p2 = cv_prob(Dr.values[keep], y)
    pk = cv_prob(Df[DIMS].values[keep], y)  # kernel alone, no length
    acc = lambda p: float(((p > .5) == y).mean())
    auc = lambda p: float(roc_auc_score(y, p))
    acc_ci, auc_ci = boot(y, p0, p1)
    Xs = Df.values[keep]; Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-9)
    coef = LogisticRegression(max_iter=2000).fit(Xs, y).coef_[0]
    out[s] = {"pairs": int(keep.sum()),
              "acc_len": acc(p0), "acc_len_framed": acc(p1), "acc_len_raw": acc(p2), "acc_kernel_only": acc(pk),
              "auc_len": auc(p0), "auc_len_framed": auc(p1), "auc_len_raw": auc(p2), "auc_kernel_only": auc(pk),
              "acc_gain_ci": acc_ci, "auc_gain_ci": auc_ci,
              "coef": dict(zip(["dlen"] + DIMS, np.round(coef, 3).tolist()))}
    o = out[s]
    print(f"{s:12s} pairs {o['pairs']:5d}  AUC len {o['auc_len']:.3f} +framed {o['auc_len_framed']:.3f} "
          f"(gain CI {auc_ci[0]:+.3f},{auc_ci[1]:+.3f}) +raw {o['auc_len_raw']:.3f} kernel-only {o['auc_kernel_only']:.3f} | "
          f"acc len {o['acc_len']:.3f} +framed {o['acc_len_framed']:.3f}")
    print("   coef", o["coef"])

(HERE / "data" / "pairwise.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


# ---------- robustness (added): drop single-segment pairs; add a repetition feature ----------
import gzip, re
SRC = HERE.parent.parent / "public_corpora" / "helpsteer2"
resp = []
for split in ("train", "validation"):
    for line in gzip.open(SRC / f"{split}.jsonl.gz", "rt", encoding="utf-8"):
        resp.append(json.loads(line)["response"])
WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def rep4(t):
    w = WORD.findall(t.lower()); g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 0.0 if not g else 1 - len(set(g)) / len(g)


ra = np.array([rep4(resp[i]) for i in a.rid]); rb = np.array([rep4(resp[i]) for i in b.rid])
multi = ((a.framed_sentence_count > 1) & (b.framed_sentence_count > 1)).values
rob = {}
print("\nrobustness")
for s in SCORES:
    base = (a[s] != b[s]).values
    res = {}
    keep = base & multi
    y = (a[s] > b[s]).values[keep].astype(int)
    p0 = cv_prob(Df[["dlen"]].values[keep], y); p1 = cv_prob(Df.values[keep], y)
    _, auc_ci = boot(y, p0, p1)
    res["multi_segment_only"] = {"pairs": int(keep.sum()), "auc_len": float(roc_auc_score(y, p0)),
                                 "auc_len_framed": float(roc_auc_score(y, p1)), "auc_gain_ci": auc_ci}
    keep = base
    y = (a[s] > b[s]).values[keep].astype(int)
    X0 = np.column_stack([Df[["dlen"]].values, ra - rb])[keep]
    X1 = np.column_stack([Df.values, ra - rb])[keep]
    q0 = cv_prob(X0, y); q1 = cv_prob(X1, y)
    _, auc_ci2 = boot(y, q0, q1)
    res["with_rep4"] = {"pairs": int(keep.sum()), "auc_len_rep4": float(roc_auc_score(y, q0)),
                        "auc_len_rep4_framed": float(roc_auc_score(y, q1)), "auc_gain_ci": auc_ci2}
    rob[s] = res
    print(f"{s:12s} multi-seg: len {res['multi_segment_only']['auc_len']:.3f} +k {res['multi_segment_only']['auc_len_framed']:.3f} CI {np.round(auc_ci,3)} | "
          f"len+rep4 {res['with_rep4']['auc_len_rep4']:.3f} +k {res['with_rep4']['auc_len_rep4_framed']:.3f} CI {np.round(auc_ci2,3)}")
(HERE / "data" / "pairwise_robustness.json").write_text(json.dumps(rob, indent=1), encoding="utf-8")
