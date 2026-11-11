#!/usr/bin/env python3
"""Analyses A-E from PLAN.md. Reads data/readings.csv, writes data/results.json."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
DIMS = ["Δx", "rᵥ", "ρᵣ", "μ", "Δy", "xₚ", "θ", "dₛ"]
SCORES = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
B = 1000
rng = np.random.default_rng(0)

m = pd.read_csv(HERE / "data" / "readings.csv")
m["loglen"] = np.log1p(m.prose_tokens)
E = m[m.eligible].dropna(subset=[f"framed_{d}" for d in DIMS] + [f"raw_{d}" for d in DIMS]).reset_index(drop=True)
R = {"n_all": len(m), "n_eligible": len(E), "n_prompts_eligible": int(E.pid.nunique()),
     "ineligible_reasons": m.reasons[m.reasons.fillna("") != ""].str.split(";").explode().value_counts().to_dict()}


def resid(y, x):
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def boot_groups(groups):
    """Index arrays for prompt-level bootstrap."""
    codes, uniq = pd.factorize(groups)
    members = [np.where(codes == k)[0] for k in range(len(uniq))]
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([members[k] for k in pick])


def ci(vals):
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ---------- A. pooled Spearman, plain and length-partialled ----------
def pooled(df, side):
    rk = {c: rankdata(df[c]) for c in [f"{side}_{d}" for d in DIMS] + SCORES + ["loglen"]}
    L = rk["loglen"]
    out = {}
    idx_sets = list(boot_groups(df.pid.values))
    for d in DIMS:
        a = rk[f"{side}_{d}"]; ar = resid(a, L)
        for s in SCORES:
            b = rk[s]; br = resid(b, L)
            plain, part = corr(a, b), corr(ar, br)
            bs = [corr(ar[ix], br[ix]) for ix in idx_sets[:300]]  # 300 draws for the 40-cell grid
            out[f"{d}|{s}"] = {"plain": plain, "partial": part, "partial_ci": ci(bs)}
    lenc = {s: corr(L, rk[s]) for s in SCORES}
    return out, lenc

R["A_framed"], R["length_spearman"] = pooled(E, "framed")
R["A_raw"], _ = pooled(E, "raw")

# ---------- B. within pair ----------
pairs = E.groupby("pid").filter(lambda g: len(g) == 2).sort_values(["pid", "rid"])
a, b = pairs.iloc[0::2].reset_index(drop=True), pairs.iloc[1::2].reset_index(drop=True)
R["B_n_pairs"] = len(a)
dL = rankdata(a.loglen.values - b.loglen.values)
Bres = {}
idxs = [rng.integers(0, len(a), len(a)) for _ in range(300)]
for d in DIMS:
    x = resid(rankdata(a[f"framed_{d}"].values - b[f"framed_{d}"].values), dL)
    for s in SCORES:
        y = resid(rankdata(a[s].values - b[s].values), dL)
        Bres[f"{d}|{s}"] = {"partial": corr(x, y), "ci": ci([corr(x[ix], y[ix]) for ix in idxs])}
R["B_pairs"] = Bres
R["B_length_diff_spearman"] = {s: corr(rankdata(a.loglen.values - b.loglen.values), rankdata(a[s].values - b[s].values)) for s in SCORES}


# ---------- C/D. held-out prediction ----------
def heldout(df, cols, target):
    X = df[cols].values; y = df[target].values
    pred = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=5).split(X, y, df.pid.values):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        mdl = LinearRegression().fit((X[tr] - mu) / sd, y[tr])
        pred[te] = mdl.predict((X[te] - mu) / sd)
    return pred


E["loglen2"] = E.loglen ** 2
LEN = ["loglen", "loglen2"]
C = {}
pair_ix = {pid: ix for pid, ix in E.groupby("pid").indices.items() if len(ix) == 2}
for s in SCORES:
    y = E[s].values
    p0 = heldout(E, LEN, s)
    p1 = heldout(E, LEN + [f"framed_{d}" for d in DIMS], s)
    p2 = heldout(E, LEN + [f"raw_{d}" for d in DIMS], s)
    r0, r1, r2 = (rankdata(p) for p in (p0, p1, p2)); ry = rankdata(y)
    g1, g2 = [], []
    for ix in boot_groups(E.pid.values):
        c0 = corr(r0[ix], ry[ix]); g1.append(corr(r1[ix], ry[ix]) - c0); g2.append(corr(r2[ix], ry[ix]) - c0)

    # within-pair choice accuracy (pairs whose scores differ)
    def pair_acc(p):
        hits = n = 0
        for ix in pair_ix.values():
            i, j = ix
            if y[i] == y[j]:
                continue
            n += 1; hits += (p[i] - p[j]) * (y[i] - y[j]) > 0
        return hits / n, n
    acc0, npair = pair_acc(p0); acc1, _ = pair_acc(p1); acc2, _ = pair_acc(p2)
    C[s] = {"len_only": corr(r0, ry), "len_plus_framed": corr(r1, ry), "len_plus_raw": corr(r2, ry),
            "gain_framed": corr(r1, ry) - corr(r0, ry), "gain_framed_ci": ci(g1),
            "gain_raw": corr(r2, ry) - corr(r0, ry), "gain_raw_ci": ci(g2),
            "pair_acc_len": acc0, "pair_acc_framed": acc1, "pair_acc_raw": acc2, "pairs_with_diff": npair}
    E[f"pred_{s}"] = p1
R["C"] = C

# model weights on full data (standardised), for reading
W = {}
for s in SCORES:
    X = E[LEN + [f"framed_{d}" for d in DIMS]].values
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    W[s] = dict(zip(LEN + DIMS, LinearRegression().fit(Xs, E[s].values).coef_.round(4).tolist()))
R["C_weights"] = W

# ---------- E. collapse cohorts (all responses, raw reading) ----------
Ec = {}
for s in ["helpfulness", "correctness", "coherence"]:
    base = m[m.collapse_reading == "none"][s]
    for grp in ["short_complete", "short_cut", "cut_partial", "cut_code_or_math"]:
        g = m[m.collapse_reading == grp][s]
        if len(g) == 0:
            continue
        diffs = [rng.choice(g.values, len(g)).mean() - rng.choice(base.values, len(base)).mean() for _ in range(B)]
        Ec[f"{grp}|{s}"] = {"n": int(len(g)), "mean": float(g.mean()), "none_mean": float(base.mean()),
                            "diff": float(g.mean() - base.mean()), "ci": ci(diffs)}
R["E_collapse"] = Ec
R["collapse_counts"] = m.collapse_reading.value_counts().to_dict()

# ---------- residuals for reading ----------
E["resid_help"] = E.helpfulness - E.pred_helpfulness
cols = ["rid", "pid", "helpfulness", "correctness", "coherence", "complexity", "verbosity", "pred_helpfulness",
        "resid_help", "prose_tokens"] + [f"framed_{d}" for d in DIMS] + ["framed_rhetorical_state", "collapse_reading", "multi_turn"]
E.sort_values("resid_help")[cols].head(40).to_csv(HERE / "data" / "resid_low.csv", index=False)
E.sort_values("resid_help")[cols].tail(40).to_csv(HERE / "data" / "resid_high.csv", index=False)
E[cols].to_csv(HERE / "data" / "eligible_with_pred.csv", index=False)

(HERE / "data" / "results.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
print(json.dumps({k: R[k] for k in ("n_all", "n_eligible", "n_prompts_eligible", "B_n_pairs", "length_spearman")}, indent=1, default=float))
for s in SCORES:
    c = C[s]
    print(f"{s:12s} len {c['len_only']:.3f}  +framed {c['len_plus_framed']:.3f} gain {c['gain_framed']:+.3f} {np.round(c['gain_framed_ci'],3)}  +raw gain {c['gain_raw']:+.3f}  pair acc len {c['pair_acc_len']:.3f} framed {c['pair_acc_framed']:.3f} raw {c['pair_acc_raw']:.3f} (n {c['pairs_with_diff']})")
