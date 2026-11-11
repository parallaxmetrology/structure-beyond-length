#!/usr/bin/env python3
"""Arena 100k analyses A-D (PLAN.md), with AlpacaEval and MT-Bench refitted in the same fixed units.
Writes data/results.json."""
import hashlib, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DIMS = ["Δx", "rᵥ", "ρᵣ", "μ", "Δy", "xₚ", "θ", "dₛ"]
rng = np.random.default_rng(0)
R = {}

# ---------- AlpacaEval judge data: fixed direction and scaling ----------
AEres = json.loads((ROOT / "studies/alpacaeval/data/results.json").read_text())
w = np.array([AEres["J2"][c]["judge"] for c in DIMS]); w = w / np.linalg.norm(w)
am = pd.read_csv(ROOT / "studies/alpacaeval/data/readings.csv.gz")
am = am[am.folder != "NullModel"]; am["loglen"] = np.log1p(am.prose_tokens)
v = pd.DataFrame(json.loads((ROOT / "studies/alpacaeval/data/verdicts.json").read_text()))
v["iid"] = [hashlib.sha1(s.encode()).hexdigest()[:12] for s in v.instruction]
REF = "gpt4_1106_preview"
v = v[(v.generator_1 == REF) & ~v.folder.isin(["NullModel", REF])].copy()
v["pref"] = pd.to_numeric(v.preference, errors="coerce"); v = v.dropna(subset=["pref"])
cols = ["folder", "iid", "loglen", "rep4", "eligible"] + [f"framed_{d}" for d in DIMS]
ref = am[am.folder == REF][cols].drop(columns="folder").set_index("iid")
J = v[["folder", "iid", "pref"]].join(am[cols].set_index(["folder", "iid"]), on=["folder", "iid"]).join(ref, on="iid", rsuffix="_ref")
J = J[(J.eligible == True) & (J.eligible_ref == True)].dropna()
AE = pd.DataFrame({"dlen": (J.loglen - J.loglen_ref).values, "drep4": (J.rep4 - J.rep4_ref).values,
                   "y": (J.pref > 1.5).astype(int).values, "cluster": J.folder.values})
for d in DIMS:
    AE[d] = (J[f"framed_{d}"] - J[f"framed_{d}_ref"]).values
SD = {d: float(AE[d].std()) for d in DIMS}
R["fixed_units"] = {"direction": dict(zip(DIMS, np.round(w, 4).tolist())), "axis_sd_alpacaeval": SD}


def add_struct(X):
    X["struct"] = sum(w[i] * X[d] / SD[d] for i, d in enumerate(DIMS))
    return X

AE = add_struct(AE)


def fit(X, cols):
    return LogisticRegression(C=1e6, max_iter=5000).fit(X[cols].values, X.y.values).coef_[0]


def boot_fits(X, cols, B):
    cl = X.cluster.values; u = np.unique(cl); mem = {k: np.where(cl == k)[0] for k in u}
    out = []
    for _ in range(B):
        ix = np.concatenate([mem[k] for k in rng.choice(u, len(u))])
        out.append(fit(X.iloc[ix], cols))
    return np.array(out)


pc = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
BASE = ["dlen", "drep4", "struct"]


def summarise(X, B, label, extra=()):
    cols = BASE + list(extra)
    b = fit(X, cols); bs = boot_fits(X, cols, B)
    out = {"n": int(len(X)), "clusters": int(X.cluster.nunique()), "cols": cols,
           "beta": dict(zip(cols, b.tolist())), "beta_ci": {c: pc(bs[:, i]) for i, c in enumerate(cols)},
           "ratio_struct_len": float(b[2] / b[0]), "ratio_struct_len_ci": pc(bs[:, 2] / bs[:, 0]),
           "ratio_rep_len": float(b[1] / b[0]), "ratio_rep_len_ci": pc(bs[:, 1] / bs[:, 0])}
    out["_ratios"] = (bs[:, 2] / bs[:, 0]).tolist()
    print(label, out["n"], {k: round(v, 3) for k, v in out["beta"].items()}, "ratio", round(out["ratio_struct_len"], 3), np.round(out["ratio_struct_len_ci"], 3))
    return out


# ---------- Arena ----------
bt = pd.read_csv(HERE / "data" / "battles.csv")
an = pd.read_csv(HERE / "data" / "answers.csv.gz", low_memory=False)
an["bid"] = an.key.str.split("|").str[0].astype(int); an["side"] = an.key.str.split("|").str[1]
an["loglen"] = np.log1p(an.prose_tokens); an["bold"] = (an.edit_bold_lines > 0).astype(int); an["head"] = (an.edit_headers > 0).astype(int)
A_ = an[an.side == "a"].set_index("bid"); B_ = an[an.side == "b"].set_index("bid")
bt = bt.set_index("bid")
ok = A_.eligible & B_.eligible
ids = ok[ok].index
a, b, t = A_.loc[ids], B_.loc[ids], bt.loc[ids]
AR = pd.DataFrame({"dlen": (a.loglen - b.loglen).values, "drep4": (a.rep4 - b.rep4).values,
                   "dbold": (a.bold - b.bold).values, "dhead": (a["head"] - b["head"]).values,
                   "y": (t.winner == "model_a").astype(int).values, "cluster": t.question_id.values,
                   "model_a": t.model_a.values, "model_b": t.model_b.values,
                   "is_code": t.is_code.values, "is_refusal": t.is_refusal.values})
for d in DIMS:
    AR[d] = (a[f"framed_{d}"] - b[f"framed_{d}"]).values
AR = add_struct(AR)

# cohorts
style_share = pd.concat([pd.DataFrame({"m": A_.loc[ids].index.map(bt.model_a), "b": a.bold.values}),
                         pd.DataFrame({"m": B_.loc[ids].index.map(bt.model_b), "b": b.bold.values})]).groupby("m").b.mean()
sty = set(style_share[style_share >= 0.2].index)
AR["both_style"] = AR.model_a.isin(sty) & AR.model_b.isin(sty)
AR["neither_style"] = ~AR.model_a.isin(sty) & ~AR.model_b.isin(sty)
