#!/usr/bin/env python3
"""Style-control baseline (PLAN.md). Writes results.json."""
import json, runpy
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent; ST = HERE.parent; ROOT = ST.parent
rng = np.random.default_rng(0)
g = runpy.run_path(str(ST / "arena100k" / "analyze_core.py"))
AR = g["AR"].reset_index(drop=True); ids = list(g["ids"])

# ---- Arena style-control features per battle
d = pd.read_parquet(ROOT / "public_corpora/arena100k/arena-explorer-preference-100k.parquet", columns=["language", "turn", "winner", "conv_metadata"])
d = d[(d.language == "English") & (d.turn == 1) & d.winner.isin(["model_a", "model_b"])].reset_index(drop=True)
nd = lambda a, b: 0.0 if a + b == 0 else (a - b) / (a + b)
rows = []
for i in ids:
    m = d.conv_metadata.iloc[i]
    rows.append({"sc_len": nd(m["sum_assistant_a_tokens"], m["sum_assistant_b_tokens"]),
                 "sc_head": nd(sum(m["header_count_a"].values()), sum(m["header_count_b"].values())),
                 "sc_bold": nd(sum(m["bold_count_a"].values()), sum(m["bold_count_b"].values())),
                 "sc_list": nd(sum(m["list_count_a"].values()), sum(m["list_count_b"].values()))})
AR = pd.concat([AR, pd.DataFrame(rows)], axis=1)
SC = ["sc_len", "sc_head", "sc_bold", "sc_list"]; K = ["dlen", "drep4", "struct"]

# ---- labels on the 5,794-battle sample
S = pd.read_csv(ST / "arena_judge" / "sample.csv")[["row", "cohort"]]
S = S.join(AR, on="row")
V = pd.read_csv(ST / "arena_judge" / "results" / "verdicts.csv")
V["r"] = V.custom_id.str.split("-").str[0].astype(int); V["o"] = V.custom_id.str.split("-").str[1]
W = V.pivot_table(index="r", columns="o", values="verdict", aggfunc="first")
comb = lambda ab, ba: "a" if (ab, ba) == ("A", "B") else ("b" if (ab, ba) == ("B", "A") else "tie")
S["judge"] = S.row.map({r: comb(x.get("ab"), x.get("ba")) for r, x in W.iterrows()})
for rm in ["armorm", "gpm", "pairrm"]:
    X = pd.read_csv(ST / "rm_audit" / "results" / f"{rm}.csv")
    X["m"] = X.score_a - X.score_b if rm == "armorm" else (X.pref_ab - X.pref_ba) / 2
    S[rm] = S.row.map(dict(zip(X.row, X.m)))

pc = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def coefs(X, ycol, cols):
    A = X[cols].values
    return LogisticRegression(C=1e6, max_iter=5000).fit(A, X[ycol].values).coef_[0]


def boot_struct(X, ycol, B=400):
    cols = SC + K; i = cols.index("struct")
    u = X.cluster.unique(); mem = {k: np.where(X.cluster.values == k)[0] for k in u}
    est = coefs(X, ycol, cols)[i]
    bs = [coefs(X.iloc[np.concatenate([mem[k] for k in rng.choice(u, len(u))])], ycol, cols)[i] for _ in range(B)]
    return float(est), pc(bs)


def heldout(X, ycol, cols):
    p = np.zeros(len(X))
    for tr, te in GroupKFold(5).split(X, X[ycol], X.cluster):
        A = X[cols].values; mu, sd = A[tr].mean(0), A[tr].std(0) + 1e-9
        p[te] = LogisticRegression(max_iter=5000).fit((A[tr] - mu) / sd, X[ycol].values[tr]).predict_proba((A[te] - mu) / sd)[:, 1]
    return p


def auc_gain(X, ycol, base, full, B=300):
    y = X[ycol].values; p0, p1 = heldout(X, ycol, base), heldout(X, ycol, full)
    u = X.cluster.unique(); mem = {k: np.where(X.cluster.values == k)[0] for k in u}
    gs = []
    for _ in range(B):
        ix = np.concatenate([mem[k] for k in rng.choice(u, len(u))])
        gs.append(roc_auc_score(y[ix], p1[ix]) - roc_auc_score(y[ix], p0[ix]))
    return {"auc_base": float(roc_auc_score(y, p0)), "auc_full": float(roc_auc_score(y, p1)), "gain_ci": pc(gs)}


def block(X, ycol):
    X = X.reset_index(drop=True)
    est, ci = boot_struct(X, ycol)
    return {"n": int(len(X)), "struct_beta_with_SC": est, "struct_ci_with_SC": ci,
            "SC_to_SCK": auc_gain(X, ycol, SC, SC + K), "K_to_KSC": auc_gain(X, ycol, K, K + SC),
            "sc_coefs_in_full": dict(zip(SC, coefs(X, ycol, SC + K)[:4].round(4).tolist()))}


R = {"voters_all": block(AR, "y")}
for c in ["neither", "both"]:
    Z = S[S.cohort == c].copy()
    R[f"voters_{c}"] = block(Z, "y")
    J = Z[Z.judge != "tie"].copy(); J["yj"] = (J.judge == "a").astype(int)
    R[f"judge_{c}"] = block(J, "yj")
    for rm in ["armorm", "gpm", "pairrm"]:
        Q = Z[Z[rm].notna() & (Z[rm] != 0)].copy(); Q["yr"] = (Q[rm] > 0).astype(int)
        R[f"{rm}_{c}"] = block(Q, "yr")
for rm in ["armorm", "gpm", "pairrm"]:
    Q = S[S[rm].notna() & (S[rm] != 0)].copy(); Q["yr"] = (Q[rm] > 0).astype(int)
    R[f"{rm}_all5794"] = block(Q, "yr")

R["predictions"] = {
    "B1 voters all: struct > 0 with SC": R["voters_all"]["struct_ci_with_SC"][0] > 0,
    "B2 each RM both-styled: struct > 0 with SC": all(R[f"{rm}_both"]["struct_ci_with_SC"][0] > 0 for rm in ["armorm", "gpm", "pairrm"]),
    "B3 voters both-styled: struct CI includes 0 with SC": R["voters_both"]["struct_ci_with_SC"][0] <= 0 <= R["voters_both"]["struct_ci_with_SC"][1],
    "B4 voters all: SC+K beats SC held out": R["voters_all"]["SC_to_SCK"]["gain_ci"][0] > 0,
    "B5 each RM: SC+K beats SC held out (5,794)": all(R[f"{rm}_all5794"]["SC_to_SCK"]["gain_ci"][0] > 0 for rm in ["armorm", "gpm", "pairrm"]),
}
(HERE / "results.json").write_text(json.dumps(R, indent=1, default=float))
for k, v in R.items():
    if k == "predictions": continue
    a, b = v["SC_to_SCK"], v["K_to_KSC"]
    print(f"{k:16s} n={v['n']:6d} struct|SC {v['struct_beta_with_SC']:+.3f} {np.round(v['struct_ci_with_SC'],3)}  "
          f"AUC SC {a['auc_base']:.3f} -> SC+K {a['auc_full']:.3f} {np.round(a['gain_ci'],3)} | K {b['auc_base']:.3f} -> K+SC {b['auc_full']:.3f} {np.round(b['gain_ci'],3)}")
for k, v in R["predictions"].items():
    print(("SUPPORTED     " if v else "not supported ") + k)
