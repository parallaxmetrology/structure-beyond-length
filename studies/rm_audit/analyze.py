#!/usr/bin/env python3
"""M1-M5 from PLAN.md. Reads results/{armorm,gpm,pairrm}.csv (whichever exist), the judge-run sample
and the AlpacaEval style vectors. Writes results/analysis.json and prints a summary.
  python3 analyze.py [results_dir]"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
ROOT = HERE.parent
DIMS = ["Δx", "rᵥ", "ρᵣ", "μ", "Δy", "xₚ", "θ", "dₛ"]
COLS = ["dlen", "drep4", "struct"]
rng = np.random.default_rng(0)
S = pd.read_csv(ROOT / "arena_judge" / "sample.csv")
fit = lambda X, y, c=COLS: LogisticRegression(C=1e6, max_iter=5000).fit(X[c].values, X[y].values).coef_[0]
pc = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

# axis differences for the fingerprint direction (raw kernel axes from the Arena reading)
import runpy
g = runpy.run_path(str(ROOT / "arena100k" / "analyze_core.py"))
AR = g["AR"].reset_index(drop=True)
AXD = AR.loc[S.row.values, DIMS].reset_index(drop=True)
S = pd.concat([S.reset_index(drop=True), AXD], axis=1)

R = {}
models = {}
for rm in ["armorm", "gpm", "pairrm"]:
    p = RES / f"{rm}.csv"
    if not p.exists():
        R[rm] = {"status": "missing"}; continue
    X = pd.read_csv(p)
    if rm == "armorm":
        X["m"] = X.score_a - X.score_b
    else:
        X["m"] = (X.pref_ab - X.pref_ba) / 2  # a-over-b in both orders
    D = S.merge(X[["row", "m"]], on="row")
    D = D[D.m != 0].copy(); D["yr"] = (D.m > 0).astype(int)
    models[rm] = D
    out = {"status": "ok", "n": int(len(D))}
    for c in ["both", "neither"]:
        Z = D[D.cohort == c].reset_index(drop=True)
        def f(W):
            r_, h = fit(W, "yr"), fit(W, "y")
            return np.r_[r_, h, r_[2] / r_[0] - h[2] / h[0], r_[1] / r_[0] - h[1] / h[0], r_[2] / r_[0], h[2] / h[0], r_[1] / r_[0], h[1] / h[0]]
        est = f(Z); u = Z.cluster.unique(); mem = {k: np.where(Z.cluster.values == k)[0] for k in u}
        bs = np.array([f(Z.iloc[np.concatenate([mem[k] for k in rng.choice(u, len(u))])]) for _ in range(500)])
        out[c] = {"n": int(len(Z)),
                  "rm_beta": dict(zip(COLS, est[:3].round(4).tolist())), "rm_beta_ci": {k: pc(bs[:, i]) for i, k in enumerate(COLS)},
                  "human_beta": dict(zip(COLS, est[3:6].round(4).tolist())), "human_beta_ci": {k: pc(bs[:, 3 + i]) for i, k in enumerate(COLS)},
                  "struct_ratio_rm": float(est[8]), "struct_ratio_human": float(est[9]), "struct_ratio_diff_ci": pc(bs[:, 6]),
                  "rep_ratio_rm": float(est[10]), "rep_ratio_human": float(est[11]), "rep_ratio_diff_ci": pc(bs[:, 7]),
                  "agreement": float((Z.yr == Z.y).mean())}
    Zb, Zn = D[D.cohort == "both"], D[D.cohort == "neither"]
    ag = lambda Z: float((Z.yr == Z.y).mean())
    dd = [ag(Zb.sample(frac=1, replace=True, random_state=int(rng.integers(1e9)))) - ag(Zn.sample(frac=1, replace=True, random_state=int(rng.integers(1e9)))) for _ in range(1000)]
    out["agreement_both_minus_neither_ci"] = pc(dd)
    # 8-axis direction on all battles (standardised axis differences, length and repetition included as controls)
    Dz = D.copy()
    for d in DIMS:
        Dz[d] = (Dz[d] - Dz[d].mean()) / Dz[d].std()
    out["direction_8"] = dict(zip(DIMS, fit(Dz, "yr", ["dlen", "drep4"] + DIMS)[2:].round(4).tolist()))
    R[rm] = out

# M5 fingerprint: tuned model shift from Meta-Llama-3-8B-Instruct vs each reward model's direction
style = pd.read_csv(ROOT / "alpacaeval" / "data" / "style_by_model.csv", index_col=0)
Z = [f"z_{d}" for d in DIMS]
tuned = {"REBEL-Llama-3-8B-Instruct-Armo": "armorm", "SPPO-Llama-3-8B-Instruct-GPM-2B": "gpm", "GPO-Llama-3-8B-Instruct-GPM-2B": "gpm",
         "SPPO-Llama-3-Instruct-8B-PairRM": "pairrm"}
base = "Meta-Llama-3-8B-Instruct"
fp = {}
dirs = {rm: np.array([R[rm]["direction_8"][d] for d in DIMS]) for rm in R if R[rm].get("status") == "ok"}
unit = lambda v: v / np.linalg.norm(v)
if base in style.index and dirs:
    for m, own in tuned.items():
        if m not in style.index or own not in dirs:
            continue
        shift = style.loc[m, Z].values.astype(float) - style.loc[base, Z].values.astype(float)
        cos = {rm: float(unit(shift) @ unit(v)) for rm, v in dirs.items()}
        fp[m] = {"own": own, "cosines": cos, "own_is_highest": max(cos, key=cos.get) == own if len(cos) > 1 else None}
R["fingerprint"] = fp

ok = [rm for rm in ["armorm", "gpm", "pairrm"] if R[rm].get("status") == "ok"]
cip = lambda rm, c, k: R[rm][c]["rm_beta_ci"][k]
R["predictions"] = {
    "R1 ArmoRM rewards structure between styled answers": ("armorm" in ok) and cip("armorm", "both", "struct")[0] > 0,
    "R2 >=2 RMs struct/length ratio above voters, both styled": sum(R[rm]["both"]["struct_ratio_diff_ci"][0] > 0 for rm in ok) >= 2,
    "R3 all RMs reward structure between unstyled answers": len(ok) == 3 and all(cip(rm, "neither", "struct")[0] > 0 for rm in ok),
    "R4 each RM agrees less with voters between styled answers": len(ok) > 0 and all(R[rm]["agreement_both_minus_neither_ci"][1] < 0 for rm in ok),
    "R5 each RM penalises repetition more than voters (one cohort)": len(ok) > 0 and all(any(R[rm][c]["rep_ratio_diff_ci"][1] < 0 for c in ["both", "neither"]) for rm in ok),
    "R6 own RM direction highest for >=3 tuned models": sum(1 for v in fp.values() if v["own_is_highest"]) >= 3,
}
(RES / "analysis.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=float))
for rm in ok:
    for c in ["neither", "both"]:
        r = R[rm][c]
        print(f"{rm:7s} {c:8s} n={r['n']:5d} struct {r['rm_beta']['struct']:+.3f} {np.round(r['rm_beta_ci']['struct'],3)} human {r['human_beta']['struct']:+.3f} "
              f"| ratio rm {r['struct_ratio_rm']:+.2f} hum {r['struct_ratio_human']:+.2f} diff {np.round(r['struct_ratio_diff_ci'],2)} "
              f"| rep rm {r['rep_ratio_rm']:+.1f} hum {r['rep_ratio_human']:+.1f} diff {np.round(r['rep_ratio_diff_ci'],1)} | agree {r['agreement']:.2f}")
for m, v in fp.items():
    print("fingerprint", m, v["own"], {k: round(x, 2) for k, x in v["cosines"].items()}, "own highest:", v["own_is_highest"])
for k, v in R["predictions"].items():
    print(("SUPPORTED     " if v else "not supported ") + k)
