#!/usr/bin/env python3
"""Added checks (not in the frozen plan), after reading the M1-M5 results.
C1  Do the three reward models share one direction? Cosines between their 8-axis directions, and with the
    AlpacaEval judge and HelpSteer2 human directions.
C2  Self-preference: rerun the both-styled structure weight without battles involving a model tuned on that
    reward model's preferences (ArmoRM: gemma-2-9b-it-simpo; PairRM/GPM: none present in Arena 2024 styled set,
    checked by name).
C3  Without code battles.
Writes results/added_checks.json."""
import json, runpy
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
R = json.loads((HERE / "results" / "analysis.json").read_text())
DIMS = ["Δx", "rᵥ", "ρᵣ", "μ", "Δy", "xₚ", "θ", "dₛ"]
AE = json.loads((ROOT / "alpacaeval" / "data" / "results.json").read_text())["J2"]
unit = lambda v: np.asarray(v, float) / np.linalg.norm(v)
dirs = {rm: unit([R[rm]["direction_8"][d] for d in DIMS]) for rm in ["armorm", "gpm", "pairrm"]}
dirs["judge_alpacaeval"] = unit([AE[d]["judge"] for d in DIMS]); dirs["human_helpsteer2"] = unit([AE[d]["human"] for d in DIMS])
names = list(dirs)
C1 = {a: {b: round(float(dirs[a] @ dirs[b]), 3) for b in names} for a in names}

S = pd.read_csv(ROOT / "arena_judge" / "sample.csv")
bt = pd.read_csv(ROOT / "arena100k" / "data" / "battles.csv").set_index("bid")
S["is_code"] = S.bidx.map(bt.is_code).astype(bool)
rng = np.random.default_rng(3)
COLS = ["dlen", "drep4", "struct"]
fit = lambda X, y: LogisticRegression(C=1e6, max_iter=5000).fit(X[COLS].values, X[y].values).coef_[0]
pc = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def struct_ci(X, y):
    u = X.cluster.unique(); m = {k: np.where(X.cluster.values == k)[0] for k in u}
    est = fit(X, y)[2]
    bs = [fit(X.iloc[np.concatenate([m[k] for k in rng.choice(u, len(u))])], y)[2] for _ in range(400)]
    return {"n": int(len(X)), "beta_struct": float(est), "ci": pc(bs)}


students = {"armorm": ["simpo", "wpo"], "gpm": ["gpm"], "pairrm": ["pairrm", "sppo"]}
out = {"C1_cosines": C1, "C2": {}, "C3": {}}
for rm in ["armorm", "gpm", "pairrm"]:
    X = pd.read_csv(HERE / "results" / f"{rm}.csv")
    X["m"] = X.score_a - X.score_b if rm == "armorm" else (X.pref_ab - X.pref_ba) / 2
    D = S.merge(X[["row", "m"]], on="row"); D = D[D.m != 0].copy(); D["yr"] = (D.m > 0).astype(int)
    B = D[D.cohort == "both"]
    pat = "|".join(students[rm])
    stud = B.model_a.str.lower().str.contains(pat) | B.model_b.str.lower().str.contains(pat)
    out["C2"][rm] = {"student_models_in_both_cohort": sorted(set(B[stud].model_a) | set(B[stud].model_b) & set(B.model_a) | set()),
                     "battles_with_student": int(stud.sum()),
                     "without_students": struct_ci(B[~stud].reset_index(drop=True), "yr")}
    out["C3"][rm] = {c: struct_ci(D[(D.cohort == c) & ~D.is_code].reset_index(drop=True), "yr") for c in ["both", "neither"]}
    out["C3"][rm]["human_both_nocode"] = struct_ci(D[(D.cohort == "both") & ~D.is_code].reset_index(drop=True), "y")
(HERE / "results" / "added_checks.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
print("C1 cosines"); print(pd.DataFrame(C1).round(2).to_string())
for rm in ["armorm", "gpm", "pairrm"]:
    c2 = out["C2"][rm]; w = c2["without_students"]
    print(f"C2 {rm}: student battles {c2['battles_with_student']}; without: {w['beta_struct']:+.3f} {np.round(w['ci'],3)} (n={w['n']})")
    c3 = out["C3"][rm]
    print(f"C3 {rm} no code: both {c3['both']['beta_struct']:+.3f} {np.round(c3['both']['ci'],3)}, neither {c3['neither']['beta_struct']:+.3f}, human both {c3['human_both_nocode']['beta_struct']:+.3f} {np.round(c3['human_both_nocode']['ci'],3)}")
