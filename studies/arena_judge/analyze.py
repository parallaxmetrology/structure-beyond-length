#!/usr/bin/env python3
"""K1-K5 from PLAN.md on the collected judge verdicts. Writes results/analysis.json and prints a summary.
  python3 analyze.py [path/to/verdicts.csv]"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
V = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else HERE / "results" / "verdicts.csv")
S = pd.read_csv(HERE / "sample.csv")
rng = np.random.default_rng(0)
COLS = ["dlen", "drep4", "struct"]

V["row"] = V.custom_id.str.split("-").str[0].astype(int); V["order"] = V.custom_id.str.split("-").str[1]
W = V.pivot_table(index="row", columns="order", values="verdict", aggfunc="first")
def combine(ab, ba):  # ab: A = answer a; ba: A = answer b
    if ab == "A" and ba == "B": return "a"
    if ab == "B" and ba == "A": return "b"
    return "tie"
W["judge"] = [combine(r.get("ab"), r.get("ba")) for _, r in W.iterrows()]
W["consistent"] = [(r.get("ab"), r.get("ba")) in {("A", "B"), ("B", "A"), ("tie", "tie")} for _, r in W.iterrows()]
D = S.set_index("row").join(W[["judge", "consistent"]], how="inner").reset_index()
D["yj"] = (D.judge == "a").astype(int)
R = {"battles_with_verdicts": int(len(D)), "unparsed_requests": int((V.verdict == "unparsed").sum()),
     "failed_requests": int((V.type != "succeeded").sum())}


def fit(X, ycol):
    return LogisticRegression(C=1e6, max_iter=5000).fit(X[COLS].values, X[ycol].values).coef_[0]


def boot(X, f, B=1000):
    u = X.cluster.unique(); m = {k: np.where(X.cluster.values == k)[0] for k in u}
    return np.array([f(X.iloc[np.concatenate([m[k] for k in rng.choice(u, len(u))])]) for _ in range(B)])


pc = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
for c in ["both", "neither"]:
    X = D[(D.cohort == c) & (D.judge != "tie")].reset_index(drop=True)
    def both_fits(Z):
        j, h = fit(Z, "yj"), fit(Z, "y")
        return np.r_[j, h, j[2] / j[0] - h[2] / h[0]]
    est = both_fits(X); bs = boot(X, both_fits)
    allc = D[D.cohort == c]
    R[c] = {"n_judge_nontie": int(len(X)), "judge_beta": dict(zip(COLS, est[:3].round(4).tolist())),
            "judge_beta_ci": {k: pc(bs[:, i]) for i, k in enumerate(COLS)},
            "human_beta_same_battles": dict(zip(COLS, est[3:6].round(4).tolist())),
            "human_beta_ci": {k: pc(bs[:, 3 + i]) for i, k in enumerate(COLS)},
            "ratio_judge": float(est[2] / est[0]), "ratio_human": float(est[5] / est[3]),
            "ratio_diff": float(est[6]), "ratio_diff_ci": pc(bs[:, 6]),
            "agreement_judge_human": float((X.yj == X.y).mean()),
            "position_consistency": float(allc.consistent.mean()), "tie_rate": float((allc.judge == "tie").mean())}
ag = lambda Z: float((Z.yj == Z.y).mean())
Xb, Xn = D[(D.cohort == "both") & (D.judge != "tie")], D[(D.cohort == "neither") & (D.judge != "tie")]
diffs = [ag(Xb.sample(frac=1, replace=True, random_state=int(rng.integers(1e9)))) - ag(Xn.sample(frac=1, replace=True, random_state=int(rng.integers(1e9)))) for _ in range(1000)]
R["agreement_both_minus_neither"] = {"diff": ag(Xb) - ag(Xn), "ci": pc(diffs)}
R["predictions"] = {
    "Q1 judge rewards structure between two styled answers": R["both"]["judge_beta_ci"]["struct"][0] > 0,
    "Q2 judge rewards structure between two unstyled answers": R["neither"]["judge_beta_ci"]["struct"][0] > 0,
    "Q3 judge ratio above human ratio, both styled": R["both"]["ratio_diff_ci"][0] > 0,
    "Q4 agreement lower in both-styled than neither": R["agreement_both_minus_neither"]["ci"][1] < 0,
    "Q5 judge rewards length and penalises repetition (both cohorts)": all(R[c]["judge_beta_ci"]["dlen"][0] > 0 and R[c]["judge_beta_ci"]["drep4"][1] < 0 for c in ["both", "neither"]),
}
out = HERE / "results" / ("analysis.json" if len(sys.argv) == 1 else "analysis_test.json")
out.write_text(json.dumps(R, indent=1, default=float))
for c in ["both", "neither"]:
    r = R[c]
    print(f"{c:8s} n={r['n_judge_nontie']:5d}  judge struct {r['judge_beta']['struct']:+.3f} {np.round(r['judge_beta_ci']['struct'],3)}  "
          f"human struct {r['human_beta_same_battles']['struct']:+.3f}  ratio judge {r['ratio_judge']:+.3f} human {r['ratio_human']:+.3f} "
          f"diff CI {np.round(r['ratio_diff_ci'],3)}  agree {r['agreement_judge_human']:.2f}  consistent {r['position_consistency']:.2f}")
for k, v in R["predictions"].items():
    print(("SUPPORTED     " if v else "not supported ") + k)
print("wrote", out)
