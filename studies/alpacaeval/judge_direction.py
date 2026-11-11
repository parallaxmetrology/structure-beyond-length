#!/usr/bin/env python3
"""Added analysis H (not in the frozen plan), after reading the 2024-H2 cohort.

Judge direction = the judge's standardised coefficients on the 8 axes (J2).
Human direction = the HelpSteer2 human coefficients on the same 8 axes.
For each model: projection of its style vector (prompt-relative z means) onto each direction.
Questions:
  H1  Does the drift between cohorts point along the judge direction more than the human one? (cosine)
  H2  Does a model's projection on the judge direction track its LC win rate, within each cohort?
  H3  Is 2024-H2 split in two along the judge direction (bimodal), and is the split the bold-title style?
Writes data/judge_direction.json and data/judge_direction_models.csv."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DIMS = ["Δx", "rᵥ", "ρᵣ", "μ", "Δy", "xₚ", "θ", "dₛ"]
Z = [f"z_{d}" for d in DIMS]
R = json.loads((HERE / "data" / "results.json").read_text())
rng = np.random.default_rng(0)

s = pd.read_csv(HERE / "data" / "style_by_model.csv", index_col=0)
lb = pd.read_csv(ROOT / "public_corpora/alpaca_eval/src/alpaca_eval/leaderboards/data_AlpacaEval_2/weighted_alpaca_eval_gpt4_turbo_leaderboard.csv", index_col=0)
s = s.join(lb[["mode", "length_controlled_winrate"]])
jd = np.array([R["J2"][d]["judge"] for d in DIMS]); hd = np.array([R["J2"][d]["human"] for d in DIMS])
unit = lambda v: v / np.linalg.norm(v)
cos = lambda a, b: float(unit(a) @ unit(b))

out = {"cos_judge_human": cos(jd, hd)}
cen = {c: s[s.cohort == c][Z].mean().values for c in ["2023-H2", "2024-H1", "2024-H2"]}
drift = cen["2024-H2"] - cen["2023-H2"]
out["drift"] = dict(zip(DIMS, np.round(drift, 3).tolist()))
out["cos_drift_judge"], out["cos_drift_human"] = cos(drift, jd), cos(drift, hd)
bj, bh = [], []
for _ in range(1000):
    a = s[s.cohort == "2023-H2"].sample(frac=1, replace=True, random_state=int(rng.integers(1e9)))[Z].mean().values
    b = s[s.cohort == "2024-H2"].sample(frac=1, replace=True, random_state=int(rng.integers(1e9)))[Z].mean().values
    bj.append(cos(b - a, jd)); bh.append(cos(b - a, hd))
out["cos_drift_judge_ci"] = [float(np.percentile(bj, q)) for q in (2.5, 97.5)]
out["cos_drift_human_ci"] = [float(np.percentile(bh, q)) for q in (2.5, 97.5)]
out["cos_drift_minus_human_ci"] = [float(np.percentile(np.array(bj) - np.array(bh), q)) for q in (2.5, 97.5)]

s["proj_judge"] = s[Z].values @ unit(jd)
s["proj_human"] = s[Z].values @ unit(hd)
H2 = {}
for c in ["2023-H2", "2024-H1", "2024-H2", "all"]:
    g = s if c == "all" else s[s.cohort == c]
    g = g.dropna(subset=["length_controlled_winrate"])
    H2[c] = {"n": int(len(g)),
             "spearman_lc_judge_proj": float(g.proj_judge.corr(g.length_controlled_winrate, method="spearman")),
             "spearman_lc_human_proj": float(g.proj_human.corr(g.length_controlled_winrate, method="spearman")),
             "spearman_lc_bold": float(g.bold.corr(g.length_controlled_winrate, method="spearman"))}
out["H2"] = H2

g = s[s.cohort == "2024-H2"].sort_values("proj_judge")
out["H3_2024H2"] = g[["mode", "proj_judge", "bold", "tokens", "length_controlled_winrate"]].round(3).reset_index().to_dict("records")
hi = g[g.bold >= 0.2]; lo = g[g.bold < 0.2]
out["H3_split"] = {"bold_ge_0.2": {"n": len(hi), "median_lc": float(hi.length_controlled_winrate.median()), "median_proj": float(hi.proj_judge.median())},
                   "bold_lt_0.2": {"n": len(lo), "median_lc": float(lo.length_controlled_winrate.median()), "median_proj": float(lo.proj_judge.median())}}
# spread within each half of 2024-H2 vs 2023-H2 (is the divergence a split?)
from itertools import combinations
mp = lambda X: float(np.mean([np.linalg.norm(X[i] - X[j]) for i, j in combinations(range(len(X)), 2)]))
out["H3_spread"] = {"2023-H2_all": mp(s[s.cohort == "2023-H2"][Z].values), "2024-H2_all": mp(g[Z].values),
                    "2024-H2_bold": mp(hi[Z].values), "2024-H2_plain": mp(lo[Z].values)}
s.to_csv(HERE / "data" / "judge_direction_models.csv")
(HERE / "data" / "judge_direction.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k != "H3_2024H2"}, indent=1, ensure_ascii=False, default=float))
print(g[["mode", "proj_judge", "bold", "tokens", "length_controlled_winrate"]].round(2).to_string())
