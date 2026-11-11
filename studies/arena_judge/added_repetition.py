#!/usr/bin/env python3
"""Added analysis R (not in the frozen plan): on the same battles, the judge's repetition-to-length ratio
vs humans', per cohort, with a paired bootstrap over prompts. Also counts refusals by cohort.
Writes results/added_repetition.json."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
V = pd.read_csv(HERE / "results" / "verdicts.csv"); S = pd.read_csv(HERE / "sample.csv")
V["row"] = V.custom_id.str.split("-").str[0].astype(int); V["order"] = V.custom_id.str.split("-").str[1]
W = V.pivot_table(index="row", columns="order", values="verdict", aggfunc="first")
comb = lambda ab, ba: "a" if (ab, ba) == ("A", "B") else ("b" if (ab, ba) == ("B", "A") else "tie")
W["judge"] = [comb(r.get("ab"), r.get("ba")) for _, r in W.iterrows()]
D = S.set_index("row").join(W[["judge"]], how="inner").reset_index(); D["yj"] = (D.judge == "a").astype(int)
bt = pd.read_csv(HERE.parent / "arena100k" / "data" / "battles.csv").set_index("bid")
D["is_code"] = D.bidx.map(bt.is_code).astype(bool)
COLS = ["dlen", "drep4", "struct"]; rng = np.random.default_rng(1)
fit = lambda X, y: LogisticRegression(C=1e6, max_iter=5000).fit(X[COLS].values, X[y].values).coef_[0]
pc = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
out = {}
for c in ["both", "neither", "both_nocode", "neither_nocode"]:
    base = c.split("_")[0]
    X = D[(D.cohort == base) & (D.judge != "tie") & (~D.is_code if c.endswith("nocode") else True)].reset_index(drop=True)
    def f(Z):
        j, h = fit(Z, "yj"), fit(Z, "y"); return np.array([j[1] / j[0], h[1] / h[0]])
    est = f(X); u = X.cluster.unique(); m = {k: np.where(X.cluster.values == k)[0] for k in u}
    bs = np.array([f(X.iloc[np.concatenate([m[k] for k in rng.choice(u, len(u))])]) for _ in range(1000)])
    out[c] = {"rep_ratio_judge": float(est[0]), "rep_ratio_judge_ci": pc(bs[:, 0]), "rep_ratio_human": float(est[1]),
              "rep_ratio_human_ci": pc(bs[:, 1]), "diff_judge_minus_human": float(est[0] - est[1]), "diff_ci": pc(bs[:, 0] - bs[:, 1])}
    print(c, {k: (round(v, 2) if isinstance(v, float) else [round(x, 2) for x in v]) for k, v in out[c].items()})
ref = V[V.stop == "refusal"].row.map(S.set_index("row").cohort).value_counts().to_dict()
out["refusals_by_cohort"] = {k: int(v) for k, v in ref.items()}
print("refusals", out["refusals_by_cohort"])
(HERE / "results" / "added_repetition.json").write_text(json.dumps(out, indent=1))
