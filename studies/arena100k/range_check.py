#!/usr/bin/env python3
"""Added check E: is the vanished structure reward in both-style battles just a narrower range?
Compares the SD of the structure-index difference across cohorts, and refits the neither-style
cohort on a subsample whose struct-difference distribution is matched to the both-style cohort."""
import json, runpy
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
g = runpy.run_path(str(HERE / "analyze_core.py"))
AR, fit = g["AR"], g["fit"]
rng = np.random.default_rng(1)
both, nei = AR[AR.both_style].reset_index(drop=True), AR[AR.neither_style].reset_index(drop=True)
out = {"sd_struct_both": float(both.struct.std()), "sd_struct_neither": float(nei.struct.std()),
       "sd_dlen_both": float(both.dlen.std()), "sd_dlen_neither": float(nei.dlen.std())}
# match |struct| distribution by binning
bins = np.quantile(np.abs(AR.struct), np.linspace(0, 1, 11)); bins[-1] += 1e-9
hb = np.histogram(np.abs(both.struct), bins)[0]
nb = np.digitize(np.abs(nei.struct), bins) - 1
betas = []
for _ in range(200):
    pick = np.concatenate([rng.choice(np.where(nb == i)[0], hb[i], replace=True) for i in range(10) if hb[i] > 0 and (nb == i).any()])
    betas.append(fit(nei.iloc[pick], ["dlen", "drep4", "struct"]))
betas = np.array(betas)
out["neither_matched_struct_beta"] = float(np.median(betas[:, 2]))
out["neither_matched_struct_beta_ci"] = [float(np.percentile(betas[:, 2], 2.5)), float(np.percentile(betas[:, 2], 97.5))]
out["both_struct_beta"] = float(fit(both, ["dlen", "drep4", "struct"])[2])
(HERE / "data" / "range_check.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
