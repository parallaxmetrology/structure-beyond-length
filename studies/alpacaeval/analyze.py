#!/usr/bin/env python3
"""Studies 1 (style over time) and 2 (the judge) from PLAN.md. Writes data/results.json."""
import gzip, hashlib, json, re
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DIMS = ["Δx", "rᵥ", "ρᵣ", "μ", "Δy", "xₚ", "θ", "dₛ"]
REF = "gpt4_1106_preview"
EXCLUDE = ["NullModel"]
rng = np.random.default_rng(0)
R = {"excluded_models": EXCLUDE}

m = pd.read_csv(HERE / "data" / "readings.csv.gz")
m = m[~m.folder.isin(EXCLUDE)]
m["loglen"] = np.log1p(m.prose_tokens)
dates = json.loads((HERE / "data" / "added_dates.json").read_text())
lb = pd.read_csv(ROOT / "public_corpora/alpaca_eval/src/alpaca_eval/leaderboards/data_AlpacaEval_2/weighted_alpaca_eval_gpt4_turbo_leaderboard.csv", index_col=0)
R["n_answers"] = int(len(m)); R["n_models"] = int(m.folder.nunique()); R["eligible_share"] = float(m.eligible.mean())

E = m[m.eligible].dropna(subset=[f"framed_{d}" for d in DIMS]).copy()
# ---------- prompt-relative z ----------
Z = [f"z_{d}" for d in DIMS]
for d in DIMS + ["loglen"]:
    col = f"framed_{d}" if d in DIMS else d
    g = E.groupby("iid")[col]
    E[f"z_{d}"] = (E[col] - g.transform("mean")) / (g.transform("std") + 1e-9)
counts = E.groupby("folder").size()
keep = counts[counts >= 400].index
E = E[E.folder.isin(keep)]
style = E.groupby("folder")[Z + ["z_loglen"]].mean()
style["date"] = [dates.get(f) for f in style.index]
style = style.dropna(subset=["date"])


def cohort(d):
    return "2023-H2" if d < "2024-01-01" else ("2024-H1" if d < "2024-07-01" else "2024-H2")


style["cohort"] = style.date.map(cohort)
R["cohort_sizes"] = style.cohort.value_counts().to_dict()
R["models_in_style_study"] = int(len(style))
k = int(style.cohort.value_counts().min())
COH = ["2023-H2", "2024-H1", "2024-H2"]


def mean_pdist(X):
    return float(np.mean([np.linalg.norm(X[i] - X[j]) for i, j in combinations(range(len(X)), 2)]))


def s1(S, cols, B=1000):
    out = {}
    draws = {c: [] for c in COH}
    for _ in range(B):
        for c in COH:
            X = S[S.cohort == c][cols].values
            draws[c].append(mean_pdist(X[rng.choice(len(X), k, replace=True)]))
    for c in COH:
        out[c] = {"mean": float(np.mean(draws[c])), "ci": [float(np.percentile(draws[c], q)) for q in (2.5, 97.5)]}
    diff = np.array(draws["2024-H2"]) - np.array(draws["2023-H2"])
    out["diff_2024H2_minus_2023H2"] = {"mean": float(diff.mean()), "ci": [float(np.percentile(diff, q)) for q in (2.5, 97.5)]}
    return out

R["k_subsample"] = k
R["S1"] = s1(style, Z)

# S4: partial length out of each prompt-relative axis, rebuild style vectors
Ep = E.copy()
x = Ep["z_loglen"].values
for d in DIMS:
    y = Ep[f"z_{d}"].values
    b = np.polyfit(x, y, 1)
    Ep[f"z_{d}"] = y - np.polyval(b, x)
style_p = Ep.groupby("folder")[Z].mean().loc[style.index]
style_p["cohort"] = style.cohort
R["S4"] = s1(style_p, Z)

# S2: per-prompt spread across a cohort's answers
def s2(B=200):
    out = {c: [] for c in COH}
    piv = {d: E.pivot_table(index="iid", columns="folder", values=f"z_{d}") for d in DIMS}
    for _ in range(B):
        for c in COH:
            fs = rng.choice(style[style.cohort == c].index.values, k, replace=True)
            out[c].append(float(np.mean([np.nanmean(piv[d][fs].std(axis=1)) for d in DIMS])))
    return {c: {"mean": float(np.mean(v)), "ci": [float(np.percentile(v, q)) for q in (2.5, 97.5)]} for c, v in out.items()}
R["S2"] = s2()

# S3: drift
mm = m[m.eligible & m.folder.isin(style.index)].copy()
mm["cohort"] = mm.folder.map(style.cohort)
per_model = mm.groupby("folder").agg(tokens=("prose_tokens", "median"), rv=("framed_rᵥ", "mean"), rep4=("rep4", "mean"),
                                    code=("edit_code_blocks", lambda s: (s > 0).mean()), headers=("edit_headers", lambda s: (s > 0).mean()),
                                    bold=("edit_bold_lines", lambda s: (s > 0).mean()))
per_model = per_model.join(style[["cohort"] + Z])
S3 = {}
for col in ["tokens", "rv", "rep4", "headers", "bold"] + Z:
    S3[col] = {c: float(per_model[per_model.cohort == c][col].median()) for c in COH}
    a, b = per_model[per_model.cohort == "2023-H2"][col].values, per_model[per_model.cohort == "2024-H2"][col].values
    diffs = [np.median(rng.choice(b, len(b))) - np.median(rng.choice(a, len(a))) for _ in range(1000)]
    S3[col]["diff_ci"] = [float(np.percentile(diffs, q)) for q in (2.5, 97.5)]
R["S3"] = S3
per_model.assign(date=style.date).to_csv(HERE / "data" / "style_by_model.csv")

# ---------- Study 2: the judge ----------
v = pd.DataFrame(json.loads((HERE / "data" / "verdicts.json").read_text()))
v["iid"] = [hashlib.sha1(s.encode()).hexdigest()[:12] for s in v.instruction]
v = v[(v.generator_1 == REF) & ~v.folder.isin(EXCLUDE + [REF])].copy()
v["pref"] = pd.to_numeric(v.preference, errors="coerce")
v = v.dropna(subset=["pref"])
cols = ["folder", "iid", "loglen", "rep4", "eligible"] + [f"framed_{d}" for d in DIMS]
ref = m[m.folder == REF][cols].drop(columns="folder").set_index("iid")
mod = m[cols].set_index(["folder", "iid"])
J = v[["folder", "iid", "pref"]].join(mod, on=["folder", "iid"]).join(ref, on="iid", rsuffix="_ref")
J = J[(J.eligible == True) & (J.eligible_ref == True)].dropna()
X = pd.DataFrame({"dlen": J.loglen - J.loglen_ref, "drep4": J.rep4 - J.rep4_ref})
for d in DIMS:
    X[d] = J[f"framed_{d}"] - J[f"framed_{d}_ref"]
y = (J.pref > 1.5).astype(int).values
groups = J.folder.values
R["J_n"] = int(len(J)); R["J_models"] = int(J.folder.nunique()); R["J_model_win_share"] = float(y.mean())


def cv(cols):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        A = X[cols].values; mu, sd = A[tr].mean(0), A[tr].std(0) + 1e-9
        p[te] = LogisticRegression(max_iter=3000).fit((A[tr] - mu) / sd, y[tr]).predict_proba((A[te] - mu) / sd)[:, 1]
    return p

sets = {"len": ["dlen"], "len+kernel": ["dlen"] + DIMS, "len+rep": ["dlen", "drep4"], "len+rep+kernel": ["dlen", "drep4"] + DIMS}
P = {k_: cv(c) for k_, c in sets.items()}
J1 = {k_: float(roc_auc_score(y, p)) for k_, p in P.items()}
uniq = np.unique(groups); members = {g: np.where(groups == g)[0] for g in uniq}
gains = {"kernel_over_len": [], "kernel_over_len_rep": []}
for _ in range(200):
    ix = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq))])
    a = {k_: roc_auc_score(y[ix], p[ix]) for k_, p in P.items()}
    gains["kernel_over_len"].append(a["len+kernel"] - a["len"])
    gains["kernel_over_len_rep"].append(a["len+rep+kernel"] - a["len+rep"])
J1["gain_ci"] = {g: [float(np.percentile(v_, q)) for q in (2.5, 97.5)] for g, v_ in gains.items()}
R["J1"] = J1


def std_coefs(Xd, yv, cols):
    A = Xd[cols].values; A = (A - A.mean(0)) / (A.std(0) + 1e-9)
    return LogisticRegression(max_iter=3000).fit(A, yv).coef_[0]

CC = ["dlen", "drep4"] + DIMS
judge_coef = std_coefs(X, y, CC)
jb = []
for _ in range(100):
    ix = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq))])
    jb.append(std_coefs(X.iloc[ix], y[ix], CC))
jb = np.array(jb)

# human side: HelpSteer2 helpfulness pairs, same features
hs = pd.read_csv(ROOT / "studies/helpsteer2/data/readings.csv")
hs = hs[hs.eligible].copy(); hs["loglen"] = np.log1p(hs.prose_tokens)
texts = []
for split in ("train", "validation"):
    for line in gzip.open(ROOT / "public_corpora/helpsteer2" / f"{split}.jsonl.gz", "rt", encoding="utf-8"):
        texts.append(json.loads(line)["response"])
WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
def rep4(t):
    w = WORD.findall(t.lower()); g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 0.0 if not g else 1 - len(set(g)) / len(g)
hs["rep4"] = [rep4(texts[i]) for i in hs.rid]
pairs = hs.groupby("pid").filter(lambda g: len(g) == 2).sort_values(["pid", "rid"])
a_, b_ = pairs.iloc[0::2].reset_index(drop=True), pairs.iloc[1::2].reset_index(drop=True)
keep_h = (a_.helpfulness != b_.helpfulness).values
Xh = pd.DataFrame({"dlen": a_.loglen - b_.loglen, "drep4": a_.rep4 - b_.rep4})
for d in DIMS:
    Xh[d] = a_[f"framed_{d}"] - b_[f"framed_{d}"]
Xh = Xh[keep_h].reset_index(drop=True); yh = (a_.helpfulness > b_.helpfulness).values[keep_h].astype(int)
human_coef = std_coefs(Xh, yh, CC)
hb = np.array([std_coefs(Xh.iloc[ix], yh[ix], CC) for ix in (rng.integers(0, len(yh), len(yh)) for _ in range(100))])
R["J2"] = {c: {"judge": float(judge_coef[i]), "judge_ci": [float(np.percentile(jb[:, i], 2.5)), float(np.percentile(jb[:, i], 97.5))],
               "human": float(human_coef[i]), "human_ci": [float(np.percentile(hb[:, i], 2.5)), float(np.percentile(hb[:, i], 97.5))]}
           for i, c in enumerate(CC)}
R["J2_human_pairs"] = int(len(yh))

# J3: model level, LC win rate vs length + style (leave-one-out)
ml = style.join(lb[["length_controlled_winrate", "avg_length", "mode"]], how="inner").drop(index=[REF], errors="ignore")
ml = ml.dropna(subset=["length_controlled_winrate", "avg_length"])
ml["logavg"] = np.log(ml.avg_length)
def loo_r2(cols):
    A = ml[cols].values; yy = ml.length_controlled_winrate.values; pred = np.zeros(len(yy))
    for i in range(len(yy)):
        tr = np.arange(len(yy)) != i
        pred[i] = LinearRegression().fit(A[tr], yy[tr]).predict(A[i:i + 1])[0]
    return float(1 - ((yy - pred) ** 2).sum() / ((yy - yy.mean()) ** 2).sum())
r2_len, r2_full = loo_r2(["logavg"]), loo_r2(["logavg"] + Z)
bs = []
for _ in range(300):
    ix = rng.integers(0, len(ml), len(ml)); sub = ml.iloc[ix]
    def fit_r2(cols):
        A = sub[cols].values; yy = sub.length_controlled_winrate.values
        return LinearRegression().fit(A, yy).score(A, yy)
    bs.append(fit_r2(["logavg"] + Z) - fit_r2(["logavg"]))
coefs3 = LinearRegression().fit((ml[["logavg"] + Z] - ml[["logavg"] + Z].mean()) / ml[["logavg"] + Z].std(), ml.length_controlled_winrate).coef_
R["J3"] = {"n_models": int(len(ml)), "loo_r2_len": r2_len, "loo_r2_len_style": r2_full,
           "insample_gain_boot_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
           "std_coefs": dict(zip(["logavg"] + DIMS, np.round(coefs3, 3).tolist())),
           "spearman_lc_by_axis": {d: float(ml[f"z_{d}"].corr(ml.length_controlled_winrate, method="spearman")) for d in DIMS}}
ml.to_csv(HERE / "data" / "model_level.csv")

(HERE / "data" / "results.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
print(json.dumps({k_: R[k_] for k_ in ["n_models", "models_in_style_study", "cohort_sizes", "k_subsample", "S1", "S4", "S2", "J_n", "J_models", "J_model_win_share", "J1", "J3"]}, indent=1, default=float))
for c in CC:
    print(f"{c:6s} judge {R['J2'][c]['judge']:+.3f} {np.round(R['J2'][c]['judge_ci'],3)}  human {R['J2'][c]['human']:+.3f} {np.round(R['J2'][c]['human_ci'],3)}")
for col, s in S3.items():
    print(col, {c: round(s[c], 3) for c in COH}, np.round(s["diff_ci"], 3))
