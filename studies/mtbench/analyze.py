#!/usr/bin/env python3
"""MT-Bench same-pairs analysis (PLAN.md M-a to M-e). Writes data/readings.csv and data/results.json."""
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "reading_frame"))
import reading_frame as RF

DIMS = ["Δx", "rᵥ", "ρᵣ", "μ", "Δy", "xₚ", "θ", "dₛ"]
rng = np.random.default_rng(0)
WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def rep4(t):
    w = WORD.findall(t.lower()); g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 0.0 if not g else 1 - len(set(g)) / len(g)


hum = pd.read_parquet(ROOT / "public_corpora/mtbench/human.parquet")
gpt = pd.read_parquet(ROOT / "public_corpora/mtbench/gpt4_pair.parquet")


def answer(conv, turn):
    msgs = [m["content"] for m in conv if m["role"] == "assistant"]
    return msgs[turn - 1] if len(msgs) >= turn else ""


# ---- read every distinct (question, turn, model) answer once ----
texts = {}
for d in (hum, gpt):
    for r in d.itertuples():
        texts[(r.question_id, r.turn, r.model_a)] = answer(r.conversation_a, r.turn)
        texts[(r.question_id, r.turn, r.model_b)] = answer(r.conversation_b, r.turn)
reader = RF.Reader()
rows = []
for (q, t, mdl), txt in texts.items():
    row = RF.flat_row(reader.read(txt)); row.update(qid=q, turn=t, model=mdl, rep4=rep4(txt)); rows.append(row)
A = pd.DataFrame(rows)
(HERE / "data").mkdir(exist_ok=True)
A.to_csv(HERE / "data" / "readings.csv", index=False)
A["loglen"] = np.log1p(A.prose_tokens)
A = A.set_index(["qid", "turn", "model"])
R = {"answers_read": len(A), "answers_eligible": int(A.eligible.sum())}


def build(d, who):
    d = d[d.winner.isin(["model_a", "model_b"])].copy()
    a = A.loc[list(zip(d.question_id, d.turn, d.model_a))].reset_index(drop=True)
    b = A.loc[list(zip(d.question_id, d.turn, d.model_b))].reset_index(drop=True)
    X = pd.DataFrame({"dlen": a.loglen.values - b.loglen.values, "drep4": a.rep4.values - b.rep4.values})
    for c in DIMS:
        X[c] = a[f"framed_{c}"].values - b[f"framed_{c}"].values
    X["y"] = (d.winner.values == "model_a").astype(int)
    X["qid"] = d.question_id.values; X["turn"] = d.turn.values
    X["pair"] = [f"{q}|{t}|{'|'.join(sorted([ma, mb]))}" for q, t, ma, mb in zip(d.question_id, d.turn, d.model_a, d.model_b)]
    X["sign"] = [1 if ma < mb else -1 for ma, mb in zip(d.model_a, d.model_b)]  # orient to sorted pair
    X["elig"] = a.eligible.values & b.eligible.values
    X["who"] = who
    return X

H, G = build(hum, "human"), build(gpt, "gpt4")
shared = set(H[H.elig].pair) & set(G[G.elig].pair)
H, G = H[H.elig & H.pair.isin(shared)].reset_index(drop=True), G[G.elig & G.pair.isin(shared)].reset_index(drop=True)
R.update(shared_pairs=len(shared), human_votes=len(H), gpt4_verdicts=len(G), human_judges=int(hum.judge.nunique()))

# ---- structure index: AlpacaEval judge direction, fixed ----
AE = json.loads((ROOT / "studies/alpacaeval/data/results.json").read_text())["J2"]
w = np.array([AE[c]["judge"] for c in DIMS]); w = w / np.linalg.norm(w)
R["structure_direction"] = dict(zip(DIMS, np.round(w, 4).tolist()))
both = pd.concat([H, G])
sd = {c: both[c].std() for c in ["dlen", "drep4"] + DIMS}
for X in (H, G):
    X["struct"] = sum((X[c] / sd[c]) * w[i] for i, c in enumerate(DIMS))
    for c in ["dlen", "drep4"] + DIMS:
        X[f"s_{c}"] = X[c] / sd[c]   # same scaling for both judges


def fit(X, cols):
    return LogisticRegression(C=1e6, max_iter=5000).fit(X[cols].values, X.y.values).coef_[0]


def boot(X, cols, B=1000):
    q = X.qid.unique(); groups = {k: np.where(X.qid.values == k)[0] for k in q}
    for _ in range(B):
        ix = np.concatenate([groups[k] for k in rng.choice(q, len(q))])
        yield ix


def ratios(XH, XG, cols, num, B=1000):
    ch, cg = fit(XH, cols), fit(XG, cols)
    i_len, i_num = cols.index("s_dlen"), cols.index(num)
    base = {"human_beta": ch.tolist(), "gpt4_beta": cg.tolist(), "cols": cols,
            "r_human": float(ch[i_num] / ch[i_len]), "r_gpt4": float(cg[i_num] / cg[i_len])}
    qs = sorted(set(XH.qid) | set(XG.qid))
    gh = {k: np.where(XH.qid.values == k)[0] for k in qs}; gg = {k: np.where(XG.qid.values == k)[0] for k in qs}
    rh, rg, bl_h, bl_g = [], [], [], []
    for _ in range(B):
        pick = rng.choice(qs, len(qs))
        ih = np.concatenate([gh[k] for k in pick]); ig = np.concatenate([gg[k] for k in pick])
        a, b = fit(XH.iloc[ih], cols), fit(XG.iloc[ig], cols)
        rh.append(a[i_num] / a[i_len]); rg.append(b[i_num] / b[i_len]); bl_h.append(a[i_len]); bl_g.append(b[i_len])
    rh, rg = np.array(rh), np.array(rg)
    pc = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    base.update(r_human_ci=pc(rh), r_gpt4_ci=pc(rg), diff=float(base["r_gpt4"] - base["r_human"]), diff_ci=pc(rg - rh),
                beta_len_human_ci=pc(bl_h), beta_len_gpt4_ci=pc(bl_g))
    return base

COLS = ["s_dlen", "s_drep4", "struct"]
R["Ma_struct"] = ratios(H, G, COLS, "struct")
R["Ma_rep"] = ratios(H, G, COLS, "s_drep4")

# ---- M-b: all 8 axes ----
C8 = ["s_dlen", "s_drep4"] + [f"s_{c}" for c in DIMS]
bh, bg = fit(H, C8), fit(G, C8)
cosv = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
R["Mb"] = {"human": dict(zip(C8, np.round(bh, 4).tolist())), "gpt4": dict(zip(C8, np.round(bg, 4).tolist())),
           "cos_8axes": cosv(bh[2:], bg[2:])}
per_axis = {}
for c in DIMS:
    r = ratios(H, G, C8, f"s_{c}", B=300)
    per_axis[c] = {k: r[k] for k in ("r_human", "r_human_ci", "r_gpt4", "r_gpt4_ci", "diff", "diff_ci")}
R["Mb_per_axis"] = per_axis
cb = []
qs = sorted(set(H.qid)); gh = {k: np.where(H.qid.values == k)[0] for k in qs}; gg = {k: np.where(G.qid.values == k)[0] for k in qs}
for _ in range(300):
    pick = rng.choice(qs, len(qs))
    a = fit(H.iloc[np.concatenate([gh[k] for k in pick])], C8); b = fit(G.iloc[np.concatenate([gg[k] for k in pick])], C8)
    cb.append(cosv(a[2:], b[2:]))
R["Mb"]["cos_8axes_ci"] = [float(np.percentile(cb, 2.5)), float(np.percentile(cb, 97.5))]

# ---- M-c: held-out AUC by question folds ----
def cv_auc(X, cols):
    p = np.zeros(len(X))
    for tr, te in GroupKFold(5).split(X, X.y, X.qid):
        p[te] = LogisticRegression(max_iter=5000).fit(X.iloc[tr][cols], X.y.iloc[tr]).predict_proba(X.iloc[te][cols])[:, 1]
    return float(roc_auc_score(X.y, p))
R["Mc"] = {who: {"len": cv_auc(X, ["s_dlen"]), "len+rep": cv_auc(X, ["s_dlen", "s_drep4"]),
                 "len+rep+struct": cv_auc(X, COLS), "len+rep+kernel8": cv_auc(X, C8)} for who, X in (("human", H), ("gpt4", G))}

# ---- M-d: human majority vote per pair ----
H2 = H.copy(); H2["y_or"] = np.where(H2.sign == 1, H2.y, 1 - H2.y)   # winner = sorted-first model?
agg = H2.groupby("pair").agg(m=("y_or", "mean"), n=("y_or", "size")).reset_index()
agg = agg[agg.m != 0.5]
first = H2.drop_duplicates("pair").set_index("pair")
Hm = first.loc[agg.pair].reset_index()
flip = Hm.sign.values  # rows oriented as stored; re-orient so features match the sorted-first model
for c in ["s_dlen", "s_drep4", "struct"]:
    Hm[c] = Hm[c] * flip
Hm["y"] = (agg.m.values > 0.5).astype(int)
G2 = G.copy()
for c in ["s_dlen", "s_drep4", "struct"]:
    G2[c] = G2[c] * G2.sign
G2["y"] = np.where(G2.sign == 1, G2.y, 1 - G2.y)
R["Md"] = ratios(Hm, G2, COLS, "struct")
R["Md"]["pairs_majority"] = int(len(Hm))

# ---- M-e: agreement ----
H2["y_sorted"] = H2.y_or
multi = H2.groupby("pair").filter(lambda g: len(g) >= 2)
hh = []
for _, g in multi.groupby("pair"):
    v = g.y_sorted.values
    hh += [v[i] == v[j] for i in range(len(v)) for j in range(i + 1, len(v))]
gmap = dict(zip(G2.pair, G2.y))
hg = [gmap[p] == y for p, y in zip(H2.pair, H2.y_sorted) if p in gmap]
R["Me"] = {"human_human_agreement": float(np.mean(hh)), "human_human_pairs_of_votes": len(hh),
           "human_gpt4_agreement": float(np.mean(hg)), "human_votes_compared": len(hg)}

(HERE / "data" / "results.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
pr = lambda k: print(k, json.dumps({x: (round(v, 3) if isinstance(v, float) else v) for x, v in R[k].items() if x not in ("human_beta", "gpt4_beta", "cols")}, ensure_ascii=False, default=float))
print({k: R[k] for k in ("answers_read", "answers_eligible", "shared_pairs", "human_votes", "gpt4_verdicts")})
for k in ("Ma_struct", "Ma_rep", "Md", "Me", "Mc"):
    pr(k)
print("cos8", round(R["Mb"]["cos_8axes"], 3), R["Mb"]["cos_8axes_ci"])
for c in DIMS:
    p = R["Mb_per_axis"][c]; print(f"{c:3s} human {p['r_human']:+.3f} gpt4 {p['r_gpt4']:+.3f} diff {p['diff']:+.3f} {np.round(p['diff_ci'],3)}")
print("betas human", R["Mb"]["human"]); print("betas gpt4", R["Mb"]["gpt4"])
