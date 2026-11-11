#!/usr/bin/env python3
"""Generates numbers.tex (LaTeX macros for every number in the paper) and figures/*.pdf
from the four studies' result files. Run before compiling main.tex."""
import json, runpy
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parent
S = P.parent
L = lambda p: json.loads((S / p).read_text())
HS, HP, HR, HT = L("helpsteer2/data/results.json"), L("helpsteer2/data/pairwise.json"), L("helpsteer2/data/pairwise_robustness.json"), L("helpsteer2/data/theta_check.json")
AE, AJ = L("alpacaeval/data/results.json"), L("alpacaeval/data/judge_direction.json")
MT = L("mtbench/data/results.json")
AR, ARC = L("arena100k/data/results.json"), L("arena100k/data/range_check.json")

# ---------------- numbers.tex ----------------
N = {}
def put(k, v): N[k] = v
f2 = lambda x: f"{x:.2f}"; f3 = lambda x: f"{x:.3f}"; s2 = lambda x: f"{x:+.2f}".replace("-", "$-$"); s3 = lambda x: f"{x:+.3f}".replace("-", "$-$")
ci = lambda c, n=2: ("[" + f"{c[0]:+.{n}f}, {c[1]:+.{n}f}" + "]").replace("-", "$-$")
pct = lambda x: f"{100*x:.0f}\\%"
# HelpSteer2
put("hsPairs", f"{HS['B_n_pairs']:,}"); put("hsN", f"{HS['n_all']:,}")
for s, k in (("helpfulness", "Help"), ("correctness", "Corr"), ("coherence", "Coh"), ("complexity", "Cplx"), ("verbosity", "Verb")):
    put(f"hs{k}Len", f3(HP[s]["auc_len"])); put(f"hs{k}K", f3(HP[s]["auc_len_framed"]))
    put(f"hs{k}LenRep", f3(HR[s]["with_rep4"]["auc_len_rep4"])); put(f"hs{k}LenRepK", f3(HR[s]["with_rep4"]["auc_len_rep4_framed"]))
    put(f"hs{k}GainCI", ci(HR[s]["with_rep4"]["auc_gain_ci"], 3))
put("hsDsPooled", s3(HS["A_framed"]["dₛ|complexity"]["partial"])); put("hsDsPair", s3(HS["B_pairs"]["dₛ|complexity"]["partial"]))
for d, k in (("ρᵣ", "Rho"), ("xₚ", "Xp"), ("rᵥ", "Rv"), ("dₛ", "Ds"), ("θ", "Theta"), ("μ", "Mu")):
    put(f"hsPair{k}", s3(HS["B_pairs"][f"{d}|helpfulness"]["partial"])); put(f"hsPair{k}CI", ci(HS["B_pairs"][f"{d}|helpfulness"]["ci"], 3))
put("hsThetaRep", s2(HT["pooled_spearman_theta_rep4"]))
put("hsThetaCoh", s3(HT["coherence"]["theta_given_len"])); put("hsThetaCohRep", s3(HT["coherence"]["theta_given_len_rep4"])); put("hsThetaCohRepCI", ci(HT["coherence"]["theta_given_len_rep4_ci"], 3))
# AlpacaEval
S1, S3, J1, J2, J3 = AE["S1"], AE["S3"], AE["J1"], AE["J2"], AE["J3"]
put("aeModels", str(AE["n_models"])); put("aeAnswers", f"{AE['n_answers']:,}"); put("aeVerdicts", f"{AE['J_n']:,}"); put("aeStyleModels", str(AE["models_in_style_study"]))
put("aeTokA", f"{S3['tokens']['2023-H2']:.0f}"); put("aeTokC", f"{S3['tokens']['2024-H2']:.0f}")
put("aeBoldA", pct(S3["bold"]["2023-H2"])); put("aeBoldB", pct(S3["bold"]["2024-H1"])); put("aeBoldC", pct(S3["bold"]["2024-H2"]))
put("aeSpreadA", f2(S1["2023-H2"]["mean"])); put("aeSpreadC", f2(S1["2024-H2"]["mean"])); put("aeSpreadDiffCI", ci(S1["diff_2024H2_minus_2023H2"]["ci"]))
put("aeSpreadLenDiffCI", ci(AE["S4"]["diff_2024H2_minus_2023H2"]["ci"]))
put("aeJLen", f3(J1["len"])); put("aeJK", f3(J1["len+kernel"])); put("aeJLenRep", f3(J1["len+rep"])); put("aeJLenRepK", f3(J1["len+rep+kernel"]))
put("aeJGainRepCI", ci(J1["gain_ci"]["kernel_over_len_rep"], 3))
put("aeCosJH", f2(AJ["cos_judge_human"])); put("aeCosDriftJ", f2(AJ["cos_drift_judge"])); put("aeCosDriftH", f2(AJ["cos_drift_human"]))
put("aeLooLen", f2(J3["loo_r2_len"])); put("aeLooStyle", f2(J3["loo_r2_len_style"])); put("aeJthreeN", str(J3["n_models"]))
put("aeRhoLCJudge", s2(AJ["H2"]["all"]["spearman_lc_judge_proj"]))
put("aeCohA", str(AE["cohort_sizes"]["2023-H2"])); put("aeCohB", str(AE["cohort_sizes"]["2024-H1"])); put("aeCohC", str(AE["cohort_sizes"]["2024-H2"]))
# MT-Bench
Ma, Md, Me, Mc, Mb = MT["Ma_struct"], MT["Md"], MT["Me"], MT["Mc"], MT["Mb"]
put("mtPairs", str(MT["shared_pairs"])); put("mtHumanVotes", f"{MT['human_votes']:,}"); put("mtGptVerdicts", str(MT["gpt4_verdicts"]))
put("mtRH", s3(Ma["r_human"])); put("mtRG", s3(Ma["r_gpt4"])); put("mtRDiffCI", ci(Ma["diff_ci"])); put("mtMajDiffCI", ci(Md["diff_ci"]))
put("mtAgreeHH", pct(Me["human_human_agreement"])); put("mtAgreeHG", pct(Me["human_gpt4_agreement"]))
put("mtAucHLen", f3(Mc["human"]["len"])); put("mtAucHRep", f3(Mc["human"]["len+rep"])); put("mtAucGLen", f3(Mc["gpt4"]["len"])); put("mtAucGRep", f3(Mc["gpt4"]["len+rep"]))
bh, bg = Mb["human"], Mb["gpt4"]
put("mtScaleLen", f2(bg["s_dlen"] / bh["s_dlen"])); put("mtScaleRep", f2(bg["s_drep4"] / bh["s_drep4"])); put("mtScaleRv", f2(bg["s_rᵥ"] / bh["s_rᵥ"]))
put("mtScaleXp", f"{bg['s_xₚ']/bh['s_xₚ']:.1f}"); put("mtScaleDs", f"{bg['s_dₛ']/bh['s_dₛ']:.1f}")
# Arena + fixed units
A, B, BH, D, C = AR["arena_A"], AR["arena_B"], AR["arena_B_head"], AR["arena_D"], AR["C"]
put("arBattles", f"{AR['arena_counts']['battles_both_eligible']:,}"); put("arFiltered", f"{AR['arena_counts']['battles_filtered']:,}")
put("arBoldShare", pct(AR["arena_counts"]["answers_with_bold_title"])); put("arHeadShare", pct(AR["arena_counts"]["answers_with_headers"]))
put("arStruct", s3(A["beta"]["struct"])); put("arStructCI", ci(A["beta_ci"]["struct"], 3)); put("arRatio", s3(A["ratio_struct_len"])); put("arRatioCI", ci(A["ratio_struct_len_ci"], 3))
put("arRep", s3(A["beta"]["drep4"])); put("arRepCI", ci(A["beta_ci"]["drep4"], 3)); put("arLen", s3(A["beta"]["dlen"]))
put("arBold", s3(B["beta"]["dbold"])); put("arBoldCI", ci(B["beta_ci"]["dbold"], 3)); put("arHead", s3(BH["beta"]["dhead"])); put("arHeadCI", ci(BH["beta_ci"]["dhead"], 3))
put("arNeiStruct", s3(D["neither_style_model"]["beta"]["struct"])); put("arNeiStructCI", ci(D["neither_style_model"]["beta_ci"]["struct"], 3)); put("arNeiN", f"{D['neither_style_model']['n']:,}")
put("arBothStruct", s3(D["both_style_models"]["beta"]["struct"])); put("arBothStructCI", ci(D["both_style_models"]["beta_ci"]["struct"], 3)); put("arBothN", f"{D['both_style_models']['n']:,}")
put("arSdBoth", f2(ARC["sd_struct_both"])); put("arSdNei", f2(ARC["sd_struct_neither"])); put("arMatched", s3(ARC["neither_matched_struct_beta"])); put("arMatchedCI", ci(ARC["neither_matched_struct_beta_ci"], 3))
put("arStyleModels", str(len(AR["style_models"])))
put("fuAE", s3(AR["alpacaeval_judge"]["ratio_struct_len"])); put("fuAECI", ci(AR["alpacaeval_judge"]["ratio_struct_len_ci"], 3))
put("fuMTH", s3(AR["mtbench_human"]["ratio_struct_len"])); put("fuMTG", s3(AR["mtbench_gpt4"]["ratio_struct_len"]))
put("fuAEminusAR", ci(C["alpacaeval_judge_minus_arena_human"]["ci"])); put("fuARminusMT", ci(C["arena_human_minus_mtbench_human"]["ci"]))
# coin flip among both-styled
g = runpy.run_path(str(S / "arena100k/analyze_core.py"))
AR_ = g["AR"]; bs = AR_[AR_.both_style]
from sklearn.linear_model import LogisticRegression as _LR
mixed = AR_[~AR_.both_style & ~AR_.neither_style].reset_index(drop=True)
_f = lambda X: _LR(C=1e6, max_iter=5000).fit(X[["dlen", "drep4", "struct"]].values, X.y.values).coef_[0]
_rng = np.random.default_rng(3); _u = mixed.cluster.unique(); _m = {k: np.where(mixed.cluster.values == k)[0] for k in _u}
_bs = [_f(mixed.iloc[np.concatenate([_m[k] for k in _rng.choice(_u, len(_u))])])[2] for _ in range(300)]
put("arMixN", f"{len(mixed):,}"); put("arMixStruct", s3(_f(mixed)[2])); put("arMixStructCI", ci([float(np.percentile(_bs, 2.5)), float(np.percentile(_bs, 97.5))], 3))
put("arBothHigherWins", pct(float((np.where(bs.y == 1, bs.struct, -bs.struct) > 0).mean())))
# worked examples
import gzip, re, sys
sys.path.insert(0, str(S.parent / "reading_frame")); import reading_frame as RF
W_ = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
def rep4(x):
    w = W_.findall(x.lower()); g_ = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 0.0 if not g_ else 1 - len(set(g_)) / len(g_)
hs_t = []
for sp in ("train", "validation"):
    for line in gzip.open(S.parent / "public_corpora/helpsteer2" / f"{sp}.jsonl.gz", "rt", encoding="utf-8"):
        hs_t.append(json.loads(line)["response"])
rd = RF.Reader()
for tag, i in (("A", 13950), ("B", 13951)):
    r_ = RF.flat_row(rd.read(hs_t[i]))
    put(f"exOneRep{tag}", f"{rep4(hs_t[i]):.2f}"); put(f"exOneTok{tag}", str(r_["prose_tokens"])); put(f"exOneTheta{tag}", f"{r_['framed_θ']:.2f}")
ae_q = "Why can I see the moon during the day?"
for tag, folder in (("A", "claude-2"), ("B", "gemma-2-9b-it-SimPO")):
    out = {r_["instruction"]: r_["output"] for r_ in json.load(open(S.parent / f"public_corpora/alpaca_eval/results/{folder}/model_outputs.json"))}[ae_q]
    r_ = RF.flat_row(rd.read(out))
    put(f"exTwoTok{tag}", str(r_["prose_tokens"])); put(f"exTwoRep{tag}", f"{rep4(out):.2f}")
    for d, k in (("rᵥ", "Rv"), ("ρᵣ", "Rho"), ("xₚ", "Xp"), ("dₛ", "Ds"), ("θ", "Theta")):
        put(f"exTwo{k}{tag}", f"{r_['framed_' + d]:.2f}")
put("exThreeStruct", f"{AR_.iloc[16443].struct:+.2f}".replace("-", "$-$"))
# Arena judge run
JA = L("arena_judge/results/analysis.json"); JR = L("arena_judge/results/added_repetition.json")
put("jBattles", f"{sum(1 for _ in open(S / 'arena_judge/sample.csv')) - 1:,}")
put("jRequests", f"{sum(1 for _ in open(S / 'arena_judge/requests.jsonl')):,}")
put("jRefusals", str(JA["unparsed_requests"]))
for c, k in (("both", "B"), ("neither", "N")):
    r = JA[c]
    put(f"j{k}n", f"{r['n_judge_nontie']:,}"); put(f"j{k}Cons", pct(r["position_consistency"]))
    put(f"j{k}Struct", s3(r["judge_beta"]["struct"])); put(f"j{k}StructCI", ci(r["judge_beta_ci"]["struct"], 3))
    put(f"j{k}HStruct", s3(r["human_beta_same_battles"]["struct"])); put(f"j{k}HStructCI", ci(r["human_beta_ci"]["struct"], 3))
    put(f"j{k}Ratio", s2(r["ratio_judge"])); put(f"j{k}HRatio", s2(r["ratio_human"])); put(f"j{k}RatioDiffCI", ci(r["ratio_diff_ci"]))
    put(f"j{k}Agree", pct(r["agreement_judge_human"]))
    put(f"j{k}RepJ", f"{JR[c]['rep_ratio_judge']:+.1f}".replace("-", "$-$")); put(f"j{k}RepH", f"{JR[c]['rep_ratio_human']:+.1f}".replace("-", "$-$"))
    put(f"j{k}RepDiffCI", ci(JR[c]["diff_ci"], 1)); put(f"j{k}RepDiffNoCodeCI", ci(JR[c + "_nocode"]["diff_ci"], 1))
put("jAgreeDiffCI", ci(JA["agreement_both_minus_neither"]["ci"], 3))
# Reward-model audit
RA = L("rm_audit/results/analysis.json"); RC = L("rm_audit/results/added_checks.json"); RV = L("rm_audit/results/voters_full_cohort.json")
put("rmVotersB", s3(RV["both"][0])); put("rmVotersBCI", ci(RV["both"][1], 3)); put("rmVotersN", s3(RV["neither"][0]))
for rm, k in (("armorm", "Armo"), ("gpm", "Gpm"), ("pairrm", "Pair")):
    for c, cc in (("neither", "N"), ("both", "B")):
        r = RA[rm][c]
        put(f"rm{k}{cc}", s3(r["rm_beta"]["struct"])); put(f"rm{k}{cc}CI", ci(r["rm_beta_ci"]["struct"], 3))
        put(f"rm{k}{cc}Agree", pct(r["agreement"]))
        put(f"rm{k}{cc}RatioRM", s2(r["struct_ratio_rm"])); put(f"rm{k}{cc}RatioH", s2(r["struct_ratio_human"])); put(f"rm{k}{cc}RatioDiffCI", ci(r["struct_ratio_diff_ci"]))
        put(f"rm{k}{cc}RepRM", f"{r['rep_ratio_rm']:+.1f}".replace("-", "$-$")); put(f"rm{k}{cc}RepH", f"{r['rep_ratio_human']:+.1f}".replace("-", "$-$"))
    put(f"rm{k}NoStudent", s3(RC["C2"][rm]["without_students"]["beta_struct"]))
    put(f"rm{k}NoCode", s3(RC["C3"][rm]["both"]["beta_struct"]))
put("rmArmoStudentBattles", str(RC["C2"]["armorm"]["battles_with_student"]))
put("rmCosArmoJudge", f2(RC["C1_cosines"]["armorm"]["judge_alpacaeval"])); put("rmCosArmoHuman", f2(RC["C1_cosines"]["armorm"]["human_helpsteer2"]))
put("rmCosGpmPair", f2(RC["C1_cosines"]["gpm"]["pairrm"]))
put("rmFpOwn", str(sum(1 for v in RA["fingerprint"].values() if v["own_is_highest"]))); put("rmFpN", str(len(RA["fingerprint"])))
# Style-control baseline
SB = L("style_baseline/results.json")
for key, k in (("voters_all", "VAll"), ("voters_neither", "VN"), ("voters_both", "VB"), ("judge_neither", "JN"), ("judge_both", "JB"),
               ("armorm_neither", "AN"), ("armorm_both", "AB"), ("gpm_neither", "GN"), ("gpm_both", "GB"), ("pairrm_neither", "PN"), ("pairrm_both", "PB")):
    r = SB[key]
    put(f"sb{k}", s3(r["struct_beta_with_SC"])); put(f"sb{k}CI", ci(r["struct_ci_with_SC"], 3))
    put(f"sb{k}AucSC", f3(r["SC_to_SCK"]["auc_base"])); put(f"sb{k}AucSCK", f3(r["SC_to_SCK"]["auc_full"])); put(f"sb{k}GainCI", ci(r["SC_to_SCK"]["gain_ci"], 3))
    put(f"sb{k}AucK", f3(r["K_to_KSC"]["auc_base"])); put(f"sb{k}RevGainCI", ci(r["K_to_KSC"]["gain_ci"], 3))
put("sbVAllN", f"{SB['voters_all']['n']:,}")
(P / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in N.items()), encoding="utf-8")

# ---------------- figures ----------------
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e4e2dc", "grid.linewidth": 0.6, "pdf.fonttype": 42})
C1, C2, C3, MUT = "#2a78d6", "#eb6834", "#1baf7a", "#7a7974"
COH = {"2023-H2": C1, "2024-H1": C2, "2024-H2": C3}
FIG = P / "figures"; FIG.mkdir(exist_ok=True)

M = pd.read_csv(S / "alpacaeval/data/judge_direction_models.csv", index_col=0)
dates = json.loads((S / "alpacaeval/data/added_dates.json").read_text())
M["date"] = pd.to_datetime(M.index.map(dates))
# Fig 1: drift by submission date
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
for c, col in COH.items():
    g_ = M[M.cohort == c]
    ax[0].scatter(g_.date, 100 * g_.bold, s=10, color=col, alpha=0.85, lw=0.3, edgecolor="white", label=c)
    ax[1].scatter(g_.date, g_.proj_judge, s=10, color=col, alpha=0.85, lw=0.3, edgecolor="white")
ax[0].set_ylabel("bold title line (% of answers)"); ax[1].set_ylabel("structure index")
import matplotlib.dates as mdates
for a in ax:
    a.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); a.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    a.set_xlabel("date answers were added to AlpacaEval")
ax[0].legend(frameon=False, loc="upper left", fontsize=7.5)
ax[0].set_title("(a) formatting", fontsize=8.5, loc="left"); ax[1].set_title("(b) structure index (model style)", fontsize=8.5, loc="left")
fig.tight_layout(); fig.savefig(FIG / "fig_drift.pdf"); fig.savefig(FIG / "fig_drift.svg"); plt.close(fig)

# Fig 2: style vs LC win rate
fig, ax = plt.subplots(figsize=(3.3, 2.6))
for c, col in COH.items():
    g_ = M[M.cohort == c].dropna(subset=["length_controlled_winrate"])
    ax.scatter(g_.proj_judge, g_.length_controlled_winrate, s=10, color=col, alpha=0.85, lw=0.3, edgecolor="white", label=c)
ax.set_xlabel("position along structure direction"); ax.set_ylabel("LC win rate (%)")
ax.legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "fig_lc.pdf"); fig.savefig(FIG / "fig_lc.svg"); plt.close(fig)

# Fig 3: MT-Bench weights
fig, ax = plt.subplots(figsize=(3.3, 3.0))
names = {"s_dlen": "length", "s_drep4": "repetition"}
xs = np.array([bh[k] for k in bh]); ys = np.array([bg[k] for k in bh])
lim = [-2.7, 1.9]
ax.plot(lim, lim, ls="--", color=MUT, lw=0.8, label="equal weights")
sc = bg["s_dlen"] / bh["s_dlen"]
ax.plot([lim[0], lim[1] / sc], [lim[0] * sc, lim[1]], color=C2, lw=1.2, label=f"GPT-4 = {sc:.2f} × human")
ax.scatter(xs, ys, s=16, color=C1, zorder=3, edgecolor="white", lw=0.4)
off = {"s_xₚ": (-14, 2), "s_dₛ": (6, -6), "s_θ": (5, -3), "s_Δy": (5, -3), "s_dlen": (-34, -2), "s_drep4": (5, -2)}
for k, (dx, dy) in off.items():
    ax.annotate(names.get(k, k.replace("s_", "")), (bh[k], bg[k]), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color="#52514e")
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel("human weight"); ax.set_ylabel("GPT-4 weight (same pairs, same scaling)")
ax.legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "fig_mtbench.pdf"); fig.savefig(FIG / "fig_mtbench.svg"); plt.close(fig)

# Fig 4: fixed-unit ratios
rows = [("MT-Bench, humans (2023)", AR["mtbench_human"], C1), ("MT-Bench, GPT-4 (same pairs)", AR["mtbench_gpt4"], C2),
        ("Arena, humans (mid-2024)", A, C1), ("Arena: neither model styled", D["neither_style_model"], C1),
        ("Arena: both models styled", D["both_style_models"], C1), ("AlpacaEval, GPT-4-turbo judge", AR["alpacaeval_judge"], C2)]
fig, ax = plt.subplots(figsize=(5.2, 2.2))
for i, (n, g_, col) in enumerate(rows[::-1]):
    r, c = g_["ratio_struct_len"], g_["ratio_struct_len_ci"]
    ax.plot(c, [i, i], color=col, lw=1.6); ax.scatter([r], [i], color=col, s=22, zorder=3, edgecolor="white", lw=0.5)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([n for n, _, _ in rows[::-1]])
ax.axvline(0, color=MUT, lw=0.8); ax.set_xlabel("structure weight / length weight (fixed units)")
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(FIG / "fig_ratios.pdf"); fig.savefig(FIG / "fig_ratios.svg"); plt.close(fig)
# Fig 5: structure weight by rater, unstyled vs styled
JA2 = L("arena_judge/results/analysis.json")
raters = [("Arena voters", RV["neither"], RV["both"], C1),
          ("Claude Opus 5.5 judge", (JA2["neither"]["judge_beta"]["struct"], JA2["neither"]["judge_beta_ci"]["struct"]), (JA2["both"]["judge_beta"]["struct"], JA2["both"]["judge_beta_ci"]["struct"]), C2)]
for rm, n in (("armorm", "ArmoRM-Llama3-8B"), ("gpm", "GPM-Gemma-2B"), ("pairrm", "PairRM")):
    raters.append((n, (RA[rm]["neither"]["rm_beta"]["struct"], RA[rm]["neither"]["rm_beta_ci"]["struct"]), (RA[rm]["both"]["rm_beta"]["struct"], RA[rm]["both"]["rm_beta_ci"]["struct"]), C3))
fig, ax = plt.subplots(figsize=(5.4, 2.6))
for i, (n, a, b, col) in enumerate(raters[::-1]):
    for (v, c), dy, filled in ((a, 0.15, False), (b, -0.15, True)):
        ax.plot(c, [i + dy, i + dy], color=col, lw=1.6, alpha=1 if filled else 0.5)
        ax.scatter([v], [i + dy], s=24, zorder=3, color=col if filled else "white", edgecolor=col, lw=1.4)
ax.set_yticks(range(len(raters))); ax.set_yticklabels([r[0] for r in raters[::-1]])
ax.axvline(0, color=MUT, lw=0.8); ax.set_xlabel("structure weight (fixed units), beyond length and repetition"); ax.grid(axis="y", visible=False)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec="#52514e", label="neither model styled"),
                   Line2D([], [], marker="o", ls="", color="#52514e", label="both models styled")], frameon=False, fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(FIG / "fig_rm.pdf"); fig.savefig(FIG / "fig_rm.svg"); plt.close(fig)
print("numbers:", len(N), "figures:", sorted(p.name for p in FIG.iterdir()))
