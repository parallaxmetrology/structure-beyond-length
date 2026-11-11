#!/usr/bin/env python3
"""Builds LABBOOK.html from data/results.json, data/judge_direction.json and data/judge_direction_models.csv."""
import html, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
D = HERE / "data"
R = json.loads((D / "results.json").read_text())
JD = json.loads((D / "judge_direction.json").read_text())
M = pd.read_csv(D / "judge_direction_models.csv", index_col=0)
DIMS = ["Δx", "rᵥ", "ρᵣ", "μ", "Δy", "xₚ", "θ", "dₛ"]
COH = ["2023-H2", "2024-H1", "2024-H2"]
e = html.escape
f2 = lambda x: f"{x:.2f}"; f3 = lambda x: f"{x:.3f}"; s3 = lambda x: f"{x:+.3f}"
ci = lambda c, n=3: f"[{c[0]:+.{n}f}, {c[1]:+.{n}f}]"
S1, S4, S2, S3, J1, J2, J3 = R["S1"], R["S4"], R["S2"], R["S3"], R["J1"], R["J2"], R["J3"]
CVAR = {"2023-H2": "--s1", "2024-H1": "--s2", "2024-H2": "--s3"}


def scatter():
    W, H, L, B, T0, Rm = 720, 380, 56, 40, 16, 16
    g = M.dropna(subset=["length_controlled_winrate", "proj_judge"])
    xlo, xhi = np.floor(g.proj_judge.min()), np.ceil(g.proj_judge.max())
    x = lambda v: L + (v - xlo) / (xhi - xlo) * (W - L - Rm)
    y = lambda v: H - B - v / 80 * (H - B - T0)
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Judge-direction score vs LC win rate per model">']
    for t in range(0, 81, 20):
        o.append(f'<line x1="{L}" x2="{W-Rm}" y1="{y(t):.1f}" y2="{y(t):.1f}" class="grid"/><text x="{L-8}" y="{y(t)+4:.1f}" class="tick" text-anchor="end">{t}</text>')
    for t in np.arange(xlo, xhi + 0.01, 1):
        o.append(f'<text x="{x(t):.1f}" y="{H-B+16}" class="tick" text-anchor="middle">{t:+.0f}</text>')
    o.append(f'<text x="{(L+W-Rm)/2}" y="{H-6}" class="lab" text-anchor="middle">position along the judge\'s preferred structural direction (model style, prompt-relative)</text>')
    o.append(f'<text x="14" y="{(H-B)/2}" class="lab" transform="rotate(-90 14 {(H-B)/2})" text-anchor="middle">LC win rate %</text>')
    for c in COH:
        for name, r in g[g.cohort == c].iterrows():
            o.append(f'<circle cx="{x(r.proj_judge):.1f}" cy="{y(r.length_controlled_winrate):.1f}" r="4.5" fill="var({CVAR[c]})" fill-opacity="0.85" stroke="var(--surface)" stroke-width="1.5">'
                     f'<title>{e(name)} · {c} · score {r.proj_judge:+.2f} · LC {r.length_controlled_winrate:.1f}% · bold titles {r.bold:.0%}</title></circle>')
    o.append("</svg>")
    leg = "".join(f'<span class="key"><i style="background:var({CVAR[c]})"></i>{c} (n={(g.cohort==c).sum()})</span>' for c in COH)
    return f'<div class="legend">{leg}</div>' + "".join(o)


def coef_chart():
    feats = ["dlen", "drep4"] + DIMS
    names = {"dlen": "length", "drep4": "repetition"}
    W, rowh, L, Rm, T0 = 720, 30, 110, 20, 20
    H = T0 + rowh * len(feats) + 24
    lo, hi = -0.5, 1.0
    x = lambda v: L + (v - lo) / (hi - lo) * (W - L - Rm)
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Judge vs human standardised weights">']
    for t in [-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0]:
        o.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{T0-6}" y2="{H-22}" class="{"axis0" if t == 0 else "grid"}"/><text x="{x(t):.1f}" y="{H-6}" class="tick" text-anchor="middle">{t:+.2f}</text>')
    for i, f in enumerate(feats):
        yy = T0 + i * rowh + 10
        o.append(f'<text x="{L-10}" y="{yy+4}" class="lab" text-anchor="end">{names.get(f, f)}</text>')
        for who, var, dy in (("judge", "--s2", -5), ("human", "--s1", 5)):
            v, c = J2[f][who], J2[f][who + "_ci"]
            o.append(f'<line x1="{x(c[0]):.1f}" x2="{x(c[1]):.1f}" y1="{yy+dy}" y2="{yy+dy}" stroke="var({var})" stroke-width="2"/>'
                     f'<circle cx="{x(v):.1f}" cy="{yy+dy}" r="4.5" fill="var({var})" stroke="var(--surface)" stroke-width="1.5"><title>{names.get(f,f)} · {who}: {v:+.3f} {ci(c)}</title></circle>')
    o.append("</svg>")
    leg = '<span class="key"><i style="background:var(--s2)"></i>GPT-4-turbo judge (AlpacaEval)</span><span class="key"><i style="background:var(--s1)"></i>Humans (HelpSteer2 helpfulness)</span>'
    return f'<div class="legend">{leg}</div>' + "".join(o)


def drift_rows():
    lab = {"tokens": "median prose tokens", "rv": "rᵥ (raw, mean)", "rep4": "repetition (4-gram)", "headers": "share of answers with # headers", "bold": "share of answers with bold title lines"}
    out = ""
    for k in ["tokens", "rv", "rep4", "headers", "bold"] + [f"z_{d}" for d in DIMS]:
        name = lab.get(k, k.replace("z_", "") + " (prompt-relative z)")
        v = S3[k]
        out += f"<tr><td>{e(name)}</td>" + "".join(f"<td>{v[c]:.3f}</td>" for c in COH) + f"<td>{ci(v['diff_ci'])}</td></tr>"
    return out


def spread_rows(S):
    return "".join(f"<tr><td>{c}</td><td>{f3(S[c]['mean'])}</td><td>{ci(S[c]['ci'])}</td></tr>" for c in COH)


preds = [
    ("S-a", "Spread between models shrinks, 2023-H2 → 2024-H2", f"difference {s3(S1['diff_2024H2_minus_2023H2']['mean'])} {ci(S1['diff_2024H2_minus_2023H2']['ci'])}", "Not supported: spread grows"),
    ("S-b", "Convergence survives length partialling", f"length-partialled difference {s3(S4['diff_2024H2_minus_2023H2']['mean'])} {ci(S4['diff_2024H2_minus_2023H2']['ci'])}", "Not applicable (no convergence); the growth in spread survives"),
    ("S-c", "Answers get longer and more segmented", f"median tokens {S3['tokens']['2023-H2']:.0f} → {S3['tokens']['2024-H2']:.0f}; rᵥ {S3['rv']['2023-H2']:.3f} → {S3['rv']['2024-H2']:.3f}", "Supported"),
    ("J-a", "Length dominates the judge", f"length-only AUC {f3(J1['len'])}", "Supported"),
    ("J-b", "Kernel adds held-out signal beyond length", f"AUC {f3(J1['len'])} → {f3(J1['len+kernel'])}, gain CI {ci(J1['gain_ci']['kernel_over_len'])}", "Supported"),
    ("J-c", "Judge rewards rᵥ, ρᵣ more than humans", f"rᵥ judge {s3(J2['rᵥ']['judge'])} vs human {s3(J2['rᵥ']['human'])}; ρᵣ {s3(J2['ρᵣ']['judge'])} vs {s3(J2['ρᵣ']['human'])}", "Holds as stated; not interpretable as a judge-specific bias (same direction as humans, cos 0.92; sizes not comparable across datasets)"),
    ("J-d", "Judge penalises repetition", f"judge {s3(J2['drep4']['judge'])} {ci(J2['drep4']['judge_ci'])}; human {s3(J2['drep4']['human'])}", "Supported; same size as humans"),
    ("J-e", "Style explains LC win rate beyond length", f"leave-one-out R² {f3(J3['loo_r2_len'])} → {f3(J3['loo_r2_len_style'])} (n={J3['n_models']})", "Supported"),
]
pred_html = "".join(f"<tr><td>{a}</td><td>{e(b)}</td><td class='num'>{e(c)}</td><td>{e(d)}</td></tr>" for a, b, c, d in preds)
h2 = JD["H2"]; sp = JD["H3_spread"]; split = JD["H3_split"]
h3rows = "".join(f"<tr><td>{e(r['folder'])}</td><td>{e(str(r['mode']))}</td><td>{r['proj_judge']:+.2f}</td><td>{r['bold']:.0%}</td><td>{r['tokens']:.0f}</td><td>{r['length_controlled_winrate']:.1f}</td></tr>" for r in JD["H3_2024H2"])
axis_lc = "".join(f"<tr><td>{d}</td><td>{J3['spearman_lc_by_axis'][d]:+.2f}</td></tr>" for d in DIMS)

page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AlpacaEval Lab Book</title>
<style>
:root{{--surface:#fcfcfb;--bg:#f6f5f2;--ink:#0b0b0b;--ink2:#52514e;--muted:#7a7974;--line:#e4e2dc;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])){{--surface:#1a1a19;--bg:#121211;--ink:#f0efec;--ink2:#c3c2b7;--muted:#9a998f;--line:#33322f;--s1:#3987e5;--s2:#d95926;--s3:#199e70}}}}
:root[data-theme="dark"]{{--surface:#1a1a19;--bg:#121211;--ink:#f0efec;--ink2:#c3c2b7;--muted:#9a998f;--line:#33322f;--s1:#3987e5;--s2:#d95926;--s3:#199e70}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,system-ui,"Segoe UI",sans-serif}}
main{{max-width:880px;margin:0 auto;padding:28px 16px 80px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:19px;margin:36px 0 8px;padding-top:12px;border-top:1px solid var(--line)}} h3{{font-size:15px;margin:20px 0 6px}}
.sub,.muted{{color:var(--ink2)}} .card{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0;display:block;overflow-x:auto}} th,td{{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{color:var(--ink2);font-weight:600}} td.num{{font-size:12px;font-variant-numeric:tabular-nums}}
.chart{{width:100%;height:auto;background:var(--surface);border:1px solid var(--line);border-radius:10px;margin:6px 0}}
.chart .grid{{stroke:var(--line)}} .chart .axis0{{stroke:var(--muted)}} .chart .tick{{fill:var(--muted);font-size:11px}} .chart .lab{{fill:var(--ink2);font-size:12px}}
.legend{{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;color:var(--ink2);margin-top:8px}} .key i{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}}
details summary{{cursor:pointer;color:var(--ink2);font-size:13px}} pre{{white-space:pre-wrap;font-size:12px;background:var(--surface);padding:10px;border-radius:6px}}
</style></head><body><main>
<h1>AlpacaEval vs the v19 kernel</h1>
<p class="sub">Lab book · 2026-11-11 · frozen v19 (sha256 1f63ee85…) through reading frame r1 · predictions frozen in <code>PLAN.md</code> before the first reading · observational, no new human data</p>

<div class="card"><b>Findings</b><ul>
<li><b>Model answers changed style between 2023 and late 2024.</b> Median length {S3['tokens']['2023-H2']:.0f} → {S3['tokens']['2024-H2']:.0f} tokens; answers with bold title lines {S3['bold']['2023-H2']:.0%} → {S3['bold']['2024-H2']:.0%} (median model); denser (ρᵣ), rarer vocabulary (xₚ), more segmented (rᵥ), less repetition, less connective cohesion (μ).</li>
<li><b>Models did not converge; they split.</b> Spread between models grew from {f2(S1['2023-H2']['mean'])} to {f2(S1['2024-H2']['mean'])}, also with length removed. In 2024-H2 the split partly follows the base model: every Gemma-2 and Llama-3.1/3.2 derivative (and Meta's own Llama-3.1-Instruct) uses bold title lines and sits far along the judge's preferred direction; every Infinity-Instruct, Qwen2 and Mistral entry stays plain; Llama-3.0 derivatives appear on both sides (spread {f2(sp['2024-H2_plain'])} vs {f2(sp['2024-H2_bold'])}).</li>
<li><b>The judge rewards the same structure humans do, in the same direction.</b> Judge and human weights on the 8 axes have cosine {JD['cos_judge_human']:.2f}. Both penalise repetition equally ({s3(J2['drep4']['judge'])} vs {s3(J2['drep4']['human'])}). Within a prompt, the kernel adds {s3(J1['len+kernel']-J1['len'])} AUC to length for predicting the judge.</li>
<li><b>The length-controlled leaderboard still carries a structural style.</b> Across {J3['n_models']} models, style explains LC win rate well beyond length (leave-one-out R² {f2(J3['loo_r2_len'])} → {f2(J3['loo_r2_len_style'])}); a model's position along the judge's preferred direction has Spearman {h2['all']['spearman_lc_judge_proj']:+.2f} with LC win rate. These data cannot separate style from quality.</li>
</ul></div>

<h2>1 · Data</h2>
<p>{R['n_models']} models (NullModel excluded: it games the judge) × 805 instructions, {R['n_answers']:,} answers, {R['eligible_share']:.1%} eligible after the reading frame. Style study: {R['models_in_style_study']} models with ≥ 400 eligible answers and a submission date. Cohorts by the date the model's answers were committed to the AlpacaEval repo (not release date): {", ".join(f"{c} {R['cohort_sizes'][c]}" for c in COH)}. 2024-H2 is mostly community entries (30 of 36).</p>
<p>Judge: {R['J_n']:,} verdicts on eligible pairs across {R['J_models']} models; each verdict compares the model's answer with the fixed reference <code>gpt4_1106_preview</code>. The model's answer won {R['J_model_win_share']:.1%} of them.</p>

<h2>2 · Frozen predictions and outcomes</h2>
<table><tr><th>#</th><th>Prediction</th><th>Result</th><th>Outcome</th></tr>{pred_html}</table>

<h2>3 · Style over time</h2>
<h3>Spread between models (S1; S4 with length partialled)</h3>
<p class="muted">Mean pairwise distance between model style vectors, cohorts resampled to {R['k_subsample']} models.</p>
<table><tr><th>cohort</th><th>S1 spread</th><th>95% CI</th></tr>{spread_rows(S1)}</table>
<table><tr><th>cohort</th><th>S4 spread, length removed</th><th>95% CI</th></tr>{spread_rows(S4)}</table>
<h3>Spread within a prompt (S2)</h3>
<p class="muted">Standard deviation across a cohort's answers to the same instruction, averaged over axes and instructions. Answers to one prompt barely tighten.</p>
<table><tr><th>cohort</th><th>per-prompt spread</th><th>95% CI</th></tr>{spread_rows(S2)}</table>
<h3>Drift (S3, medians over models; CI for 2024-H2 − 2023-H2)</h3>
<table><tr><th>measure</th><th>2023-H2</th><th>2024-H1</th><th>2024-H2</th><th>diff CI</th></tr>{drift_rows()}</table>

<h3>Added analysis H: the 2024-H2 split</h3>
<p>Reading the 2024-H2 cohort model by model showed two groups; the 20% threshold was chosen after looking. The bold group is FuseChat, Gemma-2 (SimPO, WPO, DPO, SPPO, mixture-of-agents) and Llama-3.1/3.2 derivatives plus Meta's Llama-3.1-Instruct; the plain group is the Infinity-Instruct series, Qwen2, Mistral, gpt-4o-mini, higgs-llama-3 and Llama-3.0 preference-tuned models (WPO-HB-v2, PBM-Nova, blendaxai-l3), several of which also score high. Llama-3.0 derivatives also appear in the bold group (GPO, SPPO, REBEL, RainbowPO, SkillMix). So the split partly tracks the base model each entry was built on; for Llama-3.0 the fine-tune decides. Models with bold title lines in ≥ 20% of answers: n={split['bold_ge_0.2']['n']}, median LC win rate {split['bold_ge_0.2']['median_lc']:.1f}%. The rest: n={split['bold_lt_0.2']['n']}, median {split['bold_lt_0.2']['median_lc']:.1f}%. Spread within the plain group ({f2(sp['2024-H2_plain'])}) is lower than 2023-H2 overall ({f2(sp['2023-H2_all'])}); no interval, post hoc.</p>
<details><summary>2024-H2 models, sorted by judge-direction score</summary><table><tr><th>model</th><th>mode</th><th>score</th><th>bold titles</th><th>tokens</th><th>LC %</th></tr>{h3rows}</table></details>

<h2>4 · The judge</h2>
<h3>J1: predicting the verdict within a prompt</h3>
<p class="muted">Logistic model on differences from the reference answer; cross-validation holds out whole models.</p>
<table><tr><th>features</th><th>AUC</th></tr>
<tr><td>length</td><td>{f3(J1['len'])}</td></tr><tr><td>length + kernel</td><td>{f3(J1['len+kernel'])}</td></tr>
<tr><td>length + repetition</td><td>{f3(J1['len+rep'])}</td></tr><tr><td>length + repetition + kernel</td><td>{f3(J1['len+rep+kernel'])}</td></tr></table>
<p>Kernel gain over length {ci(J1['gain_ci']['kernel_over_len'])}; over length + repetition {ci(J1['gain_ci']['kernel_over_len_rep'])} (bootstrap over models). For HelpSteer2 humans, the gain over length + repetition was about +0.013.</p>
<h3>J2: judge weights vs human weights</h3>
{coef_chart()}
<p>Same features, standardised within each dataset. The two patterns point the same way (cosine {JD['cos_judge_human']:.2f}). The judge's weights are larger, but AlpacaEval pairs (any model vs one strong reference) span a much wider range than HelpSteer2 pairs, and the human labels are 5-point ratings, so sizes are not comparable across the two datasets. A size comparison needs the same pairs judged by both.</p>
<h3>J3 and H: style and the length-controlled leaderboard</h3>
{scatter()}
<p>Position along the judge's preferred direction vs LC win rate, Spearman: {", ".join(f"{c} {h2[c]['spearman_lc_judge_proj']:+.2f}" for c in COH)}; all {h2['all']['spearman_lc_judge_proj']:+.2f}. Along the human direction: all {h2['all']['spearman_lc_human_proj']:+.2f}. Drift from 2023-H2 to 2024-H2 points along both (cosine with judge direction {JD['cos_drift_judge']:.2f}, human {JD['cos_drift_human']:.2f}).</p>
<details><summary>Per-axis Spearman with LC win rate (model level)</summary><table><tr><th>axis (prompt-relative)</th><th>ρ</th></tr>{axis_lc}</table></details>

<h2>5 · What this does and does not show</h2>
<ul>
<li>Observational throughout. Cohort dates are commit dates; the 2024-H2 cohort is mostly community submissions to this leaderboard.</li>
<li>The structure the judge rewards is the structure humans reward, by direction. Whether the judge over-rewards it cannot be settled with these two datasets.</li>
<li>Style and quality move together at model level; these data cannot pull them apart.</li>
<li>Many of the 2024-H2 leaders inherit a structural signature from their base models (Gemma-2, Llama-3.1): bold titles, density, rare vocabulary. The length control does not remove it. Base model and leaderboard success are entangled here.</li>
</ul>

<h2>6 · Reproduce</h2>
<pre>cd studies/alpacaeval
python3 fetch_verdicts.py   # judge verdicts → data/verdicts.json (~1 GB streamed)
python3 read_all.py         # 183,528 readings → data/readings.csv.gz (~70 s, 8 processes)
python3 analyze.py          # S1-S4, J1-J3 → data/results.json
python3 judge_direction.py  # added analysis H → data/judge_direction.json
python3 build_labbook.py    # this page</pre>
<p class="muted">Commit dates: <code>git log --diff-filter=A</code> on <code>results/*/model_outputs.json</code> in the AlpacaEval repo → data/added_dates.json.</p>
</main></body></html>"""
(HERE / "LABBOOK.html").write_text(page, encoding="utf-8")
print("wrote", len(page))
