#!/usr/bin/env python3
"""Builds LABBOOK.html from data/*.json. Numbers come from these files, except a few counts computed during the session (single-segment n, fenced n, residual means), which are written in directly."""
import html, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE / "data"
R = json.loads((D / "results.json").read_text())
P = json.loads((D / "pairwise.json").read_text())
RB = json.loads((D / "pairwise_robustness.json").read_text())
T = json.loads((D / "theta_check.json").read_text())
EX = json.loads((D / "theta_examples.json").read_text())
DIMS = ["Δx", "rᵥ", "ρᵣ", "μ", "Δy", "xₚ", "θ", "dₛ"]
SC = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
e = html.escape
f3 = lambda x: f"{x:.3f}"
s3 = lambda x: f"{x:+.3f}"
cis = lambda c: f"[{c[0]:+.3f}, {c[1]:+.3f}]"
excl = lambda c: c[0] > 0 or c[1] < 0


def cell(key, grid, cik):
    c = grid[key]
    return c["partial"], c[cik]


# ---------- chart 1: pairwise AUC dot rows ----------
def auc_chart():
    series = [("Length only", "--s1", lambda s: P[s]["auc_len"]),
              ("Length + kernel", "--s2", lambda s: P[s]["auc_len_framed"]),
              ("Length + repetition", "--s3", lambda s: RB[s]["with_rep4"]["auc_len_rep4"]),
              ("Length + repetition + kernel", "--s4", lambda s: RB[s]["with_rep4"]["auc_len_rep4_framed"])]
    W, H, L, Rm, T0, rowh = 720, 40 + 52 * len(SC), 120, 24, 30, 52
    lo, hi = 0.45, 1.0
    x = lambda v: L + (v - lo) / (hi - lo) * (W - L - Rm)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Pairwise AUC by score and feature set" class="chart">']
    for t in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{T0-8}" y2="{H-14}" class="grid"/>'
                   f'<text x="{x(t):.1f}" y="{H-2}" class="tick" text-anchor="middle">{t:.1f}</text>')
    out.append(f'<text x="{x(0.5):.1f}" y="{T0-14}" class="tick" text-anchor="middle">chance</text>')
    for i, s in enumerate(SC):
        y0 = T0 + i * rowh + 14
        out.append(f'<text x="{L-10}" y="{y0+12}" class="lab" text-anchor="end">{s}</text>')
        out.append(f'<line x1="{L}" x2="{W-Rm}" y1="{y0+8}" y2="{y0+8}" class="row"/>')
        for j, (name, var, fn) in enumerate(series):
            v = fn(s); yy = y0 + 2 + j * 4
            out.append(f'<circle cx="{x(v):.1f}" cy="{yy}" r="5" fill="var({var})" stroke="var(--surface)" stroke-width="2">'
                       f'<title>{s} · {name}: AUC {v:.3f}</title></circle>')
    out.append("</svg>")
    leg = "".join(f'<span class="key"><i style="background:var({v})"></i>{n}</span>' for n, v, _ in series)
    rows = "".join(f"<tr><td>{s}</td>" + "".join(f"<td>{f3(fn(s))}</td>" for _, _, fn in series) + "</tr>" for s in SC)
    tbl = ("<details><summary>Table view</summary><table><tr><th>score</th>" + "".join(f"<th>{n}</th>" for n, _, _ in series)
           + "</tr>" + rows + "</table></details>")
    return f'<div class="legend">{leg}</div>' + "".join(out) + tbl


# ---------- chart 2: within-pair heatmap ----------
def heat():
    W, cw, ch, L, T0 = 720, 116, 30, 60, 26
    H = T0 + ch * len(DIMS) + 6
    vmax = 0.10
    def col(v):
        t = max(-1, min(1, v / vmax)); a = abs(t)
        pole = "var(--pos)" if t > 0 else "var(--neg)"
        return pole, a
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Within-pair partial correlations" class="chart">']
    for j, s in enumerate(SC):
        out.append(f'<text x="{L+j*cw+cw/2}" y="{T0-8}" class="lab" text-anchor="middle">{s}</text>')
    for i, d in enumerate(DIMS):
        y = T0 + i * ch
        out.append(f'<text x="{L-10}" y="{y+ch/2+5}" class="lab" text-anchor="end">{d}</text>')
        for j, s in enumerate(SC):
            v, c = cell(f"{d}|{s}", R["B_pairs"], "ci")
            pole, a = col(v)
            out.append(f'<rect x="{L+j*cw+1}" y="{y+1}" width="{cw-2}" height="{ch-2}" rx="3" fill="var(--mid)"/>'
                       f'<rect x="{L+j*cw+1}" y="{y+1}" width="{cw-2}" height="{ch-2}" rx="3" fill="{pole}" fill-opacity="{a*0.85:.2f}">'
                       f'<title>{d} × {s}: {v:+.3f}, 95% CI {cis(c)}</title></rect>'
                       f'<text x="{L+j*cw+cw/2}" y="{y+ch/2+5}" class="val" text-anchor="middle">{v:+.3f}{"" if excl(c) else " ·"}</text>')
    out.append("</svg>")
    return "".join(out)


def pred_rows():
    A, B = R["A_framed"], R["B_pairs"]
    def both(k):
        a, b = A[k], B[k]
        return (f"pooled {s3(a['partial'])} {cis(a['partial_ci'])}; within pair {s3(b['partial'])} {cis(b['ci'])}")
    C = R["C"]; Ec = R["E_collapse"]
    rows = [
        ("P1", "dₛ rises with complexity", both("dₛ|complexity"), "Mixed: opposite sign pooled, supported within pair"),
        ("P2", "xₚ rises with complexity", both("xₚ|complexity"), "Supported"),
        ("P3", "μ rises with coherence", both("μ|coherence"), "Supported pooled (small); not within pair"),
        ("P4", "θ falls with coherence", both("θ|coherence"), "Not supported: θ rises with coherence"),
        ("P5", "verbosity is mostly length", f"kernel gain {s3(C['verbosity']['gain_framed'])} {cis(C['verbosity']['gain_framed_ci'])}", "Supported"),
        ("P6", "kernel adds held-out signal for helpfulness (C)", f"gain {s3(C['helpfulness']['gain_framed'])} {cis(C['helpfulness']['gain_framed_ci'])}; pooled model's pair choice {f3(C['helpfulness']['pair_acc_len'])} → {f3(C['helpfulness']['pair_acc_framed'])}", "Supported as written; the pooled gain does not reach pair choice (see F)"),
        ("P7", "cut responses score lower (helpfulness)",
         "; ".join(f"{g} n={Ec[g+'|helpfulness']['n']} {s3(Ec[g+'|helpfulness']['diff'])} {cis(Ec[g+'|helpfulness']['ci'])}" for g in ["short_cut", "cut_partial", "cut_code_or_math"]),
         "Supported for short_cut and cut_code_or_math; cut_partial CI crosses 0"),
        ("P8", "framed tracks scores at least as well as raw", f"helpfulness gain framed {s3(C['helpfulness']['gain_framed'])} vs raw {s3(C['helpfulness']['gain_raw'])}; pair AUC framed {f3(P['helpfulness']['auc_len_framed'])} vs raw {f3(P['helpfulness']['auc_len_raw'])}", "No difference (HelpSteer2 has little markdown)"),
    ]
    return "".join(f"<tr><td>{a}</td><td>{e(b)}</td><td class='num'>{e(c)}</td><td>{e(d)}</td></tr>" for a, b, c, d in rows)


def theta_rows():
    out = ""
    for r in T["theta_deciles"]:
        out += (f"<tr><td>{int(r['theta_dec'])+1}</td><td>{r['theta']:.2f}</td><td>{r['rep4']:.3f}</td>"
                f"<td>{r['coherence']:.2f}</td><td>{r['helpfulness']:.2f}</td><td>{int(r['tokens'])}</td></tr>")
    return out


def examples():
    out = ""
    for x in EX[:4]:
        lo, hi = x["low_theta"], x["high_theta"]
        out += (f"<div class='ex'><p class='muted'>Prompt (tail): {e(x['prompt'][-200:])}</p>"
                f"<div class='pair'><div><b>θ {lo['theta']:.2f} · coherence {lo['coherence']} · repetition {lo['rep4']:.2f}</b><pre>{e(lo['text'][:420])}</pre></div>"
                f"<div><b>θ {hi['theta']:.2f} · coherence {hi['coherence']} · repetition {hi['rep4']:.2f}</b><pre>{e(hi['text'][:420])}</pre></div></div></div>")
    return out


Cr = R["C"]
crows = "".join(f"<tr><td>{s}</td><td>{f3(Cr[s]['len_only'])}</td><td>{f3(Cr[s]['len_plus_framed'])}</td><td>{s3(Cr[s]['gain_framed'])} {cis(Cr[s]['gain_framed_ci'])}</td>"
                f"<td>{f3(Cr[s]['pair_acc_len'])} → {f3(Cr[s]['pair_acc_framed'])}</td></tr>" for s in SC)
prows = "".join(f"<tr><td>{s}</td><td>{P[s]['pairs']}</td><td>{f3(P[s]['auc_len'])}</td><td>{f3(P[s]['auc_len_framed'])}</td><td>{cis(P[s]['auc_gain_ci'])}</td>"
                f"<td>{f3(P[s]['auc_kernel_only'])}</td><td>{f3(RB[s]['multi_segment_only']['auc_len'])} → {f3(RB[s]['multi_segment_only']['auc_len_framed'])}</td>"
                f"<td>{f3(RB[s]['with_rep4']['auc_len_rep4'])} → {f3(RB[s]['with_rep4']['auc_len_rep4_framed'])} {cis(RB[s]['with_rep4']['auc_gain_ci'])}</td></tr>" for s in SC)
trows = "".join(f"<tr><td>{s}</td><td>{s3(T[s]['theta_given_len'])} {cis(T[s]['theta_given_len_ci'])}</td><td>{s3(T[s]['theta_given_len_rep4'])} {cis(T[s]['theta_given_len_rep4_ci'])}</td><td>{s3(T[s]['rep4_given_len'])} {cis(T[s]['rep4_given_len_ci'])}</td></tr>" for s in ["helpfulness", "correctness", "coherence"])
inel = ", ".join(f"{k} {v:,}" for k, v in R["ineligible_reasons"].items())
cc = R["collapse_counts"]
h, co = P["helpfulness"], P["coherence"]

page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HelpSteer2 Lab Book</title>
<style>
:root{{--surface:#fcfcfb;--bg:#f6f5f2;--ink:#0b0b0b;--ink2:#52514e;--muted:#7a7974;--line:#e4e2dc;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#6250d6;--pos:#2a78d6;--neg:#e34948;--mid:#f0efec}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])){{--surface:#1a1a19;--bg:#121211;--ink:#f0efec;--ink2:#c3c2b7;--muted:#9a998f;--line:#33322f;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#9085e9;--pos:#3987e5;--neg:#e66767;--mid:#383835}}}}
:root[data-theme="dark"]{{--surface:#1a1a19;--bg:#121211;--ink:#f0efec;--ink2:#c3c2b7;--muted:#9a998f;--line:#33322f;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#9085e9;--pos:#3987e5;--neg:#e66767;--mid:#383835}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,system-ui,"Segoe UI",sans-serif}}
main{{max-width:880px;margin:0 auto;padding:28px 16px 80px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:19px;margin:36px 0 8px;padding-top:12px;border-top:1px solid var(--line)}} h3{{font-size:15px;margin:20px 0 6px}}
.muted,.sub{{color:var(--ink2)}} .card{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0;display:block;overflow-x:auto}} th,td{{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{color:var(--ink2);font-weight:600}} td.num{{font-variant-numeric:tabular-nums;font-size:12px}}
.chart{{width:100%;height:auto;background:var(--surface);border:1px solid var(--line);border-radius:10px;margin:6px 0}}
.chart .grid{{stroke:var(--line)}} .chart .row{{stroke:var(--line);stroke-dasharray:2 4}} .chart .tick{{fill:var(--muted);font-size:11px}} .chart .lab{{fill:var(--ink2);font-size:12px}} .chart .val{{fill:var(--ink);font-size:11px}}
.legend{{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;color:var(--ink2);margin-top:8px}} .key i{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}}
pre{{white-space:pre-wrap;font-size:12px;background:var(--bg);padding:8px;border-radius:6px;max-height:180px;overflow:auto}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:10px}} @media(max-width:640px){{.pair{{grid-template-columns:1fr}}}}
.ex{{border-top:1px dashed var(--line);padding-top:8px;margin-top:8px}} details summary{{cursor:pointer;color:var(--ink2);font-size:13px}}
code{{font-size:13px}}
</style></head><body><main>
<h1>HelpSteer2 vs the v19 kernel</h1>
<p class="sub">Lab book · 2026-11-11 · frozen v19 (sha256 1f63ee85…) through reading frame r1 · predictions frozen in <code>PLAN.md</code> before the first reading</p>

<div class="card"><b>Findings</b>
<ul>
<li><b>Comparing two answers to the same prompt, the kernel helps pick the one annotators rated higher, beyond length.</b> Helpfulness AUC {f3(h['auc_len'])} (length only) → {f3(h['auc_len_framed'])} (length + kernel). Coherence {f3(co['auc_len'])} → {f3(co['auc_len_framed'])}: length says nothing about coherence, the kernel does.</li>
<li><b>Much of that signal is repetition.</b> θ (1 − word overlap between neighbouring sentences) reads low when an answer repeats itself. A direct count of repeated 4-word phrases does better than the kernel on coherence ({f3(RB['coherence']['with_rep4']['auc_len_rep4'])}). With length and repetition both in, the kernel still adds a small amount for helpfulness, correctness and coherence (CIs above 0).</li>
<li><b>Pooling across prompts mixes two things.</b> A model fitted across all answers gains held-out signal (helpfulness {s3(Cr['helpfulness']['gain_framed'])}), but it does not pick the better answer within a pair any better than length. dₛ against complexity flips sign: {s3(R['A_framed']['dₛ|complexity']['partial'])} pooled, {s3(R['B_pairs']['dₛ|complexity']['partial'])} within pair.</li>
<li><b>Verbosity is length</b> (pair AUC {f3(P['verbosity']['auc_len'])} from length alone; the kernel adds nothing).</li>
<li><b>Two v19 operating-window notes.</b> θ = 0 means "one segment", not "perfectly smooth": 262 eligible answers (1.4%) were read as a single sentence (lists without bullet markers, numbered lists on one line). And θ works as an indirect repetition reader.</li>
</ul></div>

<h2>1 · Question and data</h2>
<p>Do v19's axes line up with what human annotators scored, once length is accounted for? HelpSteer2 (NVIDIA, CC-BY-4.0): {R['n_all']:,} responses, two per prompt, five human scores 0-4. The kernel was frozen before it saw this data; annotators never saw the kernel. Length is the shared confound and is controlled in every analysis.</p>
<p>Eligible after the reading frame: {R['n_eligible']:,} responses across {R['n_prompts_eligible']:,} prompts; {R['B_n_pairs']:,} prompts have both answers eligible. Excluded (reasons can overlap): {e(inel)}.</p>
<p class="muted">Score distributions known before reading: coherence sits at the ceiling (15,298 of 21,362 score 4); complexity is mostly 1-2.</p>

<h2>2 · Frozen predictions and outcomes</h2>
<p class="muted">Correlations are Spearman partials with log length removed; 95% CIs by bootstrap over prompts (pooled) or pairs. "Supported" = sign matches and CI excludes 0.</p>
<table><tr><th>#</th><th>Prediction</th><th>Result</th><th>Outcome</th></tr>{pred_rows()}</table>

<h2>3 · Trail</h2>
<h3>Step 1 · Read everything (A, C, D)</h3>
<p>All {R['n_all']:,} responses read raw and framed in 32 s. Length alone correlates with the scores at: {", ".join(f"{s} {R['length_spearman'][s]:+.2f}" for s in SC)}.</p>
<table><tr><th>score</th><th>held-out ρ, length</th><th>length + kernel</th><th>gain [95% CI]</th><th>pair choice, pooled model</th></tr>{crows}</table>
<p>The pooled gain is real but does not show up when choosing between the two answers to one prompt. That points to the kernel separating <i>kinds of prompt</i> (a code question vs a chat reply), which also differ in typical scores.</p>

<h3>Step 2 · Hold the prompt fixed (B, and added analysis F)</h3>
<p>Within each pair: difference on each axis vs difference in each score, length difference removed. Blue = positive, red = negative; "·" marks a CI that crosses 0.</p>
{heat()}
<p>Added analysis F (not in the frozen plan): a logistic model trained on pair differences, 5-fold CV, pairs with tied scores dropped.</p>
{auc_chart()}
<table><tr><th>score</th><th>pairs</th><th>AUC length</th><th>+ kernel</th><th>gain CI</th><th>kernel alone</th><th>multi-sentence pairs only</th><th>length + repetition → + kernel [gain CI]</th></tr>{prows}</table>

<h3>Step 3 · What θ carries (added analysis G)</h3>
<p>θ had the largest within-pair weight for helpfulness and coherence, in the direction opposite to P4. θ is 1 − mean word cosine between adjacent sentences, so low θ means neighbouring sentences reuse words. Second bearing: share of repeated 4-word phrases, counted outside the kernel. Same text, different measurement, so the two are not fully independent.</p>
<p>θ and repetition: Spearman {T['pooled_spearman_theta_rep4']:+.2f} pooled, {T['pair_spearman_dtheta_drep4']:+.2f} within pair.</p>
<table><tr><th>score</th><th>θ, length removed</th><th>θ, length + repetition removed</th><th>repetition, length removed</th></tr>{trows}</table>
<p>Repetition accounts for all of θ's link to coherence and about half of its link to helpfulness.</p>
<h3>θ deciles (pooled)</h3>
<table><tr><th>decile</th><th>median θ</th><th>median repetition</th><th>mean coherence</th><th>mean helpfulness</th><th>median tokens</th></tr>{theta_rows()}</table>
<p>The lowest decile is the repetitive one. The highest decile is short answers and also scores lower, so the relation is not a straight line.</p>

<h3>Step 4 · Reading the pairs</h3>
<p>Largest θ gaps where the lower-θ answer also scored lower on coherence, lengths within 20%. All four lower-θ answers read θ = 0.00, which is v19 reading the whole answer as one segment: an unmarked list, a numbered list run onto one line, an indented outline without bullet markers, and a feature list that is also repetitive (repetition 0.40). Sorting by the largest θ gap selects these first.</p>
{examples()}
<p>Single-segment answers: 262 of {R['n_eligible']:,} eligible (1.4%), all θ = 0, median Δy 5.6 vs 0.54. Annotators also scored them lower (helpfulness 2.17 vs 2.93). Dropping pairs that contain one leaves the result unchanged (table above, "multi-sentence pairs only").</p>

<h3>Step 5 · Cohorts and residuals</h3>
<p>Collapse readings over all {R['n_all']:,} responses: {", ".join(f"{k} {v:,}" for k, v in cc.items())}. Mean helpfulness gap vs <code>none</code>:</p>
<table><tr><th>reading</th><th>n</th><th>helpfulness</th><th>correctness</th><th>coherence</th></tr>
{"".join(f"<tr><td>{g}</td><td>{R['E_collapse'][g+'|helpfulness']['n']}</td>" + "".join(f"<td>{s3(R['E_collapse'][g+'|'+s]['diff'])} {cis(R['E_collapse'][g+'|'+s]['ci'])}</td>" for s in ['helpfulness','correctness','coherence']) + "</tr>" for g in ['short_complete','short_cut','cut_partial','cut_code_or_math'])}</table>
<p>Largest residuals of the pooled helpfulness model. Rated 0 but predicted about 3.2: answers that read well and are wrong (wrong calendar dates, synonym-spun text). Rated 4 but predicted about 2: short correct answers (median 47 tokens; 35% are multi-turn replies). Both are outside what a structure reader sees.</p>

<h2>4 · Operating windows for v19</h2>
<ul>
<li><b>θ = 0 is a segmentation state.</b> v19 returns θ = 0 for any text read as one sentence. Lists without bullet markers and numbered lists on a single line land there, with very high Δy.</li>
<li><b>θ is an indirect repetition reader.</b> It tracks repeated phrasing (ρ = {T['pooled_spearman_theta_rep4']:+.2f} with a 4-gram count); a direct count reads repetition better.</li>
<li><b>Pooled vs within-prompt readings differ.</b> dₛ and complexity change sign. Cross-prompt studies need prompt-relative reading (v19's prompt-family baselines) before axis claims.</li>
<li><b>Raw vs framed made no difference here</b>; only 240 of {R['n_all']:,} responses have code fences.</li>
</ul>

<h2>5 · Reproduce</h2>
<pre>cd studies/helpsteer2
python3 read_all.py        # 21,362 readings → data/readings.csv (~30 s)
python3 analyze.py         # A-E → data/results.json
python3 pairwise.py        # F + robustness → data/pairwise*.json
python3 theta_check.py     # G → data/theta_check.json, theta_examples.json
python3 build_labbook.py   # this page</pre>
</main></body></html>"""
(HERE / "LABBOOK.html").write_text(page, encoding="utf-8")
print("wrote", HERE / "LABBOOK.html", len(page))
