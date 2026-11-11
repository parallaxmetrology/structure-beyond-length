#!/usr/bin/env python3
"""Builds LABBOOK.html from data/results.json and data/readings.csv."""
import html, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "data" / "results.json").read_text())
A = pd.read_csv(HERE / "data" / "readings.csv"); E = A[A.eligible]
DIMS = ["Δx", "rᵥ", "ρᵣ", "μ", "Δy", "xₚ", "θ", "dₛ"]
e = html.escape
f3 = lambda x: f"{x:.3f}"; s3 = lambda x: f"{x:+.3f}"
ci = lambda c: f"[{c[0]:+.3f}, {c[1]:+.3f}]"
Ma, Mr, Md, Me, Mc, Mb = R["Ma_struct"], R["Ma_rep"], R["Md"], R["Me"], R["Mc"], R["Mb"]
bh, bg = Mb["human"], Mb["gpt4"]
feats = list(bh.keys())
name = lambda k: {"s_dlen": "length", "s_drep4": "repetition"}.get(k, k.replace("s_", ""))
scale = bg["s_dlen"] / bh["s_dlen"]
bold_share = float((E.edit_bold_lines > 0).mean())


def beta_scatter():
    W, H, L, B, T0, Rm = 560, 420, 60, 46, 16, 20
    lo, hi = -2.6, 1.9
    x = lambda v: L + (v - lo) / (hi - lo) * (W - L - Rm)
    y = lambda v: H - B - (v - lo) / (hi - lo) * (H - B - T0)
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="GPT-4 weights vs human weights per feature">']
    for t in np.arange(-2.5, 2.0, 0.5):
        o.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{T0}" y2="{H-B}" class="{"axis0" if t == 0 else "grid"}"/><text x="{x(t):.1f}" y="{H-B+16}" class="tick" text-anchor="middle">{t:+.1f}</text>')
        o.append(f'<line x1="{L}" x2="{W-Rm}" y1="{y(t):.1f}" y2="{y(t):.1f}" class="{"axis0" if t == 0 else "grid"}"/><text x="{L-8}" y="{y(t)+4:.1f}" class="tick" text-anchor="end">{t:+.1f}</text>')
    o.append(f'<line x1="{x(lo):.1f}" y1="{y(lo):.1f}" x2="{x(hi):.1f}" y2="{y(hi):.1f}" class="ref"/>')
    xa, xb = lo, hi / scale
    o.append(f'<line x1="{x(xa):.1f}" y1="{y(xa*scale):.1f}" x2="{x(xb):.1f}" y2="{y(xb*scale):.1f}" class="scale"/>')
    for k in feats:
        o.append(f'<circle cx="{x(bh[k]):.1f}" cy="{y(bg[k]):.1f}" r="5" fill="var(--s1)" stroke="var(--surface)" stroke-width="1.5"><title>{e(name(k))}: human {bh[k]:+.3f}, GPT-4 {bg[k]:+.3f}</title></circle>')
        if k in ("s_dlen", "s_drep4", "s_xₚ", "s_dₛ", "s_θ", "s_Δy"):
            dy = {"s_xₚ": -6, "s_dₛ": 14}.get(k, 4)
            o.append(f'<text x="{x(bh[k])+8:.1f}" y="{y(bg[k])+dy:.1f}" class="lab">{e(name(k))}</text>')
    o.append(f'<text x="{(L+W-Rm)/2}" y="{H-8}" class="lab" text-anchor="middle">human weight (standardised)</text>')
    o.append(f'<text x="16" y="{(H-B)/2}" class="lab" transform="rotate(-90 16 {(H-B)/2})" text-anchor="middle">GPT-4 weight (same scaling)</text>')
    o.append("</svg>")
    return ('<div class="legend"><span class="key"><i class="ln" style="border-color:var(--muted)"></i>equal weights</span>'
            f'<span class="key"><i class="ln" style="border-color:var(--s2)"></i>GPT-4 = {scale:.2f} × human (the length ratio)</span></div>' + "".join(o))


preds = [
    ("P1", "Both judges' 8-axis weights point the same way (cosine > 0.5)", f"cosine {Mb['cos_8axes']:.2f} {ci(Mb['cos_8axes_ci'])}", "Supported at the point estimate; wide interval"),
    ("P2", "GPT-4 weights structure more than humans, relative to length", f"ratio human {s3(Ma['r_human'])} {ci(Ma['r_human_ci'])}; GPT-4 {s3(Ma['r_gpt4'])} {ci(Ma['r_gpt4_ci'])}; difference {s3(Ma['diff'])} {ci(Ma['diff_ci'])}", "Not supported: difference CI includes 0"),
    ("P3", "Repetition-to-length ratio similar", f"human {s3(Mr['r_human'])}, GPT-4 {s3(Mr['r_gpt4'])}, difference {ci(Mr['diff_ci'])}", "Supported"),
    ("P4", "Both reward length", f"β_length human {ci(Ma['beta_len_human_ci'])}; GPT-4 {ci(Ma['beta_len_gpt4_ci'])}", "Supported"),
    ("P5", "P2 holds under majority vote", f"difference {s3(Md['diff'])} {ci(Md['diff_ci'])} ({Md['pairs_majority']} pairs)", "Not supported"),
]
prow = "".join(f"<tr><td>{a}</td><td>{e(b)}</td><td class='num'>{e(c)}</td><td>{e(d)}</td></tr>" for a, b, c, d in preds)
axrows = "".join(f"<tr><td>{c}</td><td>{s3(R['Mb_per_axis'][c]['r_human'])}</td><td>{s3(R['Mb_per_axis'][c]['r_gpt4'])}</td><td>{ci(R['Mb_per_axis'][c]['diff_ci'])}</td></tr>" for c in DIMS)
brows = "".join(f"<tr><td>{e(name(k))}</td><td>{bh[k]:+.3f}</td><td>{bg[k]:+.3f}</td><td>{(bg[k]/bh[k]):.2f}</td></tr>" for k in feats)
mcrows = "".join(f"<tr><td>{k}</td><td>{f3(Mc['human'][k])}</td><td>{f3(Mc['gpt4'][k])}</td></tr>" for k in ["len", "len+rep", "len+rep+struct", "len+rep+kernel8"])
reasons = A.reasons.fillna("").str.split(";").explode().value_counts()
reasons = ", ".join(f"{k} {v}" for k, v in reasons.items() if k)

page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MT-Bench Lab Book</title>
<style>
:root{{--surface:#fcfcfb;--bg:#f6f5f2;--ink:#0b0b0b;--ink2:#52514e;--muted:#7a7974;--line:#e4e2dc;--s1:#2a78d6;--s2:#eb6834}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])){{--surface:#1a1a19;--bg:#121211;--ink:#f0efec;--ink2:#c3c2b7;--muted:#9a998f;--line:#33322f;--s1:#3987e5;--s2:#d95926}}}}
:root[data-theme="dark"]{{--surface:#1a1a19;--bg:#121211;--ink:#f0efec;--ink2:#c3c2b7;--muted:#9a998f;--line:#33322f;--s1:#3987e5;--s2:#d95926}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,system-ui,"Segoe UI",sans-serif}}
main{{max-width:880px;margin:0 auto;padding:28px 16px 80px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:19px;margin:36px 0 8px;padding-top:12px;border-top:1px solid var(--line)}}
.sub,.muted{{color:var(--ink2)}} .card{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0;display:block;overflow-x:auto}} th,td{{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{color:var(--ink2);font-weight:600}} td.num{{font-size:12px}}
.chart{{width:100%;max-width:560px;height:auto;background:var(--surface);border:1px solid var(--line);border-radius:10px;margin:6px 0;display:block}}
.chart .grid{{stroke:var(--line)}} .chart .axis0{{stroke:var(--muted)}} .chart .tick{{fill:var(--muted);font-size:11px}} .chart .lab{{fill:var(--ink2);font-size:12px}}
.chart .ref{{stroke:var(--muted);stroke-dasharray:4 4}} .chart .scale{{stroke:var(--s2);stroke-width:2}}
.legend{{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;color:var(--ink2);margin-top:8px}} .key .ln{{display:inline-block;width:18px;border-top:2px solid;margin-right:6px;vertical-align:4px}}
pre{{white-space:pre-wrap;font-size:12px;background:var(--surface);padding:10px;border-radius:6px}}
</style></head><body><main>
<h1>MT-Bench: humans vs GPT-4 on the same pairs</h1>
<p class="sub">Lab book · 2026-11-11 · frozen v19 (sha256 1f63ee85…) through reading frame r1 · predictions frozen in <code>PLAN.md</code> · structure index fixed from the AlpacaEval study</p>

<div class="card"><b>Findings</b><ul>
<li><b>On the same pairs, GPT-4 does not detectably weight structure more than humans do, relative to length.</b> Structure-to-length ratio: humans {s3(Ma['r_human'])}, GPT-4 {s3(Ma['r_gpt4'])}; difference {s3(Ma['diff'])}, 95% CI {ci(Ma['diff_ci'])}. Majority-vote check: {s3(Md['diff'])} {ci(Md['diff_ci'])}.</li>
<li><b>Mostly consistent with GPT-4 being a less noisy copy of the human preference.</b> Its weights are {bg['s_dlen']/bh['s_dlen']:.2f}× the human ones on length, {bg['s_drep4']/bh['s_drep4']:.2f}× on repetition and {bg['s_rᵥ']/bh['s_rᵥ']:.2f}× on segmentation (rᵥ); uniform scaling is what cleaner labels alone would produce. Rare vocabulary (xₚ, {bg['s_xₚ']/bh['s_xₚ']:.1f}×) and clause stacking (dₛ, {bg['s_dₛ']/bh['s_dₛ']:.1f}×) sit above that line, in the predicted direction, but their intervals include 0.</li>
<li><b>Repetition is the largest structural signal for both.</b> Adding it to length lifts held-out AUC from {f3(Mc['human']['len'])} to {f3(Mc['human']['len+rep'])} (humans) and {f3(Mc['gpt4']['len'])} to {f3(Mc['gpt4']['len+rep'])} (GPT-4). The kernel adds nothing held-out beyond that here.</li>
<li><b>Agreement reproduces the original paper's pattern:</b> human-human {Me['human_human_agreement']:.0%}, human-GPT-4 {Me['human_gpt4_agreement']:.0%} (non-tie votes).</li>
<li><b>Operating window.</b> These are 2023 models; none of the {len(E)} eligible answers has a bold title line ({bold_share:.0%}). The formatting style that appeared in 2024 cannot be tested on this dataset.</li>
</ul></div>

<h2>1 · Data</h2>
<p>lmsys/mt_bench_human_judgments (CC-BY-4.0). 80 questions × 2 turns × 6 models (alpaca-13b, claude-v1, gpt-3.5-turbo, gpt-4, llama-13b, vicuna-13b-v1.2). {R['answers_read']} distinct answers read, {R['answers_eligible']} eligible (excluded: {e(reasons)}). Short answers are common in MT-Bench (math, single-value replies), so about a third drop out.</p>
<p>Analysed: {R['shared_pairs']} pairs judged by both sides with a non-tie verdict and both answers eligible; {R['human_votes']} human votes ({R['human_judges']} judges in the full set) and {R['gpt4_verdicts']} GPT-4 verdicts on those pairs.</p>

<h2>2 · Frozen predictions and outcomes</h2>
<table><tr><th>#</th><th>Prediction</th><th>Result</th><th>Outcome</th></tr>{prow}</table>

<h2>3 · Weights on the same features, same scaling</h2>
{beta_scatter()}
<table><tr><th>feature</th><th>human β</th><th>GPT-4 β</th><th>GPT-4 / human</th></tr>{brows}</table>
<p class="muted">Each point is one feature. Under pure label noise, GPT-4's weights would sit on the orange line (all human weights scaled by the same factor). xₚ and dₛ sit above it; θ and Δy below. All four per-axis intervals below include 0.</p>
<h3>Per-axis ratio to length (8-axis model)</h3>
<table><tr><th>axis</th><th>human ratio</th><th>GPT-4 ratio</th><th>difference CI</th></tr>{axrows}</table>

<h2>4 · Held-out AUC (5 folds by question)</h2>
<table><tr><th>features</th><th>humans</th><th>GPT-4</th></tr>{mcrows}</table>

<h2>5 · How this fits the earlier studies</h2>
<ul>
<li>HelpSteer2: within a prompt, humans reward non-repetition and a little structure beyond length.</li>
<li>AlpacaEval: the GPT-4-turbo judge rewards the same direction, with larger weights; sizes were not comparable across datasets.</li>
<li>MT-Bench (same pairs): relative to length, GPT-4's structural weighting is not distinguishable from humans'. The larger AlpacaEval weights are consistent with wider pair differences and cleaner labels rather than a judge-specific taste. The 2024 formatting style is outside this dataset's window.</li>
</ul>

<h2>6 · Reproduce</h2>
<pre>cd studies/mtbench
python3 analyze.py        # reads 960 answers, runs M-a to M-e → data/results.json
python3 build_labbook.py  # this page</pre>
</main></body></html>"""
(HERE / "LABBOOK.html").write_text(page, encoding="utf-8")
print("wrote", len(page), "scale", round(scale, 3))
