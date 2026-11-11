#!/usr/bin/env python3
"""Builds LABBOOK.html from data/results.json and data/range_check.json."""
import html, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "data" / "results.json").read_text())
RC = json.loads((HERE / "data" / "range_check.json").read_text())
e = html.escape
f3 = lambda x: f"{x:.3f}"; s3 = lambda x: f"{x:+.3f}"
ci = lambda c: f"[{c[0]:+.3f}, {c[1]:+.3f}]"
A, B, BH, C, D, N = R["arena_A"], R["arena_B"], R["arena_B_head"], R["C"], R["arena_D"], R["arena_counts"]

groups = [("MT-Bench humans (2023)", R["mtbench_human"], "--s1"), ("MT-Bench GPT-4 (2023, same pairs)", R["mtbench_gpt4"], "--s2"),
          ("Arena humans (mid-2024)", A, "--s1"), ("Arena humans: neither model styled", D["neither_style_model"], "--s1"),
          ("Arena humans: both models styled", D["both_style_models"], "--s1"), ("AlpacaEval GPT-4-turbo judge", R["alpacaeval_judge"], "--s2")]


def ratio_chart():
    W, rowh, L, Rm, T0 = 720, 34, 250, 20, 16
    H = T0 + rowh * len(groups) + 30
    lo, hi = -0.2, 0.7
    x = lambda v: L + (v - lo) / (hi - lo) * (W - L - Rm)
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Structure-to-length ratio by judge and dataset">']
    for t in [-0.2, 0, 0.2, 0.4, 0.6]:
        o.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{T0-6}" y2="{H-26}" class="{"axis0" if t == 0 else "grid"}"/><text x="{x(t):.1f}" y="{H-10}" class="tick" text-anchor="middle">{t:+.1f}</text>')
    for i, (name, g, var) in enumerate(groups):
        yy = T0 + i * rowh + 12
        r, c = g["ratio_struct_len"], g["ratio_struct_len_ci"]
        o.append(f'<text x="{L-10}" y="{yy+4}" class="lab" text-anchor="end">{e(name)}</text>')
        o.append(f'<line x1="{x(max(c[0],lo)):.1f}" x2="{x(min(c[1],hi)):.1f}" y1="{yy}" y2="{yy}" stroke="var({var})" stroke-width="2"/>'
                 f'<circle cx="{x(r):.1f}" cy="{yy}" r="5" fill="var({var})" stroke="var(--surface)" stroke-width="1.5"><title>{e(name)}: {r:+.3f} {ci(c)}, n={g["n"]}</title></circle>')
    o.append("</svg>")
    leg = '<span class="key"><i style="background:var(--s1)"></i>human voters</span><span class="key"><i style="background:var(--s2)"></i>GPT-4 judge</span>'
    rows = "".join(f"<tr><td>{e(n)}</td><td>{g['n']:,}</td><td>{g['clusters']:,}</td><td>{s3(g['beta']['dlen'])}</td><td>{s3(g['beta']['struct'])} {ci(g['beta_ci']['struct'])}</td><td>{s3(g['ratio_struct_len'])} {ci(g['ratio_struct_len_ci'])}</td></tr>" for n, g, _ in groups)
    return (f'<div class="legend">{leg}</div>' + "".join(o) +
            "<details><summary>Table view</summary><table><tr><th>group</th><th>n</th><th>clusters</th><th>β length</th><th>β structure [CI]</th><th>ratio [CI]</th></tr>" + rows + "</table></details>")


def b(g, c):
    return f"{s3(g['beta'][c])} {ci(g['beta_ci'][c])}"

preds = [
    ("A1", "Arena humans reward length", b(A, "dlen"), "Supported"),
    ("A2", "Arena humans reward structure beyond length and repetition", b(A, "struct"), "Supported"),
    ("A3", "Arena humans penalise repetition", b(A, "drep4"), "Supported"),
    ("A4", "Bold title lines rewarded beyond length, repetition, structure", b(B, "dbold"), "Not supported (CI includes 0; point estimate negative)"),
    ("A5", "AlpacaEval judge ratio exceeds Arena humans'", f"{s3(C['alpacaeval_judge_minus_arena_human']['diff'])} {ci(C['alpacaeval_judge_minus_arena_human']['ci'])}", "Holds as stated; not attributable to judge vs human (same-pairs GPT-4 on MT-Bench matches humans)"),
]
prow = "".join(f"<tr><td>{a}</td><td>{e(b_)}</td><td class='num'>{e(c)}</td><td>{e(d)}</td></tr>" for a, b_, c, d in preds)
coh = "".join(f"<tr><td>{k.replace('_',' ')}</td><td>{g['n']:,}</td><td>{s3(g['beta']['dlen'])}</td><td>{b(g,'struct')}</td><td>{s3(g['ratio_struct_len'])} {ci(g['ratio_struct_len_ci'])}</td></tr>" for k, g in D.items())
styled = ", ".join(R["style_models"])
share = R["model_bold_share"]
share_rows = "".join(f"<tr><td>{e(m)}</td><td>{v:.0%}</td></tr>" for m, v in list(share.items())[:24])

page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Arena 100k Lab Book</title>
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
.chart{{width:100%;height:auto;background:var(--surface);border:1px solid var(--line);border-radius:10px;margin:6px 0}}
.chart .grid{{stroke:var(--line)}} .chart .axis0{{stroke:var(--muted)}} .chart .tick{{fill:var(--muted);font-size:11px}} .chart .lab{{fill:var(--ink2);font-size:12px}}
.legend{{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;color:var(--ink2);margin-top:8px}} .key i{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}}
details summary{{cursor:pointer;color:var(--ink2);font-size:13px}} pre{{white-space:pre-wrap;font-size:12px;background:var(--surface);padding:10px;border-radius:6px}}
</style></head><body><main>
<h1>Arena 100k: do humans reward the 2024 structural style?</h1>
<p class="sub">Lab book · 2026-11-11 · frozen v19 (sha256 1f63ee85…) through reading frame r1.1 · predictions frozen in <code>PLAN.md</code> · structure index and units fixed from the AlpacaEval study</p>

<div class="card"><b>Findings</b><ul>
<li><b>Mid-2024 human voters reward the structure, beyond length and repetition</b> (β {b(A,'struct')}), at a structure-to-length ratio of {s3(A['ratio_struct_len'])}, not distinguishable from MT-Bench humans in 2023 ({s3(R['mtbench_human']['ratio_struct_len'])}).</li>
<li><b>The reward disappears between two styled models.</b> When neither model writes in the 2024 style, the structure weight is {s3(D['neither_style_model']['beta']['struct'])}; when both do, {s3(D['both_style_models']['beta']['struct'])} {ci(D['both_style_models']['beta_ci']['struct'])}. This is not a narrower range: the structure difference is wider between styled models (SD {RC['sd_struct_both']:.2f} vs {RC['sd_struct_neither']:.2f}), and unstyled battles resampled to the same spread still give {s3(RC['neither_matched_struct_beta'])} {ci(RC['neither_matched_struct_beta_ci'])}.</li>
<li><b>Bold title lines are not rewarded in themselves</b> ({b(B,'dbold')}). Markdown headers are (added check: {b(BH,'dhead')}).</li>
<li><b>Judge vs human, in fixed units.</b> The AlpacaEval judge's ratio ({s3(R['alpacaeval_judge']['ratio_struct_len'])}) is above Arena humans' (difference {ci(C['alpacaeval_judge_minus_arena_human']['ci'])}). On MT-Bench's same pairs, GPT-4 ({s3(R['mtbench_gpt4']['ratio_struct_len'])}) matches humans (difference {ci(C['mtbench_gpt4_minus_mtbench_human']['ci'])}). The gap therefore tracks the dataset or judge version (GPT-4-turbo on AlpacaEval's 2023-24 models vs GPT-4 on MT-Bench's 2023 models), not judges vs humans as such.</li>
</ul></div>

<h2>1 · Data</h2>
<p>106,134 battles (June-August 2024). Filtered to English, single-turn, with a winner: {N['battles_filtered']:,}; both answers eligible: {N['battles_both_eligible']:,}. Among eligible answers, {N['answers_with_bold_title']:.0%} have a bold title line and {N['answers_with_headers']:.0%} a markdown header. Six answers could not be read by v19 (literal special tokens such as <code>&lt;|endoftext|&gt;</code>; reading frame r1.1 marks them).</p>
<p>"Styled" models: ≥ 20% of their eligible answers carry a bold title line (threshold carried over from the AlpacaEval study): {e(styled)}.</p>

<h2>2 · Frozen predictions and outcomes</h2>
<table><tr><th>#</th><th>Prediction</th><th>Result</th><th>Outcome</th></tr>{prow}</table>

<h2>3 · Structure-to-length ratio across judges and datasets (fixed units)</h2>
<p class="muted">Logistic model win ~ length + repetition + structure index in each dataset; ratio = β structure / β length. Ratios cancel the uniform shrinkage that noisy labels cause. CIs: bootstrap over prompts (Arena, MT-Bench) or models (AlpacaEval).</p>
{ratio_chart()}

<h2>4 · Formatting</h2>
<table><tr><th>model</th><th>β length</th><th>β repetition</th><th>β structure</th><th>β bold title</th><th>β header</th></tr>
<tr><td>A (planned)</td><td>{b(A,'dlen')}</td><td>{b(A,'drep4')}</td><td>{b(A,'struct')}</td><td></td><td></td></tr>
<tr><td>B (planned)</td><td>{b(B,'dlen')}</td><td>{b(B,'drep4')}</td><td>{b(B,'struct')}</td><td>{b(B,'dbold')}</td><td></td></tr>
<tr><td>B + headers (added)</td><td>{b(BH,'dlen')}</td><td>{b(BH,'drep4')}</td><td>{b(BH,'struct')}</td><td>{b(BH,'dbold')}</td><td>{b(BH,'dhead')}</td></tr></table>

<h2>5 · Cohorts (D) and the range check (added)</h2>
<table><tr><th>cohort</th><th>n</th><th>β length</th><th>β structure [CI]</th><th>ratio [CI]</th></tr>{coh}</table>
<p>Range check: SD of the structure difference {RC['sd_struct_both']:.2f} (both styled) vs {RC['sd_struct_neither']:.2f} (neither). Resampling unstyled battles to the styled cohort's |structure difference| distribution: β {s3(RC['neither_matched_struct_beta'])} {ci(RC['neither_matched_struct_beta_ci'])}.</p>
<details><summary>Share of eligible answers with a bold title line, by model (top 24)</summary><table><tr><th>model</th><th>share</th></tr>{share_rows}</table></details>

<h2>6 · Across the four studies</h2>
<ul>
<li><b>Humans reward the structure and penalise repetition</b> in all three human datasets (HelpSteer2 within pairs, MT-Bench, Arena). In fixed units the structure-to-length ratio is {s3(R['mtbench_human']['ratio_struct_len'])} (MT-Bench) and {s3(A['ratio_struct_len'])} (Arena).</li>
<li><b>GPT-4 on the same pairs matches humans</b> (MT-Bench). The higher AlpacaEval judge ratio comes with a different judge version and a 2024 model population.</li>
<li><b>The 2024 style is rewarded by humans up to a point.</b> Between two unstyled models, more structure wins votes; between two styled models, it gains nothing. Beyond length, repetition and structure, bold title lines carry no reward; headers do.</li>
</ul>

<h2>7 · Reproduce</h2>
<pre>cd studies/arena100k
python3 read_all.py       # 59,358 answers → data/answers.csv.gz (~30 s)
python3 analyze.py        # A-D plus AlpacaEval and MT-Bench refits → data/results.json
python3 range_check.py    # added range check → data/range_check.json (uses analyze_core.py)
python3 build_labbook.py  # this page</pre>
</main></body></html>"""
(HERE / "LABBOOK.html").write_text(page, encoding="utf-8")
print("wrote", len(page))
