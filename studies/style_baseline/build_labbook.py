#!/usr/bin/env python3
"""Builds LABBOOK.html from results.json."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
s3 = lambda x: f"{x:+.3f}"; ci = lambda c: f"[{c[0]:+.3f}, {c[1]:+.3f}]"
order = [("voters_all", "Arena voters", "all battles"), ("voters_neither", "Arena voters", "unstyled"), ("voters_both", "Arena voters", "styled"),
         ("judge_neither", "Claude judge", "unstyled"), ("judge_both", "Claude judge", "styled"),
         ("armorm_neither", "ArmoRM", "unstyled"), ("armorm_both", "ArmoRM", "styled"), ("gpm_neither", "GPM", "unstyled"), ("gpm_both", "GPM", "styled"),
         ("pairrm_neither", "PairRM", "unstyled"), ("pairrm_both", "PairRM", "styled"),
         ("armorm_all5794", "ArmoRM", "both cohorts"), ("gpm_all5794", "GPM", "both cohorts"), ("pairrm_all5794", "PairRM", "both cohorts")]
rows = "".join(f"<tr><td>{n}</td><td>{c}</td><td>{R[k]['n']:,}</td><td>{s3(R[k]['struct_beta_with_SC'])} {ci(R[k]['struct_ci_with_SC'])}</td>"
               f"<td>{R[k]['SC_to_SCK']['auc_base']:.3f} → {R[k]['SC_to_SCK']['auc_full']:.3f} {ci(R[k]['SC_to_SCK']['gain_ci'])}</td>"
               f"<td>{R[k]['K_to_KSC']['auc_base']:.3f} → {R[k]['K_to_KSC']['auc_full']:.3f} {ci(R[k]['K_to_KSC']['gain_ci'])}</td></tr>" for k, n, c in order)
preds = "".join(f"<tr><td>{k}</td><td>{'Supported' if v else 'Not supported'}</td></tr>" for k, v in R["predictions"].items())
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Style Control Baseline</title>
<style>:root{{--surface:#fcfcfb;--bg:#f6f5f2;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e2dc}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])){{--surface:#1a1a19;--bg:#121211;--ink:#f0efec;--ink2:#c3c2b7;--line:#33322f}}}}
:root[data-theme="dark"]{{--surface:#1a1a19;--bg:#121211;--ink:#f0efec;--ink2:#c3c2b7;--line:#33322f}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,system-ui,sans-serif}} main{{max-width:880px;margin:0 auto;padding:28px 16px 80px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:19px;margin:32px 0 8px;padding-top:12px;border-top:1px solid var(--line)}} .sub{{color:var(--ink2)}}
.card{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow-x:auto}} th,td{{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line)}} th{{color:var(--ink2)}}</style></head>
<body><main><h1>Style-control baseline</h1>
<p class="sub">Lab book · 2026-11-11 · no new data or compute: refits on saved votes, judge verdicts and reward-model scores · plan frozen in <code>PLAN.md</code> before fitting</p>
<div class="card"><b>Findings</b><ul>
<li>For Arena voters, Arena's own style-control features (length, headers, bold, lists) carry most of the structure signal: the kernel's remaining weight is {s3(R['voters_all']['struct_beta_with_SC'])} on all battles and not detectable within either cohort.</li>
<li>For the Claude judge and the three reward models the kernel captures what the markdown counts miss, most of all between styled answers (judge {s3(R['judge_both']['struct_beta_with_SC'])}; reward models {s3(R['gpm_both']['struct_beta_with_SC'])} to {s3(R['armorm_both']['struct_beta_with_SC'])}).</li>
<li>PairRM barely responds to Arena's features (AUC {R['pairrm_neither']['SC_to_SCK']['auc_base']:.3f} and {R['pairrm_both']['SC_to_SCK']['auc_base']:.3f}); the kernel explains most of what it rewards.</li>
<li>The voters-versus-reward-models gap persists between styled answers and opens between unstyled ones once standard style control is applied.</li></ul></div>
<h2>Frozen predictions</h2><table><tr><th>prediction</th><th>outcome</th></tr>{preds}</table>
<h2>Results</h2><table><tr><th>rater</th><th>cohort</th><th>n</th><th>β struct with Arena style features [CI]</th><th>AUC: style → style + kernel [gain CI]</th><th>AUC: kernel → kernel + style [gain CI]</th></tr>{rows}</table>
<p class="sub">Arena style features: normalised differences (a − b)/(a + b) of answer tokens, headers, bold and list items, from each battle's <code>conv_metadata</code>. Kernel: log-length difference, repetition difference, fixed-unit structure index. AUC by 5-fold cross-validation grouped by prompt; CIs by bootstrap over prompts.</p>
<h2>Reproduce</h2><pre>python3 analyze.py        # results.json
python3 build_labbook.py  # this page</pre></main></body></html>"""
(HERE / "LABBOOK.html").write_text(page, encoding="utf-8"); print("wrote", len(page))
