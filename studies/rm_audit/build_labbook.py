#!/usr/bin/env python3
"""Builds LABBOOK.html from results/analysis.json, results/added_checks.json and the judge run."""
import html, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
R = json.loads((HERE / "results" / "analysis.json").read_text())
C = json.loads((HERE / "results" / "added_checks.json").read_text())
J = json.loads((ROOT / "arena_judge" / "results" / "analysis.json").read_text())
S = pd.read_csv(ROOT / "arena_judge" / "sample.csv")
e = html.escape; s3 = lambda x: f"{x:+.3f}"; ci = lambda c, n=3: f"[{c[0]:+.{n}f}, {c[1]:+.{n}f}]"
RMS = [("armorm", "ArmoRM-Llama3-8B"), ("gpm", "GPM-Gemma-2B"), ("pairrm", "PairRM")]
rng = np.random.default_rng(0)
COLS = ["dlen", "drep4", "struct"]
fit = lambda X: LogisticRegression(C=1e6, max_iter=5000).fit(X[COLS].values, X["y"].values).coef_[0][2]
voters = {}
for c in ["neither", "both"]:
    X = S[S.cohort == c].reset_index(drop=True); u = X.cluster.unique(); m = {k: np.where(X.cluster.values == k)[0] for k in u}
    bs = [fit(X.iloc[np.concatenate([m[k] for k in rng.choice(u, len(u))])]) for _ in range(400)]
    voters[c] = (fit(X), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
(HERE / "results" / "voters_full_cohort.json").write_text(json.dumps(voters))

rows = [("Arena voters", voters["neither"], voters["both"], "--s1")]
rows.append(("Claude Opus 5.5 judge", (J["neither"]["judge_beta"]["struct"], J["neither"]["judge_beta_ci"]["struct"]),
             (J["both"]["judge_beta"]["struct"], J["both"]["judge_beta_ci"]["struct"]), "--s2"))
for k, n in RMS:
    rows.append((n, (R[k]["neither"]["rm_beta"]["struct"], R[k]["neither"]["rm_beta_ci"]["struct"]),
                 (R[k]["both"]["rm_beta"]["struct"], R[k]["both"]["rm_beta_ci"]["struct"]), "--s3"))


def chart():
    W, rowh, L, Rm, T0 = 720, 46, 190, 20, 26
    H = T0 + rowh * len(rows) + 30
    lo, hi = -0.15, 0.55
    x = lambda v: L + (v - lo) / (hi - lo) * (W - L - Rm)
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Structure weight by rater, unstyled vs styled battles">']
    for t in [-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        o.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{T0-10}" y2="{H-26}" class="{"axis0" if abs(t) < 1e-9 else "grid"}"/><text x="{x(t):.1f}" y="{H-10}" class="tick" text-anchor="middle">{t:+.1f}</text>')
    for i, (n, a, b, var) in enumerate(rows):
        y = T0 + i * rowh + 10
        o.append(f'<text x="{L-10}" y="{y+8}" class="lab" text-anchor="end">{e(n)}</text>')
        for (v, c), dy, shape in ((a, 0, "neither"), (b, 14, "both")):
            yy = y + dy
            o.append(f'<line x1="{x(c[0]):.1f}" x2="{x(c[1]):.1f}" y1="{yy}" y2="{yy}" stroke="var({var})" stroke-width="2" opacity="{0.45 if shape=="neither" else 1}"/>')
            mark = (f'<circle cx="{x(v):.1f}" cy="{yy}" r="5" fill="var(--surface)" stroke="var({var})" stroke-width="2">' if shape == "neither"
                    else f'<circle cx="{x(v):.1f}" cy="{yy}" r="5" fill="var({var})" stroke="var(--surface)" stroke-width="1.5">')
            o.append(mark + f'<title>{e(n)}, {shape} styled: {v:+.3f} {ci(c)}</title></circle>')
    o.append("</svg>")
    return ('<div class="legend"><span class="key"><i class="o"></i>neither model styled</span><span class="key"><i class="f"></i>both models styled</span></div>' + "".join(o))


def rm_rows():
    out = ""
    for k, n in RMS:
        for c in ["neither", "both"]:
            r = R[k][c]
            out += (f"<tr><td>{n}</td><td>{c}</td><td>{r['n']:,}</td><td>{s3(r['rm_beta']['struct'])} {ci(r['rm_beta_ci']['struct'])}</td>"
                    f"<td>{s3(r['human_beta']['struct'])}</td><td>{r['struct_ratio_rm']:+.2f} / {r['struct_ratio_human']:+.2f} {ci(r['struct_ratio_diff_ci'],2)}</td>"
                    f"<td>{r['rep_ratio_rm']:+.1f} / {r['rep_ratio_human']:+.1f} {ci(r['rep_ratio_diff_ci'],1)}</td><td>{r['agreement']:.0%}</td></tr>")
    return out


fp = "".join(f"<tr><td>{e(m)}</td><td>{v['own']}</td>" + "".join(f"<td>{v['cosines'].get(k, float('nan')):.2f}</td>" for k, _ in RMS) + f"<td>{'yes' if v['own_is_highest'] else 'no'}</td></tr>" for m, v in R["fingerprint"].items())
cos = C["C1_cosines"]; cn = list(cos)
cos_rows = "".join(f"<tr><td>{a}</td>" + "".join(f"<td>{cos[a][b]:.2f}</td>" for b in cn) + "</tr>" for a in cn)
preds = "".join(f"<tr><td>{e(k)}</td><td>{'Supported' if v else 'Not supported'}</td></tr>" for k, v in R["predictions"].items())
c2 = "".join(f"<tr><td>{n}</td><td>{C['C2'][k]['battles_with_student']}</td><td>{s3(C['C2'][k]['without_students']['beta_struct'])} {ci(C['C2'][k]['without_students']['ci'])}</td>"
             f"<td>{s3(C['C3'][k]['both']['beta_struct'])} {ci(C['C3'][k]['both']['ci'])}</td><td>{s3(C['C3'][k]['human_both_nocode']['beta_struct'])} {ci(C['C3'][k]['human_both_nocode']['ci'])}</td></tr>" for k, n in RMS)
log = (HERE / "results" / "armorm_rerun.out").read_text().strip().splitlines()[-2:]

page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reward Model Audit</title>
<style>
:root{{--surface:#fcfcfb;--bg:#f6f5f2;--ink:#0b0b0b;--ink2:#52514e;--muted:#7a7974;--line:#e4e2dc;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])){{--surface:#1a1a19;--bg:#121211;--ink:#f0efec;--ink2:#c3c2b7;--muted:#9a998f;--line:#33322f;--s1:#3987e5;--s2:#d95926;--s3:#199e70}}}}
:root[data-theme="dark"]{{--surface:#1a1a19;--bg:#121211;--ink:#f0efec;--ink2:#c3c2b7;--muted:#9a998f;--line:#33322f;--s1:#3987e5;--s2:#d95926;--s3:#199e70}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,system-ui,"Segoe UI",sans-serif}}
main{{max-width:880px;margin:0 auto;padding:28px 16px 80px}} h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:19px;margin:36px 0 8px;padding-top:12px;border-top:1px solid var(--line)}}
.sub,.muted{{color:var(--ink2)}} .card{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0;display:block;overflow-x:auto}} th,td{{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}} th{{color:var(--ink2)}}
.chart{{width:100%;height:auto;background:var(--surface);border:1px solid var(--line);border-radius:10px;margin:6px 0}}
.chart .grid{{stroke:var(--line)}} .chart .axis0{{stroke:var(--muted)}} .chart .tick{{fill:var(--muted);font-size:11px}} .chart .lab{{fill:var(--ink2);font-size:12px}}
.legend{{display:flex;gap:16px;font-size:13px;color:var(--ink2);margin-top:8px}} .key i{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px;border:2px solid var(--ink2)}} .key i.f{{background:var(--ink2)}}
pre{{white-space:pre-wrap;font-size:12px;background:var(--surface);padding:10px;border-radius:6px}}
</style></head><body><main>
<h1>Reward-model audit on Arena 2024 battles</h1>
<p class="sub">Lab book · 2026-11-11 · frozen v19 through reading frame r1.1 · same 5,794 battles as the judge run · predictions frozen in <code>PLAN.md</code> before any reward model ran (two amendments, both recorded before analysis)</p>

<div class="card"><b>Findings</b><ul>
<li><b>Reward models keep rewarding structure where voters stop.</b> Between two answers already in the 2024 style, voters give the structure index nothing ({s3(voters['both'][0])}); ArmoRM gives {s3(R['armorm']['both']['rm_beta']['struct'])}, GPM {s3(R['gpm']['both']['rm_beta']['struct'])}, PairRM {s3(R['pairrm']['both']['rm_beta']['struct'])}, all with intervals above zero. Their weight is as large or larger there than between unstyled answers. The Claude judge sits in between ({s3(J['both']['judge_beta']['struct'])}).</li>
<li><b>They penalise repetition far more than voters</b>, at least ten times more relative to length (10 to 47 times across models and cohorts; intervals wide, all excluding zero).</li>
<li><b>Not self-preference, not code.</b> Dropping battles involving ArmoRM's own student model, or all code battles, leaves the result in place.</li>
<li><b>No trainer fingerprint.</b> Llama-3-8B models tuned on each reward model's preferences moved from their base along a shared direction; ArmoRM's direction (cosine 0.94 with the AlpacaEval judge, 0.89 with HelpSteer2 humans) matched every shift best. The trainer-specific prediction (R6) is not supported.</li>
<li><b>They agree with voters less than the judge does</b>: {R['armorm']['neither']['agreement']:.0%}/{R['gpm']['neither']['agreement']:.0%}/{R['pairrm']['neither']['agreement']:.0%} between unstyled answers (ArmoRM/GPM/PairRM), against the judge's {J['neither']['agreement_judge_human']:.0%}.</li>
</ul></div>

<h2>1 · Setup</h2>
<p>Three open reward models, each used to produce preference data for a styled model on AlpacaEval (model cards): ArmoRM-Llama3-8B-v0.1 (gemma-2-9b-it-SimPO, gemma-2-9b-it-WPO-HB, REBEL-Llama-3-8B-Instruct-Armo), GPM-Gemma-2B (SPPO/GPO-Llama-3-8B-Instruct-GPM-2B), PairRM (SPPO-*-PairRM). Run as published on one rented L40S (Russell's Runpod account), libraries pinned (transformers 4.44.2), inputs up to 4,096 tokens. ArmoRM's first run stopped at battle 393 because token-level truncation removed the chat marker its gating needs; it was rerun from scratch with text-level shortening ({e(log[0])}). GPM and PairRM scored both answer orders.</p>

<h2>2 · Frozen predictions</h2>
<table><tr><th>prediction</th><th>outcome</th></tr>{preds}</table>

<h2>3 · Structure weight by rater</h2>
{chart()}
<p class="muted">Logistic model win ~ length + repetition + structure index (fixed units), per cohort; 95% bootstrap over prompts. Voters fitted on all 2,897 battles per cohort; reward models on their non-tied battles.</p>
<table><tr><th>reward model</th><th>cohort</th><th>n</th><th>β struct [CI]</th><th>voters β (same battles)</th><th>struct/len: RM / voters [diff CI]</th><th>rep/len: RM / voters [diff CI]</th><th>agreement</th></tr>{rm_rows()}</table>

<h2>4 · Added checks</h2>
<table><tr><th>reward model</th><th>battles with own student (styled cohort)</th><th>β struct without them</th><th>β struct, styled, no code</th><th>voters, styled, no code</th></tr>{c2}</table>
<h3>Direction cosines (8 axes)</h3>
<table><tr><th></th>{''.join(f'<th>{c}</th>' for c in cn)}</tr>{cos_rows}</table>

<h2>5 · Fingerprint test (R6)</h2>
<p class="muted">Shift of each tuned model's AlpacaEval style vector from Meta-Llama-3-8B-Instruct, cosine with each reward model's 8-axis direction.</p>
<table><tr><th>tuned model</th><th>trained with</th>{''.join(f'<th>{n}</th>' for _, n in RMS)}<th>own highest</th></tr>{fp}</table>

<h2>6 · Reproduce</h2>
<pre>pod/run_all.sh          # on a GPU pod: scores all battles with the three reward models
pod/score_rm.py armorm  # fixed version (text-level shortening)
python3 analyze.py      # M1-M5 -> results/analysis.json
python3 added_checks.py # C1-C3 -> results/added_checks.json
python3 build_labbook.py</pre>
</main></body></html>"""
(HERE / "LABBOOK.html").write_text(page, encoding="utf-8")
print("wrote", len(page), voters)
