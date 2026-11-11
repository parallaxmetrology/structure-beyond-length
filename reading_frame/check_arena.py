#!/usr/bin/env python3
"""Before/after check of reading frame r1 on Arena answers.

Reads the first turn of response_a and response_b for 1,500 battles (seed 0),
measures each raw and framed with frozen v19, writes a CSV of flat rows and a
markdown report to reports/.

Usage: python3 check_arena.py [path/to/arena55k/train.csv]
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reading_frame as RF

CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "public_corpora" / "arena55k" / "train.csv"
OUT = HERE / "reports"


def md_table(ct):
    cols = [str(c) for c in ct.columns]
    lines = ["| " + " | ".join([str(ct.index.name or "")] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in ct.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row.values]) + " |")
    return "\n".join(lines)

OUT.mkdir(exist_ok=True)

df = pd.read_csv(CSV).sample(1500, random_state=0)
texts = []
for _, r in df.iterrows():
    for col in ("response_a", "response_b"):
        try:
            parts = json.loads(r[col])
        except Exception:
            continue
        if parts and isinstance(parts[0], str) and parts[0].strip():
            texts.append(parts[0])

reader = RF.Reader()
rows = [RF.flat_row(reader.read(t)) for t in texts]
m = pd.DataFrame(rows)
m.to_csv(OUT / "arena_check_r1.csv", index=False)

L = []
p = L.append
p(f"# Reading frame r1: Arena check\n")
p(f"{len(m)} answers (first turn, 1,500 battles, seed 0). Kernel sha256 `{reader.kernel_sha256[:16]}…` (frozen v19).\n")

p("## Who is eligible\n")
p(f"- eligible: {m.eligible.sum()} ({m.eligible.mean():.1%})")
reasons = m.reasons[m.reasons != ""].str.split(";").explode().value_counts()
for k, v in reasons.items():
    p(f"- {k}: {v} ({v/len(m):.1%})")
p("")

p("## Collapse: v19 label vs frame reading\n")
ct = pd.crosstab(m.raw_collapse_subtype, m.collapse_reading)
p(md_table(ct) + "\n")
tail_closed = (m.raw_collapse_subtype.eq("severe") & m.ends_closed).sum()
p(f"Severe-short answers that end in closing punctuation: {tail_closed} of {m.raw_collapse_subtype.eq('severe').sum()}.\n")

E = m[m.eligible].copy()
md = ~E.framed_identical
p("## Markdown among eligible answers\n")
p(f"- framed text identical to raw (no markdown the frame touches): {E.framed_identical.sum()} ({E.framed_identical.mean():.1%})")
for k in ("code_blocks", "table_lines", "headers", "bold_lines", "links"):
    p(f"- has {k.replace('_', ' ')}: {(E[f'edit_{k}'] > 0).sum()}")
p("")

p("## Does framing close the gap? (medians, eligible answers)\n")
p("Plain = no markdown edits. Each group shows raw → framed. A frame that works moves the markdown groups toward Plain.\n")
groups = {"Plain": E.framed_identical}
for k, name in (("code_blocks", "Code"), ("headers", "Headers"), ("bold_lines", "Bold lines"), ("table_lines", "Tables")):
    groups[name] = E[f"edit_{k}"] > 0
hdr = "| group | n | " + " | ".join(RF.DIMS) + " |"
p(hdr)
p("|" + "---|" * (len(RF.DIMS) + 2))
for name, mask in groups.items():
    g = E[mask]
    if len(g) == 0:
        continue
    cells = []
    for d in RF.DIMS:
        a, b = g[f"raw_{d}"].median(), g[f"framed_{d}"].median()
        cells.append(f"{a:.2f}" if name == "Plain" else f"{a:.2f} → {b:.2f}")
    p(f"| {name} | {len(g)} | " + " | ".join(cells) + " |")
p("")

p("## Length dependence (Spearman with prose tokens, eligible)\n")
p("| dim | raw | framed |\n|---|---|---|")
for d in RF.DIMS:
    a = E[[f"raw_{d}", "prose_tokens"]].corr(method="spearman").iloc[0, 1]
    b = E[[f"framed_{d}", "prose_tokens"]].corr(method="spearman").iloc[0, 1]
    p(f"| {d} | {a:.2f} | {b:.2f} |")
p("")

p("## Rhetorical state, eligible markdown answers (raw vs framed)\n")
p(md_table(pd.crosstab(E[md].raw_rhetorical_state, E[md].framed_rhetorical_state)) + "\n")

(OUT / "arena_check_r1.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
