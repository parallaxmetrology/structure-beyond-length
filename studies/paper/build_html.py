#!/usr/bin/env python3
"""Builds paper.html from main.tex + numbers.tex with pandoc, so the HTML and the PDF share one source.
Resolves what pandoc does not handle on its own: the number macros, section/figure/table numbering
and \\ref, numeric citations, the example boxes, side-by-side minipage figures and tabularx."""
import re, subprocess
from pathlib import Path

P = Path(__file__).resolve().parent
tex = (P / "main.tex").read_text(encoding="utf-8")
title = re.search(r"\\title\{(.*?)\}\n", tex).group(1)
body = tex.split("\\begin{document}", 1)[1].split("\\end{document}", 1)[0]
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S).group(1).strip()
body = re.sub(r"\\maketitle|\\begin\{abstract\}.*?\\end\{abstract\}", "", body, flags=re.S)

# 1. number macros (longest name first so \hsHelpLen does not eat \hsHelpLenRep)
macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (P / "numbers.tex").read_text(encoding="utf-8")))
def sub_macros(s):
    for k in sorted(macros, key=len, reverse=True):
        v = macros[k].replace("$-$", "−")
        s = re.sub(r"\\" + k + r"(\{\})?(?![A-Za-z])", lambda m, v=v: v, s)
    return s
body, abstract = sub_macros(body), sub_macros(abstract)

# 2. side-by-side minipages -> two figures
def split_minipages(m):
    out = []
    for inner in re.findall(r"\\begin\{minipage\}\{[^}]*\}(.*?)\\end\{minipage\}", m.group(0), re.S):
        out.append("\\begin{figure}[htbp]\\centering" + inner.replace("\\centering", "") + "\\end{figure}")
    return "\n".join(out)
body = re.sub(r"\\begin\{figure\}\[[^\]]*\]\\centering\s*\\begin\{minipage\}.*?\\end\{figure\}", split_minipages, body, flags=re.S)

# 3. numbering: sections, appendix letters, figures, tables; then \ref
labels, sec, sub, app = {}, 0, 0, False
fig = tab = 0
tokens = re.split(r"(\\appendix|\\section\{[^}]*\}|\\subsection\{[^}]*\}|\\begin\{figure\}|\\begin\{table\}|\\label\{[^}]*\})", body)
last = None
out = []
for t in tokens:
    if t == "\\appendix":
        app, sec = True, 0; continue
    m = re.match(r"\\section\{(.*)\}", t)
    if m:
        sec += 1; sub = 0
        num = chr(64 + sec) if app else str(sec)
        last = num; out.append(f"\\section*{{{num} · {m.group(1)}}}"); continue
    m = re.match(r"\\subsection\{(.*)\}", t)
    if m:
        sub += 1; last = f"{sec}.{sub}"; out.append(f"\\subsection*{{{last} {m.group(1)}}}"); continue
    if t == "\\begin{figure}":
        fig += 1; last = str(fig); out.append(t); continue
    if t == "\\begin{table}":
        tab += 1; last = str(tab); out.append(t); continue
    m = re.match(r"\\label\{(.*)\}", t)
    if m:
        labels[m.group(1)] = last; continue
    out.append(t)
body = "".join(out)
body = re.sub(r"\\ref\{([^}]*)\}", lambda m: labels.get(m.group(1), "?"), body)
# figure/table captions get their number in front
fig_i = tab_i = 0
def cap(m):
    global fig_i, tab_i
    env = m.group(1)
    if env == "figure":
        fig_i += 1; return m.group(0).replace("\\caption{", f"\\caption{{Figure {fig_i}. ", 1)
    tab_i += 1; return m.group(0).replace("\\caption{", f"\\caption{{Table {tab_i}. ", 1)
body = re.sub(r"\\begin\{(figure|table)\}.*?\\end\{\1\}", cap, body, flags=re.S)

# 4. citations -> [n]
keys = re.findall(r"\\bibitem\{([^}]*)\}", body)
num = {k: i + 1 for i, k in enumerate(keys)}
body = re.sub(r"\\cite\{([^}]*)\}", lambda m: "[" + ", ".join(str(num[k.strip()]) for k in m.group(1).split(",")) + "]", body)
bib = re.search(r"\\begin\{thebibliography\}\{\d+\}(.*?)\\end\{thebibliography\}", body, re.S)
items = re.split(r"\\bibitem\{[^}]*\}", bib.group(1).replace("\\small", ""))[1:]
body = body.replace(bib.group(0), "\\section*{References}\n\\begin{enumerate}\n" + "\n".join("\\item " + i.strip() for i in items) + "\n\\end{enumerate}")

# 5. examples, tables, figures
body = re.sub(r"\\begin\{example\}\{(.*?)\}", lambda m: "\\begin{quote}\n\\textbf{" + m.group(1) + "}\n\n", body)
body = body.replace("\\end{example}", "\\end{quote}")
body = re.sub(r"\\begin\{tabularx\}\{\\linewidth\}\{@\{\}(\w+)@\{\}\}", lambda m: "\\begin{tabular}{" + m.group(1).replace("X", "l") + "}", body)
body = body.replace("\\end{tabularx}", "\\end{tabular}")
body = re.sub(r"(figures/fig_\w+)\.pdf", r"\1.svg", body)
body = re.sub(r"\\includegraphics\[[^\]]*\]", r"\\includegraphics", body)
body = body.replace("\\quad ", "").replace("[H]", "[htbp]").replace("\\\\[2pt]", "\n\n")

src = "\\documentclass{article}\n\\begin{document}\n" + body + "\n\\end{document}\n"
(P / "_html_src.tex").write_text(src, encoding="utf-8")
abstract_html = subprocess.run(["pandoc", "-f", "latex", "-t", "html5", "--mathml"], input=abstract, capture_output=True, text=True, check=True).stdout

css = (P / "paper.css").read_text(encoding="utf-8")
header = f'<p class="draft">Working draft · {macros.get("dummy", "")}local, not published · same source as main.pdf</p>'
subprocess.run(["pandoc", str(P / "_html_src.tex"), "-f", "latex", "-t", "html5", "--standalone", "--embed-resources",
                "--mathml", "--resource-path", str(P), "--metadata", f"title={title}", "--metadata", "pagetitle=Structure Beyond Length",
                "--css", str(P / "paper.css"), "-o", str(P / "paper.html")], check=True)
html = (P / "paper.html").read_text(encoding="utf-8")
front = ('<p class="byline">Russell Parrish · Parallax Metrology · working draft, analysis carried out with Claude (Anthropic) · local, not published</p>'
         f'<section class="abstract"><h2>Abstract</h2>{abstract_html}</section>')
html = html.replace("</header>", "</header>\n" + front, 1)
(P / "paper.html").write_text(html, encoding="utf-8")

# site version: no "local" line, PDF link, see-also and the site's copyright footer
site_front = ('<p class="byline">Russell Parrish · Parallax Metrology · working draft, November 2026 · analysis carried out with Claude (Anthropic) · '
              '<a href="Structure_Beyond_Length.pdf">PDF</a></p>'
              f'<section class="abstract"><h2>Abstract</h2>{abstract_html}</section>')
site = html.replace(front, site_front, 1)
site_end = ('<p class="seealso">See also: <a href="linguistic_kernel_pm.html">The Linguistic Kernel</a> · '
            '<a href="text_has_shape.html">Text has shape</a> · <a href="structural_residue.html">Structural residue</a></p>'
            '<footer><p>&copy; 2025–2026 Russell Parrish / A.rtist I.nfluencer / Parallax Metrology. All rights reserved. No part of this '
            'system, visual material, or accompanying documents may be reproduced, distributed, or transmitted in any form or by any '
            'means, including AI training datasets, without explicit written permission from the creator. More info: '
            '<a href="https://www.artistinfluencer.com">www.artistinfluencer.com</a> and '
            '<a href="https://www.parallaxmetrology.com">www.parallaxmetrology.com</a></p></footer>')
site = site.replace("</body>", site_end + "\n</body>", 1)
site = site.replace("<head>", '<head>\n<meta name="description" content="What human raters and LLM judges reward in model answers beyond length, read with one fixed structural instrument across HelpSteer2, AlpacaEval, MT-Bench and Chatbot Arena, with a same-battle judge run.">', 1)
(P / "site_structure_beyond_length.html").write_text(site, encoding="utf-8")
print("wrote paper.html", len(html), "labels", len(labels), "citations", len(num), "unresolved refs", html.count(">?<") + html.count(" ?)"))
