#!/usr/bin/env python3
"""Reading frame r1: a preparation layer that runs before the frozen v19 kernel.

v19 is not edited, imported from a copy, or reconfigured. This module loads it
from its folder, records its file hash, and reads every text twice:

  raw     the text exactly as given
  framed  the same text with markdown that v19 misreads neutralised

Both readings and their difference are returned, so the markdown effect is a
measured quantity rather than noise. Nothing here changes a v19 number.

What the frame changes (and nothing else):
  1. Fenced code blocks (``` or ~~~) are removed. Code is not prose.
  2. Table rows and table separator lines are removed.
  3. Header lines (# Title) and bold-only lines (**Title:**) are removed, so
     v19 does not glue a title onto the next sentence. (r1 draft made them their
     own paragraph instead; on Arena that invented paragraphs and pushed rᵥ up,
     so titles are dropped, leaving a paragraph break where markdown renders one.)
  4. Markdown links [text](url) keep their text and drop the URL.

Plain text with none of these patterns comes back byte-identical, so its two
readings are identical.

It also adds two fields v19 does not have:
  eligible        English (by script and function-word share) and at least
                  MIN_PROSE_TOKENS word tokens after framing
  collapse_reading  v19's collapse_subtype restated for chat corpora, where
                  short complete answers are common:
                    severe + tail signal + open ending -> short_cut
                    severe otherwise                   -> short_complete
                    partial                  -> cut_partial
                    math                     -> cut_code_or_math
                    none                     -> none
"""
from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

VERSION = "reading-frame-r1.1"  # r1.1: kernel errors caught and marked ineligible (kernel_error); no other change
DEFAULT_KERNEL_PATH = Path(__file__).resolve().parent.parent / "kernel" / "kernel_v19.py"
# The studies ran on the original v19 file (sha256 1f63ee858c92af466957...). This repository ships the
# published copy from github.com/parallaxmetrology/linguistic-kernel (src/lingkernel/_v19.py): same code with
# citation markers removed from its docstring and an unrelated add-on section left out. That repository's
# golden tests show identical numbers, and the reading frame gives identical readings on study data.
V19_SHA256 = "cac31dcf244fb52fb8faf1ee4a04f4bea8f2176809205907ecf3f7bfbbfbfe37"

MIN_PROSE_TOKENS = 40          # v19 severe is 1-15 tokens, partial 16-40
MIN_ASCII_LETTER_SHARE = 0.95  # share of letters that are a-z / A-Z
MIN_FUNCTION_WORD_SHARE = 0.12 # share of word tokens that are English function words
FUNCTION_WORD_CHECK_MIN_TOKENS = 20

DIMS = ["Δx", "rᵥ", "ρᵣ", "μ", "Δy", "xₚ", "θ", "dₛ"]

# Same word pattern v19 uses, so token counts agree.
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

_FENCE_RE = re.compile(r"^\s{0,3}(```|~~~)")
_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
_BOLD_LINE_RE = re.compile(r"^\s*(\*\*|__)([^*_].*?)\1\s*:?\s*$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_LINK_RE = re.compile(r"\[([^\]\n]+)\]\(([^)\s]+)\)")

_FUNCTION_WORDS = frozenset("""
a an the and or but if then so of to in on at by for with from as is are was were be been being
it its this that these those there here i you he she we they me him her us them my your our their
not no do does did have has had will would can could should may might must which who what when
where how why than also just only about into over after before because while all any each some
""".split())


@dataclass
class FrameEdits:
    code_blocks: int = 0
    code_lines: int = 0
    unclosed_fence: bool = False
    table_lines: int = 0
    headers: int = 0
    bold_lines: int = 0
    links: int = 0

    @property
    def any(self) -> bool:
        return bool(self.code_blocks or self.table_lines or self.headers or self.bold_lines or self.links)


@dataclass
class Framed:
    text: str
    edits: FrameEdits
    identical: bool
    prose_tokens: int
    ascii_letter_share: float
    function_word_share: float
    english: bool
    eligible: bool
    reasons: List[str] = field(default_factory=list)


def frame_text(text: str) -> Framed:
    """Neutralise the markdown patterns v19 misreads. Pure function, no kernel call."""
    edits = FrameEdits()
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: List[str] = []
    in_fence = False
    fence_mark = ""

    for ln in lines:
        m = _FENCE_RE.match(ln)
        if in_fence:
            edits.code_lines += 1
            if m and m.group(1) == fence_mark:
                in_fence = False
                out.append("")
            continue
        if m:
            in_fence, fence_mark = True, m.group(1)
            edits.code_blocks += 1
            edits.code_lines += 1
            out.append("")
            continue
        if _TABLE_SEP_RE.match(ln) or _TABLE_ROW_RE.match(ln):
            edits.table_lines += 1
            out.append("")
            continue
        hm = _HEADER_RE.match(ln)
        if hm:
            edits.headers += 1
            out.append("")
            continue
        bm = _BOLD_LINE_RE.match(ln)
        if bm:
            edits.bold_lines += 1
            out.append("")
            continue
        new_ln, n_links = _LINK_RE.subn(r"\1", ln)
        edits.links += n_links
        out.append(new_ln)

    edits.unclosed_fence = in_fence

    if edits.any:
        framed = "\n".join(out)
        framed = re.sub(r"\n[ \t]*(\n[ \t]*)+\n", "\n\n", framed).strip()
    else:
        framed = text

    tokens = _WORD_RE.findall(framed.lower())
    letters = [c for c in framed if c.isalpha()]
    ascii_share = (sum(1 for c in letters if c.isascii()) / len(letters)) if letters else 0.0
    fw_share = (sum(1 for t in tokens if t in _FUNCTION_WORDS) / len(tokens)) if tokens else 0.0

    reasons: List[str] = []
    if not letters:
        reasons.append("no_letters")
    elif ascii_share < MIN_ASCII_LETTER_SHARE:
        reasons.append("non_latin_script")
    elif len(tokens) >= FUNCTION_WORD_CHECK_MIN_TOKENS and fw_share < MIN_FUNCTION_WORD_SHARE:
        reasons.append("low_english_function_words")
    english = not reasons
    if len(tokens) < MIN_PROSE_TOKENS:
        reasons.append("short_prose")

    return Framed(
        text=framed, edits=edits, identical=(framed == text), prose_tokens=len(tokens),
        ascii_letter_share=round(ascii_share, 4), function_word_share=round(fw_share, 4),
        english=english, eligible=not reasons, reasons=reasons,
    )


_CLOSED_END_RE = re.compile(r"[.!?。！？…][\"'”’)\]*_]*\s*$")


def ends_closed(text: str) -> bool:
    """True when the text ends in sentence-closing punctuation (optionally followed by quotes/brackets/emphasis)."""
    return bool(_CLOSED_END_RE.search(text or ""))


def collapse_reading(evidence: Dict[str, Any], text: str = "") -> str:
    """v19's tail signal also fires on a final scaffold word ("...help with that."),
    so a severe-short text that ends in closing punctuation reads as complete."""
    sub = evidence.get("collapse_subtype", "none")
    if sub == "severe":
        cut = evidence.get("tail_incomplete_flag") and not ends_closed(text)
        return "short_cut" if cut else "short_complete"
    if sub == "partial":
        return "cut_partial"
    if sub == "math":
        return "cut_code_or_math"
    return "none"


def load_v19(path: Optional[Path] = None, check_hash: bool = True):
    """Load the frozen kernel from its own file. Refuses a file whose hash has changed."""
    p = Path(path or DEFAULT_KERNEL_PATH)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    if check_hash and sha != V19_SHA256:
        raise RuntimeError(f"kernel file hash {sha} does not match frozen v19 {V19_SHA256}")
    spec = importlib.util.spec_from_file_location("kernel_v19", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kernel_v19"] = mod  # dataclasses look the module up while it loads
    spec.loader.exec_module(mod)
    return mod, sha


class Reader:
    """Reads texts with v19, raw and framed."""

    def __init__(self, kernel_path: Optional[Path] = None):
        self.K, self.kernel_sha256 = load_v19(kernel_path)
        self.kernel = self.K.LinguisticKernel()

    def _measure(self, text: str) -> Optional[Dict[str, Any]]:
        if not text or not text.strip():
            return None
        try:
            r = self.kernel.compute(text)
        except Exception as ex:  # r1.1: v19 raises on some inputs (e.g. literal "<|endoftext|>")
            self.last_error = f"{type(ex).__name__}: {str(ex)[:120]}"
            return None
        e = r["evidence"]
        coords = {**r["kernel"], **r["aux"]}
        return {
            "coords": {d: coords.get(d) for d in DIMS},
            "basin": r.get("basin"),
            "rhetorical_state": e.get("rhetorical_state"),
            "collapse_subtype": e.get("collapse_subtype"),
            "segmentation_reliable": e.get("segmentation_reliable"),
            "tail_incomplete_flag": e.get("tail_incomplete_flag"),
            "math_expression_tail_flag": e.get("math_expression_tail_flag"),
            "token_count": e.get("token_count"),
            "sentence_count": e.get("sentence_count"),
            "paragraph_count": e.get("paragraph_count"),
        }

    def read(self, text: str) -> Dict[str, Any]:
        text = text if isinstance(text, str) else ""
        f = frame_text(text)
        self.last_error = None
        raw = self._measure(text)
        framed = raw if f.identical else self._measure(f.text)
        kernel_error = self.last_error
        if kernel_error:
            f.eligible = False
            f.reasons = f.reasons + ["kernel_error"]
        delta = None
        if raw and framed:
            delta = {d: (framed["coords"][d] - raw["coords"][d]) for d in DIMS
                     if framed["coords"][d] is not None and raw["coords"][d] is not None}
        return {
            "frame_version": VERSION,
            "kernel_sha256": self.kernel_sha256,
            "eligible": f.eligible,
            "reasons": f.reasons,
            "english": f.english,
            "prose_tokens": f.prose_tokens,
            "framed_identical": f.identical,
            "edits": asdict(f.edits),
            "collapse_reading": collapse_reading(raw, text) if raw else "empty",
            "ends_closed": ends_closed(text),
            "kernel_error": kernel_error,
            "raw": raw,
            "framed": framed,
            "delta": delta,
        }


def flat_row(reading: Dict[str, Any]) -> Dict[str, Any]:
    """One flat dict per reading, for a DataFrame or CSV."""
    row = {k: reading[k] for k in ("frame_version", "eligible", "english", "prose_tokens",
                                   "framed_identical", "collapse_reading", "ends_closed")}
    row["reasons"] = ";".join(reading["reasons"])
    for k, v in reading["edits"].items():
        row[f"edit_{k}"] = v
    for side in ("raw", "framed"):
        m = reading[side]
        if not m:
            continue
        for d in DIMS:
            row[f"{side}_{d}"] = m["coords"][d]
        for k in ("rhetorical_state", "collapse_subtype", "segmentation_reliable", "token_count", "sentence_count"):
            row[f"{side}_{k}"] = m[k]
    return row
