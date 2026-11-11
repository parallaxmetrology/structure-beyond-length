#!/usr/bin/env python3
"""Linguistic Kernel (Rebuild) v19 Russell Parrish russellgparrish@gmail.com

Behavior-only measurement of text outputs.
Canonical minimal kernel: K = [Δx, rᵥ, ρᵣ, μ]
Optional aux: Aux = [Δy, xₚ, θ, dₛ]

No training-set speculation. Deterministic. Auditable.

Deterministic Telemetry for LLM Reliability. Stop using LLMs to judge LLMs. Use a deterministic linguistic kernel to monitor structural integrity in real-time. Controlled structural excursion mapping under deterministic stress.

## Linguistic Kernel — What This Is / What It Does

This kernel is a **behavior-only instrumentation layer** for LLM text outputs. It does **not** judge whether an answer is correct, persuasive, safe, or aligned; it measures **how the output is structured**. The goal is to treat text as a measurable geometric field (a “geometry of delivery”), so models, prompts, and multi-turn workflows can be compared with **deterministic, auditable signals** rather than vibe checks or LLM-as-judge loops.

### Core idea
Given a raw output string, the kernel computes a **canonical coordinate** in a small structural space:

- **Canonical minimal kernel:** `K = [Δx, rᵥ, ρᵣ, μ]`
  - **Δx (Weight Shift):** where “informational mass” lands across the response (front-loaded vs back-loaded).
  - **rᵥ (Void Ratio):** segmentation / breathing room (paragraphing, bullets, whitespace) as a proxy for sparsity vs compression.
  - **ρᵣ (Packing Density):** concentration of content-heavy tokens vs common connective language.
  - **μ (Cohesion):** continuity of discourse scaffolding (connectors, reference continuity, transitions).

- **Optional auxiliary metrics:** `Aux = [Δy, xₚ, θ, dₛ]`
  - **xₚ (Peripheral Pull):** edge-seeking behavior (technical rarity + metaphor density / outlier markers).
  - **θ (Flow Instability):** non-linear jumps between adjacent sentences/paragraphs (requires reader synthesis; reduced smoothness).
  - **dₛ (Structural Thickness):** clause stacking / subordinate structure density.
  - **Δy (segmentation_strain):** average sentence length relative to 15-token baseline. Coordinate-dependent statistic. Plain-language alias: segmentation_strain. Tracks punctuation presence, verbosity constraints, and sentence boundary suppression. Large excursions indicate segmentation events, not content complexity. Does not imply vertical spatial geometry.

### What it explicitly does *not* claim
To prevent category mistakes, the kernel is **not** a truth judge and does not measure correctness, training data coverage, internal mechanisms, or “intelligence” as a single scalar. Two answers can be equally correct yet land in very different kernel coordinates—deliberately.

### Envelope layer (Phase-2 analog)
On top of raw coordinates, v4 adds an envelope system: fit a **neutral baseline** distribution in **z-space** (normalized coordinates), compute each output’s distance from the baseline centroid, then label outputs by bands:

- 🟢 stable range
- 🟠 elevated deviation
- 🔴 breach / excursion

This turns single-output fingerprinting into operational telemetry: breach ratio, drift magnitude, recovery, and **hysteresis** (stress→recovery path asymmetry).

### How it is built (deterministic + auditable)
Implementation is a deterministic pipeline: tokenize/segment the string, extract countable evidence features, transform evidence into coordinates, assign coarse basin labels, and (optionally) fit envelope bands for drift monitoring. The key constraint is **auditability**: every metric is backed by countable evidence features rather than a model-in-the-loop score.

The output is designed to be **engine-agnostic**: it measures only what is visible in the produced text.

---

## What It Does (Operationally)

Given an output string, the kernel produces:

1. **Raw kernel coordinates** (Δx, rᵥ, ρᵣ, μ + optional aux).
2. A **basin assignment** (coarse categorical label) derived from raw kernel space. This is a “where it lives” tag, not a value judgment.
3. A structured **evidence bundle** (counts of tokens, sentences, paragraphs, technical hits, metaphor hits, subordinator hits, transitions, etc.) supporting the computed metrics.
4. A **meta block** with versioning and reproducibility notes.

Additionally, v4 introduces a **Phase-2-style envelope layer** (language analog of the image envelope system):

- Fit an envelope on a **neutral baseline distribution** (centroid + R95 radius) in **z-space** (baseline-normalized coordinates).
- Compute each run’s **envelope distance** from the baseline centroid.

This makes it possible to quantify:
- **Breach ratio** (how often a model leaves its neutral structural envelope under stress),
- **Mean drift magnitude** (average excursion distance),
- **Recovery behavior** (does the model return toward the basin after stress),
- **Hysteresis** (stress → recovery path distance; “non-inverse return”).

---

## What It Is NOT

To prevent category mistakes, this system explicitly does **not** claim to measure:

- Ground-truth correctness or factual accuracy
- Training data coverage or “what the model has seen”
- Internal model mechanisms
- “Intelligence” as a single scalar

It is also **not** a semantic scoring rubric. Two answers can be equally correct yet have very different kernel coordinates.

This is deliberate: the kernel is meant to quantify **structural behavior**, not content truth.

### Positioning disclaimer
All geometric language in this documentation refers to induced structure in feature space
and does not imply intrinsic spatial organization within the model's internal representations.
This kernel is a structural deformation monitor for output stability, not a geometric model of language.

### Formal distinction: μ vs ρᵣ
- **μ (Cohesion):** directional structural displacement — measures continuity of discourse scaffolding (transitions, reference pronouns). A μ shift indicates movement in the connective tissue of the response.
- **ρᵣ (Packing Density):** entropy contraction / lexical clustering — measures concentration of content-heavy tokens vs connective language. A ρᵣ shift indicates compression or expansion of informational density.
These are orthogonal dimensions. μ measures direction of structural flow; ρᵣ measures how tightly packed the content is.

---

## Why This Exists (Value / Use Cases)

Most production failures in LLM deployments are not benchmark failures—they are **structural reliability failures**:

- the model breaks output format constraints (JSON/schema),
- constraint drift increases across turns,
- the response becomes brittle under pressure (over-compressed, under-specified, or structurally chaotic),
- recovery after stress is asymmetric or incomplete.

The kernel + envelope system provides a way to:

- **Compare engines** by structural “fingerprint” (default basin + drift patterns),
- **Detect structural excursions** before they become downstream failures,
- **Quantify recovery** rather than assuming “it came back”,
- Build a **controllability layer**: measure → perturb → re-measure to steer outputs into desired structural regimes.

This is the “missing middle” between:
- “works on benchmarks”
and
- “works reliably in production.”

---

## How It Was Built (Method / Implementation Notes)

Design constraints:

- **Behavior-only:** measurements are inferred strictly from produced text.
- **Deterministic:** no model-in-the-loop, no web calls, no stochastic components.
- **Auditable:** each metric is supported by countable evidence features.
- **Baseline-aware:** supports fitting prompt-family baselines and mapping to z-space for drift/envelope work.

Architecture (high-level):

1. **Tokenization + segmentation:** token, sentence, and paragraph boundaries are computed from the raw string.
2. **Evidence extraction:** count-based features are computed (technical token hits, metaphor hits, subordinator hits, transitions, etc.).
3. **Kernel computation:** evidence features are transformed into kernel coordinates (Δx, rᵥ, ρᵣ, μ).
4. **Aux computation (optional):** additional measures (xₚ, θ, dₛ, Δy) are derived from evidence and adjacency structure.
5. **Basin assignment:** a conservative, coarse classifier maps coordinates to basin labels (raw vs z-space handled separately).
6. **Envelope tooling (v4):** given a baseline mask, fit a z-space centroid + R95 radius and compute excursion bands.

Validator add-ons (v4):

- **LIS (Logical Integrity Score proxy):** a conservative structural consistency proxy (penalizes explicit reversals/contradiction markers and excessive hedging). It is **not** truth verification.
- **HG (Hallucination Gradient proxy):** a constrained-task proxy for “invented specificity.” When a reference passage is provided, it penalizes numbers/capitalized entities not present in the reference. It is **not** open-domain hallucination detection.

---

## Output Contract (What You Get Back)

The canonical `compute(text)` output is a structured dict with:

- `kernel`: the canonical kernel coordinates (Δx, rᵥ, ρᵣ, μ)
- `aux`: optional auxiliary coordinates (Δy, xₚ, θ, dₛ)
- `basin`: coarse basin label (raw-space)
- `z`: (optional) baseline-normalized coordinates for drift/envelope work
- `evidence`: audit-friendly counts backing the metrics
- `meta`: versioning and reproducibility metadata

The Phase-2 envelope layer operates over tabular runs:
- fit baseline → compute distances → band excursions → summarize breach/recovery/hysteresis.

This makes the system usable for:
- per-sample diagnostics,
- batch comparative profiling,
- and controlled stress testing under a Phase-2 envelope protocol.

Fixes vs previous upload:
- Removed duplicate PromptFamily / PromptFamilyRegistry definitions (single canonical version).
- Fixed baseline z-scoring call: KernelBaseline.z_map(...) is the API (previously called z_kernel).
- Ensured registry load/write schema matches PromptFamily.to_json/from_json.
- Minor hardening: safe handling when family exists but baseline missing.

v18 changes (2026):
- New rhetorical state: formatting_collapse. Triggered by low_token_count flag AND Δy < -0.3
  AND rᵥ > 0.45. Catches severe mid-expression truncation (1–15 token responses). Longer
  truncations (24–39 tokens) are not caught by coordinates alone; content inspection required.
- New KernelConfig thresholds: fc_dy_max, fc_rv_min (formatting_collapse gate parameters).
- Flag routing: short_sentence_count + low_token_count co-occurrence now evaluated against
  formatting_collapse thresholds before defaulting to assertive. Previously these flags had
  no downstream classification route — they fired correctly but were not acted on.
- New constraint_basis value: "formatting_collapse" (distinct from "sentence_suppression"
  and "verbosity").
- segmentation_reliable docstring updated: flag now explicitly documented as a measurement
  ceiling indicator, not a caution notice. Concentration of False in complying_under_constraint
  is expected and is itself a structural finding, not instrument malfunction.
- tiktoken_token_count scoped as Phase 2 instrument. Documented in meta block as cross-model
  comparison signal; not reported in single-model study outputs by convention. Field retained
  for pipeline compatibility.
- kernel_version string updated to "lang-kernel-rebuild-v18".

v19 changes (2026):
- Collapse detection split into three tiers, replacing the single v18 formatting_collapse gate.
  All three tiers share the rhetorical_state label "formatting_collapse" for backward
  compatibility; they are distinguished by the new collapse_subtype field in evidence.

  Tier 1 — formatting_collapse_severe (preserved from v18, unchanged):
    token_count < fc_token_max (16) AND Dy < fc_dy_max (-0.3) AND rv > fc_rv_min (0.45)
    Catches extremely short truncations (1–15 tokens). Calibration: 6/6 true positives,
    0 false positives from Gemini EMS batch corpus (n=400).

  Tier 2 — formatting_collapse_partial (new):
    16 <= token_count <= pc_token_max (40) AND Dy <= pc_dy_max (0.25)
    AND rv >= pc_rv_min (0.12) AND tail_incomplete_flag = True
    Catches mid-length truncations (16–40 tokens) that carry visible incompletion signals
    (unclosed brackets, dangling operators, mid-expression endings, scaffold-word endings).
    tail_incomplete_flag is required: threshold relaxation alone produces false positives.
    Calibration: 6/6 additional true positives, 0 false positives on GPT EMS corpus (n=400).

  Tier 3 — formatting_collapse_math (new):
    tail_incomplete_flag = True AND math_expression_tail_flag = True
    No token ceiling. Catches truncations in math/code-heavy responses with normal
    segmentation statistics (Dy 0.5–0.7) that are geometrically indistinguishable from
    complete short explanations without content inspection.
    math_expression_tail_flag fires on: open integrals, dangling LaTeX sequences,
    unbalanced math operators, code blocks without closing structure.
    CRITICAL gate: both flags must be True (conjunction). math_expression_tail_flag alone
    produces false positives on complete responses containing inline math notation.
    Calibration: 3/3 remaining true positives, 0 false positives (conjunction eliminates
    all 3 FPs that would arise from math_expression_tail_flag alone).

  Combined detector performance on GPT EMS corpus (n=400, 15 true collapses):
    TP=15, FP=0, FN=0, TN=385. Precision=1.00, Recall=1.00, F1=1.00.

- Routing order updated: revising → degenerate_input → formatting_collapse_severe →
  formatting_collapse_partial → formatting_collapse_math → complying_under_constraint
  (structural override) → dialectic → complying_under_constraint (Dy gate) → assertive.

- New KernelConfig parameters (all tunable):
    pc_token_max (40): upper token ceiling for partial collapse gate.
    pc_dy_max (0.25): Dy ceiling for partial collapse gate.
    pc_rv_min (0.12): rv floor for partial collapse gate.

- New evidence fields:
    tail_incomplete_flag: bool — heuristic detection of truncated text endings.
    math_expression_tail_flag: bool — heuristic detection of incomplete math/code endings.
    collapse_subtype: str — "severe" | "partial" | "math" | "none". Distinguishes which
      collapse tier fired. "none" when rhetorical_state is not formatting_collapse.

- constraint_basis updated: now reports "severe" | "partial" | "math" when
  rhetorical_state == "formatting_collapse", replacing the single "formatting_collapse" value.
  Downstream consumers checking constraint_basis == "formatting_collapse" should update
  to check rhetorical_state == "formatting_collapse" instead.

- kernel_version string updated to "lang-kernel-rebuild-v19".
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional batch/envelope deps
try:
    import numpy as np
except Exception:
    np = None  # envelope functions will error if called

try:
    import pandas as pd
except Exception:
    pd = None  # envelope functions will error if called

try:
    import tiktoken as _tiktoken
    import os as _os
    # If the vocab file is bundled in the same directory as this script,
    # point tiktoken at it so no network download is needed.
    # Ship "9b5ad71b2ce5302211f9c61530b329a4922fc6a4" alongside this file.
    _VOCAB_HASH = "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"
    _SCRIPT_DIR = Path(__file__).parent
    _LOCAL_VOCAB = _SCRIPT_DIR / _VOCAB_HASH
    if _LOCAL_VOCAB.exists():
        _os.environ["TIKTOKEN_CACHE_DIR"] = str(_SCRIPT_DIR)
    _TK_ENC = _tiktoken.get_encoding("cl100k_base")
except Exception:
    _tiktoken = None  # tiktoken not installed or vocab unavailable; tiktoken_token_count will be -1
    _TK_ENC = None


# ----------------------------
# Text preprocessing utilities
# ----------------------------

_ABBREV = {
    "e.g.", "i.e.", "etc.", "vs.", "mr.", "mrs.", "ms.", "dr.", "prof.",
    "sr.", "jr.", "st.", "mt.", "no.", "fig.", "sec.", "eq.",
}

_BULLET_RE = re.compile(r"^\s*(?:[-*•]+|\d+[\.)]|\([a-zA-Z]\))\s+")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_lines(text: str) -> List[str]:
    return normalize_text(text).split("\n")


def paragraph_blocks(text: str) -> List[str]:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", t) if b.strip()]
    return blocks


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text.lower())


def segment_sentences(text: str) -> List[str]:
    """Deterministic sentence segmentation with bullet-preservation.

    Guarantees:
    - Bullet lines are emitted as their own sentence-like units even if they have no punctuation.
    - Common abbreviations are protected to reduce false splits.
    - Non-bullet text is split on punctuation followed by whitespace + (A-Z|digit).

    This is intentionally deterministic and dependency-free.
    """
    text = normalize_text(text)
    if not text:
        return []

    def _split_punct(block: str) -> List[str]:
        block = block.strip()
        if not block:
            return []
        protected = block
        for ab in _ABBREV:
            protected = protected.replace(ab, ab.replace(".", "<DOT>"))
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", protected)
        out: List[str] = []
        for p in parts:
            p = p.replace("<DOT>", ".").strip()
            if p:
                out.append(p)
        return out if out else [block]

    lines = [ln.rstrip() for ln in split_lines(text) if ln.strip()]
    sents: List[str] = []
    buf: List[str] = []

    for ln in lines:
        if _BULLET_RE.match(ln):
            if buf:
                sents.extend(_split_punct(" ".join(buf)))
                buf = []
            sents.append(ln.strip())
        else:
            buf.append(ln.strip())

    if buf:
        sents.extend(_split_punct(" ".join(buf)))

    return sents


# ----------------------------
# Baseline calibration
# ----------------------------

@dataclass(frozen=True)
class BaselineStats:
    mean: float
    std: float

    def z(self, x: float) -> float:
        if self.std < 1e-9:
            return 0.0
        return (x - self.mean) / self.std


@dataclass
class KernelBaseline:
    """Per-dimension baseline statistics for z-scoring."""
    stats: Dict[str, BaselineStats]

    def z_map(self, raw: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, v in raw.items():
            st = self.stats.get(k)
            out[k] = st.z(v) if st else v
        return out

    def to_json(self) -> Dict[str, Any]:
        return {k: {"mean": v.mean, "std": v.std} for k, v in self.stats.items()}

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> "KernelBaseline":
        stats = {k: BaselineStats(mean=float(v["mean"]), std=float(v["std"])) for k, v in obj.items()}
        return KernelBaseline(stats=stats)


@dataclass
class PromptFamily:
    family_id: str
    description: str = ""
    baseline: Optional[KernelBaseline] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "family_id": self.family_id,
            "description": self.description,
            "baseline": self.baseline.to_json() if self.baseline else None,
        }

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> "PromptFamily":
        bl = KernelBaseline.from_json(obj["baseline"]) if obj.get("baseline") else None
        return PromptFamily(
            family_id=str(obj["family_id"]),
            description=str(obj.get("description", "")),
            baseline=bl,
        )


class PromptFamilyRegistry:
    """prompt_family_id -> PromptFamily"""
    def __init__(self) -> None:
        self._families: Dict[str, PromptFamily] = {}

    def upsert(self, family: PromptFamily) -> None:
        self._families[family.family_id] = family

    def get(self, family_id: Optional[str]) -> Optional[PromptFamily]:
        if not family_id:
            return None
        return self._families.get(family_id)

    def list_ids(self) -> List[str]:
        return sorted(self._families.keys())

    def to_json(self) -> Dict[str, Any]:
        return {fid: fam.to_json() for fid, fam in self._families.items()}

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> "PromptFamilyRegistry":
        reg = PromptFamilyRegistry()
        for _, fam_obj in obj.items():
            fam = PromptFamily.from_json(fam_obj)
            reg.upsert(fam)
        return reg

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.write_text(json.dumps(self.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "PromptFamilyRegistry":
        p = Path(path)
        obj = json.loads(p.read_text(encoding="utf-8"))
        return PromptFamilyRegistry.from_json(obj)


# ----------------------------
# Configuration
# ----------------------------

@dataclass
class KernelConfig:
    """All knobs live here. Freeze + version-bump on change if you want 'metric-locked' behavior."""
    stopwords: Optional[set] = None
    transitions: Optional[set] = None
    reference_pronouns: Optional[set] = None
    technical_markers: Optional[set] = None
    metaphor_patterns: Optional[List[str]] = None
    subordinators: Optional[set] = None
    contrast_markers: Optional[set] = None  # dialectical tension markers (distinct from transitions)

    # rᵥ normalization targets
    rv_paras_per_100_tokens_full: float = 4.0
    rv_bullets_per_100_tokens_full: float = 6.0
    rv_short_sentence_prop_full: float = 0.60
    rv_dense_avg_sent_len: float = 25.0
    rv_airy_avg_sent_len: float = 8.0

    # dₛ normalization target
    ds_markers_per_token_full: float = 0.12

    # μ scaling (sigmoid center)
    mu_center: float = 0.15

    # xₚ scaling (sigmoid center)
    # v17: recalibrated from 0.10 to 0.77 to account for the domain-agnostic
    # rarity proxy baseline. With the 6,355-word corpus-derived common_words list,
    # neutral English prose has rare_ratio ~0.02-0.05, placing raw ~0.07-0.18.
    # Center 0.77 puts neutral prose at xₚ~0.50 and technical text at xₚ~0.57+.
    xp_center: float = 0.77
    # v17: domain-agnostic common-word baseline for xₚ rarity proxy.
    # Words NOT in this set are treated as rare/peripheral regardless of domain.
    common_words: Optional[set] = None

    # Rhetorical state classification thresholds (tunable hyperparams, not hardcoded truths)
    rhet_dialectic_min_contrast: int = 2       # contrast_hits >= N → dialectic
    rhet_constraint_min_dy: float = 1.0        # Δy > N with no contrast → complying_under_constraint

    # formatting_collapse thresholds (v18)
    # Triggered when token count is very low AND both coordinate gates pass.
    # Calibrated against the Gemini EMS corpus (n=12 truncated, n=356 normal).
    # Gate: token_count < fc_token_max AND Dy < fc_dy_max AND rv > fc_rv_min
    # At these settings: catches 6/12 severe truncations, 1/356 normal false positive (tc=8).
    # The 1 FP is a genuine one-line complete response with identical geometry to a truncation.
    # Content inspection required to distinguish it. Documented limitation, not a design flaw.
    # Longer truncations (24-39 tokens) are outside fc_token_max and are not caught here.
    fc_token_max: int = 16     # token_count must be below this (strict severe-truncation gate)
    fc_dy_max: float = -0.3    # Dy must be below this (negative = compressed below baseline)
    fc_rv_min: float = 0.45    # rv must be above this (inflated void ratio from fragment)

    # formatting_collapse_partial thresholds (v19)
    # Second collapse tier: mid-length truncations (16-40 tokens) with visible tail signal.
    # tail_incomplete_flag is required — threshold relaxation alone produces false positives.
    # Calibration: GPT EMS corpus (n=400): catches 6 additional collapses, 0 FPs.
    pc_token_max: int = 40      # upper token ceiling (partial gate, inclusive)
    pc_dy_max: float = 0.25     # Dy must be at or below this (allows near-normal segmentation)
    pc_rv_min: float = 0.12     # rv must be at or above this (minimal floor; fragments still inflate rv)

    # segmentation_reliable thresholds
    seg_punct_per_token: float = 1.0 / 30.0   # min punctuation density
    seg_min_tokens: int = 20                   # below this, don't evaluate reliability
    seg_newline_per_token: float = 1.0 / 40.0 # newline/bullet density as alternative signal

    def __post_init__(self) -> None:
        if self.stopwords is None:
            self.stopwords = {
                "the", "a", "an", "and", "or", "but", "if", "then", "else", "when", "while",
                "to", "of", "in", "on", "for", "with", "as", "at", "by", "from", "into", "over",
                "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
                "it", "this", "that", "these", "those", "they", "them", "their", "we", "you", "i",
                "not", "no", "so", "just", "very", "also", "too", "than", "such",
                # v17: domain-crossing function words that carry no content signal
                # regardless of register. Were inflating ρᵣ in technical text because
                # they are >=4 chars and were not in the prior stopword list.
                "via", "per", "each", "both", "more", "most", "other", "some", "same",
                "thus", "hence", "well", "own", "all", "any", "few",
            }
        if self.transitions is None:
            # Conjunctive adverbs signaling discourse continuation and logical connection.
            # Excludes "however", "conversely", "nevertheless", "nonetheless" which
            # move to contrast_markers to avoid double counting.
            # Excludes weak connectives ("so", "then", "still", "rather", "instead").
            # v17: added "thus", "thereby", "subsequently", "ultimately", "generally",
            # "typically", "overall" — connectors present in technical/legal/formal
            # writing that were absent from the prior list.
            self.transitions = {
                "moreover", "furthermore", "therefore", "consequently",
                "meanwhile", "additionally", "similarly",
                "accordingly", "hence", "indeed", "specifically", "particularly",
                "notably", "thus", "thereby", "subsequently", "ultimately",
                "generally", "typically", "overall", "collectively",
            }
        if self.reference_pronouns is None:
            self.reference_pronouns = {"it", "this", "that", "these", "those", "they", "them", "their", "its"}
        if self.technical_markers is None:
            # v17: deprecated as primary xₚ signal. xₚ now uses domain-agnostic
            # word rarity proxy (common_words baseline). This set is kept empty by
            # default; callers may still pass a custom set as a supplementary signal.
            self.technical_markers = set()
        if self.common_words is None:
            # v17: domain-agnostic common-word baseline for xₚ rarity proxy.
            # Source: top 7,000 English words by Zipf frequency (wordfreq library,
            # backed by OpenSubtitles, Wikipedia, COCA, and web corpora).
            # Filtered to len>=4, alpha-only: 6,355 words covering Zipf >= 4.01.
            # Per Oxford English Corpus, top 7K lemmas cover ~90% of token usage.
            # Any 4+ char token NOT in this set is treated as rare/peripheral.
            # xp_center recalibrated to 0.77 to place neutral prose at xp~0.50.
            self.common_words = {
                "aaron", "abandon", "abandoned", "abilities", "ability", "able", "aboard", "abortion",
                "about", "above", "abroad", "absence", "absolute", "absolutely", "abstract", "abuse",
                "academic", "academy", "accent", "accept", "acceptable", "acceptance", "accepted", "accepting",
                "access", "accessible", "accident", "accidentally", "accidents", "accommodate", "accommodation", "accompanied",
                "accomplish", "accomplished", "accordance", "according", "accordingly", "account", "accounting", "accounts",
                "accuracy", "accurate", "accused", "achieve", "achieved", "achievement", "achievements", "achieving",
                "acid", "acknowledge", "acknowledged", "acquire", "acquired", "acquisition", "acres", "across",
                "acted", "acting", "action", "actions", "activated", "active", "actively", "activist",
                "activists", "activities", "activity", "actor", "actors", "actress", "acts", "actual",
                "actually", "adam", "adams", "adaptation", "adapted", "added", "addiction", "adding",
                "addition", "additional", "additionally", "address", "addressed", "addresses", "addressing", "adds",
                "adequate", "adjacent", "adjust", "adjusted", "administration", "administrative", "administrator", "admission",
                "admit", "admitted", "adopt", "adopted", "adoption", "adorable", "adult", "adults",
                "advance", "advanced", "advances", "advantage", "advantages", "adventure", "adventures", "advertising",
                "advice", "advise", "advised", "advisor", "advisory", "advocate", "aesthetic", "affair",
                "affairs", "affect", "affected", "affecting", "affects", "afford", "affordable", "afghanistan",
                "afraid", "africa", "african", "after", "afternoon", "afterwards", "again", "against",
                "aged", "agencies", "agency", "agenda", "agent", "agents", "ages", "aggressive",
                "aging", "agree", "agreed", "agreement", "agreements", "agrees", "agricultural", "agriculture",
                "ahead", "aids", "aimed", "aims", "aircraft", "airline", "airlines", "airport",
                "alabama", "alan", "alarm", "alaska", "albert", "album", "albums", "alcohol",
                "alert", "alex", "alexander", "algorithm", "alice", "alien", "aliens", "alike",
                "alive", "allegations", "alleged", "allegedly", "allen", "alliance", "allied", "allies",
                "allow", "allowed", "allowing", "allows", "ally", "almost", "alone", "along",
                "alongside", "alpha", "already", "alright", "also", "alter", "altered", "alternate",
                "alternative", "alternatives", "although", "altogether", "aluminum", "always", "amateur", "amazing",
                "amazon", "ambassador", "ambulance", "amendment", "america", "american", "americans", "among",
                "amongst", "amount", "amounts", "amsterdam", "analysis", "analyst", "anchor", "ancient",
                "anderson", "andrew", "android", "andy", "angel", "angela", "angeles", "angels",
                "anger", "angle", "angry", "animal", "animals", "animated", "animation", "anime",
                "anna", "anne", "anniversary", "announce", "announced", "announcement", "annoying", "annual",
                "annually", "anonymous", "another", "answer", "answered", "answering", "answers", "anthony",
                "anti", "anticipated", "antonio", "anxiety", "anxious", "anybody", "anymore", "anyone",
                "anything", "anytime", "anyway", "anyways", "anywhere", "apart", "apartment", "apartments",
                "apologize", "apology", "apparent", "apparently", "appeal", "appeals", "appear", "appearance",
                "appearances", "appeared", "appearing", "appears", "apple", "applicable", "application", "applications",
                "applied", "applies", "apply", "applying", "appointed", "appointment", "appreciate", "appreciated",
                "appreciation", "approach", "approached", "approaches", "approaching", "appropriate", "approval", "approve",
                "approved", "approximately", "apps", "april", "arab", "arabia", "arabic", "arch",
                "architect", "architecture", "archives", "arctic", "area", "areas", "arena", "argentina",
                "argue", "argued", "argues", "arguing", "argument", "arguments", "arizona", "arkansas",
                "armed", "arms", "army", "around", "arranged", "arrangement", "arrangements", "array",
                "arrest", "arrested", "arrival", "arrive", "arrived", "arrives", "arriving", "arrow",
                "arsenal", "arthur", "article", "articles", "artificial", "artist", "artistic", "artists",
                "arts", "artwork", "ashamed", "ashley", "asia", "asian", "aside", "asked",
                "asking", "asks", "asleep", "aspect", "aspects", "assault", "assembly", "assess",
                "assessment", "asset", "assets", "asshole", "assigned", "assignment", "assist", "assistance",
                "assistant", "assisted", "assists", "associate", "associated", "associates", "association", "associations",
                "assume", "assumed", "assuming", "assumption", "assure", "assured", "athlete", "athletes",
                "athletic", "atlanta", "atlantic", "atmosphere", "attached", "attack", "attacked", "attacking",
                "attacks", "attempt", "attempted", "attempting", "attempts", "attend", "attendance", "attended",
                "attending", "attention", "attitude", "attitudes", "attorney", "attorneys", "attract", "attracted",
                "attraction", "attractive", "attributed", "auction", "audience", "audiences", "audio", "audit",
                "august", "aunt", "austin", "australia", "australian", "austria", "authentic", "author",
                "authorities", "authority", "authorized", "authors", "auto", "automatic", "automatically", "autumn",
                "availability", "available", "avenue", "average", "aviation", "avoid", "avoided", "avoiding",
                "awake", "award", "awarded", "awards", "aware", "awareness", "away", "awesome",
                "awful", "awkward", "axis", "babe", "babies", "baby", "back", "backed",
                "background", "backing", "backs", "backup", "backwards", "bacon", "bacteria", "badge",
                "badly", "bags", "bail", "bailey", "baker", "balance", "balanced", "ball",
                "ballot", "balls", "baltimore", "band", "bands", "bang", "bank", "banking",
                "bankruptcy", "banks", "banned", "banner", "barbara", "barcelona", "bare", "barely",
                "barrel", "barrier", "barriers", "barry", "bars", "base", "baseball", "based",
                "basement", "bases", "basic", "basically", "basin", "basis", "basket", "basketball",
                "bass", "bath", "bathroom", "batman", "batteries", "battery", "battle", "battles",
                "beach", "beam", "beans", "bear", "beard", "bearing", "bears", "beast",
                "beat", "beaten", "beating", "beats", "beautiful", "beauty", "became", "because",
                "become", "becomes", "becoming", "bedroom", "beds", "beef", "been", "beer",
                "before", "began", "begin", "beginning", "begins", "begun", "behalf", "behavior",
                "behaviour", "behind", "beijing", "being", "beings", "belgium", "belief", "beliefs",
                "believe", "believed", "believes", "believing", "bell", "belly", "belong", "belonged",
                "belonging", "belongs", "beloved", "below", "belt", "bench", "bend", "beneath",
                "beneficial", "benefit", "benefits", "benjamin", "bennett", "berlin", "bernard", "beside",
                "besides", "best", "beta", "better", "between", "beyond", "bias", "bible",
                "bigger", "biggest", "bike", "bill", "billion", "bills", "billy", "binding",
                "biological", "biology", "bird", "birds", "birmingham", "birth", "birthday", "bishop",
                "bitch", "bitcoin", "bite", "bits", "bitter", "bizarre", "black", "blacks",
                "blade", "blake", "blame", "blank", "blanket", "blast", "bleeding", "bless",
                "blessed", "blessing", "blew", "blind", "block", "blocked", "blocking", "blocks",
                "blog", "blonde", "blood", "bloody", "blow", "blowing", "blown", "blows",
                "blue", "blues", "board", "boards", "boat", "boats", "bobby", "bodies",
                "body", "bold", "bolt", "bomb", "bombing", "bombs", "bond", "bonds",
                "bone", "bones", "bonus", "book", "booked", "books", "boom", "boost",
                "boot", "booth", "boots", "border", "borders", "bored", "boring", "born",
                "boss", "boston", "both", "bother", "bottle", "bottles", "bottom", "bought",
                "bounce", "bound", "boundaries", "boundary", "bout", "bowl", "bowling", "boxes",
                "boxing", "boyfriend", "boys", "brad", "bradley", "brain", "brains", "branch",
                "branches", "brand", "brandon", "brands", "brass", "brave", "brazil", "brazilian",
                "breach", "bread", "break", "breakdown", "breakfast", "breaking", "breaks", "breast",
                "breath", "breathe", "breathing", "breed", "breeding", "brian", "brick", "bride",
                "bridge", "bridges", "brief", "briefly", "bright", "brilliant", "bring", "bringing",
                "brings", "bristol", "britain", "british", "broad", "broadcast", "broadcasting", "broader",
                "broadway", "broke", "broken", "bronze", "brooklyn", "brooks", "bros", "brother",
                "brothers", "brought", "brown", "bruce", "brush", "brutal", "bubble", "buck",
                "bucket", "bucks", "buddy", "budget", "buffalo", "bugs", "build", "building",
                "buildings", "built", "bulk", "bull", "bullet", "bullets", "bullshit", "bunch",
                "burden", "bureau", "burger", "buried", "burn", "burned", "burning", "burns",
                "burnt", "burst", "buses", "bush", "business", "businesses", "bust", "busy",
                "butler", "butt", "butter", "button", "buyer", "buyers", "buying", "buzz",
                "cabin", "cabinet", "cable", "cafe", "cage", "cake", "calculated", "calendar",
                "california", "call", "called", "calling", "calls", "calm", "calories", "cambridge",
                "came", "camera", "cameras", "cameron", "camp", "campaign", "campaigns", "campbell",
                "camps", "campus", "canada", "canadian", "canal", "cancel", "cancelled", "cancer",
                "candidate", "candidates", "candy", "cannabis", "cannon", "cannot", "canon", "cant",
                "capabilities", "capability", "capable", "capacity", "cape", "capital", "caps", "captain",
                "capture", "captured", "carbon", "card", "cards", "care", "cared", "career",
                "careers", "careful", "carefully", "cares", "cargo", "caribbean", "caring", "carl",
                "carlos", "carol", "carolina", "caroline", "carpet", "carried", "carrier", "carriers",
                "carries", "carry", "carrying", "cars", "carter", "cartoon", "case", "cases",
                "cash", "casino", "cast", "casting", "castle", "casual", "catch", "catching",
                "categories", "category", "catherine", "catholic", "cats", "cattle", "caught", "cause",
                "caused", "causes", "causing", "cave", "cease", "ceiling", "celebrate", "celebrated",
                "celebrating", "celebration", "celebrities", "celebrity", "cell", "cells", "cemetery", "census",
                "cent", "center", "centers", "central", "centre", "centres", "cents", "centuries",
                "century", "ceremony", "certain", "certainly", "certificate", "certified", "chain", "chains",
                "chair", "chairman", "chairs", "challenge", "challenged", "challenges", "challenging", "chamber",
                "chambers", "champion", "champions", "championship", "championships", "chance", "chancellor", "chances",
                "change", "changed", "changes", "changing", "channel", "channels", "chaos", "chapel",
                "chapter", "chapters", "character", "characteristics", "characterized", "characters", "charge", "charged",
                "charges", "charging", "charity", "charles", "charlie", "charlotte", "charm", "charming",
                "chart", "charter", "charts", "chase", "chasing", "chat", "cheap", "cheaper",
                "cheat", "cheating", "check", "checked", "checking", "checks", "cheer", "cheers",
                "cheese", "chef", "chelsea", "chemical", "chemicals", "chemistry", "cherry", "chest",
                "chicago", "chick", "chicken", "chief", "chiefs", "child", "childhood", "children",
                "chile", "chill", "china", "chinese", "chip", "chips", "chocolate", "choice",
                "choices", "choose", "choosing", "chose", "chosen", "chris", "christ", "christian",
                "christianity", "christians", "christmas", "christopher", "chronic", "chuck", "church", "churches",
                "cigarette", "cigarettes", "cinema", "circle", "circles", "circuit", "circumstances", "cited",
                "cities", "citizen", "citizens", "citizenship", "city", "civil", "civilian", "civilians",
                "civilization", "claim", "claimed", "claiming", "claims", "clark", "clarke", "class",
                "classes", "classic", "classical", "classification", "classified", "classroom", "clay", "clean",
                "cleaned", "cleaning", "clear", "cleared", "clearing", "clearly", "cleveland", "clever",
                "click", "client", "clients", "cliff", "climate", "climb", "climbing", "clinic",
                "clinical", "clinton", "clip", "clips", "clock", "close", "closed", "closely",
                "closer", "closest", "closet", "closing", "closure", "cloth", "clothes", "clothing",
                "cloud", "clouds", "club", "clubs", "clue", "coach", "coaches", "coaching",
                "coal", "coalition", "coast", "coastal", "coat", "cock", "code", "codes",
                "coffee", "cognitive", "coin", "coins", "coke", "cold", "cole", "colin",
                "collaboration", "collapse", "collar", "colleagues", "collect", "collected", "collecting", "collection",
                "collections", "collective", "college", "colleges", "collins", "colonel", "colonial", "colony",
                "color", "colorado", "colored", "colors", "colour", "colours", "columbia", "columbus",
                "column", "combat", "combination", "combine", "combined", "come", "comeback", "comedy",
                "comes", "comfort", "comfortable", "comic", "comics", "coming", "command", "commander",
                "comment", "commentary", "commented", "comments", "commerce", "commercial", "commission", "commissioner",
                "commit", "commitment", "committed", "committee", "common", "commonly", "commons", "commonwealth",
                "communicate", "communication", "communications", "communist", "communities", "community", "compact", "companies",
                "companion", "company", "comparable", "compare", "compared", "comparing", "comparison", "compensation",
                "compete", "competing", "competition", "competitive", "competitors", "complain", "complaining", "complaint",
                "complaints", "complete", "completed", "completely", "completing", "completion", "complex", "complexity",
                "compliance", "complicated", "comply", "component", "components", "composed", "composition", "compound",
                "comprehensive", "compromise", "computer", "computers", "computing", "concentrate", "concentration", "concept",
                "concepts", "concern", "concerned", "concerning", "concerns", "concert", "concluded", "conclusion",
                "concrete", "condition", "conditions", "conduct", "conducted", "conducting", "conference", "conferences",
                "confidence", "confident", "confirm", "confirmation", "confirmed", "conflict", "conflicts", "confused",
                "confusing", "confusion", "congrats", "congratulations", "congress", "congressional", "connect", "connected",
                "connecticut", "connecting", "connection", "connections", "conscious", "consciousness", "consecutive", "consensus",
                "consent", "consequence", "consequences", "conservation", "conservative", "conservatives", "consider", "considerable",
                "consideration", "considered", "considering", "considers", "consistent", "consistently", "consisting", "consists",
                "console", "conspiracy", "constant", "constantly", "constitution", "constitutional", "constructed", "construction",
                "consultant", "consultation", "consulting", "consumer", "consumers", "consumption", "contact", "contacts",
                "contain", "contained", "container", "containing", "contains", "contemporary", "content", "contents",
                "contest", "context", "continent", "continental", "continue", "continued", "continues", "continuing",
                "continuous", "contract", "contractor", "contractors", "contracts", "contrary", "contrast", "contribute",
                "contributed", "contributing", "contribution", "contributions", "control", "controlled", "controller", "controlling",
                "controls", "controversial", "controversy", "convenience", "convenient", "convention", "conventional", "conversation",
                "conversations", "conversion", "convert", "converted", "convicted", "conviction", "convince", "convinced",
                "cook", "cooked", "cookie", "cookies", "cooking", "cool", "cooling", "cooper",
                "cooperation", "cope", "copies", "copper", "cops", "copy", "copyright", "core",
                "corn", "corner", "corners", "corp", "corporate", "corporation", "corporations", "corps",
                "correct", "correctly", "corresponding", "corrupt", "corruption", "cost", "costa", "costs",
                "costume", "cotton", "couch", "could", "council", "counsel", "count", "counted",
                "counter", "counties", "counting", "countries", "country", "counts", "county", "couple",
                "couples", "courage", "course", "courses", "court", "courtesy", "courts", "cousin",
                "cover", "coverage", "covered", "covering", "covers", "cowboys", "crack", "craft",
                "craig", "crap", "crash", "crashed", "crazy", "cream", "create", "created",
                "creates", "creating", "creation", "creative", "creator", "creature", "creatures", "credit",
                "credits", "creek", "creepy", "crew", "cricket", "cried", "crime", "crimes",
                "criminal", "criminals", "crisis", "criteria", "critical", "criticism", "critics", "crop",
                "crops", "cross", "crossed", "crossing", "crowd", "crown", "crucial", "cruel",
                "cruise", "crush", "crushed", "cruz", "crying", "crystal", "cuba", "cult",
                "cultural", "culture", "cultures", "cups", "cure", "curious", "currency", "current",
                "currently", "curriculum", "curse", "curve", "custody", "custom", "customer", "customers",
                "customs", "cute", "cuts", "cutting", "cycle", "cycling", "daddy", "daily",
                "dairy", "dakota", "dallas", "damage", "damaged", "damages", "damn", "dance",
                "dancing", "danger", "dangerous", "daniel", "danny", "dare", "dark", "darkness",
                "darling", "data", "database", "date", "dated", "dates", "dating", "daughter",
                "daughters", "dave", "david", "davis", "dawn", "days", "dead", "deadline",
                "deadly", "deal", "dealer", "dealing", "deals", "dealt", "dean", "dear",
                "death", "deaths", "debate", "debt", "debut", "decade", "decades", "december",
                "decent", "decide", "decided", "decides", "deciding", "decision", "decisions", "deck",
                "declaration", "declare", "declared", "decline", "declined", "decrease", "dedicated", "deemed",
                "deep", "deeper", "deeply", "deer", "default", "defeat", "defeated", "defence",
                "defend", "defender", "defending", "defense", "defensive", "deficit", "define", "defined",
                "definitely", "definition", "degree", "degrees", "delay", "delayed", "delete", "deleted",
                "delhi", "deliberately", "delicious", "delighted", "deliver", "delivered", "delivering", "delivery",
                "delta", "demand", "demanded", "demanding", "demands", "demo", "democracy", "democrat",
                "democratic", "democrats", "demon", "demonstrate", "demonstrated", "demonstration", "denied", "denmark",
                "dennis", "density", "dental", "denver", "deny", "department", "departments", "departure",
                "depend", "dependent", "depending", "depends", "deployed", "deposit", "depressed", "depression",
                "depth", "deputy", "derby", "derived", "describe", "described", "describes", "describing",
                "description", "desert", "deserve", "deserved", "deserves", "design", "designated", "designed",
                "designer", "designers", "designs", "desire", "desired", "desk", "desperate", "despite",
                "destination", "destiny", "destroy", "destroyed", "destroying", "destruction", "detail", "detailed",
                "details", "detect", "detected", "detection", "detective", "determination", "determine", "determined",
                "detroit", "develop", "developed", "developer", "developers", "developing", "development", "developments",
                "device", "devices", "devil", "devoted", "diabetes", "diagnosed", "diagnosis", "dialogue",
                "diameter", "diamond", "diamonds", "diana", "diary", "dick", "dictionary", "didnt",
                "died", "diego", "dies", "diesel", "diet", "differ", "difference", "differences",
                "different", "differently", "difficult", "difficulties", "difficulty", "digital", "dignity", "dimension",
                "dimensional", "dimensions", "dining", "dinner", "diplomatic", "direct", "directed", "direction",
                "directions", "directly", "director", "directors", "dirt", "dirty", "disability", "disabled",
                "disagree", "disappear", "disappeared", "disappointed", "disappointing", "disaster", "disc", "discipline",
                "discount", "discover", "discovered", "discovery", "discrimination", "discuss", "discussed", "discussing",
                "discussion", "discussions", "disease", "diseases", "disgusting", "dish", "dishes", "dismissed",
                "disney", "disorder", "disorders", "display", "displayed", "displays", "dispute", "distance",
                "distant", "distinct", "distinction", "distinguished", "distributed", "distribution", "district", "districts",
                "disturbing", "dive", "diverse", "diversity", "divide", "divided", "divine", "diving",
                "division", "divisions", "divorce", "divorced", "doctor", "doctors", "document", "documentary",
                "documents", "does", "doesnt", "dogs", "doing", "doll", "dollar", "dollars",
                "domain", "domestic", "dominant", "dominated", "donald", "donate", "donated", "donations",
                "done", "dont", "door", "doors", "dose", "double", "doubt", "doug",
                "douglas", "down", "download", "downtown", "dozen", "dozens", "draft", "drag",
                "dragon", "dragons", "drain", "drama", "dramatic", "draw", "drawing", "drawings",
                "drawn", "draws", "dream", "dreams", "dress", "dressed", "dresses", "dressing",
                "drew", "dried", "drill", "drink", "drinking", "drinks", "drive", "driven",
                "driver", "drivers", "drives", "driving", "drop", "dropped", "dropping", "drops",
                "drove", "drug", "drugs", "drum", "drunk", "dual", "dubai", "dublin",
                "duck", "dude", "duke", "dumb", "dump", "duncan", "duration", "during",
                "dust", "dutch", "duties", "duty", "dying", "dylan", "dynamic", "dynamics",
                "each", "eager", "eagle", "eagles", "earl", "earlier", "earliest", "early",
                "earn", "earned", "earning", "earnings", "ears", "earth", "earthquake", "ease",
                "easier", "easily", "east", "easter", "eastern", "easy", "eaten", "eating",
                "ebay", "echo", "economic", "economics", "economy", "eddie", "edge", "edges",
                "edinburgh", "edit", "edited", "editing", "edition", "editor", "editorial", "editors",
                "educated", "education", "educational", "edward", "edwards", "effect", "effective", "effectively",
                "effectiveness", "effects", "efficiency", "efficient", "effort", "efforts", "eggs", "egypt",
                "egyptian", "eight", "eighth", "either", "elaborate", "elder", "elderly", "elect",
                "elected", "election", "elections", "electoral", "electric", "electrical", "electricity", "electronic",
                "electronics", "element", "elementary", "elements", "elephant", "elevated", "eleven", "eligible",
                "eliminate", "eliminated", "elite", "elizabeth", "ellen", "else", "elsewhere", "email",
                "emails", "embarrassed", "embarrassing", "embassy", "embrace", "emerged", "emergency", "emerging",
                "emily", "emissions", "emma", "emotion", "emotional", "emotionally", "emotions", "emperor",
                "emphasis", "empire", "employed", "employee", "employees", "employer", "employers", "employment",
                "empty", "enable", "enabled", "encounter", "encountered", "encourage", "encouraged", "encouraging",
                "ended", "ending", "endless", "ends", "enemies", "enemy", "energy", "enforcement",
                "engage", "engaged", "engagement", "engaging", "engine", "engineer", "engineering", "engineers",
                "engines", "england", "english", "enhance", "enhanced", "enjoy", "enjoyed", "enjoying",
                "enormous", "enough", "ensure", "enter", "entered", "entering", "enterprise", "entertaining",
                "entertainment", "enthusiasm", "entire", "entirely", "entitled", "entity", "entrance", "entries",
                "entry", "environment", "environmental", "environments", "epic", "episode", "episodes", "equal",
                "equality", "equally", "equation", "equipment", "equipped", "equity", "equivalent", "eric",
                "error", "errors", "escape", "escaped", "especially", "essay", "essays", "essence",
                "essential", "essentially", "establish", "established", "establishing", "establishment", "estate", "estimate",
                "estimated", "estimates", "eternal", "ethical", "ethics", "ethnic", "euro", "europe",
                "european", "evaluation", "evans", "even", "evening", "event", "events", "eventually",
                "ever", "every", "everybody", "everyday", "everyone", "everything", "everywhere", "evidence",
                "evil", "evolution", "evolved", "exact", "exactly", "exam", "examination", "examine",
                "examined", "example", "examples", "excellence", "excellent", "except", "exception", "exceptional",
                "excess", "excessive", "exchange", "excited", "excitement", "exciting", "exclusive", "exclusively",
                "excuse", "executed", "execution", "executive", "executives", "exercise", "exercises", "exhausted",
                "exhibit", "exhibition", "exist", "existed", "existence", "existing", "exists", "exit",
                "expand", "expanded", "expanding", "expansion", "expect", "expectations", "expected", "expecting",
                "expense", "expenses", "expensive", "experience", "experienced", "experiences", "experiencing", "experiment",
                "experimental", "experiments", "expert", "expertise", "experts", "explain", "explained", "explaining",
                "explains", "explanation", "exploration", "explore", "exploring", "explosion", "export", "exports",
                "exposed", "exposure", "express", "expressed", "expression", "extend", "extended", "extending",
                "extension", "extensive", "extent", "external", "extra", "extract", "extraordinary", "extreme",
                "extremely", "eyes", "fabric", "fabulous", "face", "facebook", "faced", "faces",
                "facial", "facilities", "facility", "facing", "fact", "factor", "factors", "factory",
                "facts", "faculty", "fail", "failed", "failing", "fails", "failure", "fair",
                "fairly", "fairy", "faith", "faithful", "fake", "fall", "fallen", "falling",
                "falls", "false", "fame", "familiar", "families", "family", "famous", "fancy",
                "fans", "fantastic", "fantasy", "farm", "farmer", "farmers", "farming", "farms",
                "fascinating", "fashion", "fast", "faster", "fastest", "fatal", "fate", "father",
                "fathers", "fault", "favor", "favorite", "favorites", "favour", "favourite", "fear",
                "fears", "feature", "featured", "features", "featuring", "february", "federal", "federation",
                "feed", "feedback", "feeding", "feel", "feeling", "feelings", "feels", "fees",
                "feet", "fell", "fellow", "felt", "female", "females", "feminist", "fence",
                "ferry", "festival", "fever", "fewer", "fiber", "fiction", "field", "fields",
                "fifa", "fifteen", "fifth", "fifty", "fight", "fighter", "fighters", "fighting",
                "fights", "figure", "figured", "figures", "file", "filed", "files", "filing",
                "fill", "filled", "filling", "film", "filmed", "filming", "films", "filter",
                "final", "finale", "finally", "finals", "finance", "financial", "financing", "find",
                "finding", "findings", "finds", "fine", "finest", "finger", "fingers", "finish",
                "finished", "finishing", "fire", "fired", "fires", "firing", "firm", "firms",
                "first", "fiscal", "fish", "fisher", "fishing", "fitness", "fits", "fitted",
                "fitting", "five", "fixed", "fixing", "flag", "flags", "flame", "flames",
                "flash", "flat", "flavor", "fled", "fleet", "flesh", "flew", "flexibility",
                "flexible", "flies", "flight", "flights", "flip", "floating", "flood", "floor",
                "floors", "florida", "flow", "flower", "flowers", "flowing", "flows", "fluid",
                "flying", "focus", "focused", "focuses", "focusing", "fold", "folk", "folks",
                "follow", "followed", "followers", "following", "follows", "food", "foods", "fool",
                "foot", "footage", "football", "force", "forced", "forces", "forcing", "ford",
                "forecast", "foreign", "forest", "forests", "forever", "forget", "forgive", "forgot",
                "forgotten", "fork", "form", "formal", "format", "formation", "formed", "former",
                "formerly", "forming", "forms", "formula", "fort", "forth", "fortune", "forty",
                "forum", "forward", "foster", "fought", "foul", "found", "foundation", "founded",
                "founder", "four", "fourth", "frame", "framework", "france", "franchise", "francis",
                "francisco", "frank", "franklin", "frankly", "fraud", "freak", "freaking", "fred",
                "free", "freedom", "freely", "freeze", "freezing", "french", "frequency", "frequent",
                "frequently", "fresh", "friday", "fridge", "fried", "friend", "friendly", "friends",
                "friendship", "from", "front", "frozen", "fruit", "fruits", "frustrated", "fuck",
                "fucked", "fuckin", "fucking", "fuel", "full", "fully", "function", "functional",
                "functioning", "functions", "fund", "fundamental", "funded", "funding", "funds", "funeral",
                "funny", "furniture", "further", "furthermore", "fusion", "future", "gain", "gained",
                "gaining", "gains", "galaxy", "gallery", "gambling", "game", "games", "gaming",
                "gang", "garage", "garbage", "garden", "gardens", "gary", "gate", "gates",
                "gather", "gathered", "gathering", "gauge", "gave", "gear", "gender", "gene",
                "general", "generally", "generate", "generated", "generation", "generations", "generic", "generous",
                "genes", "genetic", "genius", "genre", "gentle", "gentleman", "gentlemen", "gently",
                "genuine", "genuinely", "george", "georgia", "german", "germans", "germany", "gets",
                "getting", "ghost", "giant", "giants", "gift", "gifts", "girl", "girlfriend",
                "girls", "give", "given", "gives", "giving", "glad", "glasgow", "glass",
                "glasses", "global", "globe", "glorious", "glory", "gloves", "goal", "goals",
                "goat", "gods", "goes", "going", "gold", "golden", "golf", "gone",
                "gonna", "good", "goodbye", "goodness", "goods", "google", "gordon", "gorgeous",
                "gospel", "gotta", "gotten", "governance", "governing", "government", "governments", "governor",
                "grab", "grabbed", "grace", "grade", "grades", "gradually", "graduate", "graduated",
                "graduates", "graduation", "graham", "grain", "grammar", "grand", "grandfather", "grandma",
                "grandmother", "grant", "granted", "grants", "graph", "graphic", "graphics", "grass",
                "grateful", "grave", "gravity", "gray", "great", "greater", "greatest", "greatly",
                "greece", "greek", "green", "greg", "grew", "grey", "grid", "grief",
                "grip", "grocery", "gross", "ground", "grounds", "group", "groups", "grow",
                "growing", "grown", "grows", "growth", "guarantee", "guaranteed", "guard", "guardian",
                "guards", "guess", "guest", "guests", "guidance", "guide", "guided", "guidelines",
                "guilt", "guilty", "guinea", "guitar", "gulf", "guns", "guys", "habit",
                "habits", "hack", "haha", "hair", "half", "halfway", "hall", "halloween",
                "hamilton", "hammer", "hand", "handed", "handful", "handle", "handled", "handling",
                "hands", "handsome", "hang", "hanging", "hannah", "happen", "happened", "happening",
                "happens", "happier", "happily", "happiness", "happy", "harbor", "hard", "harder",
                "hardest", "hardly", "hardware", "harm", "harmony", "harold", "harper", "harris",
                "harrison", "harry", "harsh", "hart", "harvard", "harvest", "harvey", "hate",
                "hated", "hates", "hatred", "have", "haven", "having", "hawaii", "head",
                "headed", "heading", "headline", "headquarters", "heads", "heal", "healing", "health",
                "healthcare", "healthy", "hear", "heard", "hearing", "heart", "hearts", "heat",
                "heated", "heating", "heaven", "heavily", "heavy", "heck", "heels", "height",
                "heights", "held", "helen", "helicopter", "hell", "hello", "helmet", "help",
                "helped", "helpful", "helping", "helps", "hence", "henry", "here", "heritage",
                "hero", "heroes", "herself", "hidden", "hide", "hiding", "high", "higher",
                "highest", "highlight", "highlights", "highly", "highway", "hilarious", "hill", "hillary",
                "hills", "himself", "hint", "hire", "hired", "hiring", "historic", "historical",
                "historically", "history", "hitler", "hits", "hitting", "hockey", "hold", "holder",
                "holders", "holding", "holds", "hole", "holes", "holiday", "holidays", "holland",
                "hollywood", "holmes", "holy", "home", "homeless", "homes", "homework", "honest",
                "honestly", "honey", "hong", "honor", "honored", "honour", "hood", "hook",
                "hope", "hoped", "hopefully", "hopes", "hoping", "horizon", "horn", "horrible",
                "horror", "horse", "horses", "hospital", "hospitals", "host", "hosted", "hostile",
                "hosting", "hosts", "hotel", "hotels", "hour", "hours", "house", "household",
                "households", "houses", "housing", "houston", "howard", "however", "http", "https",
                "hudson", "huge", "hughes", "hull", "human", "humanity", "humans", "humble",
                "humor", "hundred", "hundreds", "hung", "hunger", "hungry", "hunt", "hunter",
                "hunting", "hurricane", "hurry", "hurt", "hurting", "hurts", "husband", "hybrid",
                "hype", "icon", "iconic", "idea", "ideal", "ideas", "identical", "identification",
                "identified", "identify", "identifying", "identity", "ideology", "idiot", "idiots", "ignorance",
                "ignorant", "ignore", "ignored", "ignoring", "illegal", "illinois", "illness", "illustrated",
                "image", "images", "imagination", "imagine", "imagined", "immediate", "immediately", "immigrant",
                "immigrants", "immigration", "immune", "impact", "impacts", "imperial", "implement", "implementation",
                "implemented", "implications", "import", "importance", "important", "importantly", "imposed", "impossible",
                "impressed", "impression", "impressive", "improve", "improved", "improvement", "improvements", "improving",
                "inch", "inches", "incident", "incidents", "include", "included", "includes", "including",
                "inclusion", "income", "incorporated", "increase", "increased", "increases", "increasing", "increasingly",
                "incredible", "incredibly", "indeed", "independence", "independent", "independently", "index", "india",
                "indian", "indiana", "indians", "indicate", "indicated", "indicates", "indicating", "indication",
                "indigenous", "individual", "individuals", "indonesia", "indoor", "induced", "industrial", "industries",
                "industry", "inevitable", "infected", "infection", "infinite", "inflation", "influence", "influenced",
                "influences", "influential", "info", "inform", "information", "informed", "infrastructure", "ingredients",
                "inhabitants", "initial", "initially", "initiative", "initiatives", "injured", "injuries", "injury",
                "inner", "innings", "innocent", "innovation", "innovative", "input", "inquiry", "insane",
                "insects", "inside", "insight", "insist", "insisted", "inspection", "inspector", "inspiration",
                "inspired", "inspiring", "instagram", "install", "installation", "installed", "instance", "instances",
                "instant", "instantly", "instead", "institute", "institution", "institutions", "instruction", "instructions",
                "instructor", "instrument", "instruments", "insurance", "integrated", "integration", "integrity", "intel",
                "intellectual", "intelligence", "intelligent", "intend", "intended", "intense", "intensity", "intensive",
                "intent", "intention", "intentions", "inter", "interact", "interaction", "interactions", "interactive",
                "interest", "interested", "interesting", "interests", "interface", "interior", "intermediate", "internal",
                "international", "internet", "interpretation", "intervention", "interview", "interviews", "intimate", "into",
                "introduce", "introduced", "introducing", "introduction", "invasion", "invented", "invention", "inventory",
                "invest", "invested", "investigate", "investigated", "investigating", "investigation", "investigations", "investing",
                "investment", "investments", "investors", "invisible", "invitation", "invite", "invited", "involve",
                "involved", "involvement", "involves", "involving", "iowa", "ipad", "iphone", "iran",
                "iranian", "iraq", "ireland", "irish", "iron", "isis", "islam", "islamic",
                "island", "islands", "isolated", "isolation", "israel", "israeli", "issue", "issued",
                "issues", "italian", "italy", "item", "items", "itself", "jack", "jacket",
                "jackson", "jacob", "jail", "jake", "james", "jamie", "jane", "january",
                "japan", "japanese", "jason", "jazz", "jealous", "jean", "jeans", "jeff",
                "jennifer", "jeremy", "jerry", "jersey", "jerusalem", "jesse", "jessica", "jesus",
                "jets", "jewelry", "jewish", "jews", "jimmy", "jobs", "john", "johnny",
                "johnson", "join", "joined", "joining", "joins", "joint", "joke", "jokes",
                "joking", "jonathan", "jones", "jordan", "jose", "joseph", "josh", "journal",
                "journalism", "journalist", "journalists", "journey", "juan", "judge", "judges", "judging",
                "judgment", "judicial", "juice", "julia", "julie", "july", "jump", "jumped",
                "jumping", "june", "jungle", "junior", "junk", "jurisdiction", "jury", "just",
                "justice", "justified", "justify", "justin", "kansas", "karen", "kate", "keen",
                "keep", "keeper", "keeping", "keeps", "keith", "kelly", "kennedy", "kenny",
                "kent", "kentucky", "kenya", "kept", "kevin", "keyboard", "keys", "khan",
                "kick", "kicked", "kicking", "kicks", "kidding", "kidney", "kids", "kill",
                "killed", "killer", "killing", "kills", "kind", "kinda", "kinds", "king",
                "kingdom", "kings", "kiss", "kitchen", "knee", "knees", "knew", "knife",
                "knight", "knights", "knock", "knocked", "know", "knowing", "knowledge", "known",
                "knows", "kong", "korea", "korean", "kyle", "label", "labels", "labor",
                "laboratory", "labour", "lack", "lacking", "ladder", "ladies", "lady", "laid",
                "lake", "lakes", "lamb", "lamp", "land", "landed", "landing", "lands",
                "landscape", "lane", "language", "languages", "laptop", "large", "largely", "larger",
                "largest", "larry", "laser", "last", "lasted", "lasting", "late", "lately",
                "later", "latest", "latin", "latter", "laugh", "laughed", "laughing", "laughs",
                "laughter", "launch", "launched", "laundry", "laura", "lawrence", "laws", "lawsuit",
                "lawyer", "lawyers", "layer", "layers", "laying", "layout", "lazy", "lead",
                "leader", "leaders", "leadership", "leading", "leads", "leaf", "league", "leak",
                "lean", "learn", "learned", "learning", "lease", "least", "leather", "leave",
                "leaves", "leaving", "lecture", "left", "legacy", "legal", "legally", "legend",
                "legendary", "legends", "legislation", "legislative", "legislature", "legitimate", "legs", "lemon",
                "length", "lens", "lesbian", "less", "lesser", "lesson", "lessons", "lets",
                "letter", "letters", "letting", "level", "levels", "lewis", "lgbt", "liability",
                "liar", "liberal", "liberals", "liberty", "libraries", "library", "license", "licensed",
                "lied", "lies", "lieutenant", "life", "lifestyle", "lifetime", "lift", "lifted",
                "lifting", "light", "lighter", "lighting", "lightning", "lights", "like", "liked",
                "likely", "likes", "likewise", "limit", "limitations", "limited", "limits", "lincoln",
                "linda", "line", "linear", "lined", "lines", "lineup", "link", "linked",
                "links", "lion", "lions", "lips", "liquid", "lisa", "list", "listed",
                "listen", "listened", "listening", "listing", "lists", "literally", "literary", "literature",
                "little", "live", "lived", "liver", "liverpool", "lives", "living", "lloyd",
                "lmao", "load", "loaded", "loading", "loads", "loan", "loans", "lobby",
                "local", "locally", "located", "location", "locations", "lock", "locked", "lodge",
                "logan", "logic", "logical", "logo", "london", "lonely", "long", "longer",
                "longest", "look", "looked", "looking", "looks", "loop", "loose", "lord",
                "lords", "lose", "loses", "losing", "loss", "losses", "lost", "lots",
                "loud", "louis", "louisiana", "love", "loved", "lovely", "lover", "lovers",
                "loves", "loving", "lower", "lowest", "loyal", "loyalty", "lucas", "luck",
                "lucky", "lucy", "luke", "lunch", "lung", "luxury", "lying", "lyrics",
                "machine", "machinery", "machines", "made", "madison", "madrid", "magazine", "magazines",
                "magic", "magical", "magnetic", "magnificent", "mail", "main", "maine", "mainly",
                "mainstream", "maintain", "maintained", "maintaining", "maintenance", "majesty", "major", "majority",
                "make", "maker", "makers", "makes", "makeup", "making", "malaysia", "male",
                "males", "mall", "mama", "manage", "managed", "management", "manager", "managers",
                "managing", "manchester", "mandatory", "manhattan", "manner", "mansion", "manual", "manufactured",
                "manufacturer", "manufacturers", "manufacturing", "many", "maps", "marathon", "marc", "march",
                "marcus", "margaret", "margin", "maria", "marie", "marijuana", "marine", "mario",
                "mark", "marked", "market", "marketing", "markets", "marks", "marriage", "married",
                "marry", "mars", "marshall", "martial", "martin", "marvel", "mary", "maryland",
                "mask", "mason", "mass", "massachusetts", "massive", "master", "masters", "match",
                "matches", "matching", "mate", "material", "materials", "math", "mathematical", "mathematics",
                "matrix", "matt", "matter", "matters", "matthew", "mature", "maximum", "maybe",
                "mayor", "meal", "meals", "mean", "meaning", "meaningful", "means", "meant",
                "meantime", "meanwhile", "measure", "measured", "measurement", "measurements", "measures", "measuring",
                "meat", "mechanical", "mechanics", "mechanism", "mechanisms", "medal", "media", "median",
                "medical", "medication", "medicine", "medieval", "medium", "meet", "meeting", "meetings",
                "meets", "mega", "melbourne", "member", "members", "membership", "memorial", "memories",
                "memory", "mental", "mentally", "mention", "mentioned", "mentions", "menu", "merchant",
                "mercury", "mercy", "mere", "merely", "merit", "mess", "message", "messages",
                "metal", "meter", "meters", "method", "methods", "metres", "metro", "metropolitan",
                "mexican", "mexico", "miami", "mice", "michael", "michelle", "michigan", "micro",
                "microsoft", "middle", "midnight", "might", "mighty", "migration", "mike", "milan",
                "mild", "mile", "miles", "military", "milk", "mill", "miller", "million",
                "millions", "mills", "mind", "minded", "minds", "mine", "mineral", "mines",
                "mini", "minimal", "minimum", "mining", "minister", "ministers", "ministry", "minnesota",
                "minor", "minority", "minute", "minutes", "miracle", "mirror", "miserable", "miss",
                "missed", "missile", "missing", "mission", "missions", "mississippi", "missouri", "mistake",
                "mistakes", "mitchell", "mixed", "mixing", "mixture", "mobile", "mobility", "mode",
                "model", "models", "moderate", "modern", "modest", "modified", "molecular", "moment",
                "moments", "momentum", "monday", "monetary", "money", "monitor", "monitoring", "monkey",
                "monster", "monsters", "montana", "month", "monthly", "months", "montreal", "mood",
                "moon", "moore", "moral", "more", "moreover", "morgan", "morning", "morris",
                "mortgage", "moscow", "most", "mostly", "mother", "mothers", "motion", "motivated",
                "motivation", "motor", "motorcycle", "motors", "mount", "mountain", "mountains", "mounted",
                "mouse", "mouth", "move", "moved", "movement", "movements", "moves", "movie",
                "movies", "moving", "much", "multi", "multiple", "municipal", "murder", "murdered",
                "murders", "murphy", "murray", "muscle", "muscles", "museum", "music", "musical",
                "musicians", "muslim", "muslims", "must", "mutual", "myself", "mysterious", "mystery",
                "myth", "nail", "nails", "naked", "name", "named", "namely", "names",
                "nancy", "narrative", "narrow", "nasa", "nasty", "nathan", "nation", "national",
                "nations", "nationwide", "native", "natural", "naturally", "nature", "naval", "navigation",
                "navy", "nazi", "near", "nearby", "nearest", "nearly", "necessarily", "necessary",
                "neck", "need", "needed", "needing", "needs", "negative", "negotiations", "neighbor",
                "neighborhood", "neighbors", "neil", "neither", "nelson", "nerve", "nerves", "nervous",
                "nest", "netflix", "netherlands", "network", "networks", "neutral", "nevada", "never",
                "nevertheless", "newcastle", "newest", "newly", "news", "newspaper", "newspapers", "next",
                "nice", "nicely", "nicholas", "nick", "nigeria", "nigerian", "night", "nightmare",
                "nights", "nine", "nintendo", "ninth", "noble", "nobody", "noise", "nominated",
                "nomination", "none", "nonetheless", "nonsense", "noon", "nope", "normal", "normally",
                "norman", "north", "northeast", "northern", "northwest", "norway", "nose", "notable",
                "note", "noted", "notes", "nothing", "notice", "noticed", "notion", "novel",
                "novels", "november", "nowadays", "nowhere", "nuclear", "number", "numbers", "numerous",
                "nurse", "nurses", "nursing", "nutrition", "nuts", "obama", "object", "objective",
                "objectives", "objects", "obligation", "observation", "observations", "observe", "observed", "obsessed",
                "obtain", "obtained", "obvious", "obviously", "occasion", "occasional", "occasionally", "occasions",
                "occupation", "occupied", "occur", "occurred", "occurs", "ocean", "october", "odds",
                "offense", "offensive", "offer", "offered", "offering", "offers", "office", "officer",
                "officers", "offices", "official", "officially", "officials", "often", "ohio", "okay",
                "oklahoma", "older", "oldest", "olive", "oliver", "olympic", "olympics", "once",
                "ones", "ongoing", "online", "only", "ontario", "onto", "open", "opened",
                "opening", "openly", "opens", "opera", "operate", "operated", "operates", "operating",
                "operation", "operational", "operations", "operator", "operators", "opinion", "opinions", "opponent",
                "opponents", "opportunities", "opportunity", "opposed", "opposing", "opposite", "opposition", "optical",
                "option", "options", "oral", "orange", "orbit", "order", "ordered", "ordering",
                "orders", "ordinary", "oregon", "organ", "organic", "organisation", "organisations", "organised",
                "organization", "organizations", "organized", "orientation", "oriented", "origin", "original", "originally",
                "origins", "orlando", "orleans", "oscar", "other", "others", "otherwise", "ought",
                "ours", "ourselves", "outcome", "outcomes", "outdoor", "outer", "outfit", "outlook",
                "output", "outside", "outstanding", "outta", "oven", "over", "overall", "overcome",
                "overnight", "overseas", "overview", "overwhelming", "owen", "owned", "owner", "owners",
                "ownership", "owns", "oxford", "oxygen", "pace", "pacific", "pack", "package",
                "packed", "packing", "page", "pages", "paid", "pain", "painful", "paint",
                "painted", "painting", "paintings", "pair", "pairs", "pakistan", "palace", "pale",
                "palestinian", "palm", "palmer", "panel", "panels", "panic", "pants", "paper",
                "papers", "parade", "paradise", "paragraph", "parallel", "parent", "parents", "paris",
                "parish", "park", "parker", "parking", "parks", "parliament", "parliamentary", "part",
                "partial", "partially", "participants", "participate", "participating", "participation", "particles", "particular",
                "particularly", "parties", "partly", "partner", "partners", "partnership", "parts", "party",
                "pass", "passage", "passed", "passenger", "passengers", "passes", "passing", "passion",
                "passionate", "passive", "passport", "password", "past", "pastor", "patch", "patent",
                "path", "pathetic", "paths", "patience", "patient", "patients", "patrick", "patrol",
                "pattern", "patterns", "paul", "paying", "payment", "payments", "pays", "peace",
                "peaceful", "peak", "pearl", "peer", "peers", "penalties", "penalty", "pending",
                "peninsula", "penis", "pennsylvania", "penny", "pension", "people", "peoples", "pepper",
                "perceived", "percent", "percentage", "perception", "perfect", "perfectly", "perform", "performance",
                "performances", "performed", "performing", "perhaps", "period", "periods", "permanent", "permanently",
                "permission", "permit", "permitted", "perry", "person", "personal", "personality", "personally",
                "personnel", "persons", "perspective", "pete", "peter", "petition", "pets", "petty",
                "phase", "phenomenon", "phil", "philadelphia", "philip", "philippines", "phillips", "philosophy",
                "phoenix", "phone", "phones", "photo", "photograph", "photographer", "photographs", "photography",
                "photos", "phrase", "physical", "physically", "physician", "physicians", "physics", "piano",
                "pick", "picked", "picking", "picks", "pics", "picture", "pictures", "piece",
                "pieces", "pile", "pill", "pills", "pilot", "pilots", "pine", "pink",
                "pipe", "pipeline", "piss", "pissed", "pitch", "pittsburgh", "pizza", "place",
                "placed", "placement", "places", "placing", "plain", "plan", "plane", "planes",
                "planet", "planets", "planned", "planning", "plans", "plant", "planted", "plants",
                "plastic", "plate", "plates", "platform", "platforms", "platinum", "play", "played",
                "player", "players", "playing", "playoff", "playoffs", "plays", "pleasant", "please",
                "pleased", "pleasure", "plenty", "plot", "plug", "plus", "pocket", "pockets",
                "podcast", "poem", "poems", "poet", "poetry", "point", "pointed", "pointing",
                "points", "poison", "pokemon", "poland", "pole", "police", "policies", "policy",
                "polish", "political", "politically", "politician", "politicians", "politics", "poll", "polls",
                "pollution", "pool", "poor", "poorly", "pope", "popular", "popularity", "population",
                "populations", "pork", "porn", "port", "portfolio", "portion", "portland", "portrait",
                "ports", "portugal", "portuguese", "pose", "position", "positions", "positive", "possess",
                "possession", "possibilities", "possibility", "possible", "possibly", "post", "posted", "poster",
                "posting", "posts", "potato", "potatoes", "potential", "potentially", "potter", "pound",
                "pounds", "pour", "poverty", "powder", "power", "powered", "powerful", "powers",
                "practical", "practically", "practice", "practices", "praise", "praised", "pray", "prayer",
                "prayers", "precious", "precise", "precisely", "predict", "predicted", "prefer", "preference",
                "preferred", "pregnancy", "pregnant", "preliminary", "premier", "premiere", "premises", "premium",
                "preparation", "prepare", "prepared", "preparing", "presence", "present", "presentation", "presented",
                "presenting", "presents", "preserve", "preserved", "presidency", "president", "presidential", "presidents",
                "press", "pressed", "pressing", "pressure", "presumably", "pretend", "pretending", "pretty",
                "prevent", "prevented", "preventing", "prevention", "previous", "previously", "price", "prices",
                "pricing", "pride", "priest", "primarily", "primary", "prime", "prince", "princess",
                "principal", "principle", "principles", "print", "printed", "printing", "prints", "prior",
                "priorities", "priority", "prison", "prisoner", "prisoners", "privacy", "private", "privilege",
                "prize", "probability", "probably", "problem", "problems", "procedure", "procedures", "proceed",
                "proceedings", "proceeds", "process", "processes", "processing", "produce", "produced", "producer",
                "producers", "produces", "producing", "product", "production", "productive", "productivity", "products",
                "prof", "profession", "professional", "professionals", "professor", "profile", "profit", "profitable",
                "profits", "program", "programme", "programmes", "programming", "programs", "progress", "progressive",
                "project", "projects", "prominent", "promise", "promised", "promises", "promising", "promote",
                "promoted", "promoting", "promotion", "pronounced", "proof", "propaganda", "proper", "properly",
                "properties", "property", "proportion", "proposal", "proposals", "proposed", "prosecution", "prospect",
                "prospects", "protect", "protected", "protecting", "protection", "protective", "protein", "proteins",
                "protest", "protests", "protocol", "proud", "prove", "proved", "proven", "proves",
                "provide", "provided", "provider", "providers", "provides", "providing", "province", "provinces",
                "provincial", "provision", "provisions", "psychological", "psychology", "public", "publication", "publications",
                "publicly", "publish", "published", "publisher", "publishers", "publishing", "puerto", "pull",
                "pulled", "pulling", "pulse", "pump", "punch", "punishment", "punk", "pupils",
                "purchase", "purchased", "purchases", "purchasing", "pure", "purely", "purple", "purpose",
                "purposes", "pursue", "pursuit", "push", "pushed", "pushing", "pussy", "putin",
                "puts", "putting", "qualified", "qualify", "qualities", "quality", "quantity", "quantum",
                "quarter", "quarters", "queen", "queens", "queensland", "quest", "question", "questioned",
                "questioning", "questions", "quick", "quickly", "quiet", "quietly", "quit", "quite",
                "quote", "quoted", "quotes", "rabbit", "race", "races", "rachel", "racial",
                "racing", "racism", "racist", "radar", "radiation", "radical", "radio", "rage",
                "raid", "rail", "railroad", "railway", "rain", "rainbow", "raise", "raised",
                "raises", "raising", "rally", "ralph", "ranch", "random", "range", "rangers",
                "ranges", "ranging", "rank", "ranked", "ranking", "ranks", "rape", "raped",
                "rapid", "rapidly", "rare", "rarely", "rate", "rated", "rates", "rather",
                "rating", "ratings", "ratio", "rational", "rats", "rays", "reach", "reached",
                "reaches", "reaching", "react", "reaction", "reactions", "read", "reader", "readers",
                "reading", "reads", "ready", "real", "realise", "realised", "realistic", "reality",
                "realize", "realized", "really", "realm", "rear", "reason", "reasonable", "reasonably",
                "reasons", "rebel", "rebels", "recall", "receive", "received", "receiver", "receives",
                "receiving", "recent", "recently", "reception", "recipe", "recognised", "recognition", "recognize",
                "recognized", "recommend", "recommendation", "recommendations", "recommended", "record", "recorded", "recording",
                "records", "recover", "recovered", "recovery", "recreation", "reduce", "reduced", "reducing",
                "reduction", "reed", "refer", "reference", "references", "referred", "referring", "refers",
                "reflect", "reflected", "reflection", "reflects", "reform", "reforms", "refugees", "refuse",
                "refused", "refuses", "regard", "regarded", "regarding", "regardless", "regards", "regime",
                "region", "regional", "regions", "register", "registered", "registration", "regret", "regular",
                "regularly", "regulation", "regulations", "regulatory", "reign", "reject", "rejected", "relate",
                "related", "relating", "relation", "relations", "relationship", "relationships", "relative", "relatively",
                "relatives", "relax", "release", "released", "releases", "releasing", "relevant", "reliable",
                "relief", "religion", "religious", "rely", "remain", "remainder", "remained", "remaining",
                "remains", "remarkable", "remarks", "remember", "remembered", "remind", "reminded", "reminder",
                "reminds", "remote", "removal", "remove", "removed", "removing", "renewed", "rent",
                "rental", "repair", "repairs", "repeat", "repeated", "repeatedly", "replace", "replaced",
                "replacement", "replacing", "replied", "reply", "report", "reported", "reportedly", "reporter",
                "reporting", "reports", "represent", "representation", "representative", "representatives", "represented", "representing",
                "represents", "republic", "republican", "republicans", "reputation", "request", "requested", "requests",
                "require", "required", "requirement", "requirements", "requires", "requiring", "rescue", "research",
                "researchers", "reserve", "reserved", "reserves", "residence", "resident", "residential", "residents",
                "resignation", "resigned", "resist", "resistance", "resolution", "resolve", "resolved", "resort",
                "resource", "resources", "respect", "respected", "respective", "respectively", "respond", "responded",
                "responding", "response", "responses", "responsibilities", "responsibility", "responsible", "rest", "restaurant",
                "restaurants", "restoration", "restore", "restored", "restricted", "restrictions", "result", "resulted",
                "resulting", "results", "resume", "retail", "retain", "retained", "retire", "retired",
                "retirement", "retreat", "return", "returned", "returning", "returns", "reveal", "revealed",
                "reveals", "revenge", "revenue", "revenues", "reverse", "review", "reviewed", "reviews",
                "revised", "revolution", "revolutionary", "reward", "rewards", "rhythm", "rice", "rich",
                "richard", "richmond", "rick", "ride", "rider", "rides", "ridge", "ridiculous",
                "riding", "rifle", "right", "rights", "ring", "rings", "riot", "ripped",
                "rise", "rises", "rising", "risk", "risks", "ritual", "rival", "river",
                "rivers", "road", "roads", "robert", "roberts", "robin", "robinson", "robot",
                "rock", "rocket", "rocks", "rocky", "roger", "rogers", "role", "roles",
                "roll", "rolled", "rolling", "rolls", "roman", "romance", "romantic", "rome",
                "roof", "rookie", "room", "rooms", "root", "roots", "rope", "rose",
                "ross", "roster", "rotation", "rough", "roughly", "round", "rounds", "route",
                "routes", "routine", "royal", "rubber", "rude", "rugby", "ruin", "ruined",
                "ruins", "rule", "ruled", "rules", "ruling", "runner", "running", "runs",
                "rural", "rush", "russell", "russia", "russian", "russians", "ruth", "ryan",
                "sacred", "sacrifice", "sadly", "safe", "safely", "safer", "safety", "said",
                "saint", "saints", "sake", "salad", "salary", "sale", "sales", "salt",
                "same", "sample", "samples", "samsung", "samuel", "sand", "sandwich", "sandy",
                "sang", "santa", "sarah", "satellite", "satisfaction", "satisfied", "saturday", "sauce",
                "saudi", "savage", "save", "saved", "saves", "saving", "savings", "saying",
                "says", "scale", "scan", "scandal", "scare", "scared", "scary", "scenario",
                "scene", "scenes", "schedule", "scheduled", "scheme", "schemes", "scholars", "scholarship",
                "school", "schools", "science", "sciences", "scientific", "scientist", "scientists", "scope",
                "score", "scored", "scores", "scoring", "scotland", "scott", "scottish", "scratch",
                "scream", "screaming", "screen", "screening", "screens", "screw", "script", "seal",
                "sealed", "sean", "search", "searching", "season", "seasons", "seat", "seats",
                "seattle", "second", "secondary", "seconds", "secret", "secretary", "secrets", "section",
                "sections", "sector", "sectors", "secure", "secured", "securities", "security", "seed",
                "seeds", "seeing", "seek", "seeking", "seeks", "seem", "seemed", "seemingly",
                "seems", "seen", "sees", "segment", "seized", "select", "selected", "selection",
                "self", "selfish", "sell", "selling", "sells", "semester", "semi", "senate",
                "senator", "senators", "send", "sending", "sends", "senior", "sense", "sensitive",
                "sent", "sentence", "sentenced", "sentences", "sentiment", "separate", "separated", "separately",
                "separation", "sept", "september", "sequence", "sergeant", "serial", "series", "serious",
                "seriously", "servant", "servants", "serve", "served", "server", "servers", "serves",
                "service", "services", "serving", "session", "sessions", "sets", "setting", "settings",
                "settle", "settled", "settlement", "setup", "seven", "seventh", "several", "severe",
                "severely", "sexual", "sexually", "sexy", "shade", "shadow", "shadows", "shake",
                "shaking", "shall", "shallow", "shame", "shanghai", "shape", "shaped", "share",
                "shared", "shares", "sharing", "shark", "sharp", "shaw", "shed", "sheep",
                "sheet", "sheets", "shelf", "shell", "shelter", "sheriff", "shield", "shift",
                "shine", "ship", "shipped", "shipping", "ships", "shirt", "shirts", "shit",
                "shitty", "shock", "shocked", "shocking", "shoe", "shoes", "shook", "shoot",
                "shooter", "shooting", "shop", "shopping", "shops", "shore", "short", "shorter",
                "shortly", "shorts", "shot", "shots", "should", "shoulder", "shoulders", "shout",
                "show", "showed", "shower", "showing", "shown", "shows", "shut", "siblings",
                "sick", "side", "sides", "sierra", "sight", "sign", "signal", "signals",
                "signature", "signed", "significance", "significant", "significantly", "signing", "signs", "silence",
                "silent", "silk", "silly", "silver", "similar", "similarly", "simon", "simple",
                "simply", "simultaneously", "since", "sing", "singapore", "singer", "singh", "singing",
                "single", "singles", "sink", "sins", "sister", "sisters", "site", "sites",
                "sits", "sitting", "situated", "situation", "situations", "sixth", "size", "sized",
                "sizes", "sketch", "skill", "skilled", "skills", "skin", "skip", "skull",
                "slam", "slave", "slavery", "slaves", "sleep", "sleeping", "slept", "slide",
                "slight", "slightly", "slim", "slip", "slot", "slow", "slowly", "small",
                "smaller", "smart", "smartphone", "smash", "smell", "smells", "smile", "smiling",
                "smith", "smoke", "smoking", "smooth", "snake", "snap", "snow", "soap",
                "soccer", "social", "socialist", "societies", "society", "socks", "soda", "soft",
                "software", "soil", "solar", "sold", "soldier", "soldiers", "sole", "solely",
                "solid", "solo", "solution", "solutions", "solve", "solved", "some", "somebody",
                "someday", "somehow", "someone", "something", "sometime", "sometimes", "somewhat", "somewhere",
                "song", "songs", "sons", "sony", "soon", "sooner", "sophisticated", "sorry",
                "sort", "sorts", "sought", "soul", "souls", "sound", "sounded", "sounds",
                "soup", "source", "sources", "south", "southeast", "southern", "southwest", "soviet",
                "space", "spaces", "spain", "spam", "span", "spanish", "spare", "speak",
                "speaker", "speakers", "speaking", "speaks", "special", "specialist", "specialized", "species",
                "specific", "specifically", "specified", "spectacular", "spectrum", "speech", "speed", "speeds",
                "spell", "spencer", "spend", "spending", "spent", "spider", "spin", "spirit",
                "spirits", "spiritual", "spite", "split", "spoke", "spoken", "sponsor", "sponsored",
                "sport", "sporting", "sports", "spot", "spots", "spotted", "spray", "spread",
                "spreading", "spring", "springs", "squad", "square", "stability", "stable", "stack",
                "stadium", "staff", "stage", "stages", "stairs", "stake", "stakes", "stamp",
                "stan", "stance", "stand", "standard", "standards", "standing", "stands", "stanley",
                "star", "staring", "starring", "stars", "start", "started", "starter", "starting",
                "starts", "state", "stated", "statement", "statements", "states", "static", "stating",
                "station", "stations", "statistical", "statistics", "stats", "statue", "status", "stay",
                "stayed", "staying", "stays", "steady", "steal", "stealing", "steam", "steel",
                "stem", "step", "stephen", "stepped", "steps", "sterling", "steve", "steven",
                "stewart", "stick", "sticking", "sticks", "still", "stock", "stocks", "stole",
                "stolen", "stomach", "stone", "stones", "stood", "stop", "stopped", "stopping",
                "stops", "storage", "store", "stored", "stores", "stories", "storm", "story",
                "straight", "strain", "strange", "stranger", "strangers", "strategic", "strategies", "strategy",
                "streak", "stream", "streaming", "streams", "street", "streets", "strength", "strengthen",
                "stress", "stressed", "stretch", "strict", "strictly", "strike", "strikes", "striking",
                "string", "strings", "strip", "stroke", "strong", "stronger", "strongest", "strongly",
                "struck", "structural", "structure", "structures", "struggle", "struggles", "struggling", "stuart",
                "stuck", "student", "students", "studied", "studies", "studio", "studios", "study",
                "studying", "stuff", "stunning", "stupid", "style", "styles", "subject", "subjects",
                "submission", "submit", "submitted", "subscribe", "subscription", "subsequent", "subsequently", "substance",
                "substantial", "substitute", "subtle", "subway", "succeed", "succeeded", "success", "successful",
                "successfully", "such", "suck", "sucks", "sudden", "suddenly", "suffer", "suffered",
                "suffering", "sufficient", "sugar", "suggest", "suggested", "suggesting", "suggestion", "suggestions",
                "suggests", "suicide", "suit", "suitable", "suite", "suited", "suits", "summary",
                "summer", "summit", "sunday", "sunny", "sunset", "sunshine", "super", "superior",
                "superman", "supplement", "supplied", "supplies", "supply", "support", "supported", "supporter",
                "supporters", "supporting", "supportive", "supports", "suppose", "supposed", "supreme", "sure",
                "surely", "surface", "surfaces", "surgeon", "surgery", "surgical", "surprise", "surprised",
                "surprising", "surprisingly", "surrender", "surrounded", "surrounding", "surveillance", "survey", "survival",
                "survive", "survived", "surviving", "survivors", "susan", "suspect", "suspected", "suspended",
                "suspension", "suspicious", "sustainable", "sustained", "swear", "sweat", "sweden", "swedish",
                "sweet", "swift", "swim", "swimming", "swing", "swiss", "switch", "switched",
                "switching", "switzerland", "sword", "sydney", "symbol", "symbols", "symptoms", "syndrome",
                "syria", "syrian", "system", "systems", "table", "tables", "tablet", "tackle",
                "tactics", "tail", "taiwan", "take", "taken", "takes", "taking", "tale",
                "talent", "talented", "tales", "talk", "talked", "talking", "talks", "tall",
                "tank", "tanks", "tape", "target", "targeted", "targets", "task", "tasks",
                "taste", "tastes", "tattoo", "taught", "taxes", "taxi", "taylor", "teach",
                "teacher", "teachers", "teaching", "team", "teams", "tear", "tears", "tech",
                "technical", "technically", "technique", "techniques", "technological", "technologies", "technology", "teen",
                "teenage", "teenager", "teens", "teeth", "telephone", "television", "tell", "telling",
                "tells", "temperature", "temperatures", "temple", "temporarily", "temporary", "tend", "tender",
                "tends", "tennessee", "tennis", "tension", "tent", "term", "terminal", "terms",
                "terrible", "territories", "territory", "terror", "terrorism", "terrorist", "terrorists", "terry",
                "test", "testament", "tested", "testimony", "testing", "tests", "texas", "text",
                "texts", "thai", "thailand", "than", "thank", "thanks", "thanksgiving", "that",
                "thats", "theater", "theatre", "thee", "theft", "their", "them", "theme",
                "themes", "themselves", "then", "theories", "theory", "therapy", "there", "thereby",
                "therefore", "thermal", "these", "thesis", "they", "thick", "thin", "thing",
                "things", "think", "thinking", "thinks", "third", "thirty", "this", "thomas",
                "thompson", "thoroughly", "those", "thou", "though", "thought", "thoughts", "thousand",
                "thousands", "thread", "threat", "threatened", "threatening", "threats", "three", "threshold",
                "threw", "throat", "throne", "through", "throughout", "throw", "throwing", "thrown",
                "throws", "thunder", "thursday", "thus", "ticket", "tickets", "tide", "tied",
                "tier", "ties", "tiger", "tigers", "tight", "till", "time", "timeline",
                "times", "timing", "tiny", "tips", "tire", "tired", "tires", "tissue",
                "title", "titled", "titles", "tobacco", "today", "todd", "together", "toilet",
                "tokyo", "told", "tolerance", "toll", "tommy", "tomorrow", "tone", "tongue",
                "tonight", "tons", "tony", "took", "tool", "tools", "tooth", "topic",
                "topics", "tops", "torn", "toronto", "torture", "total", "totally", "touch",
                "touched", "touching", "tough", "tour", "tourism", "tourist", "tourists", "tournament",
                "tours", "toward", "towards", "tower", "towers", "town", "towns", "toxic",
                "toys", "trace", "track", "tracking", "tracks", "trade", "traded", "trades",
                "trading", "tradition", "traditional", "traditionally", "traditions", "traffic", "tragedy", "tragic",
                "trail", "trailer", "train", "trained", "trainer", "training", "trains", "trans",
                "transaction", "transactions", "transfer", "transferred", "transfers", "transform", "transformation", "transformed",
                "transit", "transition", "translated", "translation", "transmission", "transport", "transportation", "trap",
                "trapped", "trash", "trauma", "travel", "traveled", "traveling", "travelling", "travels",
                "treasure", "treasury", "treat", "treated", "treating", "treatment", "treatments", "treats",
                "treaty", "tree", "trees", "tremendous", "trend", "trends", "trial", "trials",
                "tribal", "tribe", "tribes", "tribute", "trick", "tricks", "tried", "tries",
                "trigger", "trip", "triple", "trips", "troops", "trophy", "tropical", "trouble",
                "truck", "trucks", "true", "truly", "trump", "trunk", "trust", "trusted",
                "truth", "trying", "tube", "tuesday", "tune", "tunnel", "turkey", "turkish",
                "turn", "turned", "turner", "turning", "turns", "tweet", "tweeted", "twelve",
                "twenty", "twice", "twin", "twins", "twist", "twitter", "tyler", "type",
                "types", "typical", "typically", "ugly", "ukraine", "ukrainian", "ultimate", "ultimately",
                "ultra", "unable", "unbelievable", "uncertainty", "uncle", "uncomfortable", "under", "underground",
                "underlying", "underneath", "understand", "understanding", "understands", "understood", "unemployment", "unexpected",
                "unfair", "unfortunate", "unfortunately", "unhappy", "uniform", "union", "unions", "unique",
                "unit", "united", "units", "unity", "universal", "universe", "universities", "university",
                "unknown", "unless", "unlike", "unlikely", "unlimited", "unnecessary", "until", "unusual",
                "upcoming", "update", "updated", "updates", "upgrade", "upon", "upper", "upset",
                "urban", "urge", "urgent", "usage", "used", "useful", "useless", "user",
                "users", "uses", "using", "usual", "usually", "utah", "utility", "utterly",
                "vacation", "vacuum", "valid", "valley", "valuable", "value", "valued", "values",
                "valve", "vampire", "vancouver", "variable", "variation", "variations", "varied", "variety",
                "various", "vary", "vast", "vegas", "vegetables", "vehicle", "vehicles", "venture",
                "venue", "verdict", "verse", "version", "versions", "versus", "vertical", "very",
                "vessel", "vessels", "veteran", "veterans", "vice", "victim", "victims", "victor",
                "victoria", "victorian", "victory", "video", "videos", "vietnam", "view", "viewed",
                "viewers", "viewing", "views", "villa", "village", "villages", "vincent", "vintage",
                "violation", "violence", "violent", "viral", "virgin", "virginia", "virtual", "virtually",
                "virus", "visa", "visible", "vision", "visit", "visited", "visiting", "visitors",
                "visits", "visual", "vital", "vocal", "voice", "voices", "voltage", "volume",
                "volumes", "volunteer", "volunteers", "vote", "voted", "voter", "voters", "votes",
                "voting", "vulnerable", "wage", "wages", "wait", "waited", "waiting", "wake",
                "wales", "walk", "walked", "walker", "walking", "walks", "wall", "wallace",
                "wallet", "walls", "walter", "wanna", "want", "wanted", "wanting", "wants",
                "ward", "warehouse", "warfare", "warm", "warming", "warn", "warned", "warner",
                "warning", "warrant", "warren", "warrior", "warriors", "wars", "wash", "washed",
                "washing", "washington", "waste", "wasted", "wasting", "watch", "watched", "watches",
                "watching", "water", "waters", "watson", "wave", "waves", "wayne", "ways",
                "weak", "weakness", "wealth", "wealthy", "weapon", "weapons", "wear", "wearing",
                "wears", "weather", "website", "websites", "wedding", "wednesday", "weed", "week",
                "weekend", "weekends", "weekly", "weeks", "weight", "weird", "welcome", "welfare",
                "well", "wells", "welsh", "went", "were", "west", "western", "what",
                "whatever", "wheel", "wheels", "when", "whenever", "where", "whereas", "wherever",
                "whether", "which", "while", "whilst", "white", "whites", "whoever", "whole",
                "whom", "whose", "wide", "widely", "wider", "widespread", "widow", "width",
                "wife", "wild", "wildlife", "will", "william", "williams", "willing", "wilson",
                "wind", "window", "windows", "winds", "wine", "wing", "wings", "winner",
                "winners", "winning", "wins", "winter", "wire", "wireless", "wisconsin", "wisdom",
                "wise", "wish", "wishes", "witch", "with", "withdraw", "withdrawal", "within",
                "without", "witness", "witnessed", "witnesses", "wives", "woke", "wolf", "woman",
                "women", "wonder", "wondered", "wonderful", "wondering", "wont", "wood", "wooden",
                "woods", "word", "words", "wore", "work", "worked", "worker", "workers",
                "working", "workout", "workplace", "works", "workshop", "world", "worlds", "worldwide",
                "worn", "worried", "worries", "worry", "worrying", "worse", "worship", "worst",
                "worth", "worthy", "would", "wound", "wounded", "wounds", "wrap", "wrapped",
                "wrestling", "wright", "write", "writer", "writers", "writes", "writing", "written",
                "wrong", "wrote", "xbox", "yard", "yards", "yeah", "year", "years",
                "yelling", "yellow", "yesterday", "yield", "yoga", "york", "yorkshire", "young",
                "younger", "youngest", "your", "yours", "yourself", "youth", "youtube", "zealand",
                "zero", "zone", "zones",
            }
        if self.metaphor_patterns is None:
            self.metaphor_patterns = [
                r"\bas if\b",
                r"\bthink of\b",
                r"\bimagine\b",
                r"\bpicture\b",
                r"\bgravity\b",
                r"\bpull\b",
                r"\bmomentum\b",
                r"\bturbulence\b",
                r"\bstall\b",
                r"\bflight[- ]envelope\b",
            ]
        if self.subordinators is None:
            # True subordinating conjunctions only: introduce dependent clauses
            # and create genuine clause stacking / structural thickness.
            # Excludes conjunctive adverbs ("however", "thus", "rather", "instead")
            # which are handled by transitions or contrast_markers.
            # Excludes "since" (ambiguous causal/temporal) and "because", "but"
            # (common connective tissue, not reasoning markers).
            self.subordinators = {
                "although", "though", "unless", "whereas", "while",
            }
        if self.contrast_markers is None:
            # Dialectical tension markers: signal two positions held in structural relation.
            # Distinct from transitions (continuation) and subordinators (clause dependency).
            # These indicate the model is working with opposing ideas rather than just
            # connecting or subordinating them. Key signal for dialectical reasoning structure.
            self.contrast_markers = {
                "however", "yet", "nevertheless", "nonetheless", "conversely",
                "on the other hand", "by contrast", "rather than", "but",
            }


# ----------------------------
# Evidence payload
# ----------------------------

@dataclass
class KernelEvidence:
    token_count: int          # surface word tokens (kernel's own tokenizer)
    sentence_count: int
    paragraph_count: int
    bullet_line_count: int
    transitions_count: int
    reference_pronouns_count: int
    content_word_count: int
    stopword_count: int
    adjacent_cosine_mean: float
    technical_hits: int
    metaphor_hits: int
    punctuation_clause_markers: int
    subordinator_hits: int
    contrast_marker_hits: int = 0   # dialectical tension markers (opposing ideas in structural relation)
    segmentation_reliable: bool = True  # False if punctuation density too low to trust Δy/sentence-derived coords.
                                        # v18: documented as measurement ceiling indicator, not a caution notice.
                                        # Concentration of False in complying_under_constraint is structurally
                                        # expected: rv collapse in that regime reduces sentence count, which
                                        # reduces the boundary material this flag depends on. The kernel is not
                                        # malfunctioning when this fires on constrained responses -- it is
                                        # accurately reporting the point where structural compression is most severe.
    rhetorical_state: str = "assertive"  # output mode: assertive | dialectic | revising | complying_under_constraint | formatting_collapse
    # Basis decomposition: traceability layer — what drove each classification
    seg_basis: str = "punct"            # what makes segmentation_reliable: punct | newline | both | none
    contrast_basis: str = "none"        # what drove contrast hits: token | phrase | both | none
    constraint_basis: str = "none"      # what drove complying_under_constraint: sentence_suppression | verbosity | none
                                        # v19: when rhetorical_state == "formatting_collapse", reports
                                        # "severe" | "partial" | "math" instead of "formatting_collapse".
                                        # Downstream consumers: check rhetorical_state for collapse, use
                                        # constraint_basis for collapse subtype.
    # Normalized companion fields (shadow metrics — do not replace canonical kernel values)
    # Safe to use for invariance testing and reviewer challenge; canonical coords unchanged.
    ds_per_100_tokens: float = 0.0        # dₛ raw marker count normalized per 100 word tokens
    contrast_per_100_tokens: float = 0.0  # contrast marker hits normalized per 100 word tokens
    subordinators_per_100_tokens: float = 0.0  # subordinator hits normalized per 100 word tokens
    tiktoken_token_count: int = -1  # BPE token count (cl100k_base); -1 if tiktoken not installed
    # v19 collapse detection fields
    tail_incomplete_flag: bool = False   # heuristic: text ends with incomplete structure
                                         # (unclosed brackets, dangling operators, scaffold-word endings,
                                         #  mid-expression endings). Required for partial and math collapse gates.
    math_expression_tail_flag: bool = False  # heuristic: text ends with incomplete math/code expression
                                              # (open integrals, dangling LaTeX, code without closing structure).
                                              # Must be combined with tail_incomplete_flag for collapse detection —
                                              # standalone math_expression_tail_flag produces false positives on
                                              # complete responses that contain inline math notation.
    collapse_subtype: str = "none"       # "severe" | "partial" | "math" | "none"
                                         # "none" when rhetorical_state != "formatting_collapse"


# ----------------------------
# Kernel computer
# ----------------------------

class LinguisticKernel:
    """Compute kernel + aux metrics, with prompt-family baseline support."""

    def __init__(
        self,
        config: Optional[KernelConfig] = None,
        registry: Optional[PromptFamilyRegistry] = None,
    ) -> None:
        self.cfg = config or KernelConfig()
        self.registry = registry or PromptFamilyRegistry()

    @staticmethod
    def _sigmoid01(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _cosine(a: Counter, b: Counter) -> float:
        if not a or not b:
            return 0.0
        inter = set(a.keys()) & set(b.keys())
        dot = sum(a[t] * b[t] for t in inter)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return dot / (na * nb)

    # ---- primitive metrics ----

    def compute_delta_x(self, sentences: List[str]) -> Tuple[float, Dict[str, Any]]:
        """Δx: weighted centroid of informational weight along sentence index, centered at 0."""
        n = len(sentences)
        if n <= 1:
            return 0.0, {"weights": [1.0], "centroid": 0.5}

        weights: List[float] = []
        positions: List[float] = []

        for i, sent in enumerate(sentences):
            wds = tokenize_words(sent)
            if not wds:
                w = 0.0
            else:
                content = sum(1 for w_ in wds if (w_ not in self.cfg.stopwords and len(w_) >= 4))
                w = content / max(len(wds), 1)
            pos = i / (n - 1)
            weights.append(w)
            positions.append(pos)

        wsum = sum(weights)
        centroid = 0.5 if wsum <= 1e-12 else sum(w * p for w, p in zip(weights, positions)) / wsum
        dx = float(centroid - 0.5)
        return dx, {"weights": weights, "centroid": centroid}

    def compute_rho_r(self, tokens: List[str]) -> Tuple[float, Dict[str, Any]]:
        """ρᵣ: content-token proportion (packing density analog)."""
        if not tokens:
            return 0.0, {"content": 0, "stop": 0}

        stop = sum(1 for t in tokens if t in self.cfg.stopwords)
        content = sum(1 for t in tokens if (t not in self.cfg.stopwords and len(t) >= 4))
        rho = content / max(len(tokens), 1)
        rho = max(0.0, min(1.0, rho))
        return float(rho), {"content": content, "stop": stop}

    def compute_mu(self, tokens: List[str]) -> Tuple[float, Dict[str, Any]]:
        """μ: cohesion in [0,1] (transition + reference density, sigmoid-scaled)."""
        if not tokens:
            return 0.0, {"transitions": 0, "refs": 0}

        transitions = sum(1 for t in tokens if t in self.cfg.transitions)  # set membership, order irrelevant
        refs = sum(1 for t in tokens if t in self.cfg.reference_pronouns)

        t_rate = transitions / len(tokens)
        r_rate = refs / len(tokens)

        raw = (8.0 * t_rate) + (4.0 * r_rate)
        mu = self._sigmoid01(raw - self.cfg.mu_center)
        return float(mu), {"transitions": transitions, "refs": refs, "t_rate": t_rate, "r_rate": r_rate}

    def compute_r_v(self, text: str, tokens: List[str], sentences: List[str]) -> Tuple[float, Dict[str, Any]]:
        """rᵥ: void/breathing-room analog, composite of formatting + rhythm."""
        if not tokens:
            return 0.0, {"paragraphs": 0, "bullets": 0, "short_prop": 0.0, "avg_sent_len": 0.0}

        paras = len(paragraph_blocks(text))
        bullet_lines = sum(1 for ln in split_lines(text) if _BULLET_RE.match(ln))

        sent_lens = [len(tokenize_words(s)) for s in sentences] if sentences else []
        avg_sent_len = sum(sent_lens) / max(len(sent_lens), 1)
        short_prop = sum(1 for L in sent_lens if L <= 10) / max(len(sent_lens), 1)

        tok_n = len(tokens)
        para_rate = paras / max(tok_n / 100.0, 1e-9)
        bullet_rate = bullet_lines / max(tok_n / 100.0, 1e-9)

        para_norm = min(1.0, para_rate / max(self.cfg.rv_paras_per_100_tokens_full, 1e-9))
        bullet_norm = min(1.0, bullet_rate / max(self.cfg.rv_bullets_per_100_tokens_full, 1e-9))
        short_norm = min(1.0, short_prop / max(self.cfg.rv_short_sentence_prop_full, 1e-9))

        airy = self.cfg.rv_airy_avg_sent_len
        dense = self.cfg.rv_dense_avg_sent_len
        if avg_sent_len <= airy:
            inv_len_norm = 1.0
        elif avg_sent_len >= dense:
            inv_len_norm = 0.0
        else:
            inv_len_norm = (dense - avg_sent_len) / (dense - airy)

        rv = (0.30 * para_norm) + (0.25 * bullet_norm) + (0.25 * short_norm) + (0.20 * inv_len_norm)
        rv = max(0.0, min(1.0, rv))

        return float(rv), {
            "paragraphs": paras,
            "bullets": bullet_lines,
            "short_prop": short_prop,
            "avg_sent_len": avg_sent_len,
            "para_norm": para_norm,
            "bullet_norm": bullet_norm,
            "short_norm": short_norm,
            "inv_len_norm": inv_len_norm,
        }

    # ---- aux/analyzers ----

    def compute_delta_y(self, sentences: List[str]) -> Tuple[float, Dict[str, Any]]:
        """Δy (segmentation_strain): average sentence length relative to 15-token baseline.
        
        Honest description: this is a segmentation and punctuation proxy, not a positional
        token variation statistic. It measures how far average sentence length deviates from
        a 15-token neutral baseline. Positive = longer sentences (less segmented). Negative
        = shorter sentences (more fragmented).
        
        Named Δy for coordinate consistency across all published studies and documentation.
        Plain-language alias: segmentation_strain.
        
        Will track: punctuation presence/absence, verbosity constraints, run-on compliance,
        any prompt that suppresses sentence boundaries. Treat large Δy excursions as
        segmentation events, not content complexity signals.
        """
        if not sentences:
            return 0.0, {"avg_sent_len": 0.0, "segmentation_strain": 0.0}
        lens = [len(tokenize_words(s)) for s in sentences]
        avg_len = sum(lens) / max(len(lens), 1)
        dy = (avg_len - 15.0) / 15.0
        return float(dy), {"avg_sent_len": avg_len, "segmentation_strain": round(dy, 4)}

    def compute_theta(self, sentences: List[str]) -> Tuple[float, Dict[str, Any]]:
        """θ: flow instability = 1 - mean adjacent cosine similarity."""
        if len(sentences) <= 1:
            return 0.0, {"adj_cos_mean": 1.0, "sims": []}

        vecs = [Counter(tokenize_words(s)) for s in sentences]
        sims: List[float] = []
        for i in range(len(vecs) - 1):
            sims.append(self._cosine(vecs[i], vecs[i + 1]))

        adj_mean = sum(sims) / max(len(sims), 1)
        theta = 1.0 - max(0.0, min(1.0, adj_mean))
        return float(theta), {"adj_cos_mean": adj_mean, "sims": sims}

    def compute_d_s(self, sentences: List[str]) -> Tuple[float, Dict[str, Any]]:
        """dₛ: structural thickness proxy (punctuation + subordinators per token)."""
        if not sentences:
            return 0.0, {"punct": 0, "subs": 0, "raw": 0.0}

        punct = 0
        subs = 0
        tok_total = 0
        for s in sentences:
            # v17: semicolons/colons weighted 1.5x — near-pure structural markers.
            # Commas weighted 1.0x — conflate clause boundaries with list separators.
            # Reduces dₛ inflation from technical enumeration text.
            commas = s.count(",")
            heavy = s.count(";") + s.count(":")
            punct += commas + int(heavy * 1.5)
            w = tokenize_words(s)
            tok_total += len(w)
            subs += sum(1 for t in w if t in self.cfg.subordinators)

        if tok_total <= 0:
            return 0.0, {"punct": punct, "subs": subs, "raw": 0.0}

        raw = (punct + subs) / tok_total
        ds = min(1.0, raw / max(self.cfg.ds_markers_per_token_full, 1e-9))
        return float(ds), {"punct": punct, "subs": subs, "raw": raw}

    def compute_x_p(self, text: str, tokens: List[str], sentences: List[str]) -> Tuple[float, Dict[str, Any]]:
        """xₚ: peripheral pull/extremity proxy.

        v17 redesign: domain-agnostic rarity proxy replaces v16 domain-locked
        technical_markers list. Any token (len>=4) absent from cfg.common_words
        is rare by this measure — works across semiconductor, medical, legal,
        financial, or any other technical register without hardcoded vocabulary.

        Formula:
            rare_ratio  = rare_tokens / content_tokens
            met_rate    = metaphor_hits / sentence_count
            raw         = (3.5 * rare_ratio) + (0.15 * met_rate)
            xₚ          = sigmoid(raw - xp_center)

        Supplementary: cfg.technical_markers still accepted as a caller override;
        if non-empty, those hits augment rare_ratio at 0.5x weight.
        """
        if not tokens:
            return 0.0, {"rare_words": 0, "rare_ratio": 0.0,
                         "tech": 0, "met": 0, "met_rate": 0.0}

        common = self.cfg.common_words or set()
        rare = sum(1 for t in tokens if len(t) >= 4 and t not in common)
        rare_ratio = rare / max(len(tokens), 1)

        tech = sum(1 for t in tokens if t in self.cfg.technical_markers)
        legacy_ratio = tech / max(len(tokens), 1)

        lower = text.lower()
        met = 0
        for pat in self.cfg.metaphor_patterns:
            met += len(re.findall(pat, lower))
        met_rate = met / max(len(sentences), 1)

        raw = (3.5 * rare_ratio) + (0.5 * legacy_ratio) + (0.15 * met_rate)
        xp = self._sigmoid01(raw - self.cfg.xp_center)
        return float(xp), {
            "rare_words": rare,
            "rare_ratio": round(rare_ratio, 4),
            "tech": tech,
            "met": met,
            "met_rate": round(met_rate, 4),
        }

    # ---- collapse tail detectors (v19) ----

    @staticmethod
    def _tail_incomplete_flag(text: str) -> bool:
        """Detect whether a response ends with structural incompletion signals.

        Fires on:
        - Unclosed brackets, parens, or braces: (, [, {  without matching close
        - Text ending with a dangling operator: =, +, -, *, /, ,, :
        - Dangling LaTeX: \\(, \\frac{, \\int, \\begin{, $  at tail
        - Code block open without close: ``` without pair
        - Scaffold-word endings: last word is since, so, thus, where, if, when,
          because, and, or, for, with, as, to, of, the, a (grammatically incomplete)
        - Parenthesized expression ending with open paren content: "... lcm(" or "Since ("

        Returns True if any signal fires. False otherwise.

        Design intent: conservative heuristic. Should have low false positive rate on
        complete prose (no dangling operators, balanced brackets). Will fire on math
        and code fragments that are genuinely truncated.
        """
        t = text.strip()
        if not t:
            return False

        # 1. Unbalanced brackets / parens / braces
        if t.count("(") > t.count(")"):
            return True
        if t.count("[") > t.count("]"):
            return True
        if t.count("{") > t.count("}"):
            return True

        # 2. Ends with a dangling operator or punctuation that implies continuation
        #    Strip trailing whitespace and check final non-space character.
        last_char = t[-1] if t else ""
        if last_char in {"=", "+", "-", "*", "/", ",", "\\"}:
            return True

        # 3. Ends with an open LaTeX expression
        latex_tails = (
            "\\(", "\\[", "\\frac{", "\\int", "\\int_", "\\begin{",
            "\\left(", "\\left[", "\\left\\{", "\\propto",
        )
        t_lower = t.lower()
        for lt in latex_tails:
            if t_lower.endswith(lt):
                return True

        # 4. Unmatched inline math delimiters: odd number of $ signs
        dollar_count = t.count("$") - t.count("$$") * 2
        if dollar_count % 2 != 0:
            return True

        # 5. Unclosed code fences
        if t.count("```") % 2 != 0:
            return True

        # 6. Scaffold-word endings (grammatically open final word)
        #    Minimum length 2: single-letter tokens are math variables (A, B, x, n),
        #    not scaffold words, and must not trigger this check.
        scaffold_endings = {
            "since", "so", "thus", "where", "if", "when", "because",
            "and", "or", "for", "with", "as", "to", "of", "the", "a",
            "that", "which", "while", "although", "though", "unless",
            "whereas", "whose", "whom", "what", "how",
        }
        words = re.findall(r"[A-Za-z]+", t)
        if words and len(words[-1]) >= 2 and words[-1].lower() in scaffold_endings:
            return True

        # 7. Ends with an open function/expression call pattern: "word(" or "symbol("
        #    e.g. "lcm(" or "Since (" or "N(r)\\propto N(\\pi)\\left"
        if re.search(r"[A-Za-z0-9_]\s*\($", t):
            return True

        return False

    @staticmethod
    def _math_expression_tail_flag(text: str) -> bool:
        """Detect whether a response ends with an incomplete math or code expression.

        More specific than tail_incomplete_flag. Fires on patterns that are characteristic
        of mathematical or code truncation specifically:
        - Ends with an open integral, sum, product operator
        - Ends with a standalone math operator or relation
        - Ends with dangling LaTeX sequences (\\frac, \\sqrt, \\int, \\sum, \\prod)
        - Ends with an incomplete fraction numerator
        - Ends inside a LaTeX environment that was not closed
        - Code ends mid-function (last non-empty line is a partial statement)

        IMPORTANT: This flag produces false positives when used alone on complete responses
        that contain inline math. It must be combined with tail_incomplete_flag for reliable
        collapse detection. The conjunction (tail AND math_tail) is the operative gate.
        """
        t = text.strip()
        if not t:
            return False

        # 1. Unmatched LaTeX delimiters (\\( without \\), \\[ without \\])
        if t.count("\\(") > t.count("\\)"):
            return True
        if t.count("\\[") > t.count("\\]"):
            return True

        # 2. Unbalanced curly braces — catches truncated LaTeX arguments and
        #    C/code blocks opened but not closed.
        #    ID 262: \\sqrt{\\kappa truncated — { opened but never closed.
        #    ID 109: int factorial(int n) { — code body truncated mid-block.
        if t.count("{") > t.count("}"):
            return True

        # 3. Odd number of code fences — catches code blocks opened but not closed.
        #    ID 109: ``` opened before C code, block was truncated mid-function.
        if t.count("```") % 2 != 0:
            return True

        # 4. Ends with a pure math operator or relation symbol (tail is operator)
        math_operator_tail = re.search(
            r"[\+\-\*\/\=\<\>\~\^\|\\]\s*$", t
        )
        if math_operator_tail:
            return True

        # 5. Ends with dangling LaTeX commands (command without its argument)
        dangling_latex = re.search(
            r"\\(?:frac|sqrt|int|sum|prod|lim|partial|nabla|infty|"
            r"alpha|beta|gamma|delta|epsilon|zeta|eta|theta|lambda|mu|nu|xi|"
            r"pi|rho|sigma|tau|phi|chi|psi|omega|"
            r"Delta|Gamma|Lambda|Phi|Pi|Sigma|Theta|Omega|"
            r"left|right|bigl|bigr|Bigl|Bigr|"
            r"mathrm|mathbf|mathbb|mathcal|text|"
            r"begin|end|over|underset|overset|"
            r"propto|approx|equiv|sim|"
            r"hbar|ell|cdot|times|otimes|oplus)\s*$",
            t
        )
        if dangling_latex:
            return True

        # 6. Ends with an opening LaTeX brace argument (command{  with no closing })
        #    e.g. "\\frac{3" or "\\sqrt{n+1"
        if re.search(r"\\[A-Za-z]+\{[^}]*$", t):
            return True

        # 7. Ends inside a LaTeX \begin{} environment that has no matching \end{}
        begins = len(re.findall(r"\\begin\{", t))
        ends = len(re.findall(r"\\end\{", t))
        if begins > ends:
            return True

        # 8. Code block: last non-empty line ends with an operator or open paren
        #    (indicates a mid-expression code truncation)
        lines = [ln.rstrip() for ln in t.split("\n") if ln.strip()]
        if lines:
            last_code_line = lines[-1]
            if re.search(r"[=\+\-\*\/(,{]\s*$", last_code_line):
                return True
            # last line is a keyword that requires a body
            if re.search(r"\b(if|for|while|def|class|return|else|elif|try|except|with)\s*:?\s*$",
                         last_code_line):
                return True

        return False

    # ---- basin assignment ----


    @staticmethod
    def assign_basin(dx: float, rv: float) -> str:
        """Coarse 4-basin map in Δx vs rᵥ."""
        if abs(dx) < 0.10 and rv < 0.60:
            return "B0_centered_compact"
        if dx >= 0.10 and rv < 0.60:
            return "B1_right_bias_compact"
        if dx <= -0.10 and rv < 0.60:
            return "B2_left_bias_compact"
        return "B3_high_void_or_high_displacement"


    @staticmethod
    def assign_basin_z(z_dx: float, z_rv: float) -> str:
        """Coarse 4-basin map in z(Δx) vs z(rᵥ).

        This avoids applying raw-space thresholds to z-space.
        Thresholds are intentionally simple and should be treated as interpretive labels.
        """
        if abs(z_dx) < 0.5 and z_rv < 0.0:
            return "ZB0_centered_below_mean_void"
        if z_dx >= 0.5 and z_rv < 0.0:
            return "ZB1_right_bias_below_mean_void"
        if z_dx <= -0.5 and z_rv < 0.0:
            return "ZB2_left_bias_below_mean_void"
        return "ZB3_high_void_or_high_displacement"

    # ---- canonical output payload ----

    def compute(
        self,
        text: str,
        *,
        prompt: Optional[str] = None,
        prompt_family_id: Optional[str] = None,
        numbers_off: bool = False,
        include_detail: bool = True,
    ) -> Dict[str, Any]:
        """Compute kernel, optionally z-score under prompt-family baseline."""
        text_n = normalize_text(text)
        sents = segment_sentences(text_n)
        toks = tokenize_words(text_n)

        dx, dx_ev = self.compute_delta_x(sents)
        rv, rv_ev = self.compute_r_v(text_n, toks, sents)
        rho, rho_ev = self.compute_rho_r(toks)
        mu, mu_ev = self.compute_mu(toks)

        dy, dy_ev = self.compute_delta_y(sents)
        xp, xp_ev = self.compute_x_p(text_n, toks, sents)
        theta, th_ev = self.compute_theta(sents)
        ds, ds_ev = self.compute_d_s(sents)

        kernel = {"Δx": dx, "rᵥ": rv, "ρᵣ": rho, "μ": mu}
        aux = {"Δy": dy, "xₚ": xp, "θ": theta, "dₛ": ds}

        fam = self.registry.get(prompt_family_id) if prompt_family_id else None
        z = None
        if fam is not None and fam.baseline is not None:
            z = {
                "kernel": fam.baseline.z_map(kernel),
                "aux": fam.baseline.z_map(aux),
            }

        # Basin assignment
        # IMPORTANT: Raw-space basin uses raw thresholds; z-space basin uses z-thresholds.
        basin_raw = self.assign_basin(dx, rv)
        basin_z = None
        if z is not None:
            basin_z = self.assign_basin_z(z["kernel"]["Δx"], z["kernel"]["rᵥ"])

        # Preserve legacy key: "basin" = raw basin (stable across calibration state)
        basin = basin_raw

        flags: List[str] = []
        if not sents:
            flags.append("empty_text")
        if len(toks) < 20:
            flags.append("low_token_count")
        if len(sents) < 4:
            # short_sentence_count: fewer than 4 sentences means Δx position values
            # span the full [0.0, 1.0] range with no middle to distribute mass into.
            # Δx excursions on short texts are partly a sample-size artifact.
            # Treat any Δx reading > 0.10 on flagged texts with additional caution.
            flags.append("short_sentence_count")
        if len(paragraph_blocks(text_n)) == 1 and len(toks) > 180:
            flags.append("wall_of_text")
        if fam is not None and fam.baseline is None:
            flags.append("family_has_no_baseline")

        # BPE token count via tiktoken (cl100k_base / GPT-4 encoding)
        # Runs alongside the kernel's own word tokenizer — does NOT replace it.
        # Punctuation handling in the kernel stays intact; this is a parallel signal.
        # -1 means tiktoken is not installed.
        tk_count = len(_TK_ENC.encode(text_n)) if _TK_ENC is not None else -1

        # segmentation_reliable: True if either punctuation density OR paragraph/bullet
        # structure is sufficient to trust Δy and sentence-count-derived coordinates.
        # Some outputs are legitimately segmented via newlines/bullets with minimal punctuation.
        # When False, treat Δy, rᵥ sentence components, and sentence_count with caution.
        #
        # IMPORTANT: uses sentence-terminating punctuation (.!?) not dₛ clause markers (,;:).
        # dₛ measures clause thickness/subordination. segmentation_reliable measures whether
        # sentence boundary detection is trustworthy. These are different things.
        # Ordinary prose with few commas but normal sentence endings must not be flagged unreliable.
        seg_term_count = text_n.count(".") + text_n.count("!") + text_n.count("?")
        tok_count_for_seg = max(len(toks), 1)
        bullet_count = sum(1 for ln in split_lines(text_n) if _BULLET_RE.match(ln))
        newline_count = text_n.count("\n")
        structural_breaks = bullet_count + newline_count
        punct_ok = (seg_term_count / tok_count_for_seg) >= self.cfg.seg_punct_per_token
        newline_ok = (structural_breaks / tok_count_for_seg) >= self.cfg.seg_newline_per_token
        seg_reliable = (punct_ok or newline_ok) if tok_count_for_seg >= self.cfg.seg_min_tokens else True

        # Rhetorical state: synthesized output mode label derived from existing signals.
        # Not a new computation — reads contrast_markers, LIS signals, and segmentation_strain.
        # Assertive:                    low contrast, low revision markers, normal segmentation
        # Dialectic:                    elevated contrast markers, stable LIS
        # Revising:                     explicit self-correction markers present
        # Complying_under_constraint:   segmentation_strain spike (Δy > 1.0) with low contrast

        # Contrast marker count: dialectical tension signal
        # Counts words/phrases that hold two positions in structural opposition.
        # Single words: matched against word-tokenized list (exact token match, no subword hits).
        # Multi-word phrases: matched with word-boundary regex to prevent partial overlaps.
        #   e.g. "on the other hand" counts as 1 hit, not also counting "other" separately.
        # Each phrase/word counted independently; overlapping phrases not double-counted
        # because phrases are checked against the raw string, words against token list.
        text_lower = text_n.lower()
        # Normalize punctuation to spaces before phrase matching to handle:
        # em dashes, commas mid-phrase, quotes, line breaks, formatting noise.
        # Light normalization only — replaces non-alphanumeric non-space chars with space,
        # then collapses multiple spaces. Word-level tokenization is unaffected.
        text_normalized = re.sub(r"[^\w\s]", " ", text_lower)
        text_normalized = re.sub(r"\s+", " ", text_normalized).strip()
        contrast_hits = 0
        for marker in sorted(self.cfg.contrast_markers):  # sorted for stable logging
            if " " in marker:
                # word-boundary anchored phrase match on normalized text
                pat = r"\b" + re.escape(marker) + r"\b"
                contrast_hits += len(re.findall(pat, text_normalized))
            else:
                # exact token match via tokenizer (prevents subword hits)
                contrast_hits += sum(1 for t in toks if t == marker)

        # Derive rhetorical_state from signals computed so far
        dy_val, _ = self.compute_delta_y(sents)
        revision_markers = (
            len(re.findall(r"\bI was wrong\b|\bcorrection\b|\bretract\b|\bI misspoke\b", text_lower)) +
            len(re.findall(r"\bactually,\b|\bwait,\b", text_lower))
        )

        # v19: compute tail signals before routing — needed by partial and math collapse gates.
        tail_flag = self._tail_incomplete_flag(text_n)
        math_tail_flag = self._math_expression_tail_flag(text_n)

        # Routing order (v19):
        # 1. revising                              — explicit self-correction markers
        # 2. degenerate_input                      — single symbol or empty
        # 3. formatting_collapse_severe            — v18 gate (token < 16, Dy < -0.3, rv > 0.45)
        # 4. formatting_collapse_partial           — v19: mid-length + tail_flag
        # 5. formatting_collapse_math              — v19: normal segmentation + tail_flag + math_tail_flag
        # 6. complying_under_constraint (struct)   — schema/JSON structural override
        # 7. dialectic                             — contrast marker elevation
        # 8. complying_under_constraint (Dy gate)  — segmentation strain spike
        # 9. assertive                             — default

        collapse_subtype = "none"

        if revision_markers > 0:
            rhet_state = "revising"

        elif len(toks) == 0 or (len(toks) == 1 and not sents):
            # Degenerate input: single symbol or empty after normalization.
            rhet_state = "formatting_collapse"
            collapse_subtype = "severe"

        elif (
            len(toks) < self.cfg.fc_token_max
            and dy_val < self.cfg.fc_dy_max
            and rv > self.cfg.fc_rv_min
        ):
            # formatting_collapse_severe (v18 gate, unchanged):
            # Very short text (< 16 tokens) with compressed sentence structure and inflated void ratio.
            # Calibration: catches 6/6 severe truncations, 0 FPs in GPT EMS corpus.
            rhet_state = "formatting_collapse"
            collapse_subtype = "severe"

        elif (
            self.cfg.fc_token_max <= len(toks) <= self.cfg.pc_token_max
            and dy_val <= self.cfg.pc_dy_max
            and rv >= self.cfg.pc_rv_min
            and tail_flag
        ):
            # formatting_collapse_partial (v19):
            # Mid-length text (16–40 tokens) with near-normal or slightly compressed segmentation,
            # minimal void ratio floor, AND a visible tail incompletion signal.
            # tail_flag is the required gate — threshold relaxation alone produces false positives.
            # Calibration: catches 6/6 additional truncations, 0 FPs in GPT EMS corpus.
            rhet_state = "formatting_collapse"
            collapse_subtype = "partial"

        elif tail_flag and math_tail_flag:
            # formatting_collapse_math (v19):
            # No token ceiling. Catches math/code-heavy truncations with normal segmentation
            # statistics that are geometrically indistinguishable from complete responses.
            # CONJUNCTION REQUIRED: math_expression_tail_flag alone fires on complete responses
            # containing inline math. tail_flag AND math_tail_flag together have 0 FPs.
            # Calibration: catches 3/3 remaining misses, 0 FPs in GPT EMS corpus.
            rhet_state = "formatting_collapse"
            collapse_subtype = "math"

        elif rv < 0.08 and rho > 0.60 and dy_val > 1.5:
            # Structural override: heavily constrained schema/JSON output.
            # JSON/schema outputs: near-zero rv, elevated rho, AND very long lines (Dy > 1.5).
            # v17 FIX: added Dy > 1.5 gate. Dense technical paragraphs also hit the
            # rv+rho condition but are assertive writing, not constraint compliance.
            rhet_state = "complying_under_constraint"

        elif contrast_hits >= self.cfg.rhet_dialectic_min_contrast:
            rhet_state = "dialectic"

        elif dy_val > self.cfg.rhet_constraint_min_dy and contrast_hits == 0:
            rhet_state = "complying_under_constraint"

        else:
            rhet_state = "assertive"

        # Basis decomposition: traceability — what drove each classification
        # seg_basis: which signal(s) make segmentation reliable
        if punct_ok and newline_ok:
            seg_basis = "both"
        elif punct_ok:
            seg_basis = "punct"
        elif newline_ok:
            seg_basis = "newline"
        else:
            seg_basis = "none"

        # contrast_basis: which kind of markers fired
        token_contrast = sum(1 for m in self.cfg.contrast_markers if " " not in m
                             for t in toks if t == m)
        phrase_contrast = contrast_hits - token_contrast
        if token_contrast > 0 and phrase_contrast > 0:
            contrast_basis = "both"
        elif token_contrast > 0:
            contrast_basis = "token"
        elif phrase_contrast > 0:
            contrast_basis = "phrase"
        else:
            contrast_basis = "none"

        # constraint_basis: what kind of constraint compliance or collapse
        # v19: when rhetorical_state == "formatting_collapse", reports the subtype
        # ("severe" | "partial" | "math") rather than the single "formatting_collapse" value.
        # This makes constraint_basis more granular and avoids conflating collapse tiers.
        # Downstream consumers: check rhetorical_state == "formatting_collapse" for collapse
        # detection; use constraint_basis (or evidence.collapse_subtype) for subtype.
        if rhet_state == "formatting_collapse":
            constraint_basis = collapse_subtype  # "severe" | "partial" | "math"
        elif rhet_state == "complying_under_constraint":
            if dy_val > self.cfg.rhet_constraint_min_dy:
                constraint_basis = "sentence_suppression"
            elif len(toks) > 150:
                constraint_basis = "verbosity"
            else:
                constraint_basis = "sentence_suppression"
        else:
            constraint_basis = "none"

        evidence = KernelEvidence(
            token_count=len(toks),
            sentence_count=len(sents),
            paragraph_count=len(paragraph_blocks(text_n)),
            bullet_line_count=sum(1 for ln in split_lines(text_n) if _BULLET_RE.match(ln)),
            transitions_count=int(mu_ev.get("transitions", 0)),
            reference_pronouns_count=int(mu_ev.get("refs", 0)),
            content_word_count=int(rho_ev.get("content", 0)),
            stopword_count=int(rho_ev.get("stop", 0)),
            adjacent_cosine_mean=float(th_ev.get("adj_cos_mean", 0.0)),
            technical_hits=int(xp_ev.get("tech", 0)),
            metaphor_hits=int(xp_ev.get("met", 0)),
            punctuation_clause_markers=int(ds_ev.get("punct", 0)),
            subordinator_hits=int(ds_ev.get("subs", 0)),
            contrast_marker_hits=contrast_hits,
            segmentation_reliable=seg_reliable,
            rhetorical_state=rhet_state,
            seg_basis=seg_basis,
            contrast_basis=contrast_basis,
            constraint_basis=constraint_basis,
            ds_per_100_tokens=round(ds_ev.get("raw", 0.0) * 100, 3),
            contrast_per_100_tokens=round(contrast_hits / max(len(toks), 1) * 100, 3),
            subordinators_per_100_tokens=round(ds_ev.get("subs", 0) / max(len(toks), 1) * 100, 3),
            tiktoken_token_count=tk_count,
            tail_incomplete_flag=tail_flag,
            math_expression_tail_flag=math_tail_flag,
            collapse_subtype=collapse_subtype,
        )

        def maybe_hide(d: Dict[str, float]) -> Dict[str, Any]:
            return d if not numbers_off else {k: None for k in d.keys()}

        out: Dict[str, Any] = {
            "kernel": maybe_hide(kernel),
            "aux": maybe_hide(aux),
            "z": None if z is None else {"kernel": maybe_hide(z["kernel"]), "aux": maybe_hide(z["aux"])},
            "basin": basin,
            "basin_raw": basin_raw,
            "basin_z": basin_z,
            "flags": flags,
            "evidence": asdict(evidence),
            "meta": {
                "prompt_family_id": prompt_family_id,
                "prompt": prompt if prompt is not None else None,
                "kernel_version": "lang-kernel-rebuild-v19",
                "tiktoken_encoding": "cl100k_base" if _TK_ENC is not None else None,
                "tiktoken_version": getattr(_tiktoken, "__version__", None) if _tiktoken is not None else None,
                # v18 scope note: tiktoken_token_count is computed but scoped as a Phase 2
                # cross-model comparison instrument. In single-model studies, this field is
                # present in the output for pipeline compatibility but is not reported in
                # analysis. BPE tokenization comparison is meaningful only when Gemini, GPT,
                # and Claude outputs are placed side by side with matched prompt sets.
                "numbers_off": numbers_off,
                "positioning": "observational structural telemetry; all geometric language refers to induced structure in feature space only",
                "length_norm_factor": {
                    # Traceability block: documents the normalization basis for this output.
                    # Not new computation — derived from evidence already collected.
                    # If normalization is challenged, these values explain the denominators.
                    "effective_token_count": len(toks),
                    "sentence_count": len(sents),
                    "avg_sentence_length": round(dy_ev.get("avg_sent_len", 0.0), 2),
                    "tiktoken_token_count": tk_count,
                    "bpe_to_word_ratio": round(tk_count / max(len(toks), 1), 3) if tk_count > 0 else None,
                },
            },
        }

        if include_detail and not numbers_off:
            out["detail"] = {
                "Δx": dx_ev,
                "rᵥ": rv_ev,
                "ρᵣ": rho_ev,
                "μ": mu_ev,
                "Δy": dy_ev,
                "xₚ": xp_ev,
                "θ": th_ev,
                "dₛ": ds_ev,
            }

        return out

    def fit_family_baseline(
        self,
        family_id: str,
        corpus_texts: List[str],
        *,
        description: str = "",
        include_aux: bool = True,
    ) -> PromptFamily:
        """Fit and register a baseline for a prompt family (neutral corpus recommended)."""
        if not corpus_texts:
            raise ValueError("Cannot fit baseline on empty corpus")

        keys = ["Δx", "rᵥ", "ρᵣ", "μ"]
        if include_aux:
            keys += ["Δy", "xₚ", "θ", "dₛ"]

        vals: Dict[str, List[float]] = {k: [] for k in keys}
        for t in corpus_texts:
            res = self.compute(t, prompt_family_id=None, include_detail=False)
            raw: Dict[str, Any] = {}
            raw.update(res["kernel"])
            raw.update(res["aux"])
            for k in keys:
                if isinstance(raw.get(k), (int, float)):
                    vals[k].append(float(raw[k]))

        stats: Dict[str, BaselineStats] = {}
        for k, xs in vals.items():
            if not xs:
                continue
            mean = sum(xs) / len(xs)
            if len(xs) == 1:
                std = 0.0
            else:
                var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
                std = math.sqrt(var)
            stats[k] = BaselineStats(mean=mean, std=std)

        fam = PromptFamily(family_id=family_id, description=description, baseline=KernelBaseline(stats=stats))
        self.registry.upsert(fam)
        return fam


# ----------------------------
# CLI
# ----------------------------

def _read_text_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _read_corpus(path: str | Path) -> List[str]:
    """Read corpus from .txt (blank-line separated blocks) or .jsonl (one JSON per line with 'text')."""
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        out: List[str] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" in obj and isinstance(obj["text"], str):
                out.append(obj["text"])
        return out

    raw = p.read_text(encoding="utf-8")
    return paragraph_blocks(raw)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Linguistic Kernel (Rebuild - Fixed v2)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_fit = sub.add_parser("fit", help="Fit a prompt-family baseline from a neutral corpus")
    p_fit.add_argument("--family", required=True, help="Prompt family id")
    p_fit.add_argument("--corpus", required=True, help="Path to corpus .txt (blank-line blocks) or .jsonl")
    p_fit.add_argument("--desc", default="", help="Family description")
    p_fit.add_argument("--registry_out", required=True, help="Where to write registry JSON")

    p_compute = sub.add_parser("compute", help="Compute kernel for a single text input")
    p_compute.add_argument("--file", default=None, help="Path to a .txt file")
    p_compute.add_argument("--text", default=None, help="Inline text (use quotes)")
    p_compute.add_argument("--family", default=None, help="Optional prompt family id")
    p_compute.add_argument("--registry", default=None, help="Optional registry JSON path")
    p_compute.add_argument("--numbers_off", action="store_true", help="Suppress values (structure only)")
    p_compute.add_argument("--json_out", default=None, help="Optional output JSON file")

    p_batch = sub.add_parser("batch", help="Compute kernel for corpus (jsonl) and emit jsonl outputs")
    p_batch.add_argument("--corpus", required=True, help="Path to corpus .jsonl with {'text', optional 'prompt','family'}")
    p_batch.add_argument("--registry", default=None, help="Optional registry JSON path")
    p_batch.add_argument("--out", required=True, help="Output .jsonl")

    args = ap.parse_args(argv)

    if args.cmd == "fit":
        reg = PromptFamilyRegistry()
        lk = LinguisticKernel(registry=reg)
        corpus = _read_corpus(args.corpus)
        lk.fit_family_baseline(args.family, corpus, description=args.desc, include_aux=True)
        reg.save(args.registry_out)
        return 0

    if args.cmd == "compute":
        reg = PromptFamilyRegistry.load(args.registry) if args.registry else PromptFamilyRegistry()
        lk = LinguisticKernel(registry=reg)
        txt = _read_text_file(args.file) if args.file else (args.text or "")
        if not txt:
            raise SystemExit("compute: provide --file or --text")
        out = lk.compute(txt, prompt_family_id=args.family, numbers_off=args.numbers_off, include_detail=True)
        s = json.dumps(out, indent=2, ensure_ascii=False)
        if args.json_out:
            Path(args.json_out).write_text(s, encoding="utf-8")
        else:
            print(s)
        return 0

    if args.cmd == "batch":
        reg = PromptFamilyRegistry.load(args.registry) if args.registry else PromptFamilyRegistry()
        lk = LinguisticKernel(registry=reg)
        inp = Path(args.corpus).read_text(encoding="utf-8").splitlines()
        out_lines: List[str] = []
        for line in inp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            prompt = obj.get("prompt")
            family = obj.get("family")
            res = lk.compute(text, prompt=prompt, prompt_family_id=family, numbers_off=False, include_detail=False)
            out_lines.append(json.dumps({**obj, "kernel_out": res}, ensure_ascii=False))
        Path(args.out).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
