# Reading frame r1.1

A preparation layer that runs before the frozen v19 kernel when reading found corpora (chat logs, benchmark answers). v19 is not edited, copied or reconfigured: `reading_frame.py` loads `../kernel/kernel_v19.py` by path and refuses to run if the file's SHA-256 differs from the recorded hash. The studies ran on the original v19 file; this repository ships the published copy from linguistic-kernel, which gives identical readings.

Every text is read twice, **raw** (exactly as given) and **framed** (markdown that v19 misreads neutralised). Both readings and their difference are kept, so the markdown effect is a measured quantity.

## Files

| File | What it is |
|---|---|
| `reading_frame.py` | The layer. `frame_text(text)` (pure, no kernel), `Reader().read(text)` (raw + framed v19 readings), `flat_row(reading)` (one dict per text, for CSV). |
| `test_reading_frame.py` | 18 tests. Includes: plain text comes back byte-identical and its readings equal a direct v19 call. Run `python3 test_reading_frame.py`. |
| `check_arena.py` | Before/after check on 3,000 Arena answers. Writes `reports/arena_check_r1.md` and `.csv`. |

## What the frame changes

Only these patterns. Everything else, bullets and inline bold included, is left alone.

1. **Fenced code blocks** (``` or ~~~) are removed. An unclosed fence is recorded (`edit_unclosed_fence`).
2. **Table rows and separator lines** are removed.
3. **Header lines** (`# Title`) and **bold-only lines** (`**Title:**`) are removed, leaving a paragraph break.
4. **Markdown links** keep their text and drop the URL.

A first draft turned each header into its own one-sentence paragraph. On Arena that added paragraphs that were not there (headers rᵥ 0.39 → 0.60, against 0.27 for plain answers), so r1 drops titles instead (rᵥ 0.42 → 0.30).

## Fields the frame adds

- **`eligible`**: English and at least 40 word tokens after framing. English means at least 95% of letters are Latin a-z, and (for 20+ tokens) at least 12% of tokens are English function words. `reasons` lists what failed: `no_letters`, `non_latin_script`, `low_english_function_words`, `short_prose`.
- **`collapse_reading`**: v19's `collapse_subtype` restated for chat corpora, where short complete answers are common. It always uses the **raw** reading, because removing code also removes the evidence of a cut-off code answer.

| v19 `collapse_subtype` | `collapse_reading` |
|---|---|
| severe, tail signal, text does not end in closing punctuation | `short_cut` |
| severe, otherwise | `short_complete` |
| partial | `cut_partial` |
| math | `cut_code_or_math` |
| none | `none` |

## Arena check (3,000 answers, v19 unchanged)

**Eligibility.** 2,368 (78.9%) eligible. Excluded: short prose 580 (19.3%), low English function words 53, non-Latin script 42, no letters 23. These reasons can overlap.

**Collapse.** v19 labelled 209 answers `severe`. 206 read as `short_complete` and 3 as `short_cut`. The 43 `math` and 11 `partial` labels are kept as cuts.

**Markdown gap (medians, eligible answers; plain answers n = 2,085).**

| group | n | Δy raw → framed | dₛ raw → framed | rᵥ raw → framed |
|---|---|---|---|---|
| Plain | 2,085 | 0.47 | 0.56 | 0.27 |
| Code | 232 | 1.11 → 0.40 | 0.83 → 0.52 | 0.32 → 0.36 |
| Tables | 11 | 4.76 → 0.63 | 0.49 → 0.50 | 0.11 → 0.30 |
| Headers | 21 | 0.51 → 0.58 | 0.63 → 0.55 | 0.42 → 0.30 |
| Bold lines | 26 | 0.46 → 0.42 | 0.73 → 0.67 | 0.38 → 0.30 |

Δx, ρᵣ, μ, xₚ and θ move by 0.08 or less in every group. Full table in `reports/arena_check_r1.md`.

**Length.** Among eligible answers, Δy's correlation with length is 0.19 raw and 0.27 framed (Spearman). The first pass, over all 3,000 answers including short and non-English ones, gave 0.50.

## v19 behaviours this surfaced (recorded, not changed)

- **The scaffold-word tail rule fires on closed sentences.** `tail_incomplete_flag` checks the last word against a list that includes "that", "what", "to", "for". "I'm sorry, but I can't help with that." is therefore flagged as having an incomplete tail. The frame's `collapse_reading` checks closing punctuation for this reason.
- **Code and table answers read as `complying_under_constraint`.** 103 of the 283 eligible markdown answers carry that state raw, against 8.3% of plain answers. Framed, 78 of the 103 read `assertive`.
- **`formatting_collapse` on markdown answers is code truncation.** All 19 eligible markdown answers in that state are subtype `math`, and 15 have an unclosed code fence. Framed, 15 of the 19 read `assertive` (13 of those have an unclosed fence), because the cut-off code is gone. That is why truncation is taken from the raw reading.

## Changelog

- **r1.1 (2026-11-11).** v19 raises a `ValueError` from its tiktoken token count when a text contains a literal special token such as `<|endoftext|>` (found in 6 of 59,358 Arena answers). The frame now catches any kernel error, records it in `kernel_error`, and marks the text ineligible with reason `kernel_error`. Texts that read before read identically.

## Use

```python
import sys; sys.path.insert(0, "reading_frame")  # from the repository root
import reading_frame as RF
reader = RF.Reader()
r = reader.read(text)          # dict: eligible, reasons, edits, collapse_reading, raw, framed, delta
row = RF.flat_row(r)           # flat dict: raw_Δy, framed_Δy, edit_code_blocks, ...
```

Prompt-relative reading (comparing an answer with other answers to the same prompt) needs no new code: v19's prompt-family baselines already support it.
