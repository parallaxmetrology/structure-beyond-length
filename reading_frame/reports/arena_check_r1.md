# Reading frame r1: Arena check

3000 answers (first turn, 1,500 battles, seed 0). Kernel sha256 `1f63ee858c92af46…` (frozen v19).

## Who is eligible

- eligible: 2368 (78.9%)
- short_prose: 580 (19.3%)
- low_english_function_words: 53 (1.8%)
- non_latin_script: 42 (1.4%)
- no_letters: 23 (0.8%)

## Collapse: v19 label vs frame reading

| raw_collapse_subtype | cut_code_or_math | cut_partial | none | short_complete | short_cut |
|---|---|---|---|---|---|
| math | 43 | 0 | 0 | 0 | 0 |
| none | 0 | 0 | 2737 | 0 | 0 |
| partial | 0 | 11 | 0 | 0 | 0 |
| severe | 0 | 0 | 0 | 206 | 3 |

Severe-short answers that end in closing punctuation: 146 of 209.

## Markdown among eligible answers

- framed text identical to raw (no markdown the frame touches): 2085 (88.0%)
- has code blocks: 232
- has table lines: 11
- has headers: 21
- has bold lines: 26
- has links: 4

## Does framing close the gap? (medians, eligible answers)

Plain = no markdown edits. Each group shows raw → framed. A frame that works moves the markdown groups toward Plain.

| group | n | Δx | rᵥ | ρᵣ | μ | Δy | xₚ | θ | dₛ |
|---|---|---|---|---|---|---|---|---|---|
| Plain | 2085 | -0.00 | 0.27 | 0.54 | 0.50 | 0.47 | 0.41 | 0.74 | 0.56 |
| Code | 232 | -0.00 → -0.00 | 0.32 → 0.36 | 0.55 → 0.52 | 0.49 → 0.50 | 1.11 → 0.40 | 0.45 → 0.42 | 0.70 → 0.69 | 0.83 → 0.52 |
| Headers | 21 | 0.00 → 0.00 | 0.42 → 0.30 | 0.59 → 0.57 | 0.49 → 0.50 | 0.51 → 0.58 | 0.44 → 0.44 | 0.80 → 0.77 | 0.63 → 0.55 |
| Bold lines | 26 | 0.01 → 0.00 | 0.38 → 0.30 | 0.59 → 0.59 | 0.49 → 0.49 | 0.46 → 0.42 | 0.45 → 0.44 | 0.78 → 0.77 | 0.73 → 0.67 |
| Tables | 11 | 0.00 → 0.00 | 0.11 → 0.30 | 0.63 → 0.55 | 0.48 → 0.49 | 4.76 → 0.63 | 0.42 → 0.40 | 0.72 → 0.78 | 0.49 → 0.50 |

## Length dependence (Spearman with prose tokens, eligible)

| dim | raw | framed |
|---|---|---|
| Δx | 0.10 | 0.10 |
| rᵥ | 0.05 | 0.02 |
| ρᵣ | 0.17 | 0.20 |
| μ | 0.05 | 0.01 |
| Δy | 0.19 | 0.27 |
| xₚ | 0.11 | 0.14 |
| θ | -0.04 | -0.04 |
| dₛ | 0.10 | 0.16 |

## Rhetorical state, eligible markdown answers (raw vs framed)

| raw_rhetorical_state | assertive | complying_under_constraint | dialectic | formatting_collapse |
|---|---|---|---|---|
| assertive | 134 | 2 | 0 | 0 |
| complying_under_constraint | 78 | 25 | 0 | 0 |
| dialectic | 5 | 0 | 20 | 0 |
| formatting_collapse | 15 | 1 | 1 | 2 |
