# HelpSteer2 vs v19: plan (frozen 2026-11-11, before any kernel reading of this data)

## Question

Do v19's structural axes line up with what human annotators scored, once length is accounted for? HelpSteer2 human scores are an independent bearing: the kernel was frozen before it saw this data, and the annotators never saw the kernel.

## Data

`public_corpora/helpsteer2/` train + validation: 21,362 responses, 10,680 prompts, two responses per prompt (one prompt has four). Five human scores, 0-4: helpfulness, correctness, coherence, complexity, verbosity.
Known before reading: coherence sits at the ceiling (15,298 of 21,362 score 4). Complexity is mostly 1-2. 6,232 prompts are multi-turn; only the final response is measured.

## Instrument

Frozen v19 (sha256 1f63ee85…) through reading frame r1. Main analysis uses **framed** readings on **eligible** responses. Raw readings are run alongside and compared.

## Length control

Length is the known dominant signal. Every axis result is reported both plain and as a partial correlation with log(prose tokens) removed. The pair analysis also removes the length difference.

## Analyses

- **A. Pooled.** Spearman of each of the 8 axes with each of the 5 scores, plain and length-partialled. Bootstrap 95% CI by prompt.
- **B. Within pair.** For the two responses to one prompt, the difference on each axis vs the difference in each score, length difference partialled. This holds the prompt fixed.
- **C. Prediction.** 5-fold cross-validation grouped by prompt. Linear model per score: length only vs length + 8 axes. Report held-out Spearman and the gain.
- **D. Raw vs framed.** Repeat C with raw readings.
- **E. Cohorts.** Responses with collapse readings (`short_cut`, `cut_partial`, `cut_code_or_math`) vs `none`: compare their scores. Then pull the largest residuals and read them.

## Frozen predictions (direction, after length is removed)

| # | Prediction | Kind |
|---|---|---|
| P1 | dₛ (clause stacking) rises with **complexity** | guard |
| P2 | xₚ (rare vocabulary) rises with **complexity** | guard |
| P3 | μ (cohesion) rises with **coherence** | guard |
| P4 | θ (flow instability) falls with **coherence** | guard |
| P5 | **verbosity** is mostly length; the axes add little to it in C | guard |
| P6 | Within a pair, the kernel adds held-out signal for **helpfulness** beyond length (C gain > 0 with CI above 0) | seek |
| P7 | Responses read as cut (`short_cut`, `cut_partial`, `cut_code_or_math`) score lower on helpfulness and correctness than `none` | seek |
| P8 | Framed readings track the human scores at least as well as raw readings (D) | guard for the frame |

A prediction is "supported" when the sign matches and the 95% CI excludes 0. Effect sizes are reported regardless; a small supported effect is reported as small.
