# AlpacaEval vs v19: plan (frozen 2026-11-11, before any kernel reading of this data)

## Questions

1. **Style over time.** Do model answers converge on one structural style? Observational, no labels.
2. **What the judge rewards.** Beyond length, which structure does the GPT-4-turbo judge reward? Compare its pattern with the human pattern from the HelpSteer2 study.

## Data

`public_corpora/alpaca_eval/`: `results/*/model_outputs.json`, 228 models × 805 instructions.
`studies/alpacaeval/data/verdicts.json`: per-answer judge verdicts (`weighted_alpaca_eval_gpt4_turbo`), continuous preference 1-2 against the fixed reference `gpt4_1106_preview` (2 = model's answer preferred).
`studies/alpacaeval/data/added_dates.json`: date each model's answers were first committed to the repo, 2023-06-09 to 2024-12-28. This is a **submission date**, not a release date. Many entries are community fine-tunes of open models.
Leaderboard CSV: win rate, length-controlled (LC) win rate, average length, mode (verified / community / minimal).

Exclusions, fixed now: the reference model itself (as a judged model), and entries that game the judge (NullModel and any leaderboard entry whose name marks it as a null or cheating model). Excluded names are listed in the lab book.

## Instrument

Frozen v19 (sha256 1f63ee85…) through reading frame r1, framed readings, eligible answers only. Raw readings kept for the frame comparison. Repetition = share of repeated word 4-grams (from the HelpSteer2 study).

## Prompt-relative coordinates

For each instruction, each axis is z-scored across all models' answers to that instruction. A model's **style vector** = mean prompt-relative z per axis over its eligible answers. This removes what the prompt itself demands.

## Study 1: style over time

- **S1.** Spread between models: for each half-year submission cohort (2023-H2, 2024-H1, 2024-H2; 2023-06 joins 2023-H2), mean pairwise distance between style vectors, with models subsampled to equal cohort size; bootstrap over models.
- **S2.** Per-prompt spread: for each instruction, standard deviation across a cohort's answers per axis, averaged; equal-size subsamples.
- **S3.** Drift: cohort centroid per axis, and length per cohort.
- **S4.** Same as S1 with log length partialled out of each prompt-relative axis, to separate convergence from length.

## Study 2: the judge

- **J1.** Pairwise model: each row = one instruction × one judged model; features = differences from the reference answer (log length, 8 axes, repetition); target = verdict > 1.5. Cross-validation holds out whole models (5 folds by model). AUC for length only, + kernel, + repetition, + both.
- **J2.** Standardised coefficients compared in sign with the HelpSteer2 human helpfulness model.
- **J3.** Model level: does a model's style vector explain its LC win rate beyond average length? Leave-one-out over models, R² for length only vs length + style.

## Frozen predictions

| # | Prediction | Kind |
|---|---|---|
| S-a | Spread between models is smaller in 2024-H2 than 2023-H2 (S1), CI on the difference excludes 0 | seek |
| S-b | The S-a convergence survives length partialling (S4) | seek |
| S-c | Answers get longer and more segmented over time (length and rᵥ rise across cohorts) | guard |
| J-a | Length dominates the judge's verdict (length-only AUC well above 0.5) | guard |
| J-b | Beyond length, the kernel adds held-out signal for the judge (J1 gain CI above 0) | guard |
| J-c | The judge rewards segmentation and density (rᵥ, ρᵣ) more than the HelpSteer2 humans did: positive judge coefficients where the human ones were near 0 or smaller | seek |
| J-d | Repetition is penalised by the judge as by humans (negative coefficient) | guard |
| J-e | Style explains LC win rate beyond average length (J3, R² gain > 0) | seek |

"Supported" = sign matches and 95% CI excludes 0 (or, for J-c, the stated comparison holds). Effects are reported at their size.
