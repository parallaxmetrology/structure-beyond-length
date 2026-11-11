# MT-Bench: humans vs GPT-4 on the same pairs (frozen 2026-11-11, before any kernel reading of this data)

## Question

On the same answer pairs, does GPT-4 reward structure more than humans do, relative to how much each rewards length?

The AlpacaEval study found the GPT-4 judge's structural weights point the same way as human weights (cosine 0.92) but are larger. Different datasets made the sizes incomparable. Here both judge the same pairs.

## Data

`public_corpora/mtbench/` (lmsys/mt_bench_human_judgments, CC-BY-4.0). 80 questions × 2 turns; 6 models from 2023 (alpaca-13b, claude-v1, gpt-3.5-turbo, gpt-4, llama-13b, vicuna-13b-v1.2).
- `human.parquet`: 3,355 votes by 65 judges (experts and authors).
- `gpt4_pair.parquet`: 2,400 GPT-4 pairwise verdicts.

Unit: one pair = (question, turn, model A, model B). The judged text is the assistant answer for that turn (turn 1: first answer; turn 2: second answer). Ties are dropped, including GPT-4's "tie (inconsistent)". Analyses use pairs judged by both sides with a non-tie verdict from each, both answers eligible under reading frame r1.

## Instrument and features

Frozen v19 through reading frame r1, framed readings. Differences A − B: log prose tokens (`dlen`), repetition (`drep4`, share of repeated word 4-grams), and the 8 axes.

**Structure index**, fixed now from the AlpacaEval study (no refitting on MT-Bench): the unit vector of the AlpacaEval judge's standardised weights on the 8 axes (`studies/alpacaeval/data/results.json`, J2). Axis differences are standardised by their SD over the shared pairs, then projected onto that vector.

## Noise

Human votes are noisier than GPT-4 verdicts, which shrinks all human weights roughly in proportion. So sizes are compared as **ratios to each judge's own length weight**, which proportional shrinkage leaves unchanged.

## Analyses

- **M-a.** Logistic per judge: win ~ dlen + drep4 + structure index. Human model on individual votes; GPT-4 model on verdicts. Ratio r = β_structure / β_length per judge. Difference r_GPT4 − r_human, 95% CI by bootstrap over questions (80 clusters).
- **M-b.** Same with all 8 axes: cosine between the two judges' 8-axis weight vectors; per-axis ratio to length weight, CIs by question bootstrap.
- **M-c.** Held-out AUC per judge (5 folds by question): length; + repetition; + kernel. Descriptive (noise affects it).
- **M-d.** Sensitivity: human majority vote per pair instead of individual votes.
- **M-e.** Agreement: human-human (pairs with ≥ 2 human votes) and human-GPT-4.

## Frozen predictions

| # | Prediction | Kind |
|---|---|---|
| P1 | Both judges' 8-axis weights point the same way (cosine > 0.5) | guard |
| P2 | GPT-4 weights structure more than humans relative to length: r_GPT4 − r_human > 0, CI above 0 | seek |
| P3 | Repetition-to-length ratio is similar for both (CI on the difference includes 0) | guard |
| P4 | Both judges reward length (β_length > 0, CI above 0) | guard |
| P5 | P2 holds under majority vote (M-d) | seek |

"Supported" = sign matches and CI excludes 0 as stated. A small sample (80 questions) is expected to give wide intervals; results are reported at their size.
