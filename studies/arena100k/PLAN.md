# Arena 100k: do humans reward the 2024 structural style? (frozen 2026-11-11, before any kernel reading of this data)

## Question

The 2024 answer style (bold title lines, density, rare vocabulary, segmentation) tracks the GPT-4 judge's preferred direction and the length-controlled AlpacaEval leaderboard. MT-Bench could not test it (2023 models, no bold titles). Do human Arena voters in mid-2024 reward that style beyond length, and by how much relative to length, compared with the GPT-4 judge?

## Data

`public_corpora/arena100k/arena-explorer-preference-100k.parquet` (lmarena-ai/arena-human-preference-100k; prompts CC-BY-4.0, outputs under provider terms; research use). 106,134 battles, June-August 2024.

Filters fixed now: `language == "English"`, `turn == 1` (single-turn battles, so the judged answer is unambiguous), winner `model_a` or `model_b` (ties dropped), both answers eligible under reading frame r1. Refusal and code battles are kept and reported as cohorts.

## Features (fixed units)

Differences A − B. `dlen` = log(1 + prose tokens). `drep4` = share of repeated word 4-grams. **Structure index** = Σ wᵢ · Δaxisᵢ / sdᵢ, where w is the AlpacaEval judge's unit weight vector on the 8 axes and sdᵢ is the SD of axis differences in the AlpacaEval judge data. Both are taken from the AlpacaEval study and not refitted. `dbold` = difference in having any bold title line (0/1).

The same fixed-unit model (win ~ dlen + drep4 + struct) is refitted on the AlpacaEval judge data and on MT-Bench (humans, GPT-4) for comparison.

## Analyses

- **A.** Logistic on Arena human votes: win ~ dlen + drep4 + struct. Bootstrap CIs over prompts (question_id clusters).
- **B.** Add `dbold`: is bold-title formatting rewarded beyond length, repetition and the structure index?
- **C.** Ratio r = β_struct / β_len for Arena humans; the same ratio for the AlpacaEval judge (bootstrap over models), MT-Bench humans and GPT-4 (bootstrap over questions). Difference AlpacaEval judge − Arena humans with CI from independent bootstraps.
- **D.** Cohorts: battles where both answers come from models with the 2024 formatting style vs neither; refusal and code battles.

## Frozen predictions

| # | Prediction | Kind |
|---|---|---|
| A1 | Arena humans reward length (β_len CI above 0) | guard |
| A2 | Arena humans reward the structure index beyond length and repetition (β_struct CI above 0) | seek |
| A3 | Arena humans penalise repetition (β_rep CI below 0) | guard |
| A4 | Bold-title formatting is rewarded beyond length, repetition and structure (β_bold CI above 0) | seek |
| A5 | The AlpacaEval judge's struct/length ratio exceeds Arena humans' (difference CI above 0) | seek |

Cross-dataset caveat, stated now: prompts, models and voters differ between Arena and AlpacaEval, so A5 compares preferences in fixed units, not on the same pairs.
