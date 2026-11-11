# Style-control baseline: plan (frozen 2026-11-11, before any fitting)

## Question

Does the v19 structure index still predict preferences once Chatbot Arena's own style-control features are in the model? And does the saturation gap (voters stop rewarding structure between styled answers; reward models do not) survive that control?

## Features

**Arena style control (SC)**, from each battle's `conv_metadata` in the Arena 100k file, as in Arena's published style control: for answer tokens, headers (h1-h6 summed), bold (`**` + `__`) and lists (ordered + unordered), the normalised difference (a − b) / (a + b), 0 when both are 0.
**Kernel (K)**: dlen, drep4 and the fixed-unit structure index, unchanged from the Arena studies.

## Raters and battles

- Voters on all 23,120 Arena battles (the Arena study's set), and on the 5,794 battles of the judge and reward-model runs (both-styled and neither-styled cohorts).
- Claude Opus 5.5 judge (non-tie verdicts) and ArmoRM, GPM, PairRM on the same 5,794 battles.

## Models

- **SC**: logistic on the four SC features.
- **SC + K**: SC plus dlen, drep4, struct.
- Reported: β_struct with 95% bootstrap CI over prompts in SC + K; held-out AUC (5 folds by prompt) for SC vs SC + K, with a bootstrap CI on the gain; and, the other way round, the AUC gain of adding SC to K.

## Frozen predictions

| # | Prediction | Kind |
|---|---|---|
| B1 | Voters, all 23,120 battles: β_struct in SC + K above 0 (CI above 0) | guard |
| B2 | Each reward model, both-styled cohort: β_struct in SC + K above 0 (CI above 0) | seek |
| B3 | Voters, both-styled cohort: β_struct in SC + K CI includes 0 | guard |
| B4 | Voters, all battles: SC + K has higher held-out AUC than SC (gain CI above 0) | seek |
| B5 | For each reward model, SC + K has higher held-out AUC than SC on the 5,794 battles (gain CI above 0) | seek |

The reverse comparison (SC added to K) is reported without a prediction.
