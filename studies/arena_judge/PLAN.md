# Arena 2024 judge run: plan (frozen 2026-11-11, before any judge call)

## Question

In mid-2024 Arena votes, the structure index predicts the human winner between two unstyled models (β = +0.234) and between a styled and an unstyled one (+0.111), but not between two styled models (+0.002, the more structured answer wins 50%). Does an LLM judge, on the same battles, keep rewarding the structure direction between two styled answers?

## Data

The Arena 100k battles already read in `studies/arena100k` (English, single-turn, human winner, both answers eligible). Sample, fixed seed 0:
- **both styled**: all 2,897 battles
- **neither styled**: 2,897 sampled from 8,878
- **mixed**: 2,897 sampled from 11,345

Total 8,691 battles. Each is judged twice, with the answers in both orders (17,382 requests).

## Judge

Pairwise prompt in the style of MT-Bench's judge (impartial comparison of two answers to the user's question; output A, B or tie on the last line). Model and settings are recorded in `results/meta.json`. A verdict counts only when both orders agree; otherwise it is a tie, as in MT-Bench. Ties are dropped from the logistic models.

## Analyses

- **K1.** Per cohort, judge model: win ~ dlen + drep4 + struct (fixed units from the AlpacaEval study, unchanged). Bootstrap CIs over prompts.
- **K2.** Same model on the human votes for exactly the same battles (judge non-ties), for a same-pairs comparison.
- **K3.** Ratio β_struct / β_len, judge − human, per cohort, CI by paired bootstrap over prompts.
- **K4.** Judge-human agreement per cohort.
- **K5.** Position consistency rate per cohort.

## Frozen predictions

| # | Prediction | Kind |
|---|---|---|
| Q1 | Between two styled answers, the judge's structure weight is above 0 (CI above 0), where humans' is not | seek |
| Q2 | Between two unstyled answers, the judge rewards structure (CI above 0) | guard |
| Q3 | Judge struct/length ratio exceeds humans' in the both-styled cohort (paired CI above 0) | seek |
| Q4 | Judge-human agreement is lower in the both-styled cohort than in the neither-styled cohort | seek |
| Q5 | The judge rewards length and penalises repetition in every cohort | guard |

## Amendment (2026-11-11, before any judge call)

- **Cohorts reduced to two:** both styled and neither styled. The mixed cohort is dropped to cut cost; it is not needed for Q1-Q4. Q4 and K4 compare these two cohorts only.
- **Claude-family battles excluded from the primary analysis:** the judge is a Claude model and Arena 2024 includes Claude answers, so any battle with a model whose name contains "claude" is left out of the judged sample (self-preference risk).
- **Pilot first:** 10 battles (20 requests) are sent synchronously to check verdict parsing and token use before the batch. Pilot verdicts are not used in the analysis.
- Sample per cohort: up to 2,897 battles after the exclusion, seed 0.

## Amendment 2 (2026-11-11, after a 20-request pilot, before the batch)

The first pilot (claude-opus-5-5; `results/pilot1_lastline.json`) left 2 of 20 verdicts unparsed and used 250-950 output tokens per request (estimate ~$110 for the batch). The judge now returns its verdict through structured output (JSON field `verdict`: A, B or tie), with the system prompt's last sentence changed to match. The comparison instructions are unchanged. Position consistency in the pilot was 7 of 10 battles; the tie rule for inconsistent pairs stays as planned. Pilot verdicts are not used in the analysis.

## Amendment 3 (2026-11-11, second pilot, before the batch)

Second pilot (claude-opus-5-5, structured verdict, effort low; `results/pilot_claude-opus-5-5.json`): 0 of 20 unparsed; 13 of 20 requests used 12 output tokens (verdict without visible deliberation); position-consistent 6 of 10 battles; batch estimate ~$56. The batch runs in this verdict-only mode, which resembles AlpacaEval 2's single-verdict judge. Expected usable (position-consistent, non-tie) battles: roughly 60% per cohort. Position consistency per cohort is reported (K5).
