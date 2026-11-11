# Reward-model audit: plan (frozen 2026-11-11, before any reward model is downloaded or run)

## Question

Several of the late-2024 AlpacaEval models with the bold-title, dense style were tuned on preferences produced by specific open reward models. In 2024 Arena votes, humans reward the structure direction except between two styled answers. Do the reward models that trained those models keep rewarding it there? And did each tuned model move in its own reward model's preferred direction?

## Reward models

| Reward model | Used to tune (model cards) | Size |
|---|---|---|
| RLHFlow/ArmoRM-Llama3-8B-v0.1 | gemma-2-9b-it-SimPO (data: gemma2-ultrafeedback-armorm), gemma-2-9b-it-WPO-HB (pairs chosen by ArmoRM score), REBEL-Llama-3-8B-Instruct-Armo | 8B |
| general-preference/GPM-Gemma-2B | SPPO-Llama-3-8B-Instruct-GPM-2B, GPO-Llama-3-8B-Instruct-GPM-2B | 2B |
| llm-blender/PairRM | SPPO-Gemma-2-9B-It-PairRM, SPPO-Llama-3-Instruct-8B-PairRM, SPPO-Mistral7B-PairRM | 0.4B |

Each is run as published, no fine-tuning. If one cannot be run, that is recorded and the others proceed.

## Data

The same 5,794 Arena 100k battles as the judge run (`studies/arena_judge/sample.csv`: 2,897 both-styled, 2,897 neither-styled; Claude battles excluded), so reward models, the LLM judge and human voters are compared on identical pairs. AlpacaEval style vectors from `studies/alpacaeval/data/style_by_model.csv`.

## Scoring

ArmoRM and GPM: a scalar score per answer (prompt + answer); preference = which answer scores higher. PairRM: pairwise, both orders, averaged. Ties (exactly equal scores) dropped.

## Analyses (same frozen instrument and fixed units as the Arena studies)

- **M1.** Per reward model and cohort: win ~ dlen + drep4 + struct; human model on the same battles. Bootstrap over prompts.
- **M2.** Ratio β_struct/β_len, reward model − human, per cohort, paired bootstrap.
- **M3.** Agreement with voters per cohort.
- **M4.** Repetition-to-length ratio, reward model vs human (now planned, after the judge run found a large gap).
- **M5 (fingerprint).** For each reward model, fit an 8-axis weight vector on all 5,794 battles. For each tuned model with its base in AlpacaEval (Meta-Llama-3-8B-Instruct for the Llama-3 tunes), compute the shift of its prompt-relative style vector from the base. Cosine of that shift with its own reward model's direction vs with the other reward models' directions.

## Frozen predictions

| # | Prediction | Kind |
|---|---|---|
| R1 | ArmoRM rewards structure between two styled answers (β_struct CI above 0) | seek |
| R2 | For at least two of the three reward models, the struct/length ratio exceeds voters' in the both-styled cohort (paired CI above 0) | seek |
| R3 | All three reward models reward structure between unstyled answers (CI above 0) | guard |
| R4 | Each reward model's agreement with voters is lower in the both-styled cohort than in the neither-styled cohort | seek |
| R5 | Each reward model's repetition-to-length ratio is more negative than voters' (CI excludes 0 in at least one cohort) | seek |
| R6 | Tuned Llama-3-8B models shift from their base more toward their own reward model's direction than toward the others' (own cosine highest for at least 3 of the tuned models with a base) | seek |

## Amendment 1 (2026-11-11, before any scoring)

Runs on Russell's own Runpod account (one PyTorch GPU pod), not the shared space. Libraries pinned (transformers 4.44.2) so the reward models' published code loads. GPM uses the model card's pipeline with attention set to sdpa instead of flash_attention_2. Inputs truncated at 4,096 tokens (ArmoRM, GPM); PairRM uses its own default truncation. Each model is smoke-tested on 5 battles before the full run; smoke outputs are not analysed. Bundle: `rm_audit_pod.zip`.

## Amendment 2 (2026-11-11, during scoring)

ArmoRM's full run stopped at battle 393: when prompt plus answer exceeded 4,096 tokens, truncating token ids removed the chat-template marker ArmoRM's gating layer looks for. Fix: over-long inputs are shortened as text (answer up to half the budget, prompt the rest) and re-templated, so the marker survives. At least one of the first 393 battles was also over the limit and had been cut by token ids, so ArmoRM is rerun from scratch on all 5,794 battles with the fix; the partial file is discarded. GPM and PairRM finished all 5,794 battles. The count of shortened answers is logged.
