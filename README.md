# Structure Beyond Length

**What human raters, LLM judges and reward models reward in model answers, beyond length.**

Paper: [PDF](studies/paper/Structure_Beyond_Length.pdf) · [HTML](https://www.parallaxmetrology.com/structure_beyond_length.html) · Instrument: [linguistic-kernel](https://github.com/parallaxmetrology/linguistic-kernel)

Leaderboards correct chat-model evaluations for answer length, but answers also differ in structure. This repository reads four public preference corpora (HelpSteer2, AlpacaEval 2, MT-Bench human judgments, Chatbot Arena 100k) with one fixed, rule-based instrument, the Linguistic Kernel v19, and compares the two answers to the same prompt. Small logistic models fitted on top relate the kernel's eight axes to who won; the axes are fixed and only those weights are learned. Every study's predictions were written to a `PLAN.md` before its data was read, and every number in the paper is regenerated from the files here.

## Findings

| | |
|---|---|
| Within prompts and beyond length, human raters penalise repetition and reward denser, rarer-worded, more segmented, more clause-rich answers | HelpSteer2, MT-Bench, Arena |
| On identical pairs, GPT-4's weight on that structure relative to length is not distinguishable from humans' (wide interval) | MT-Bench |
| Between 2023 and late 2024, AlpacaEval answers moved strongly along that direction, and models spread apart rather than converging, partly by base model | AlpacaEval, 227 models |
| Mid-2024 Arena voters reward it between unstyled answers and stop between two styled answers (the more structured one wins 50%) | Arena, 23,120 battles |
| An LLM judge on the same battles keeps a small structure reward there (+0.075) and penalises repetition several times more than voters | Claude Opus 5.5, 5,794 battles |
| Three open reward models used to train some of those 2024 models keep rewarding structure between styled answers (+0.24 to +0.42 against voters' +0.00) and penalise repetition at least ten times more | ArmoRM, GPM-Gemma-2B, PairRM |
| Against Arena's own style-control features (length, headers, bold, lists), voters' remaining structure reward is small; the judge's and reward models' is not | all raters |

Predictions that were not supported are reported alongside (paper, Appendix A). Notable ones: models did not converge in style; a trainer-specific "fingerprint" of each reward model on its tuned models did not hold.

## What's here

```
kernel/kernel_v19.py        the kernel (published copy from linguistic-kernel; identical readings)
reading_frame/              preparation layer for chat-style text (markdown, eligibility, collapse), with tests
studies/
  helpsteer2/               human 0-4 scores, two answers per prompt
  alpacaeval/               227 models x 805 prompts, GPT-4-turbo judge, style over time
  mtbench/                  humans and GPT-4 on the same pairs
  arena100k/                Chatbot Arena votes, June-August 2024
  arena_judge/              Claude Opus 5.5 judge on 5,794 Arena battles (verdicts included)
  rm_audit/                 ArmoRM, GPM-Gemma-2B, PairRM on the same battles (scores included; pod/ = GPU scripts)
  style_baseline/           every rater refitted with Arena's style-control features
  paper/                    LaTeX source, figures, number generator, PDF and HTML
public_corpora/             empty; fetch_corpora.sh downloads the four corpora here
```

Each study folder has its frozen `PLAN.md` (amendments are dated and recorded before the analyses they affect), one script per analysis, the derived data, and a self-contained `LABBOOK.html`.

## Reproduce

```bash
pip install -r requirements.txt
bash public_corpora/fetch_corpora.sh          # about 1 GB; HelpSteer2, AlpacaEval, MT-Bench, Arena 100k
gunzip -k studies/alpacaeval/data/verdicts.json.gz
python3 reading_frame/test_reading_frame.py   # 18 tests
```

Then, in each study folder, the scripts named in its lab book's "Reproduce" section (for example `studies/helpsteer2`: `read_all.py`, `analyze.py`, `pairwise.py`, `theta_check.py`, `build_labbook.py`). Finally:

```bash
cd studies/paper && python3 make_assets.py && tectonic main.tex && python3 build_html.py
```

The two steps that cost money are not needed to reproduce the analyses, because their outputs are included: the judge verdicts (`studies/arena_judge/results/verdicts.csv`, Anthropic Message Batches, about $55) and the reward-model scores (`studies/rm_audit/results/`, one rented L40S GPU, about $2). To rerun them: `studies/arena_judge/judge.py` (needs `ANTHROPIC_API_KEY`) and `studies/rm_audit/make_battles.py` followed by `pod/run_all.sh` on a GPU machine.

## Not included

- The corpora themselves (downloaded by `fetch_corpora.sh` from their sources).
- Anything containing model answers from Arena or AlpacaEval (`studies/arena_judge/requests.jsonl`, `studies/rm_audit/battles.jsonl`): they fall under each model provider's terms. Scripts rebuild them from the downloaded corpora.

## Licenses

Code: [MIT](LICENSE). Paper, lab books and derived data produced here: [CC BY 4.0](LICENSE-CONTENT). Sources keep their own terms: HelpSteer2 (CC BY 4.0), AlpacaEval (Apache 2.0), MT-Bench human judgments (CC BY 4.0), Chatbot Arena 100k (prompts CC BY 4.0; model outputs under provider terms). Reward-model scores were produced with ArmoRM-Llama3-8B-v0.1, GPM-Gemma-2B and PairRM, run as published under their licenses. Short answer excerpts in the paper and lab books are quoted for analysis.

## Cite

See [CITATION.cff](CITATION.cff). Russell Parrish, *Structure Beyond Length: What Human Raters, LLM Judges and Reward Models Reward in Model Answers*, working draft, Parallax Metrology, November 2026. Analysis carried out with Claude (Anthropic).

Part of [Parallax Metrology](https://www.parallaxmetrology.com).
