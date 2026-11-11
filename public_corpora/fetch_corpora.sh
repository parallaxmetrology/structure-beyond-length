#!/usr/bin/env bash
# Download the four public corpora used by the studies into this folder (about 1 GB).
set -e
cd "$(dirname "$0")"
H=https://huggingface.co/datasets
mkdir -p helpsteer2 mtbench arena100k arena55k
for f in train.jsonl.gz validation.jsonl.gz preference/preference.jsonl.gz; do
  curl -sfL -o helpsteer2/$(basename $f) $H/nvidia/HelpSteer2/resolve/main/$f
done
curl -sfL -o mtbench/human.parquet "$H/lmsys/mt_bench_human_judgments/resolve/main/data/human-00000-of-00001-25f4910818759289.parquet"
curl -sfL -o mtbench/gpt4_pair.parquet "$H/lmsys/mt_bench_human_judgments/resolve/main/data/gpt4_pair-00000-of-00001-c0b431264a82ddc0.parquet"
curl -sfL -o arena100k/arena-explorer-preference-100k.parquet "$H/lmarena-ai/arena-human-preference-100k/resolve/main/data/arena-explorer-preference-100k.parquet"
curl -sfL -o arena55k/train.csv "$H/lmarena-ai/arena-human-preference-55k/resolve/main/train.csv"   # only for reading_frame/check_arena.py
if [ ! -d alpaca_eval ]; then
  git clone -q --filter=blob:none --sparse https://github.com/tatsu-lab/alpaca_eval.git alpaca_eval
  (cd alpaca_eval && git sparse-checkout set --no-cone '/results/*/model_outputs.json' '/src/alpaca_eval/leaderboards/')
fi
# AlpacaEval commit dates per model (style-over-time cohorts); data/added_dates.json already has them
echo "done: $(du -sh . | cut -f1)"
