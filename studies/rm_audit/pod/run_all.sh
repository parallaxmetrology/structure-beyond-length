#!/usr/bin/env bash
# Reward-model audit: run on a Runpod PyTorch GPU pod. Usage: bash run_all.sh
# Installs pinned libraries, smoke-tests each reward model on 5 battles, then scores all 5,794.
# Each reward model runs on its own; a failure in one is logged and the others continue.
# Safe to re-run: finished battles are skipped.
cd "$(dirname "$0")"
mkdir -p results
exec > >(tee -a results/run_log.txt) 2>&1
echo "=== start $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv

pip install -q "transformers==4.44.2" "accelerate>=0.33" safetensors huggingface_hub sentencepiece protobuf
pip install -q git+https://github.com/yuchenlin/LLM-Blender.git || echo "!! llm-blender install failed (PairRM will be skipped)"
pip install -q "transformers==4.44.2"   # re-pin in case llm-blender changed it
python -c "import transformers, torch; print('transformers', transformers.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"

for rm in armorm gpm pairrm; do
  echo "=== smoke test: $rm"
  if python score_rm.py $rm --limit 5; then
    echo "=== full run: $rm"
    python score_rm.py $rm || echo "!! $rm full run failed"
  else
    echo "!! $rm smoke test failed; skipping"
  fi
done

tar -czf results.tar.gz results
echo "=== finished $(date). Download results.tar.gz, then STOP and TERMINATE the pod."
