#!/usr/bin/env bash
# A7 stage 3: evaluate the four pruned Bolt builds (HE+, MBPP+, BFCL tool calls, measured memory at
# 32K and 128K context), the Bolt-24GB tool-call baseline, decode/prefill speed for all five, then
# the summary that applies the pre-registered criterion choice per tier.
set -euo pipefail
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
cd "$(dirname "$0")/.."
SRV=tools/llama.cpp-upstream/build/bin/llama-server
BENCH=tools/llama.cpp-upstream/build/bin/llama-bench
BOLT=models/nemotron-3.5-bolt-24gb-gguf/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf
R=runs/bolt
TAGS="reap50 gate_mass50 reap25 gate_mass25"
gguf(){ echo "models/bolt-pruned/Nemotron-3.5-Bolt-74B-A9B-$1-IQ2_XXS.gguf"; }
rm -f $R/S3_DONE $R/S3_FAILED
trap 'echo "bolt_stage3 failed at line $LINENO" > runs/bolt/S3_FAILED' ERR

uv run moep toolcall --run $R --server $SRV --gguf "$BOLT" --tag bolt24_iq2xxs
for tag in $TAGS; do
  for ds in humaneval mbpp; do
    uv run moep codeval --run $R --server $SRV --reasoning-format deepseek --gguf "$(gguf $tag)" \
        --tag bolt_$tag --dataset $ds --port 8094
  done
  uv run moep toolcall --run $R --server $SRV --gguf "$(gguf $tag)" --tag bolt_$tag
  uv run python scripts/measure_mem.py $R bolt_$tag "$(gguf $tag)" $SRV 32768 131072
done

models="$BOLT"
for tag in $TAGS; do models="$models,$(gguf $tag)"; done
$BENCH -m "$models" -ngl 99 -p 512 -n 128 -r 3 -o json > $R/bench.json
uv run python scripts/bolt_summary.py
echo done > $R/S3_DONE
