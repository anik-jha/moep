#!/usr/bin/env bash
# A7 stage 2: routing statistics for Nemotron 3.5 Bolt-24GB over the full calibration mix (each
# conversation truncated to 4096 tokens), then REAP and routing-mass keep-sets at 50% and 25%
# keep, then byte-copy pruned GGUFs, each verified against the source selection.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
SRC=models/nemotron-3.5-bolt-24gb-gguf/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf
R=runs/bolt
STATS=$R/calib/stats_full.json
rm -f $R/S2_DONE $R/S2_FAILED
trap 'echo "bolt_stage2 failed at line $LINENO" > runs/bolt/S2_FAILED' ERR

native/moe_stats/moe_stats -m "$SRC" -f $R/calib/calib_bolt.txt -o "$STATS" -c 4096 -b 4096 -ub 512 -ngl 999

for crit in reap gate_mass; do
  for kp in 0.5:50 0.25:25; do
    keep=${kp%%:*}
    tag="${crit}${kp##*:}"
    uv run moep gguf-score --run $R --stats "$STATS" --gguf "$SRC" --criterion $crit --keep-ratio $keep \
        --out $R/selections/bolt-$tag.json
    uv run moep gguf-prune --run $R --src "$SRC" --selection $R/selections/bolt-$tag.json \
        --out models/bolt-pruned/Nemotron-3.5-Bolt-74B-A9B-$tag-IQ2_XXS.gguf
  done
done
echo done > $R/S2_DONE
