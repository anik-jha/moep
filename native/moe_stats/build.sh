#!/usr/bin/env bash
# Build moe_stats against the pinned upstream llama.cpp worktree and its shared libraries.
set -euo pipefail
cd "$(dirname "$0")"
L=$(realpath ../../tools/llama.cpp-upstream)
g++ -O3 -march=native -std=c++17 -o moe_stats moe_stats.cpp \
    -I"$L/include" -I"$L/common" -I"$L/ggml/include" -I"$L/vendor" \
    -L"$L/build/bin" -lllama-common -lllama -lggml -lggml-base -Wl,-rpath,"$L/build/bin"
echo "built $(pwd)/moe_stats"
