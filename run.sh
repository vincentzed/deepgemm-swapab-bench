#!/usr/bin/env bash
# Reproduce the SwapAB benchmarks from scratch on an SM100-family GPU
# (measured numbers in README/logs came from an NVIDIA B300, sm_103).
#
# Prereqs: torch + flashinfer + cupti-python installed, CUDA toolkit with nvcc
# that supports your arch (the numbers here used the lmsysorg/sglang
# nightly-dev-cu13 container image).
set -euxo pipefail

WORK=${WORK:-$PWD/work}
mkdir -p "$WORK"

# 1. Clone the sgl-project DeepGEMM fork (commit used for the README numbers)
if [ ! -d "$WORK/DeepGEMM-swapab" ]; then
    git clone --recursive https://github.com/sgl-project/DeepGEMM.git "$WORK/DeepGEMM-swapab"
    git -C "$WORK/DeepGEMM-swapab" checkout a348cd95a8ddfd3ef337b595d8e8b841c0d9cf0f
    git -C "$WORK/DeepGEMM-swapab" submodule update --init --recursive
fi

# 2. Apply the two-knob heuristic patch (DG_FORCE_SWAP_AB / DG_FORCE_MULTICAST)
git -C "$WORK/DeepGEMM-swapab" apply --check "$PWD/patches/0001-dg-force-swap-ab-multicast-knobs.patch" &&
git -C "$WORK/DeepGEMM-swapab" apply "$PWD/patches/0001-dg-force-swap-ab-multicast-knobs.patch" || true

# 3. Build (compiles the host-side _C extension; kernels themselves are JIT)
(cd "$WORK/DeepGEMM-swapab" && bash develop.sh)

# 4. Install the only extra dependency flashinfer's CUPTI timer needs
pip install cupti-python

# 5. Run: one process per dtype (use separate GPUs if you have them)
mkdir -p logs
export DG_ROOT="$WORK/DeepGEMM-swapab"
for dtype in fp8 fp4 bf16; do
    DG_JIT_CACHE_DIR="$WORK/dg-cache-$dtype" DG_PRINT_CONFIGS=1 \
        python bench/bench_swapab.py --dtype "$dtype" --out "logs/$dtype.csv" \
        2>&1 | tee "logs/$dtype.log"
done
