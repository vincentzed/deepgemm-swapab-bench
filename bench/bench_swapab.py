#!/usr/bin/env python3
"""SwapAB benchmark for DeepGEMM SM100 kernels on DeepSeek-V3.2 serving shapes.

Requires the patched sgl-project/DeepGEMM build (see patches/): the patch adds
two env knobs to the SM100 layout heuristic, read fresh on every call so one
process can measure all modes:

  DG_FORCE_SWAP_AB   0 = never swap A/B, 1 = swap only, unset = heuristic
  DG_FORCE_MULTICAST 0 = single-CTA clusters only, 1 = multicast only

Timing follows the CUPTI-only house rule: flashinfer's
`bench_gpu_time_with_cupti` with `use_cuda_graph=True`, hardware-counter
kernel time, one captured call replayed per iteration, L2 flushed outside the
measured window. Every configuration is correctness-checked against a float32
matmul reference before it is timed; configurations that fail are recorded and
never timed.

Usage (inside the container, patched DeepGEMM on PYTHONPATH):
  python bench_swapab.py --dtype fp8 --out logs/fp8.csv
"""

import argparse
import csv
import os
import random
import sys

DG_ROOT = os.environ.get("DG_ROOT", "/sgl-workspace/DeepGEMM-swapab")
sys.path.insert(0, DG_ROOT)
sys.path.insert(0, os.path.join(DG_ROOT, "tests"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import deep_gemm  # noqa: E402
from deep_gemm.testing import calc_diff, count_bytes  # noqa: E402
from flashinfer.testing.utils import bench_gpu_time_with_cupti  # noqa: E402
from generators import (  # noqa: E402
    KernelType, MajorTypeAB, QuantConfig,
    generate_normal, generate_m_grouped_contiguous, generate_m_grouped_masked,
    get_ue8m0_usage,
)

import shapes  # noqa: E402


def prepack_sf(sf: torch.Tensor, mn: int, gran_mn: int) -> torch.Tensor:
    """Pre-transform float UE8M0 scales into DeepGEMM's required SM100 layout
    (packed INT, per-row, MN-major, TMA-aligned) OUTSIDE the timed function.

    This matches production serving: sglang packs weight scales once at load
    time and its activation-quant kernels emit packed scales directly, so the
    GEMM call itself must launch exactly one kernel. Passing float scales
    instead would add a transform kernel to every call (and to the timed
    window).
    """
    if gran_mn != 1:  # per-block weight scales -> broadcast to per-row
        idx = torch.arange(mn, device=sf.device) // gran_mn
        sf = sf.index_select(-2, idx)
    return deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)


def prepack_operands(dtype: str, a, b, m: int, n: int):
    """Returns (a, b, recipe_a, recipe_b) with scales pre-packed per-row."""
    quant = QUANT[dtype]
    a = (a[0], prepack_sf(a[1], m, 1))
    b_gran_mn = 1 if quant.is_fp4_b else quant.gran_k_b
    b = (b[0], prepack_sf(b[1], n, b_gran_mn))
    recipe_a = (1, quant.gran_k_a)
    recipe_b = (1, quant.gran_k_b)
    return a, b, recipe_a, recipe_b

# FP8 = per-token 1x128 activations x per-block 128x128 weights, UE8M0 scales
# (matches the DeepSeek-V3.2 checkpoint's quantization_config).
# FP4 = FP8 activations x packed-e2m1 weights with 1x32 UE8M0 scales
# (DeepGEMM's SM100 mxf8f6f4 path; QuantConfig comes from tests/generators.py).
QUANT = {
    "fp8": QuantConfig((128, 128, False, False)),
    "fp4": QuantConfig((128, 32, False, True)),
    "bf16": None,
}
DIFF_THRESHOLD = {"fp8": 0.001, "fp4": 0.01, "bf16": 1e-5}

DENSE_MODES = [
    ("swap_on", {"DG_FORCE_SWAP_AB": "1"}),
    ("swap_off", {"DG_FORCE_SWAP_AB": "0"}),
]
GROUPED_MODES = [
    ("swap_mc", {"DG_FORCE_SWAP_AB": "1"}),
    ("swap_nomc", {"DG_FORCE_SWAP_AB": "1", "DG_FORCE_MULTICAST": "0"}),
    ("swap_off", {"DG_FORCE_SWAP_AB": "0"}),
]
MODE_ENV_KEYS = ("DG_FORCE_SWAP_AB", "DG_FORCE_MULTICAST")


def set_mode(env: dict) -> None:
    for key in MODE_ENV_KEYS:
        os.environ.pop(key, None)
    os.environ.update(env)


def median_us(times_ms) -> float:
    times = sorted(float(t) for t in times_ms)
    mid = len(times) // 2
    ms = times[mid] if len(times) % 2 else 0.5 * (times[mid - 1] + times[mid])
    return ms * 1e3


COLD_L2 = True  # set by --l2; cold = production decode (weights evicted
# between reuses), warm = back-to-back microbenchmark loop


def time_fn(fn, d: torch.Tensor) -> float:
    # `d` is passed via input_args purely so the harness can find a GPU tensor
    # to size/run the cold-L2 flush against (the flush + sync happen before the
    # measured window opens).
    times = bench_gpu_time_with_cupti(
        lambda _d: fn(), dry_run_iters=10, repeat_iters=30, use_cuda_graph=True,
        cold_l2_cache=COLD_L2, input_args=(d,))
    return median_us(times)


def reseed() -> None:
    torch.manual_seed(0)
    random.seed(0)


def bench_dense(dtype: str, name: str, m: int, n: int, k: int, writer, log) -> None:
    use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)
    quant = QUANT[dtype]
    for mode, env in DENSE_MODES:
        set_mode(env)
        reseed()
        a, b, c, d, ref_d = generate_normal(
            m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor, False,
            torch.bfloat16, KernelType.Kernel1D1D,
            use_ue8m0=use_ue8m0, use_bf16=dtype == "bf16", quant_config=quant)

        if dtype == "bf16":
            fn = lambda: deep_gemm.bf16_gemm_nt(a, b, d)  # noqa: E731
        else:
            a, b, recipe_a, recipe_b = prepack_operands(dtype, a, b, m, n)
            fn = lambda: deep_gemm.fp8_fp4_gemm_nt(  # noqa: E731
                a, b, d, disable_ue8m0_cast=not use_ue8m0,
                recipe_a=recipe_a, recipe_b=recipe_b)

        fn()
        torch.cuda.synchronize()
        diff = calc_diff(d, ref_d)
        ok = diff < DIFF_THRESHOLD[dtype]
        us = time_fn(fn, d) if ok else float("nan")
        tflops = 2 * m * n * k / (us * 1e-6) / 1e12 if ok else 0.0
        gbps = count_bytes(a, b, d) / 1e9 / (us * 1e-6) if ok else 0.0
        writer.writerow(["dense", dtype, name, m, n, k, 1, mode,
                         f"{us:.3f}", f"{tflops:.1f}", f"{gbps:.1f}",
                         f"{diff:.6f}", "ok" if ok else "WRONG_RESULT"])
        log(f"dense {dtype} {name:22s} m={m:<5d} {mode:10s} "
            f"{us:9.2f} us {tflops:7.1f} TF {gbps:7.1f} GB/s diff={diff:.2e}"
            + ("" if ok else "  << WRONG RESULT, not timed"))


def bench_masked(dtype: str, name: str, groups: int, expected_m: int, n: int,
                 k: int, writer, log) -> None:
    use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)
    quant = QUANT[dtype]
    max_m = 256
    for mode, env in GROUPED_MODES:
        set_mode(env)
        alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(
            int(expected_m * 1.2))
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
        reseed()
        a, b, masked_m, psum_m, d, ref_d = generate_m_grouped_masked(
            groups, max_m, expected_m, n, k,
            use_ue8m0=use_ue8m0, use_bf16=dtype == "bf16", quant_config=quant)

        if dtype == "bf16":
            fn = lambda: deep_gemm.m_grouped_bf16_gemm_nt_masked(  # noqa: E731
                a, b, d, masked_m, expected_m)
        else:
            a, b, recipe_a, recipe_b = prepack_operands(dtype, a, b, max_m, n)
            fn = lambda: deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(  # noqa: E731
                a, b, d, masked_m, expected_m,
                disable_ue8m0_cast=not use_ue8m0,
                recipe_a=recipe_a, recipe_b=recipe_b)

        try:
            fn()
            torch.cuda.synchronize()
        except Exception as exc:  # infeasible layout under forced mode
            writer.writerow(["masked", dtype, name, expected_m, n, k, groups,
                             mode, "nan", "0", "0", "nan", f"ERROR:{type(exc).__name__}"])
            log(f"masked {dtype} {name:22s} em={expected_m:<4d} {mode:10s} ERROR: {exc}")
            continue

        diff = max(
            calc_diff(d[j, :masked_m[j].item()], ref_d[j, :masked_m[j].item()])
            for j in range(groups) if masked_m[j].item() > 0)
        ok = diff < DIFF_THRESHOLD[dtype]
        us = time_fn(fn, d) if ok else float("nan")
        valid_m = masked_m.sum().item()
        tflops = 2 * valid_m * n * k / (us * 1e-6) / 1e12 if ok else 0.0
        weight_bytes = count_bytes(b)
        act_bytes = count_bytes(a, d) * valid_m / (max_m * groups)
        gbps = (weight_bytes + act_bytes) / 1e9 / (us * 1e-6) if ok else 0.0
        writer.writerow(["masked", dtype, name, expected_m, n, k, groups, mode,
                         f"{us:.3f}", f"{tflops:.1f}", f"{gbps:.1f}",
                         f"{diff:.6f}", "ok" if ok else "WRONG_RESULT"])
        log(f"masked {dtype} {name:22s} em={expected_m:<4d} {mode:10s} "
            f"{us:9.2f} us {tflops:7.1f} TF {gbps:7.1f} GB/s diff={diff:.2e}"
            + ("" if ok else "  << WRONG RESULT, not timed"))


def bench_contiguous(dtype: str, name: str, groups: int, expected_m: int,
                     n: int, k: int, writer, log) -> None:
    use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)
    quant = QUANT[dtype]
    for mode, env in GROUPED_MODES:
        set_mode(env)
        alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
        reseed()
        m, a, b, grouped_layout, d, ref_d = generate_m_grouped_contiguous(
            groups, expected_m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
            use_ue8m0=use_ue8m0, use_bf16=dtype == "bf16", quant_config=quant)

        if dtype == "bf16":
            fn = lambda: deep_gemm.m_grouped_bf16_gemm_nt_contiguous(  # noqa: E731
                a, b, d, grouped_layout)
        else:
            a, b, recipe_a, recipe_b = prepack_operands(dtype, a, b, m, n)
            fn = lambda: deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(  # noqa: E731
                a, b, d, grouped_layout, disable_ue8m0_cast=not use_ue8m0,
                recipe_a=recipe_a, recipe_b=recipe_b)

        try:
            fn()
            torch.cuda.synchronize()
        except Exception as exc:
            writer.writerow(["contiguous", dtype, name, expected_m, n, k, groups,
                             mode, "nan", "0", "0", "nan", f"ERROR:{type(exc).__name__}"])
            log(f"contig {dtype} {name:22s} em={expected_m:<4d} {mode:10s} ERROR: {exc}")
            continue

        diff = calc_diff(d, ref_d)
        ok = diff < DIFF_THRESHOLD[dtype]
        us = time_fn(fn, d) if ok else float("nan")
        tflops = 2 * m * n * k / (us * 1e-6) / 1e12 if ok else 0.0
        gbps = count_bytes(a, b, d) / 1e9 / (us * 1e-6) if ok else 0.0
        writer.writerow(["contiguous", dtype, name, expected_m, n, k, groups,
                         mode, f"{us:.3f}", f"{tflops:.1f}", f"{gbps:.1f}",
                         f"{diff:.6f}", "ok" if ok else "WRONG_RESULT"])
        log(f"contig {dtype} {name:22s} em={expected_m:<4d} m={m:<6d} {mode:10s} "
            f"{us:9.2f} us {tflops:7.1f} TF {gbps:7.1f} GB/s diff={diff:.2e}"
            + ("" if ok else "  << WRONG RESULT, not timed"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("fp8", "fp4", "bf16"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--kinds", default="dense,masked,contiguous")
    parser.add_argument("--l2", choices=("cold", "warm"), default="cold")
    args = parser.parse_args()
    global COLD_L2
    COLD_L2 = args.l2 == "cold"
    kinds = args.kinds.split(",")

    assert "DeepGEMM-swapab" in deep_gemm.__file__, "not the patched build"
    print(f"deep_gemm: {deep_gemm.__file__}")
    print(f"device: {torch.cuda.get_device_name()}, "
          f"SMs: {deep_gemm.get_num_sms()}", flush=True)

    def log(msg: str) -> None:
        print(msg, flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kind", "dtype", "name", "m_or_expected_m", "n", "k",
                         "groups", "mode", "time_us", "tflops", "gbps", "diff",
                         "status"])
        if "dense" in kinds:
            for name, n, k, _ in shapes.DENSE_TP8:
                for m in shapes.DENSE_M_LIST:
                    bench_dense(args.dtype, name, m, n, k, writer, log)
                    f.flush()
            for name, n, k, _ in shapes.DENSE_TP1:
                for m in shapes.DENSE_M_LIST_TP1:
                    bench_dense(args.dtype, name, m, n, k, writer, log)
                    f.flush()
        if "masked" in kinds:
            for name, groups, n, k, _ in shapes.GROUPED_EP8:
                for expected_m in shapes.MASKED_EXPECTED_M_LIST:
                    bench_masked(args.dtype, name, groups, expected_m, n, k,
                                 writer, log)
                    f.flush()
        if "contiguous" in kinds:
            for name, groups, n, k, _ in shapes.GROUPED_EP8:
                for expected_m in shapes.CONTIGUOUS_EXPECTED_M_LIST:
                    bench_contiguous(args.dtype, name, groups, expected_m, n, k,
                                     writer, log)
                    f.flush()
    print("done")


if __name__ == "__main__":
    main()
