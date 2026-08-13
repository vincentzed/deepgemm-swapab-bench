#!/usr/bin/env python3
"""Record which layout the UNMODIFIED heuristic picks (no force knobs set)
for every dense shape/M/dtype, by parsing DG_PRINT_CONFIGS output.

Writes a CSV of (dtype, name, m, n, k, swap_ab, block_m, block_n, clusters).
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import shapes  # noqa: E402

WORKER = r"""
import os, sys, torch
os.environ["DG_PRINT_CONFIGS"] = "1"
for key in ("DG_FORCE_SWAP_AB", "DG_FORCE_MULTICAST"):
    os.environ.pop(key, None)
DG_ROOT = os.environ.get("DG_ROOT", "/sgl-workspace/DeepGEMM-swapab")
sys.path.insert(0, DG_ROOT)
sys.path.insert(0, os.path.join(DG_ROOT, "tests"))
sys.path.insert(0, %(here)r)
import deep_gemm
from bench_swapab import QUANT, prepack_operands
from generators import (KernelType, MajorTypeAB, generate_normal, get_ue8m0_usage)
import shapes

use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)
dtype = %(dtype)r
todo = [(name, n, k, m) for name, n, k, _ in shapes.DENSE_TP8 for m in shapes.DENSE_M_LIST]
todo += [(name, n, k, m) for name, n, k, _ in shapes.DENSE_TP1 for m in shapes.DENSE_M_LIST_TP1]
for name, n, k, m in todo:
    print(f"### PROBE {dtype} {name} m={m} n={n} k={k}", flush=True)
    a, b, c, d, ref_d = generate_normal(
        m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor, False, torch.bfloat16,
        KernelType.Kernel1D1D, use_ue8m0=use_ue8m0,
        use_bf16=dtype == "bf16", quant_config=QUANT[dtype])
    if dtype == "bf16":
        deep_gemm.bf16_gemm_nt(a, b, d)
    else:
        a, b, recipe_a, recipe_b = prepack_operands(dtype, a, b, m, n)
        deep_gemm.fp8_fp4_gemm_nt(a, b, d, disable_ue8m0_cast=False,
                                  recipe_a=recipe_a, recipe_b=recipe_b)
    torch.cuda.synchronize()
"""

LAYOUT_RE = re.compile(
    r"swap_ab=(\d+), block_m=(\d+), block_n=(\d+), block_k=\d+, "
    r"cluster_m=(\d+), cluster_n=(\d+)")


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "auto_choice.csv"
    rows = []
    for dtype in ("fp8", "fp4", "bf16"):
        code = WORKER % {"dtype": dtype, "here": HERE}
        proc = subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True, check=True)
        current = None
        seen_desc = {}
        for line in proc.stdout.splitlines():
            if line.startswith("### PROBE"):
                current = line.split()[2:]
                continue
            match = LAYOUT_RE.search(line)
            if match and current is not None:
                seen_desc[tuple(current)] = match.groups()
                current = None
        for key, (swap, bm, bn, cm, cn) in seen_desc.items():
            name = key[1]
            m = key[2].split("=")[1]
            n = key[3].split("=")[1]
            k = key[4].split("=")[1]
            rows.append((key[0], name, m, n, k, swap, bm, bn, f"{cm}x{cn}"))
        print(f"{dtype}: {len(seen_desc)} configs recorded")

    with open(out_path, "w") as f:
        f.write("dtype,name,m,n,k,swap_ab,block_m,block_n,cluster\n")
        for row in rows:
            f.write(",".join(map(str, row)) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
