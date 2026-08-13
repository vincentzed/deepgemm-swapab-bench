#!/usr/bin/env python3
"""Build the README headline figure from the benchmark CSVs in logs/."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "..", "logs")
MEDIA = os.path.join(HERE, "..", "media")
os.makedirs(MEDIA, exist_ok=True)

# Validated categorical palette (colorblind-safe on white; see bench-graphs)
BLUE, GREEN, MAGENTA, YELLOW = "#2a78d6", "#008300", "#e87ba4", "#eda100"
INK, MUTED = "#1a1a1a", "#666666"


def load(path):
    rows = {}
    for r in csv.DictReader(open(path)):
        if r["status"] == "ok":
            rows[(r["kind"], r["name"], int(r["m_or_expected_m"]), r["mode"])] = \
                float(r["time_us"])
    return rows


warm_fp8 = load(os.path.join(LOGS, "fp8-warm.csv"))
cold = {d: load(os.path.join(LOGS, f"{d}.csv")) for d in ("fp8", "fp4", "bf16")}

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(13.2, 5.1), dpi=150, facecolor="white",
    gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.24})

# ---- Panel A: dense warm-L2 speedup vs batch size, fp8 -------------------
SHAPES_A = [
    ("mtp_eh_proj", "MTP eh_proj (7168x14336)", BLUE),
    ("dense_mlp_gate_up", "dense-layer MLP gate+up (4608x7168)", GREEN),
    ("o_proj", "attention out-proj (7168x2048)", MAGENTA),
    ("lm_head", "LM head (16160x7168)", YELLOW),
]
M_LIST = [1, 4, 8, 16, 64, 128]
for name, label, color in SHAPES_A:
    ys = [warm_fp8[("dense", name, m, "swap_off")] /
          warm_fp8[("dense", name, m, "swap_on")] for m in M_LIST]
    ax1.plot(range(len(M_LIST)), ys, color=color, linewidth=2,
             marker="o", markersize=7, label=label)

ax1.axhline(1.0, color=MUTED, linewidth=1, linestyle="--", zorder=0)
ax1.text(0.0, 1.012, "same speed", fontsize=9, color=MUTED, ha="left")
ax1.set_xticks(range(len(M_LIST)), [str(m) for m in M_LIST])
ax1.set_xlabel("tokens in the batch (decode batch size per GPU)", fontsize=10,
               color=INK)
ax1.set_ylabel("how much faster the swapped kernel is\n"
               "(plain kernel time ÷ swapped kernel time)", fontsize=10,
               color=INK)
ax1.set_title(
    "Swapping A and B speeds up small-batch decode GEMMs,\n"
    "and stops helping once the batch passes ~32 tokens",
    fontsize=11.5, color=INK, loc="left", pad=10)
ax1.legend(fontsize=8.5, frameon=False, loc="upper right")
ax1.text(0, 0.845,
         "FP8, DeepSeek-V3.2 per-GPU shapes at 8-way tensor parallel,\n"
         "weights resident in L2. With a cold L2 both variants are\n"
         "equally HBM-bound and tie (full tables in README).",
         fontsize=8.5, color=MUTED, va="bottom")
ax1.set_ylim(0.8, 1.55)

# ---- Panel B: MoE prefill multicast speedup ------------------------------
DTYPES = [("fp8", "FP8", BLUE), ("fp4", "FP4 weights", GREEN),
          ("bf16", "BF16", MAGENTA)]
GROUPS = [("moe_gate_up", "expert gate+up\n(4096x7168, 32 experts)"),
          ("moe_down", "expert down\n(7168x2048, 32 experts)")]
width = 0.24
for j, (dtype, dlabel, color) in enumerate(DTYPES):
    xs, ys = [], []
    for i, (name, _) in enumerate(GROUPS):
        table = cold[dtype]
        ys.append(table[("contiguous", name, 512, "swap_nomc")] /
                  table[("contiguous", name, 512, "swap_mc")])
        xs.append(i + (j - 1) * width)
    # bars anchored at 1.0 ("same speed"): bar length = the multicast gain
    bars = ax2.bar(xs, [y - 1.0 for y in ys], bottom=1.0, width=width - 0.02,
                   color=color, label=dlabel)
    for bar, y in zip(bars, ys):
        ax2.text(bar.get_x() + bar.get_width() / 2, y + 0.006, f"{y:.2f}x",
                 ha="center", fontsize=8.5, color=INK)

ax2.axhline(1.0, color=MUTED, linewidth=1.2, zorder=0)
ax2.text(1.52, 1.006, "same speed", fontsize=8.5, color=MUTED, ha="right")
ax2.set_xticks(range(len(GROUPS)), [g[1] for g in GROUPS], fontsize=9)
ax2.set_ylabel("how much faster with TMA multicast\n"
               "(no-multicast time ÷ multicast time)", fontsize=10, color=INK)
ax2.set_title(
    "TMA multicast — only reachable in the swapped\n"
    "layout — adds up to 1.24x on MoE prefill",
    fontsize=11.5, color=INK, loc="left", pad=10)
ax2.legend(fontsize=8.5, frameon=False, loc="upper right", ncols=1,
           bbox_to_anchor=(1.0, 0.80))
ax2.set_ylim(0.985, 1.40)
ax2.text(-0.36, 1.395,
         "m-grouped contiguous GEMM, 512 tokens per expert\n"
         "(prefill). At decode sizes (1-32 tokens per expert)\n"
         "multicast is neutral.",
         fontsize=8.5, color=MUTED, va="top")

for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, zorder=-5)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=9)

fig.suptitle("SwapAB in DeepGEMM on an NVIDIA B300, measured with CUPTI",
             fontsize=13, color=INK, x=0.015, ha="left", y=1.02)
fig.tight_layout(rect=(0, 0, 1, 0.98))
for ext in ("png", "svg"):
    fig.savefig(os.path.join(MEDIA, f"swapab-headline.{ext}"),
                facecolor="white", bbox_inches="tight")
print("wrote media/swapab-headline.{png,svg}")
