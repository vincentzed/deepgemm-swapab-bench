# /// script
# requires-python = ">=3.12"
# dependencies = ["typer", "rich", "vl-convert-python", "pillow"]
# ///
"""Vega-lite blog-card plots (media/*.png) from the benchmark CSVs in logs/.

Companion to make_visuals.py (which renders the terminal-style panels);
plot helpers live in thread_visuals.py. All numbers computed live from
logs/*.csv — nothing staged.
"""

import csv
from pathlib import Path

import vl_convert as vlc

from thread_visuals import grouped_bar_plot

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
MEDIA = ROOT / "media"

FONTS = Path.home() / ".local" / "share" / "fonts"
for sub in ("liberation", "gsc", "inter"):
    if (FONTS / sub).is_dir():
        vlc.register_font_directory(str(FONTS / sub))

M_LIST = [1, 4, 8, 16, 64, 128]


def load(name: str) -> dict:
    rows = {}
    for r in csv.DictReader(open(LOGS / name)):
        if r["status"] == "ok":
            rows[(r["kind"], r["name"], int(r["m_or_expected_m"]), r["mode"])] = \
                float(r["time_us"])
    return rows


def sweep() -> None:
    facet_series = {}
    for dtype, label in (("fp8", "fp8"), ("fp4", "fp4 weights"),
                         ("bf16", "bf16")):
        warm = load(f"{dtype}-warm.csv")
        facet_series[label] = {
            "swap on": [warm[("dense", "dense_mlp_gate_up", m, "swap_on")]
                        for m in M_LIST],
            "swap off": [warm[("dense", "dense_mlp_gate_up", m, "swap_off")]
                         for m in M_LIST],
        }
    grouped_bar_plot(
        [str(m) for m in M_LIST], {}, MEDIA / "swapab-sweep.png",
        unit="kernel time, microseconds (lower is better) · "
             "dense MLP gate+up 4608×7168 · warm L2 · NVIDIA B300",
        title="Dense GEMM kernel time by batch size — swap on vs. off",
        xlabel="tokens in the batch (decode batch size per GPU)",
        label_format=".1f", mute_second=True, facet_series=facet_series,
    )
    print("wrote media/swapab-sweep.png")


def multicast() -> None:
    cold = {d: load(f"{d}.csv") for d in ("fp8", "fp4", "bf16")}
    groups, on, off = [], [], []
    for label, key in (("gate+up", "moe_gate_up"), ("down", "moe_down")):
        for d in ("fp8", "fp4", "bf16"):
            groups.append(f"{label} · {d}")
            on.append(cold[d][("contiguous", key, 512, "swap_mc")])
            off.append(cold[d][("contiguous", key, 512, "swap_nomc")])
    grouped_bar_plot(
        groups, {"multicast on": on, "multicast off": off},
        MEDIA / "moe-multicast.png",
        unit="kernel time, microseconds (lower is better) · m-grouped "
             "contiguous · 32 experts/GPU · 512 tokens/expert · NVIDIA B300",
        title="MoE prefill expert GEMM time — with vs. without TMA multicast",
        label_format=".0f", mute_second=True, width=860,
    )
    print("wrote media/moe-multicast.png")


if __name__ == "__main__":
    MEDIA.mkdir(exist_ok=True)
    sweep()
    multicast()
