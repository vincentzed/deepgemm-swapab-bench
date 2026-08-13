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

from thread_visuals import bar_plot, line_plot

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
MEDIA = ROOT / "media"

FONT_DIR = Path.home() / ".local" / "share" / "fonts" / "inter"
if FONT_DIR.is_dir():
    vlc.register_font_directory(str(FONT_DIR))

M_LIST = [1, 4, 8, 16, 64, 128]


def load(name: str) -> dict:
    rows = {}
    for r in csv.DictReader(open(LOGS / name)):
        if r["status"] == "ok":
            rows[(r["kind"], r["name"], int(r["m_or_expected_m"]), r["mode"])] = \
                float(r["time_us"])
    return rows


def ratio(table: dict, kind: str, shape: str, m: int,
          num: str, den: str) -> float:
    return table[(kind, shape, m, num)] / table[(kind, shape, m, den)]


def sweep() -> None:
    warm_fp8 = load("fp8-warm.csv")
    warm_fp4 = load("fp4-warm.csv")
    series = {
        "MTP eh_proj (fp8)": [
            ratio(warm_fp8, "dense", "mtp_eh_proj", m, "swap_off", "swap_on")
            for m in M_LIST],
        "dense MLP gate+up (fp8)": [
            ratio(warm_fp8, "dense", "dense_mlp_gate_up", m, "swap_off", "swap_on")
            for m in M_LIST],
        "LM head (fp4 weights)": [
            ratio(warm_fp4, "dense", "lm_head", m, "swap_off", "swap_on")
            for m in M_LIST],
    }
    line_plot(
        M_LIST, series, MEDIA / "swapab-sweep.png",
        xlabel="tokens in the batch (decode batch size per GPU)",
        unit="swap-off kernel time ÷ swap-on kernel time · warm L2 · "
             "per-GPU shapes at TP8 · NVIDIA B300",
        title="SwapAB speedup vs. batch size — DeepSeek-V3.2 dense GEMMs",
        zero=False, x_log2=True, x_ticks=M_LIST, y_format=".2f",
        hline=1.0, hline_label="same speed",
    )
    print("wrote media/swapab-sweep.png")


def multicast() -> None:
    cold = {d: load(f"{d}.csv") for d in ("fp8", "fp4", "bf16")}
    items = []
    for label, key in (("gate+up", "moe_gate_up"), ("down", "moe_down")):
        for d in ("fp8", "fp4", "bf16"):
            gain = (ratio(cold[d], "contiguous", key, 512,
                          "swap_nomc", "swap_mc") - 1.0) * 100
            items.append((f"{label} · {d}", round(gain)))
    bar_plot(
        items, MEDIA / "moe-multicast.png",
        unit="percent faster than no multicast · m-grouped contiguous · "
             "32 experts/GPU · 512 tokens/expert · NVIDIA B300",
        title="TMA multicast speedup — MoE prefill expert GEMMs",
        text_format=".0f",
    )
    print("wrote media/moe-multicast.png")


if __name__ == "__main__":
    MEDIA.mkdir(exist_ok=True)
    sweep()
    multicast()
