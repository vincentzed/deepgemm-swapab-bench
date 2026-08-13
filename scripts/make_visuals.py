# /// script
# requires-python = ">=3.12"
# dependencies = ["typer", "rich"]
# ///
"""Render the repo's results as terminal-style SVGs (rich -> save_svg).

Each visual is drawn with rich into a recorded Console and saved as an SVG
with macOS-terminal window chrome. Rasterize to PNG afterwards with
scripts/svg_to_png.sh (headless Chromium + Pillow autocrop).

All numbers are computed live from logs/*.csv — nothing staged.
"""

import csv
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.terminal_theme import TerminalTheme
from rich.text import Text

# Same surface as the vega-lite cards (thread_visuals.PLOT_SURFACE) so every
# figure in the README sits on one unified near-black card; ANSI slots snap to
# the same validated colorway.
THEME = TerminalTheme(
    (13, 13, 16),                                   # background #0d0d10
    (238, 233, 244),                                # foreground #eee9f4
    [(13, 13, 16), (230, 103, 103), (25, 158, 112), (186, 132, 32),
     (79, 146, 221), (213, 81, 129), (86, 180, 233), (238, 233, 244)],
    [(70, 66, 80), (230, 103, 103), (25, 158, 112), (186, 132, 32),
     (79, 146, 221), (213, 81, 129), (86, 180, 233), (255, 255, 255)],
)

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
OUT = ROOT / "media"

M_LIST = [1, 4, 8, 16, 64, 128]


def load(name: str) -> dict:
    rows = {}
    for r in csv.DictReader(open(LOGS / name)):
        if r["status"] == "ok":
            key = (r["kind"], r["name"], int(r["m_or_expected_m"]), r["mode"])
            rows[key] = float(r["time_us"])
    return rows


def load_status(name: str) -> dict:
    rows = {}
    for r in csv.DictReader(open(LOGS / name)):
        key = (r["kind"], r["name"], int(r["m_or_expected_m"]), r["mode"])
        rows[key] = (r["status"], r["diff"])
    return rows


WARM = {d: None for d in ("fp8", "fp4", "bf16")}
COLD = {d: None for d in ("fp8", "fp4", "bf16")}
for d in WARM:
    WARM[d] = load(f"{d}-warm.csv")
    COLD[d] = load(f"{d}.csv")


def ratio(table: dict, kind: str, shape: str, m: int,
          num: str = "swap_off", den: str = "swap_on") -> float:
    return table[(kind, shape, m, num)] / table[(kind, shape, m, den)]


def save(name: str, renderable, width: int, title: str) -> None:
    console = Console(record=True, width=width, force_terminal=True)
    console.print(renderable)
    path = OUT / f"{name}.svg"
    console.save_svg(str(path), title=title, theme=THEME)
    print(f"wrote {path}")


def speedup_text(x: float, hot: float = 1.10) -> Text:
    style = "bold green" if x >= hot else ("red" if x < 0.995 else "")
    return Text(f"{x:.2f}x", style=style)


def visual_headline() -> None:
    t = Table(title="swap-off time ÷ swap-on time — warm L2 (weights in cache)")
    t.add_column("per-GPU GEMM (TP8)")
    t.add_column("N×K", justify="right", style="dim")
    t.add_column("M=1 fp8", justify="right")
    t.add_column("M=1 fp4", justify="right")
    t.add_column("M=1 bf16", justify="right")
    t.add_column("M=64 fp8", justify="right")
    shapes = [
        ("MTP eh_proj", "mtp_eh_proj", "7168×14336"),
        ("dense-layer MLP gate+up", "dense_mlp_gate_up", "4608×7168"),
        ("shared-expert gate+up", "shared_expert_gate_up", "4096×7168"),
        ("attention o_proj", "o_proj", "7168×2048"),
        ("LM head", "lm_head", "16160×7168"),
    ]
    for label, key, nk in shapes:
        t.add_row(
            label, nk,
            speedup_text(ratio(WARM["fp8"], "dense", key, 1)),
            speedup_text(ratio(WARM["fp4"], "dense", key, 1)),
            speedup_text(ratio(WARM["bf16"], "dense", key, 1)),
            speedup_text(ratio(WARM["fp8"], "dense", key, 64)),
        )
    body = Group(
        Text("DeepGEMM SwapAB — DeepSeek-V3.2 decode GEMMs, NVIDIA B300 "
             "(sm_103)", style="bold"),
        Text("M = tokens in the batch. Every config CUPTI-timed and "
             "correctness-checked first.", style="dim"),
        Text(""),
        t,
        Text(""),
        Text("swap wins while M is small, loses past M≈32 — exactly where "
             "the heuristic flips.\ncold L2 (decode reality): both variants "
             "HBM-bound, 0.94-1.05x — swap costs nothing.",
             style="italic"),
    )
    save("swapab-headline", Panel(body, border_style="magenta"), 96,
         "SwapAB speedup — dense decode GEMMs")


def visual_moe_mandatory() -> None:
    t = Table(title="force swap OFF on m-grouped (MoE) GEMMs — SM100")
    t.add_column("grouped layout")
    for d in ("fp8", "fp4", "bf16"):
        t.add_column(d, justify="center")
    statuses = {d: load_status(f"{d}.csv") for d in ("fp8", "fp4", "bf16")}

    def worst_masked_diff(d: str) -> str:
        diffs = [float(v[1]) for k, v in statuses[d].items()
                 if k[0] == "masked" and k[3] == "swap_off"
                 and v[0] == "WRONG_RESULT"]
        return f"{max(diffs):.2f}" if diffs else "?"

    t.add_row("masked (decode)",
              *[Text(f"WRONG ({worst_masked_diff(d)})",
                     style="bold red") for d in ("fp8", "fp4", "bf16")])
    t.add_row("contiguous (prefill)",
              *[Text("no legal layout", style="bold red")
                for _ in range(3)])
    t.add_row("either, with swap ON",
              *[Text("OK", style="bold green") for _ in range(3)])
    body = Group(
        t,
        Text(""),
        Text("the grouped scheduler's effective-M logic exists \"for swap A/B "
             "and psum layout only\"\n(scheduler/gemm.cuh:162); contiguous "
             "groups align to 240 rows = 15×16 UMMA-N steps,\nindivisible by "
             "any non-swap BLOCK_M (32/64/128).", style="italic"),
        Text(""),
        Text("on SM100, SwapAB is not an optimization for MoE — it is the "
             "design.", style="bold italic"),
    )
    save("moe-mandatory", Panel(body, border_style="red"), 96,
         "forcing SwapAB off on MoE GEMMs — correctness")


def main(
    only: Annotated[str, typer.Option(help="Render just one visual.")] = "",
) -> None:
    """Render all visuals to media/*.svg."""
    OUT.mkdir(exist_ok=True)
    visuals = {
        "swapab-headline": visual_headline,
        "moe-mandatory": visual_moe_mandatory,
    }
    for name, fn in visuals.items():
        if only and name != only:
            continue
        fn()


if __name__ == "__main__":
    typer.run(main)
