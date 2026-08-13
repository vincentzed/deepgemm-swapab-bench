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
import re
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.terminal_theme import TerminalTheme
from rich.text import Text

# Terminal panels sit on the SAME card surface as the vega-lite plots:
# humansand's dark paper #1e1a14, ink #f1ece0, with the ANSI slots mapped to
# the humansand accents — teal #5cc9b8 (good), rust #e2836a (bad/emphasis),
# warm muted/faint grays for everything else.
THEME = TerminalTheme(
    (30, 26, 20),                                   # background #1e1a14
    (241, 236, 224),                                # foreground #f1ece0
    [(30, 26, 20), (226, 131, 106), (92, 201, 184), (179, 168, 151),
     (131, 122, 106), (92, 201, 184), (92, 201, 184), (241, 236, 224)],
    [(131, 122, 106), (226, 131, 106), (92, 201, 184), (179, 168, 151),
     (131, 122, 106), (92, 201, 184), (92, 201, 184), (255, 255, 255)],
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
    svg = path.read_text()
    # strip the cdnjs Fira Code @font-face blocks, then point the SVG at the
    # locally-installed mono (the original thread_visuals MONO_FONT)
    svg = re.sub(r"@font-face \{.*?\}\n", "", svg, flags=re.S)
    path.write_text(svg.replace("Fira Code", "Google Sans Code"))
    print(f"wrote {path}")


def speedup_text(x: float, hot: float = 1.10) -> Text:
    style = "bold green" if x >= hot else ("red" if x < 0.995 else "")
    return Text(f"{x:.2f}x", style=style)


def visual_headline() -> None:
    t = Table(title="swap-off ÷ swap-on kernel time · warm L2 · DeepSeek-V3.2 (TP8) · B300")
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
    save("dense-speedup-table", Panel(t, border_style="magenta"), 96,
         "SwapAB speedup — dense decode GEMMs")


def visual_moe_receipt() -> None:
    """Verbatim evidence: what actually happens when swap is forced off."""
    diffs_off, diffs_on = {}, {}
    for d in ("fp8", "fp4", "bf16"):
        for r in csv.DictReader(open(LOGS / f"{d}.csv")):
            if (r["kind"], r["name"], r["m_or_expected_m"]) == \
                    ("masked", "moe_gate_up", "8"):
                if r["mode"] == "swap_off":
                    diffs_off[d] = float(r["diff"])
                elif r["mode"] == "swap_mc":
                    diffs_on[d] = float(r["diff"])
    assert_line = next(
        line for line in open(LOGS / "fp8.log")
        if "swap_off   ERROR" in line and "contig" in line)
    assert_msg = assert_line.split("ERROR: ", 1)[1].strip()
    file_ref = assert_msg.split("(", 1)[1].split(")", 1)[0].split("/")[-1]

    def prompt(kind: str, note: str) -> Text:
        return Text.assemble(
            ("$ ", "dim"), ("DG_FORCE_SWAP_AB=0", "bold red"),
            (f" python bench/bench_swapab.py --kinds {kind}", "bold"),
            (f"   # {note}", "dim"))

    wrong_lines = [
        Text.assemble(
            (f"masked  {d:<5s} moe_gate_up  em=8   ", ""),
            (f"diff={diffs_off[d]:.2e}", "bold red"),
            ("   << WRONG RESULT, not timed", "red"))
        for d in ("fp8", "fp4", "bf16")]
    ok_note = Text.assemble(
        ("  same configs, swap ON:  diff = ", "dim"),
        (" · ".join(f"{diffs_on[d]:.0e}" if diffs_on[d] > 0 else "<1e-06"
                    for d in ("fp8", "fp4", "bf16")),
         "green"), ("  — OK", "green"))

    body = Group(
        prompt("masked", "the decode layout"),
        *wrong_lines,
        ok_note,
        Text(""),
        prompt("contiguous", "the prefill layout"),
        Text.assemble(("ERROR: Assertion error ", "bold red"),
                      (f"({file_ref})", "red"), (":", "bold red")),
        Text('  "DG_FORCE_SWAP_AB left no feasible layout"', style="red"),
        Text("  groups align to 240 rows = 15 × 16 UMMA-N steps — divisible "
             "by no non-swap", style="dim"),
        Text("  BLOCK_M (32 / 64 / 128); the layout cannot exist", style="dim"),
        Text(""),
        Text("on SM100, SwapAB is not an optimization for MoE — it is the "
             "design.", style="bold italic"),
    )
    save("moe-swap-off-receipt", Panel(body, border_style="red"), 96,
         "forcing SwapAB off on MoE grouped GEMMs")


def main(
    only: Annotated[str, typer.Option(help="Render just one visual.")] = "",
) -> None:
    """Render all visuals to media/*.svg."""
    OUT.mkdir(exist_ok=True)
    visuals = {
        "dense-speedup-table": visual_headline,
        "moe-swap-off-receipt": visual_moe_receipt,
    }
    for name, fn in visuals.items():
        if only and name != only:
            continue
        fn()


if __name__ == "__main__":
    typer.run(main)
