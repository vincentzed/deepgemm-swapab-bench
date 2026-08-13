# /// script
# requires-python = ">=3.12"
# dependencies = ["typer", "rich", "vl-convert-python", "pillow"]
# ///
"""Reusable visuals — vendored from vincentzed/my-skills
(measured-gpu-writeup/scripts/thread_visuals.py), lightly adapted:
PLOT_FONT -> Inter (registered from ~/.local/share/fonts/inter at render
time), line_plot grows log2-x / custom ticks / reference-line support,
bar_plot grows a text_format param.

Import the helpers (or copy this file into the writeup repo's scripts/) and
feed them MEASURED data read from logs/ — never staged numbers. Rasterize the
SVGs afterwards with svg_to_png.sh (same directory; pass the media dir).

Color rules baked in (from the dataviz method — palette is the validated
8-slot dark-mode categorical set; surface #292929):

- Slots are assigned in FIXED order, never cycled, never re-ranked.
- The full 8 slots are legal only when every mark carries its own label
  (categorical_grid REQUIRES the digit in each cell — identity is never
  color-alone). Line/bar charts cap at 3 series; fold the rest or facet.
- Magnitude comparisons (hbars) use ONE hue, not a rainbow.
- Never two y-axes. Two scales -> two charts or index to a common base.

Helpers:
  save(renderable, out_svg, ...)      rich renderable -> terminal-chrome SVG
  categorical_grid(...)               bank-map style colored digit grid
  results_table(...)                  measured-results table
  code_panel(...)                     syntax-highlighted code (monokai)
  receipt(...)                        predicted-vs-measured panel
  hbars(...)                          single-hue horizontal bars, labeled
  line_plot(...) / bar_plot(...)      blog-card vega-lite PNGs (vl-convert)

Demo/smoke-test:  uv run thread_visuals.py --demo --out /tmp/tv-demo
"""

from pathlib import Path
from typing import Annotated

from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

import typer

# validated 8-slot dark-mode categorical palette (fixed order; see docstring)
PALETTE = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
]
SURFACE = "#292929"
GOOD, BAD = "#199e70", "#e66767"  # status accents (before/after), not series


def _ink(hex_color: str) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else "white"


MONO_FONT = "Google Sans Code"  # installed locally; distinctive terminal mono


def save(renderable, out_svg: Path, width: int = 90, title: str = "") -> Path:
    console = Console(record=True, width=width, force_terminal=True)
    console.print(renderable)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    console.save_svg(str(out_svg), title=title or out_svg.stem)
    svg = out_svg.read_text()
    out_svg.write_text(svg.replace("Fira Code", MONO_FONT))
    return out_svg


def categorical_grid(
    values: list[list[int]],
    title: str,
    col_header: str = "col",
    note: str = "",
) -> Panel:
    """Colored digit grid (e.g. bank-group maps). values[r][c] in 0..7.

    The digit is printed in every cell BY DESIGN — with 8 categories in a
    grid, color alone cannot separate all pairs for CVD readers.
    """
    t = Table(
        show_header=True, header_style="dim", box=None, pad_edge=False, padding=(0, 0)
    )
    t.add_column(col_header, justify="right", style="dim")
    for c in range(len(values[0])):
        t.add_column(f" {c} ", justify="center")
    for r, row in enumerate(values):
        cells = [
            Text(f" {v} ", style=f"bold {_ink(PALETTE[v])} on {PALETTE[v]}")
            for v in row
        ]
        t.add_row(f"{r:2d} ", *cells)
    body = Group(t, *([Text(""), Text(note, style="italic")] if note else []))
    return Panel(body, title=title, border_style="cyan")


def results_table(
    columns: list[str],
    rows: list[list[str]],
    title: str,
    caption: str = "",
) -> Panel:
    t = Table(title=title)
    t.add_column(columns[0])
    for col in columns[1:]:
        t.add_column(col, justify="right")
    for row in rows:
        t.add_row(*row)
    body = Group(
        t, *([Text(""), Text(caption, style="bold italic")] if caption else [])
    )
    return Panel(body, border_style="magenta")


def code_panel(code: str, lang: str = "cpp") -> Syntax:
    return Syntax(code, lang, theme="monokai", word_wrap=False)


def freeze_code(code: str, out_svg: Path, lang: str = "cpp") -> Path:
    """Code screenshot via charmbracelet freeze: window chrome, Google Sans
    Code, dark-purple card matching the plot surface. Preferred over
    code_panel for standalone code images.

    Emits SVG — rasterize with svg_to_png.sh (freeze's own PNG rasterizer
    silently drops text for variable-weight fonts like Google Sans Code;
    Chrome resolves the installed font correctly)."""
    import subprocess
    import tempfile

    out_svg.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=f".{lang}", delete=False) as f:
        f.write(code)
        tmp = f.name
    subprocess.run(
        [
            "freeze",
            tmp,
            "--output",
            str(out_svg),
            "--language",
            lang,
            "--theme",
            "catppuccin-mocha",
            "--window",
            "--background",
            PLOT_SURFACE,
            "--border.radius",
            "12",
            "--padding",
            "24,36,24,28",
            "--font.family",
            MONO_FONT,
            "--font.size",
            "15",
            "--line-height",
            "1.35",
            "--shadow.blur",
            "0",
        ],
        check=True,
        capture_output=True,
    )
    Path(tmp).unlink()
    return out_svg


def receipt(predicted: str, measured: str, model_lines: list[str], delta: str) -> Panel:
    body = Group(
        *[Text(line, style="dim") for line in model_lines],
        Text(""),
        Text.assemble(("predicted   ", "bold"), (predicted, "bold yellow")),
        Text.assemble(("measured    ", "bold"), (measured, "bold green")),
        Text(""),
        Text(delta, style="bold italic"),
    )
    return Panel(
        body, title="pen & paper vs performance counters", border_style="yellow"
    )


def hbars(
    items: list[tuple[str, float]],
    unit: str,
    title: str,
    accent: str = PALETTE[0],
    width: int = 40,
) -> Panel:
    """Single-hue horizontal bars with direct value labels (magnitude job)."""
    peak = max(v for _, v in items)
    label_w = max(len(name) for name, _ in items)
    lines = []
    for name, v in items:
        bar = "█" * max(1, round(width * v / peak))
        lines.append(
            Text.assemble(
                (f"{name:<{label_w}}  ", "bold"),
                (bar, accent),
                (f"  {v:,g} {unit}", "default"),
            )
        )
    return Panel(Group(*lines), title=title, border_style="cyan")


# ── vega-lite plot theme (blog-card aesthetic, validated colorway) ──────────
LABEL_EXPR = (
    "datum.value >= 1000 ? format(datum.value / 1000, '.3~r') + 'k' "
    ": format(datum.value, ',.0f')"
)
PLOT_SURFACE = "#1e1a24"  # purple-tinted near-black card
PLOT_SERIES = ["#9174e8", "#ba8420", "#249f85", "#4f92dd"]  # validated 4-slot
PLOT_HERO = "#a78bfa"  # brighter emphasis purple (lone-accent use only)
PLOT_MUTED_SERIES = "#8b8496"  # gray-lavender context line (mute_rest mode)
PLOT_INK = "#eee9f4"
PLOT_INK_MUTED = "#a89fb8"
PLOT_TAN = "#d8c5a3"  # light tan — unit/subtitle accent
PLOT_GRID = "#332d3d"
PLOT_FONT = "Liberation Sans"  # Helvetica-metric stand-in for the original's Helvetica Neue


def _plot_config(width: int, height: int) -> dict:
    return {
        "background": PLOT_SURFACE,
        "width": width,
        "height": height,
        "padding": {"left": 24, "right": 96, "top": 20, "bottom": 20},
        "config": {
            "font": PLOT_FONT,
            "view": {"stroke": None},
            "title": {
                "anchor": "start",
                "color": PLOT_INK,
                "fontSize": 22,
                "fontWeight": 700,
                "offset": 26,
                "subtitleColor": PLOT_TAN,
                "subtitleFontSize": 15,
                "subtitlePadding": 8,
            },
            "axis": {
                "labelColor": PLOT_INK_MUTED,
                "labelFontSize": 14,
                "titleColor": PLOT_INK_MUTED,
                "titleFontSize": 15,
                "titleFontWeight": 400,
                "titlePadding": 12,
                "gridColor": PLOT_GRID,
                "gridWidth": 1,
                "domain": False,
                "ticks": False,
                "labelPadding": 8,
            },
            "legend": {
                "orient": "bottom",
                "title": None,
                "symbolType": "circle",
                "symbolSize": 90,
                "symbolStrokeWidth": 9,
                "labelColor": PLOT_INK_MUTED,
                "labelFontSize": 14,
                "labelLimit": 320,
                "padding": 10,
                "columnPadding": 28,
            },
        },
    }


def _render_vl(spec: dict, out_png: Path, scale: float = 3.0) -> Path:
    import vl_convert as vlc

    out_png.parent.mkdir(parents=True, exist_ok=True)
    png = vlc.vegalite_to_png(spec, scale=scale)
    out_png.write_bytes(png)
    _round_corners(out_png, radius=int(16 * scale))
    return out_png


def _round_corners(png_path: Path, radius: int) -> None:
    from PIL import Image, ImageDraw

    img = Image.open(png_path).convert("RGBA")
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, img.width, img.height), radius=radius, fill=255
    )
    img.putalpha(mask)
    img.save(png_path)


def line_plot(
    x: list[float],
    series: dict[str, list[float]],
    out_png: Path,
    xlabel: str,
    unit: str,
    title: str,
    mute_rest: bool = False,
    zero: bool = True,
    x_log2: bool = False,
    x_ticks: list[float] | None = None,
    y_format: str | None = None,
    hline: float | None = None,
    hline_label: str = "",
) -> Path:
    """Blog-card line chart (vega-lite via vl-convert; no browser, no mpl).

    Series colored by the validated 4-slot colorway in fixed order. Cap: 3
    series — fold or facet more. `mute_rest=True` (exactly 2 series) renders
    series 0 as the hero purple and series 1 as a gray-lavender context
    line — identity still carried by the legend. `unit` renders as the
    muted-tan subtitle riding the top of the y axis (reference style).
    """
    if len(series) > 3:
        raise ValueError("cap at 3 series for line plots — fold or facet")
    if mute_rest and len(series) != 2:
        raise ValueError("mute_rest is the hero-vs-context pattern: 2 series")
    names = list(series)
    colors = [PLOT_HERO, PLOT_MUTED_SERIES] if mute_rest else PLOT_SERIES[: len(names)]
    values = [
        {"x": xv, "y": ys[i], "series": name}
        for name, ys in series.items()
        for i, xv in enumerate(x)
    ]
    spec = _plot_config(760, 380) | {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": {"text": title, "subtitle": unit},
        "data": {"values": values},
        "layer": ([{
            "mark": {"type": "rule", "color": PLOT_INK_MUTED,
                     "strokeDash": [5, 5], "strokeWidth": 1.2},
            "encoding": {"y": {"datum": hline},
                         "x": None, "color": None},
        }] if hline is not None else []) + ([{
            "mark": {"type": "text", "align": "left", "dx": 4, "dy": -8,
                     "color": PLOT_INK_MUTED, "fontSize": 13},
            "encoding": {"y": {"datum": hline}, "x": {"datum": x[0]},
                         "text": {"datum": hline_label}, "color": None},
        }] if hline is not None and hline_label else []) + [{
            "mark": {
                "type": "line",
                "strokeWidth": 2.5,
                "interpolate": "linear",
                "strokeCap": "round",
            },
        }, {
            "mark": {"type": "point", "filled": True, "size": 55},
        }],
        "encoding": {
            "x": {
                "field": "x",
                "type": "quantitative",
                "title": xlabel,
                "axis": {"grid": True, "format": "~s", "tickCount": 8}
                        | ({"values": x_ticks, "format": ",.0f"} if x_ticks else {}),
                "scale": {"nice": False, "zero": False}
                         | ({"type": "log", "base": 2} if x_log2 else {}),
            },
            "y": {
                "field": "y",
                "type": "quantitative",
                "title": None,
                "axis": {"tickCount": 6, "format": y_format} if y_format
                        else {"tickCount": 6, "labelExpr": LABEL_EXPR},
                "scale": {"zero": zero},
            },
            "color": {
                "field": "series",
                "type": "nominal",
                "scale": {"domain": names, "range": colors},
                "sort": None,
            },
        },
    }
    return _render_vl(spec, out_png)


def grouped_bar_plot(
    groups: list[str],
    series: dict[str, list[float]],
    out_png: Path,
    unit: str,
    title: str,
    xlabel: str = "",
    label_format: str = ".1f",
    mute_second: bool = False,
    width: int = 760,
) -> Path:
    """Blog-card grouped vertical bars, direct value label on every bar.

    `mute_second=True` (exactly 2 series) renders series 0 as the hero purple
    and series 1 as the gray-lavender context — the hero-vs-context pattern.
    """
    names = list(series)
    colors = [PLOT_HERO, PLOT_MUTED_SERIES] if mute_second and len(names) == 2 \
        else PLOT_SERIES[: len(names)]
    values = [{"group": g, "series": name, "value": series[name][i]}
              for name in names for i, g in enumerate(groups)]
    spec = _plot_config(width, 380) | {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": {"text": title, "subtitle": unit},
        "data": {"values": values},
        "layer": [
            {"mark": {"type": "bar", "cornerRadiusEnd": 3}},
            {"mark": {"type": "text", "align": "center", "dy": -10,
                      "color": PLOT_INK, "fontSize": 12, "fontWeight": 700},
             "encoding": {"text": {"field": "value", "format": label_format}}},
        ],
        "encoding": {
            "x": {"field": "group", "type": "nominal", "sort": None,
                  "title": xlabel or None,
                  "axis": {"grid": False, "labelAngle": 0,
                           "labelFontSize": 14}},
            "xOffset": {"field": "series", "sort": None},
            "y": {"field": "value", "type": "quantitative", "title": None,
                  "axis": {"tickCount": 6, "labelExpr": LABEL_EXPR},
                  "scale": {"zero": True}},
            "color": {"field": "series", "type": "nominal",
                      "scale": {"domain": names, "range": colors},
                      "sort": None},
        },
    }
    return _render_vl(spec, out_png)


def bar_plot(
    items: list[tuple[str, float]],
    out_png: Path,
    unit: str,
    title: str,
    text_format: str = ",.4~r",
) -> Path:
    """Blog-card bar chart: one hue (magnitude job), direct value labels."""
    values = [{"name": k, "value": v} for k, v in items]
    spec = _plot_config(640, 60 * len(items) + 40) | {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": {"text": title, "subtitle": unit},
        "data": {"values": values},
        "layer": [
            {
                "mark": {
                    "type": "bar",
                    "color": PLOT_HERO,
                    "height": 30,
                    "cornerRadiusEnd": 4,
                }
            },
            {
                "mark": {
                    "type": "text",
                    "align": "left",
                    "dx": 8,
                    "color": PLOT_INK,
                    "fontSize": 14,
                    "fontWeight": 600,
                },
                "encoding": {"text": {"field": "value", "format": text_format}},
            },
        ],
        "encoding": {
            "y": {
                "field": "name",
                "type": "nominal",
                "title": None,
                "sort": None,
                "axis": {"grid": False, "labelFontSize": 15},
            },
            "x": {
                "field": "value",
                "type": "quantitative",
                "title": None,
                "axis": {"grid": True, "format": "~s", "tickCount": 6},
                "scale": {"domain": [0, max(v for _, v in items) * 1.15]},
            },
        },
    }
    return _render_vl(spec, out_png)


def main(
    demo: Annotated[
        bool, typer.Option("--demo", help="Render sample visuals.")
    ] = False,
    out: Annotated[Path, typer.Option(help="Output dir for demo files.")] = Path(
        "tv-demo"
    ),
) -> None:
    """Smoke-test the visual helpers (renders one of each to --out)."""
    if not demo:
        typer.echo("import the helpers, or run with --demo")
        raise typer.Exit()
    grid = [[c ^ (r % 8) for c in range(8)] for r in range(8)]
    save(
        categorical_grid(
            grid,
            "Swizzle<3,4,3> bank groups",
            note="warp reads a column -> all 8 groups",
        ),
        out / "demo-grid.svg",
        width=60,
    )
    save(
        results_table(
            ["variant", "time", "speedup"],
            [["naive", "3.618 ms", "1.00x"], ["swizzle", "0.463 ms", "7.81x"]],
            "demo results",
        ),
        out / "demo-table.svg",
        width=60,
    )
    save(
        receipt(
            "939,524,096",
            "939,608,900",
            ["33,554,432 instructions x 28 extra wavefronts"],
            "off by 0.009%",
        ),
        out / "demo-receipt.svg",
        width=64,
    )
    save(
        hbars([("naive", 28.7), ("swizzle", 223.9)], "TFLOP/s", "GEMM throughput"),
        out / "demo-hbars.svg",
        width=70,
    )
    import random

    rng = random.Random(0)
    steps = list(range(300, 501, 2))
    hero = [1000 + rng.gauss(0, 28) for _ in steps]
    base = [700 + rng.gauss(0, 22) for _ in steps]
    line_plot(
        steps,
        {"swizzled": hero, "naive": base},
        out / "demo-line.png",
        "Step",
        "GB/s shared-memory read",
        "Chunk-column read throughput (B300)",
        mute_rest=True,
    )
    bar_plot(
        [("naive", 28.7), ("Swizzle<3,4,3>", 223.9)],
        out / "demo-bar.png",
        "TFLOP/s",
        "mma.sync GEMM throughput (B300)",
    )
    freeze_code(
        "// the entire technology\n"
        "apply(addr) = addr ^ ((addr & 0x380) >> 3);  // ZZZ ^= YYY",
        out / "demo-freeze.svg",
    )
    typer.echo(f"demo files in {out}/ — rasterize SVGs with svg_to_png.sh {out}")


if __name__ == "__main__":
    typer.run(main)
