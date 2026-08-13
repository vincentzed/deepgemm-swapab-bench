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

from thread_visuals import (_plot_config, _render_vl, grouped_bar_plot,
                            PLOT_HERO, PLOT_INK, PLOT_INK_MUTED, PLOT_SERIES)

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
        [str(m) for m in M_LIST], {}, MEDIA / "dense-kernel-times.png",
        unit="kernel time, microseconds (lower is better) · "
             "dense MLP gate+up 4608×7168 · warm L2 · NVIDIA B300",
        title="Dense GEMM kernel time by batch size — swap on vs. off",
        xlabel="tokens in the batch (decode batch size per GPU)",
        label_format=".1f", mute_second=True, facet_series=facet_series,
    )
    print("wrote media/dense-kernel-times.png")


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
        MEDIA / "moe-multicast-times.png",
        unit="kernel time, microseconds (lower is better) · m-grouped "
             "contiguous · 32 experts/GPU · 512 tokens/expert · NVIDIA B300",
        title="MoE prefill expert GEMM time — with vs. without TMA multicast",
        label_format=".0f", mute_second=True, width=860,
    )
    print("wrote media/moe-multicast-times.png")


DENSE_SHAPES = [
    ("fused_qkv_a", 2112, 7168), ("q_b_proj", 3072, 1536),
    ("o_proj", 7168, 2048), ("indexer_wq_b", 8192, 1536),
    ("indexer_wk_wproj", 192, 7168), ("dense_mlp_gate_up", 4608, 7168),
    ("dense_mlp_down", 7168, 2304), ("shared_expert_gate_up", 4096, 7168),
    ("kv_b_proj", 4096, 512), ("lm_head", 16160, 7168),
    ("mtp_eh_proj", 7168, 14336), ("q_b_proj_tp1", 24576, 1536),
    ("kv_b_proj_tp1", 32768, 512), ("o_proj_tp1", 7168, 16384),
]
PRETTY = {
    "fused_qkv_a": "fused q_a+kv_a", "q_b_proj": "q_b_proj",
    "o_proj": "o_proj", "indexer_wq_b": "indexer wq_b",
    "indexer_wk_wproj": "indexer wk+wproj", "dense_mlp_gate_up": "dense MLP gate+up",
    "dense_mlp_down": "dense MLP down", "shared_expert_gate_up": "shared-expert gate+up",
    "kv_b_proj": "kv_b_proj", "lm_head": "LM head", "mtp_eh_proj": "MTP eh_proj",
    "q_b_proj_tp1": "q_b_proj (TP1)", "kv_b_proj_tp1": "kv_b_proj (TP1)",
    "o_proj_tp1": "o_proj (TP1)",
}


def swap_cluster_sizes() -> dict:
    """Cluster size of each shape's swap-on config, parsed from the
    DG_PRINT_CONFIGS lines captured in logs/fp8.log."""
    import re
    pat = re.compile(r"n=(\d+), k=(\d+),.*?swap_ab=1.*?cluster_m=(\d+), cluster_n=(\d+)")
    out = {}
    for line in open(LOGS / "fp8.log"):
        m = pat.search(line)
        if m:
            n, k, cm, cn = map(int, m.groups())
            out[(n, k)] = max(out.get((n, k), 1), cm * cn)
    return out


def k_deviation_bars() -> None:
    """Warm M=1 swap win per shape, sorted by K, colored by whether the
    swapped layout could use the 2-CTA multicast pairing."""
    warm = load("fp8-warm.csv")
    clusters = swap_cluster_sizes()
    rows = []
    for name, n, k in DENSE_SHAPES:
        on = warm.get(("dense", name, 1, "swap_on"))
        off = warm.get(("dense", name, 1, "swap_off"))
        if not (on and off):
            continue
        mc = clusters.get((n, k), 1) > 1
        rows.append((k, f"{PRETTY[name]} · K={k}", off / on, mc))
    rows.sort(key=lambda r: r[0])
    mc_label = {True: "2-CTA multicast available",
                False: "multicast impossible (odd 128-tile count in N)"}
    values = [{"label": lab, "ratio": round(r, 3), "eligibility": mc_label[mc]}
              for _, lab, r, mc in rows]
    spec = _plot_config(720, 30 * len(values) + 40) | {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": {"text": "The dense SwapAB win grows with K, the depth of each dot product",
                  "subtitle": "swap-off ÷ swap-on kernel time · batch size 1 · fp8 · "
                              "warm L2 · bars anchored at 1.0 = same speed · NVIDIA B300"},
        "data": {"values": values},
        "layer": [
            {"mark": {"type": "rule", "color": PLOT_INK_MUTED, "strokeWidth": 1.2},
             "encoding": {"x": {"datum": 1.0}, "y": None, "color": None}},
            {"mark": {"type": "bar", "height": 17, "cornerRadiusEnd": 3},
             "encoding": {"x2": {"datum": 1.0}}},
            {"mark": {"type": "text", "align": {"expr": "datum.ratio >= 1 ? 'left' : 'right'"},
                      "dx": {"expr": "datum.ratio >= 1 ? 6 : -6"},
                      "color": PLOT_INK, "fontSize": 12, "fontWeight": 700},
             "encoding": {"text": {"field": "ratio", "format": ".2f"}}},
        ],
        "encoding": {
            "y": {"field": "label", "type": "nominal", "sort": None, "title": None,
                  "axis": {"grid": False, "labelFontSize": 13, "labelLimit": 260}},
            "x": {"field": "ratio", "type": "quantitative", "title": None,
                  "axis": {"tickCount": 6, "format": ".1f"},
                  "scale": {"domain": [0.92, 1.42]}},
            "color": {"field": "eligibility", "type": "nominal",
                      "scale": {"domain": [mc_label[True], mc_label[False]],
                                "range": [PLOT_HERO, PLOT_SERIES[1]]},
                      "sort": None},
        },
    }
    _render_vl(spec, MEDIA / "dense-win-by-k.png")
    print("wrote media/dense-win-by-k.png")


if __name__ == "__main__":
    MEDIA.mkdir(exist_ok=True)
    sweep()
    multicast()
    k_deviation_bars()
