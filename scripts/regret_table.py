# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Heuristic regret: how much slower DeepGEMM's own layout choice is than the
best of the two forced modes, per dense configuration.

For every (dtype, shape, M): the unforced heuristic's choice comes from
logs/auto_choice.csv; its time is the forced run of that same mode (the
candidate set within a mode is identical, so the forced winner IS the
heuristic's layout); regret = chosen ÷ best-of-both − 1.

Prints summary stats and the worst rows as a markdown table.
No GPU needed — pure analysis of the committed CSVs.
"""

import csv
import statistics
from pathlib import Path

LOGS = Path(__file__).resolve().parent.parent / "logs"


def load(name: str) -> dict:
    rows = {}
    for r in csv.DictReader(open(LOGS / name)):
        if r["kind"] == "dense" and r["status"] == "ok":
            rows[(r["name"], int(r["m_or_expected_m"]), r["mode"])] = \
                float(r["time_us"])
    return rows


def main() -> None:
    auto = {}
    for r in csv.DictReader(open(LOGS / "auto_choice.csv")):
        auto[(r["dtype"], r["name"], int(r["m"]))] = int(r["swap_ab"])

    tables = {}
    for dt in ("fp8", "fp4", "bf16"):
        tables[(dt, "cold")] = load(f"{dt}.csv")
        tables[(dt, "warm")] = load(f"{dt}-warm.csv")

    rows = []
    for (dt, name, m), chose_swap in auto.items():
        for temp in ("cold", "warm"):
            t = tables[(dt, temp)]
            t_on, t_off = t.get((name, m, "swap_on")), t.get((name, m, "swap_off"))
            if not (t_on and t_off):
                continue
            t_auto = t_on if chose_swap else t_off
            best = min(t_on, t_off)
            rows.append((t_auto / best - 1, dt, name, m,
                         "swap" if chose_swap else "plain", temp, t_auto, best))
    rows.sort(reverse=True)

    for temp in ("cold", "warm"):
        regs = [r[0] for r in rows if r[5] == temp]
        print(f"{temp}: n={len(regs)}  median={statistics.median(regs) * 100:.1f}%  "
              f">=2%: {sum(1 for x in regs if x >= 0.02)}  "
              f">=5%: {sum(1 for x in regs if x >= 0.05)}  "
              f"max={max(regs) * 100:.1f}%")

    print("\n| regret | L2 | dtype | shape | M | heuristic chose | chosen µs | best µs |")
    print("|---:|---|---|---|---:|---|---:|---:|")
    for reg, dt, name, m, mode, temp, ta, tb in rows[:10]:
        print(f"| {reg * 100:.1f}% | {temp} | {dt} | {name} | {m} | {mode} "
              f"| {ta:.2f} | {tb:.2f} |")


if __name__ == "__main__":
    main()
