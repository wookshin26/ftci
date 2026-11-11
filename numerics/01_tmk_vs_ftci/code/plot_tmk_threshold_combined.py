#!/usr/bin/env python3
"""Combined threshold and L=12 TMK learning-curve figure.

This uses the dependency-free drawing primitives from
`plot_tmk_vs_tci_plain.py`, because Julia's GR backend is not available in
this execution environment.
"""

from __future__ import annotations

import csv
import math
import os
import re
import sys
from collections import defaultdict

from plot_tmk_vs_tci_plain import (
    COLORS,
    PALETTE,
    Figure,
    Panel,
    group_mean,
    mean,
    read_rows,
    threshold_summary,
)


class LogXPanel(Panel):
    def sx(self, value: float) -> float:
        lo = math.log10(self.xmin)
        hi = math.log10(self.xmax)
        return self.x + (math.log10(value) - lo) / (hi - lo) * self.w


def latest_extended_csv(result_dir: str) -> str:
    matches = [
        os.path.join(result_dir, name)
        for name in os.listdir(result_dir)
        if re.match(r"tmk_vs_tci_lowbond_extended50k_\d+_\d+\.csv$", name)
    ]
    if not matches:
        raise FileNotFoundError("no extended50k sweep CSV found")
    return max(matches, key=os.path.getmtime)


def stem_from_csv(path: str) -> str:
    match = re.search(r"lowbond_(.+)\.csv$", os.path.basename(path))
    return match.group(1) if match else "extended50k"


def short_sample_label(x: float) -> str:
    if x >= 1000:
        return f"{int(x/1000)}k"
    return str(int(x))


def save_summary_csv(rows: list[dict], outdir: str, stem: str, threshold: float) -> str:
    summary = threshold_summary(rows, [threshold])
    path = os.path.join(outdir, f"threshold0p1_samples_{stem}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            ["threshold", "L", "method", "mean_samples", "success_count", "seed_count", "max_tested"],
        )
        writer.writeheader()
        writer.writerows(summary)
    return path


def save_combined_figure(rows: list[dict], outdir: str, stem: str) -> list[str]:
    threshold = 1e-1
    summary = threshold_summary(rows, [threshold])
    lvals = sorted({r["L"] for r in rows if r["method"] == "FTCI"})
    tmk_max = max(r["samples"] for r in rows if r["method"] == "TMK")
    lambda_colors = {1: "blue", 2: "orange", 3: "green", 4: "purple"}

    fig = Figure(1280, 600, font_scale=1.25)
    p1 = Panel(
        86,
        78,
        345,
        350,
        3.5,
        12.5,
        5e2,
        8e4,
        logy=True,
        title="(a) Samples for error <= 0.1",
        xlabel="System size L",
        ylabel="Samples",
    )
    curve_Ls = [4, 8, 10, 12]
    curve_panels = []
    x0s = [570, 925, 570, 925]
    y0s = [78, 78, 330, 330]
    for i, L in enumerate(curve_Ls):
        curve_panels.append(
            LogXPanel(
                x0s[i],
                y0s[i],
                250,
                165,
                50,
                50000,
                8e-2,
                1.2e1,
                logy=True,
                title=f"(b{i + 1}) L={L}",
                xlabel="TMK samples" if i >= 2 else "",
                ylabel="Rel. L2 error" if i % 2 == 0 else "",
            )
        )

    fig.panel_axes(
        p1,
        lvals,
        [1e3, 1e4, 5e4],
        xfmt=lambda x: str(int(x)),
        yfmt=lambda y: "5e4" if abs(y - 5e4) < 1 else f"1e{int(math.log10(y))}",
    )
    fig.text(875, 42, "(b) TMK learning curves", 12, anchor="middle", weight="bold")
    for p in curve_panels:
        fig.panel_axes(
            p,
            [50, 500, 5000, 50000],
            [1e-1, 1, 10],
            xfmt=short_sample_label,
            yfmt=lambda y: "0.1" if y < 1 else str(int(y)),
        )
        fig.line(p.x, p.sy(threshold), p.x + p.w, p.sy(threshold), "red", 1.0, dash="4,4")

    ftci_pts = []
    tmk_pts = []
    missing = []
    for L in lvals:
        ftci = [
            r
            for r in summary
            if r["L"] == L and r["method"] == "FTCI" and r["threshold"] == threshold
        ][0]
        tmk = [
            r
            for r in summary
            if r["L"] == L and r["method"] == "TMK" and r["threshold"] == threshold
        ][0]
        if ftci["mean_samples"] is not None:
            ftci_pts.append((L, ftci["mean_samples"]))
        if tmk["mean_samples"] is not None:
            tmk_pts.append((L, tmk["mean_samples"]))
        else:
            missing.append((L, tmk_max))

    ftci_xy = [(p1.sx(x), p1.sy(y)) for x, y in ftci_pts]
    fig.polyline(ftci_xy, "blue", 2.0)
    for x, y in ftci_xy:
        fig.marker(x, y, "blue", "circle", 3.5)
    if tmk_pts:
        tmk_xy = [(p1.sx(x), p1.sy(y)) for x, y in tmk_pts]
        fig.polyline(tmk_xy, "orange", 2.0)
        for x, y in tmk_xy:
            fig.marker(x, y, "orange", "square", 3.5)
    for L, yval in missing:
        fig.marker(p1.sx(L), p1.sy(yval), "red", "triangle", 4.5, fill=False)

    all_lambdas = sorted({r["lambda"] for r in rows if r["method"] == "TMK"})
    for panel, L in zip(curve_panels, curve_Ls):
        subset = [r for r in rows if r["method"] == "TMK" and r["L"] == L]
        errs = group_mean(subset, ("lambda", "samples"), "rel_l2_error")
        samples = sorted({r["samples"] for r in subset})
        for lam in all_lambdas:
            color = lambda_colors.get(lam, PALETTE[(lam - 1) % len(PALETTE)])
            pts = [(n, errs[(lam, n)]) for n in samples if (lam, n) in errs]
            xy = [(panel.sx(x), panel.sy(y)) for x, y in pts]
            fig.polyline(xy, color, 1.8)
            for x, y in xy:
                fig.marker(x, y, color, "circle", 2.5)

    # Compact legend.
    fig.line(92, 545, 118, 545, "blue", 2.2)
    fig.marker(105, 545, "blue", "circle", 3.8)
    fig.text(130, 550, "FTCI", 10)
    fig.line(200, 545, 226, 545, "orange", 2.2)
    fig.marker(213, 545, "orange", "square", 3.8)
    fig.text(238, 550, "TMK reached", 10)
    fig.marker(380, 545, "red", "triangle", 4.6, fill=False)
    fig.text(396, 550, f"TMK not reached by n={tmk_max}", 10)

    lx = 585
    for idx, lam in enumerate(all_lambdas):
        color = lambda_colors.get(lam, PALETTE[idx % len(PALETTE)])
        y = 552
        x = lx + idx * 142
        fig.line(x, y - 4, x + 24, y - 4, color, 2.0)
        fig.marker(x + 12, y - 4, color, "circle", 3.0)
        fig.text(x + 33, y, f"Lambda={lam}", 10)

    csv_path = save_summary_csv(rows, outdir, stem, threshold)
    paths = [csv_path]
    for ext, saver in [("svg", fig.save_svg), ("pdf", fig.save_pdf)]:
        path = os.path.join(outdir, f"threshold0p1_and_multiL_tmk_{stem}.{ext}")
        saver(path)
        paths.append(path)
    return paths


def main() -> None:
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = latest_extended_csv(os.path.join(os.path.dirname(__file__), "results"))
    csv_path = os.path.abspath(csv_path)
    rows = read_rows(csv_path)
    paths = save_combined_figure(rows, os.path.dirname(csv_path), stem_from_csv(csv_path))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
