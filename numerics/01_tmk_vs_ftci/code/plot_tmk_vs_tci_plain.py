#!/usr/bin/env python3
"""Dependency-free vector plots for the TMK vs FTCI sweep.

This is a fallback for environments where Julia's Plots/GR backend cannot load.
It intentionally uses only the Python standard library and writes SVG/PDF
directly, so the numerical comparison remains reproducible from the CSV.
"""

from __future__ import annotations

import csv
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable


COLORS = {
    "blue": (0.05, 0.27, 0.63),
    "orange": (0.83, 0.34, 0.05),
    "green": (0.12, 0.47, 0.20),
    "red": (0.72, 0.11, 0.11),
    "purple": (0.36, 0.20, 0.65),
    "gray": (0.35, 0.35, 0.35),
    "black": (0.08, 0.08, 0.08),
    "grid": (0.86, 0.86, 0.86),
}
PALETTE = ["blue", "orange", "green", "red", "purple"]


def fmt(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".")


def svg_color(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{max(0, min(255, round(255 * c))):02x}" for c in rgb)


def pdf_color(rgb: tuple[float, float, float], stroke: bool = True) -> str:
    op = "RG" if stroke else "rg"
    return f"{rgb[0]:.4f} {rgb[1]:.4f} {rgb[2]:.4f} {op}"


def escape_svg(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def escape_pdf(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def read_rows(path: str) -> list[dict]:
    rows = []
    numeric = {
        "L",
        "seed",
        "lambda",
        "samples",
        "nfeatures",
        "rel_l2_error",
        "ftci_queries",
        "ftci_unique_queries",
        "ftci_final_bond_max",
        "coeff_energy_fraction",
        "l2_energy_fraction",
        "best_trunc_rel_l2_error",
        "time_sec",
    }
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for key, value in row.items():
                if key in numeric:
                    if value == "" or value.lower() == "missing":
                        parsed[key] = None
                    elif key in {"L", "seed", "lambda", "samples", "nfeatures"}:
                        parsed[key] = int(float(value))
                    else:
                        parsed[key] = float(value)
                else:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def mean(values: Iterable[float]) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def group_mean(rows: Iterable[dict], keys: tuple[str, ...], value_key: str) -> dict[tuple, float]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row[value_key])
    return {key: mean(vals) for key, vals in grouped.items()}


@dataclass
class Panel:
    x: float
    y: float
    w: float
    h: float
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    logy: bool = False
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""

    def sx(self, value: float) -> float:
        if self.xmax == self.xmin:
            return self.x
        return self.x + (value - self.xmin) / (self.xmax - self.xmin) * self.w

    def sy(self, value: float) -> float:
        if self.logy:
            value = max(value, self.ymin)
            lo = math.log10(self.ymin)
            hi = math.log10(self.ymax)
            frac = (math.log10(value) - lo) / (hi - lo)
        else:
            frac = (value - self.ymin) / (self.ymax - self.ymin)
        return self.y + self.h - frac * self.h


class Figure:
    def __init__(self, width: float, height: float, font_scale: float = 1.0):
        self.width = width
        self.height = height
        self.font_scale = font_scale
        self.svg: list[str] = []
        self.pdf: list[str] = []

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = "black",
        width: float = 1.0,
        dash: str | None = None,
    ) -> None:
        rgb = COLORS[color]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.svg.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{svg_color(rgb)}" stroke-width="{width:.2f}"{dash_attr}/>'
        )
        self.pdf += [pdf_color(rgb), f"{width:.3f} w"]
        if dash:
            pattern = " ".join(dash.split(","))
            self.pdf.append(f"[{pattern}] 0 d")
        else:
            self.pdf.append("[] 0 d")
        self.pdf.append(f"{x1:.3f} {self.height - y1:.3f} m {x2:.3f} {self.height - y2:.3f} l S")

    def polyline(
        self,
        points: list[tuple[float, float]],
        color: str,
        width: float = 1.5,
        dash: str | None = None,
    ) -> None:
        if len(points) < 2:
            return
        rgb = COLORS[color]
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.svg.append(
            f'<polyline points="{pts}" fill="none" stroke="{svg_color(rgb)}" '
            f'stroke-width="{width:.2f}" stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>'
        )
        self.pdf += [pdf_color(rgb), f"{width:.3f} w", "[] 0 d" if not dash else f"[{' '.join(dash.split(','))}] 0 d"]
        path = [f"{points[0][0]:.3f} {self.height - points[0][1]:.3f} m"]
        path += [f"{x:.3f} {self.height - y:.3f} l" for x, y in points[1:]]
        self.pdf.append(" ".join(path) + " S")

    def marker(
        self,
        x: float,
        y: float,
        color: str,
        shape: str = "circle",
        size: float = 4.0,
        fill: bool = True,
    ) -> None:
        rgb = COLORS[color]
        fill_color = svg_color(rgb) if fill else "white"
        stroke = svg_color(rgb)
        if shape == "square":
            self.svg.append(
                f'<rect x="{x-size:.2f}" y="{y-size:.2f}" width="{2*size:.2f}" height="{2*size:.2f}" '
                f'fill="{fill_color}" stroke="{stroke}" stroke-width="1.2"/>'
            )
            self.pdf += [pdf_color(rgb), pdf_color(rgb, stroke=False) if fill else "1 1 1 rg"]
            self.pdf.append(
                f"{x-size:.3f} {self.height-(y+size):.3f} {2*size:.3f} {2*size:.3f} re {'B' if fill else 'S'}"
            )
        elif shape == "triangle":
            pts = [(x, y - size), (x - size, y + size), (x + size, y + size)]
            self.svg.append(
                f'<polygon points="{" ".join(f"{a:.2f},{b:.2f}" for a,b in pts)}" '
                f'fill="{fill_color}" stroke="{stroke}" stroke-width="1.2"/>'
            )
            self.pdf += [pdf_color(rgb), pdf_color(rgb, stroke=False) if fill else "1 1 1 rg"]
            pdf_pts = [(a, self.height - b) for a, b in pts]
            self.pdf.append(
                f"{pdf_pts[0][0]:.3f} {pdf_pts[0][1]:.3f} m "
                f"{pdf_pts[1][0]:.3f} {pdf_pts[1][1]:.3f} l "
                f"{pdf_pts[2][0]:.3f} {pdf_pts[2][1]:.3f} l h {'B' if fill else 'S'}"
            )
        else:
            self.svg.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{size:.2f}" fill="{fill_color}" '
                f'stroke="{stroke}" stroke-width="1.2"/>'
            )
            self.pdf += [pdf_color(rgb), pdf_color(rgb, stroke=False) if fill else "1 1 1 rg"]
            k = 0.5522847498 * size
            cy = self.height - y
            self.pdf.append(
                f"{x+size:.3f} {cy:.3f} m "
                f"{x+size:.3f} {cy+k:.3f} {x+k:.3f} {cy+size:.3f} {x:.3f} {cy+size:.3f} c "
                f"{x-k:.3f} {cy+size:.3f} {x-size:.3f} {cy+k:.3f} {x-size:.3f} {cy:.3f} c "
                f"{x-size:.3f} {cy-k:.3f} {x-k:.3f} {cy-size:.3f} {x:.3f} {cy-size:.3f} c "
                f"{x+k:.3f} {cy-size:.3f} {x+size:.3f} {cy-k:.3f} {x+size:.3f} {cy:.3f} c "
                f"{'B' if fill else 'S'}"
            )

    def text(
        self,
        x: float,
        y: float,
        text: str,
        size: float = 10.0,
        color: str = "black",
        anchor: str = "start",
        rotate: float = 0.0,
        weight: str = "normal",
    ) -> None:
        size *= self.font_scale
        rgb = COLORS[color]
        attrs = f'font-size="{size:.1f}" font-family="Helvetica, Arial, sans-serif" fill="{svg_color(rgb)}"'
        if anchor != "start":
            attrs += f' text-anchor="{anchor}"'
        if weight == "bold":
            attrs += ' font-weight="700"'
        transform = f' transform="rotate({rotate:.1f} {x:.2f} {y:.2f})"' if rotate else ""
        self.svg.append(f'<text x="{x:.2f}" y="{y:.2f}" {attrs}{transform}>{escape_svg(text)}</text>')
        self.pdf += [pdf_color(rgb, stroke=False), f"/F1 {size:.2f} Tf"]
        if rotate:
            theta = math.radians(-rotate)
            c = math.cos(theta)
            s = math.sin(theta)
            self.pdf.append(
                f"BT {c:.6f} {s:.6f} {-s:.6f} {c:.6f} {x:.3f} {self.height-y:.3f} Tm ({escape_pdf(text)}) Tj ET"
            )
        else:
            self.pdf.append(
                f"BT 1 0 0 1 {x:.3f} {self.height-y:.3f} Tm ({escape_pdf(text)}) Tj ET"
            )

    def panel_axes(
        self,
        panel: Panel,
        xticks: list[float],
        yticks: list[float],
        xfmt: Callable[[float], str] = fmt,
        yfmt: Callable[[float], str] = fmt,
    ) -> None:
        self.line(panel.x, panel.y, panel.x, panel.y + panel.h, "black", 0.9)
        self.line(panel.x, panel.y + panel.h, panel.x + panel.w, panel.y + panel.h, "black", 0.9)
        for t in xticks:
            x = panel.sx(t)
            self.line(x, panel.y + panel.h, x, panel.y + panel.h + 4, "black", 0.8)
            self.text(x, panel.y + panel.h + 16, xfmt(t), 9, anchor="middle")
        for t in yticks:
            if panel.logy and t <= 0:
                continue
            y = panel.sy(t)
            self.line(panel.x - 4, y, panel.x, y, "black", 0.8)
            self.line(panel.x, y, panel.x + panel.w, y, "grid", 0.5, dash="2,3")
            self.text(panel.x - 8, y + 3, yfmt(t), 8.5, anchor="end", color="gray")
        if panel.title:
            self.text(panel.x + panel.w / 2, panel.y - 12, panel.title, 11, anchor="middle", weight="bold")
        if panel.xlabel:
            self.text(panel.x + panel.w / 2, panel.y + panel.h + 34, panel.xlabel, 10, anchor="middle")
        if panel.ylabel:
            self.text(panel.x - 44, panel.y + panel.h / 2, panel.ylabel, 10, anchor="middle", rotate=-90)

    def save_svg(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width:.0f}" height="{self.height:.0f}" '
                f'viewBox="0 0 {self.width:.0f} {self.height:.0f}">\n'
            )
            f.write('<rect width="100%" height="100%" fill="white"/>\n')
            f.write("\n".join(self.svg))
            f.write("\n</svg>\n")

    def save_pdf(self, path: str) -> None:
        stream = "\n".join(self.pdf).encode("latin-1", "replace")
        objects = []
        objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
        objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width:.0f} {self.height:.0f}] "
            f"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".encode()
        )
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
        out = [b"%PDF-1.4\n"]
        offsets = [0]
        for i, obj in enumerate(objects, start=1):
            offsets.append(sum(len(part) for part in out))
            out.append(f"{i} 0 obj\n".encode() + obj + b"\nendobj\n")
        xref = sum(len(part) for part in out)
        out.append(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
        for off in offsets[1:]:
            out.append(f"{off:010d} 00000 n \n".encode())
        out.append(f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
        with open(path, "wb") as f:
            f.write(b"".join(out))


def latest_csv(result_dir: str) -> str:
    matches = [
        os.path.join(result_dir, name)
        for name in os.listdir(result_dir)
        if re.match(r"tmk_vs_tci_lowbond_full_\d+_\d+\.csv$", name)
    ]
    if not matches:
        raise FileNotFoundError("no full sweep CSV found")
    return max(matches, key=os.path.getmtime)


def stem_from_csv(path: str) -> str:
    match = re.search(r"full_(\d+_\d+)\.csv$", os.path.basename(path))
    return f"full_{match.group(1)}" if match else "full"


def save_spread(rows: list[dict], outdir: str, stem: str) -> list[str]:
    diag = [r for r in rows if r["method"] == "TMK_BEST_TRUNCATION"]
    lvals = sorted(set(r["L"] for r in diag))
    coeff = group_mean(diag, ("L", "lambda"), "coeff_energy_fraction")
    trunc = group_mean(diag, ("L", "lambda"), "best_trunc_rel_l2_error")
    fig = Figure(720, 300)
    p1 = Panel(70, 42, 260, 185, 0, max(lvals), 0, 1.02, title="Coefficient spread", xlabel="TMK cutoff Lambda", ylabel="Energy fraction")
    p2 = Panel(430, 42, 250, 185, 0, max(lvals), 1e-4, 1.2, logy=True, title="Best possible truncation", xlabel="TMK cutoff Lambda", ylabel="Relative L2 error")
    fig.panel_axes(p1, [0, 2, 4, 6, 8, 10, 12], [0, 0.25, 0.5, 0.75, 1.0])
    fig.panel_axes(p2, [0, 2, 4, 6, 8, 10, 12], [1e-4, 1e-3, 1e-2, 1e-1, 1], yfmt=lambda y: f"1e{int(math.log10(y))}" if y < 1 else "1")
    for idx, L in enumerate(lvals):
        color = PALETTE[idx % len(PALETTE)]
        xs = sorted(k[1] for k in coeff if k[0] == L)
        pts1 = [(p1.sx(x), p1.sy(coeff[(L, x)])) for x in xs]
        pts2 = [(p2.sx(x), p2.sy(max(trunc[(L, x)], 1e-4))) for x in xs]
        fig.polyline(pts1, color, 1.8)
        fig.polyline(pts2, color, 1.8)
        for x, y in pts1:
            fig.marker(x, y, color, "circle", 3.1)
        for x, y in pts2:
            fig.marker(x, y, color, "circle", 3.1)
        lx, ly = 92 + idx * 94, 270
        fig.line(lx, ly - 4, lx + 18, ly - 4, color, 2.0)
        fig.marker(lx + 9, ly - 4, color, "circle", 3.0)
        fig.text(lx + 24, ly, f"L={L}", 9)
    paths = []
    for ext, saver in [("svg", fig.save_svg), ("pdf", fig.save_pdf)]:
        path = os.path.join(outdir, f"spread_truncation_{stem}.{ext}")
        saver(path)
        paths.append(path)
    return paths


def threshold_summary(rows: list[dict], thresholds: list[float]) -> list[dict]:
    lvals = sorted(set(r["L"] for r in rows if r["method"] == "FTCI"))
    seeds = sorted(set(r["seed"] for r in rows if r["method"] == "FTCI"))
    tmk_max = max(r["samples"] for r in rows if r["method"] == "TMK")
    summary = []
    for threshold in thresholds:
        for L in lvals:
            ftci_vals = [
                r["samples"]
                for r in rows
                if r["method"] == "FTCI" and r["L"] == L and r["rel_l2_error"] <= threshold
            ]
            summary.append(
                {
                    "threshold": threshold,
                    "L": L,
                    "method": "FTCI",
                    "mean_samples": mean(ftci_vals),
                    "success_count": len(ftci_vals),
                    "seed_count": len(seeds),
                    "max_tested": "",
                }
            )
            vals = []
            for seed in seeds:
                hits = [
                    r["samples"]
                    for r in rows
                    if r["method"] == "TMK"
                    and r["L"] == L
                    and r["seed"] == seed
                    and r["rel_l2_error"] <= threshold
                ]
                if hits:
                    vals.append(min(hits))
            summary.append(
                {
                    "threshold": threshold,
                    "L": L,
                    "method": "TMK",
                    "mean_samples": mean(vals),
                    "success_count": len(vals),
                    "seed_count": len(seeds),
                    "max_tested": tmk_max,
                }
            )
    return summary


def save_threshold(rows: list[dict], outdir: str, stem: str) -> list[str]:
    thresholds = [1e-1, 1e-2, 1e-3]
    summary = threshold_summary(rows, thresholds)
    csv_path = os.path.join(outdir, f"threshold_samples_{stem}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            ["threshold", "L", "method", "mean_samples", "success_count", "seed_count", "max_tested"],
        )
        writer.writeheader()
        writer.writerows(summary)
    fig = Figure(900, 315)
    panels = []
    for i, threshold in enumerate(thresholds):
        panels.append(
            Panel(
                68 + i * 285,
                44,
                215,
                185,
                3.5,
                12.5,
                40,
                70000,
                logy=True,
                title=f"error <= {threshold:g}",
                xlabel="System size L",
                ylabel="Samples" if i == 0 else "",
            )
        )
    for p in panels:
        fig.panel_axes(p, [4, 6, 8, 10, 12], [1e2, 1e3, 1e4], yfmt=lambda y: f"1e{int(math.log10(y))}")
    for p, threshold in zip(panels, thresholds):
        for method, color, shape in [("FTCI", "blue", "circle"), ("TMK", "orange", "square")]:
            pts = []
            missing = []
            for row in summary:
                if row["threshold"] == threshold and row["method"] == method:
                    L = row["L"]
                    if row["mean_samples"] is None:
                        missing.append((L, row["max_tested"]))
                    else:
                        pts.append((L, row["mean_samples"]))
            xy = [(p.sx(x), p.sy(y)) for x, y in pts]
            fig.polyline(xy, color, 1.8)
            for x, y in xy:
                fig.marker(x, y, color, shape, 3.3)
            for L, ymax in missing:
                fig.marker(p.sx(L), p.sy(float(ymax)), "red", "triangle", 4.0, fill=False)
    fig.line(650, 268, 670, 268, "blue", 2.0)
    fig.marker(660, 268, "blue", "circle", 3.4)
    fig.text(676, 272, "FTCI", 9)
    fig.line(725, 268, 745, 268, "orange", 2.0)
    fig.marker(735, 268, "orange", "square", 3.4)
    fig.text(751, 272, "TMK reached", 9)
    fig.marker(835, 268, "red", "triangle", 4.0, fill=False)
    fig.text(845, 272, "TMK not reached", 9)
    paths = [csv_path]
    for ext, saver in [("svg", fig.save_svg), ("pdf", fig.save_pdf)]:
        path = os.path.join(outdir, f"threshold_samples_{stem}.{ext}")
        saver(path)
        paths.append(path)
    return paths


def save_error_samples(rows: list[dict], outdir: str, stem: str, L: int = 12) -> list[str]:
    tmk = [r for r in rows if r["method"] == "TMK" and r["L"] == L]
    ftci = [r for r in rows if r["method"] == "FTCI" and r["L"] == L]
    lambdas = sorted(set(r["lambda"] for r in tmk))
    samples = sorted(set(r["samples"] for r in tmk))
    err = group_mean(tmk, ("lambda", "samples"), "rel_l2_error")
    fig = Figure(520, 330)
    p = Panel(72, 44, 380, 205, min(samples) * 0.8, max(samples) * 1.08, 1e-5, 2.0, logy=True, title=f"TMK learning curves at L={L}", xlabel="TMK samples", ylabel="Relative L2 error")
    fig.panel_axes(p, [50, 1000, 2000, 5000], [1e-5, 1e-3, 1e-1, 1], yfmt=lambda y: f"1e{int(math.log10(y))}" if y < 1 else "1")
    for idx, lam in enumerate(lambdas):
        color = PALETTE[idx % len(PALETTE)]
        pts = [(s, err[(lam, s)]) for s in samples if (lam, s) in err]
        xy = [(p.sx(x), p.sy(y)) for x, y in pts]
        fig.polyline(xy, color, 1.8)
        for x, y in xy:
            fig.marker(x, y, color, "circle", 3.0)
        ly = 274 + idx * 17
        fig.line(78, ly - 4, 98, ly - 4, color, 2.0)
        fig.marker(88, ly - 4, color, "circle", 3.0)
        fig.text(104, ly, f"Lambda={lam}", 9)
    ftci_x = mean(r["samples"] for r in ftci)
    ftci_y = max(mean(r["rel_l2_error"] for r in ftci), 1e-5)
    if ftci_x is not None and ftci_y is not None:
        fig.marker(p.sx(ftci_x), p.sy(ftci_y), "black", "square", 4.0)
        fig.text(p.sx(ftci_x) + 8, p.sy(ftci_y) - 5, "FTCI", 9)
    paths = []
    for ext, saver in [("svg", fig.save_svg), ("pdf", fig.save_pdf)]:
        path = os.path.join(outdir, f"tmk_error_vs_samples_L{L}_{stem}.{ext}")
        saver(path)
        paths.append(path)
    return paths


def main() -> None:
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = latest_csv(os.path.join(os.path.dirname(__file__), "results"))
    csv_path = os.path.abspath(csv_path)
    outdir = os.path.dirname(csv_path)
    stem = stem_from_csv(csv_path)
    rows = read_rows(csv_path)
    paths = []
    paths.extend(save_spread(rows, outdir, stem))
    paths.extend(save_threshold(rows, outdir, stem))
    paths.extend(save_error_samples(rows, outdir, stem, L=max(r["L"] for r in rows if r["method"] == "FTCI")))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
