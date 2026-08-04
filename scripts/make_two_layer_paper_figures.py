"""Generate deterministic SVG figures for the positive two-layer paper."""

from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "draft" / "pics"
PHASE50 = ROOT / "benchmark-out" / "two_layer_confirmatory_phase50.json"
PHASE51C = ROOT / "benchmark-out" / "s3dis_room_layer_validation_phase51c.json"

COLORS = {
    "candidate": "#16856B",
    "b5": "#3E6FB0",
    "m1": "#D28A24",
    "ink": "#1F2937",
    "muted": "#667085",
    "grid": "#D9DEE7",
    "paper": "#FFFFFF",
    "layer0": "#3478C7",
    "layer1": "#E28B27",
    "unsafe": "#C43D4D",
}


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 22,
    anchor: str = "start",
    weight: int = 400,
    fill: str | None = None,
) -> str:
    color = COLORS["ink"] if fill is None else fill
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}">{escape(value)}</text>'
    )


def _circle(x: float, y: float, color: str, radius: float = 5.0) -> str:
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" '
        f'fill="{color}" stroke="#FFFFFF" stroke-width="1.2"/>'
    )


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str,
    width: float = 2.0,
    dash: str | None = None,
    opacity: float = 1.0,
) -> str:
    dash_attr = "" if dash is None else f' stroke-dasharray="{dash}"'
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width:.1f}" opacity="{opacity:.3f}"'
        f"{dash_attr}/>"
    )


def _svg(width: int, height: int, body: list[str]) -> str:
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">',
            f'<rect width="{width}" height="{height}" fill="{COLORS["paper"]}"/>',
            *body,
            "</svg>",
            "",
        ]
    )


def _layer_points(
    origin_x: float, origin_y: float
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    xs = [0, 42, 84, 126, 168, 210]
    upper = [(origin_x + x, origin_y - 15 + 4 * math.sin(x / 55)) for x in xs]
    lower = [(origin_x + x, origin_y + 80 + 5 * math.sin(x / 55 + 0.4)) for x in xs]
    return upper, lower


def make_workflow() -> Path:
    body: list[str] = []
    panel_x = [35, 415, 795]
    titles = [
        "A  One-complex reconstruction",
        "B  Observed-only routing",
        "C  Per-layer connectivity",
    ]
    for x, title in zip(panel_x, titles, strict=True):
        body.append(
            f'<rect x="{x}" y="45" width="350" height="285" rx="12" '
            f'fill="#F8FAFC" stroke="{COLORS["grid"]}" stroke-width="2"/>'
        )
        body.append(_text(x + 18, 80, title, size=22, weight=700))

    upper, lower = _layer_points(95, 145)
    for row, color in ((upper, COLORS["layer0"]), (lower, COLORS["layer1"])):
        for (x1, y1), (x2, y2) in zip(row, row[1:], strict=False):
            body.append(_line(x1, y1, x2, y2, color="#9AA4B2", width=1.6))
        body.extend(_circle(x, y, color) for x, y in row)
    for index in (1, 2, 3, 4):
        body.append(
            _line(
                upper[index][0],
                upper[index][1],
                lower[index - 1][0],
                lower[index - 1][1],
                color=COLORS["unsafe"],
                width=2.6,
                dash="7 5",
            )
        )
    body.append(
        _text(
            210,
            300,
            "false cross-layer cells",
            size=18,
            anchor="middle",
            fill=COLORS["unsafe"],
        )
    )

    upper, lower = _layer_points(475, 145)
    for row, color in ((upper, COLORS["layer0"]), (lower, COLORS["layer1"])):
        body.extend(_circle(x, y, color) for x, y in row)
    body.append(_line(590, 145, 590, 225, color=COLORS["muted"], width=2.4, dash="5 5"))
    body.append(_text(601, 178, "estimated gap", size=17, fill=COLORS["muted"]))
    body.append(_text(590, 270, "SNR, balance, cross-kNN", size=18, anchor="middle"))
    body.append(
        f'<rect x="505" y="285" width="170" height="30" rx="15" '
        f'fill="#E7F5F1" stroke="{COLORS["candidate"]}"/>'
    )
    body.append(
        _text(
            590,
            307,
            "accept / rescan",
            size=17,
            anchor="middle",
            weight=700,
            fill=COLORS["candidate"],
        )
    )

    upper, lower = _layer_points(855, 145)
    for row, color in ((upper, COLORS["layer0"]), (lower, COLORS["layer1"])):
        for i in range(len(row) - 2):
            p0, p1, p2 = row[i : i + 3]
            body.append(
                f'<polygon points="{p0[0]:.1f},{p0[1]:.1f} {p1[0]:.1f},{p1[1]:.1f} '
                f'{p2[0]:.1f},{p2[1]:.1f}" fill="{color}" fill-opacity="0.15" '
                f'stroke="{color}" stroke-width="1.6"/>'
            )
        body.extend(_circle(x, y, color) for x, y in row)
    body.append(
        _text(
            970,
            300,
            "no cross-layer face is admissible",
            size=18,
            anchor="middle",
            fill=COLORS["candidate"],
        )
    )
    body.append(
        _text(
            600,
            375,
            "The method changes the admissible connectivity before scale selection.",
            size=22,
            anchor="middle",
            weight=700,
        )
    )

    path = OUT / "two_layer_workflow.svg"
    path.write_text(_svg(1200, 405, body), encoding="utf-8")
    return path


def _bar_panel(
    body: list[str],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    values: list[tuple[str, float, str]],
    maximum: float,
    formatter,
) -> None:
    body.append(_text(x, y - 20, title, size=22, weight=700))
    body.append(
        _line(x, y + height, x + width, y + height, color=COLORS["ink"], width=1.5)
    )
    body.append(_line(x, y, x, y + height, color=COLORS["ink"], width=1.5))
    for tick in range(6):
        value = maximum * tick / 5
        yy = y + height - height * tick / 5
        body.append(_line(x, yy, x + width, yy, color=COLORS["grid"], width=1.0))
        body.append(
            _text(
                x - 10,
                yy + 6,
                formatter(value),
                size=15,
                anchor="end",
                fill=COLORS["muted"],
            )
        )
    gap = width / len(values)
    bar_width = min(70.0, gap * 0.55)
    for index, (label, value, color) in enumerate(values):
        bx = x + gap * (index + 0.5) - bar_width / 2
        bh = 0.0 if maximum == 0 else height * value / maximum
        by = y + height - bh
        body.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_width:.1f}" '
            f'height="{max(bh, 1.5):.1f}" rx="4" fill="{color}"/>'
        )
        body.append(
            _text(
                bx + bar_width / 2,
                by - 8,
                formatter(value),
                size=16,
                anchor="middle",
                weight=700,
            )
        )
        body.append(
            _text(bx + bar_width / 2, y + height + 24, label, size=17, anchor="middle")
        )


def make_results() -> Path:
    phase50 = json.loads(PHASE50.read_text(encoding="utf-8"))
    phase51 = json.loads(PHASE51C.read_text(encoding="utf-8"))["core"]
    datasets = [
        ("Synthetic Phase 50 (n=144)", phase50),
        ("S3DIS Area 5 (n=63)", phase51),
    ]
    body: list[str] = []
    for index, (name, result) in enumerate(datasets):
        x = 95 + 580 * index
        f_values = [
            (
                "Layer-routed",
                float(result["candidate_mean_fscore"]),
                COLORS["candidate"],
            ),
            ("B5", float(result["b5_mean_fscore"]), COLORS["b5"]),
            ("M1", float(result["m1_mean_fscore"]), COLORS["m1"]),
        ]
        _bar_panel(
            body,
            x=x,
            y=90,
            width=440,
            height=230,
            title=f"{name}: mean F-score",
            values=f_values,
            maximum=1.0,
            formatter=lambda value: f"{value:.2f}",
        )
        topology = [
            (
                "Layer-routed",
                math.log10(float(result["candidate_topology_error_sum"]) + 1),
                COLORS["candidate"],
            ),
            (
                "B5",
                math.log10(float(result["b5_topology_error_sum"]) + 1),
                COLORS["b5"],
            ),
            (
                "M1",
                math.log10(float(result["m1_topology_error_sum"]) + 1),
                COLORS["m1"],
            ),
        ]
        _bar_panel(
            body,
            x=x,
            y=420,
            width=440,
            height=220,
            title=f"{name}: log10(topology error + 1)",
            values=topology,
            maximum=5.0,
            formatter=lambda value: f"{value:.1f}",
        )
    body.append(
        _text(
            600,
            705,
            "B5: PCA-anisotropic alpha; M1: weighted power-alpha.",
            size=18,
            anchor="middle",
            fill=COLORS["muted"],
        )
    )
    path = OUT / "two_layer_results.svg"
    path.write_text(_svg(1200, 735, body), encoding="utf-8")
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (make_workflow(), make_results()):
        print(path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
