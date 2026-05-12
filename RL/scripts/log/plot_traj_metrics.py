#!/usr/bin/env python3
"""Plot StreamWeave trajectory metrics from a verl training log.

The script intentionally avoids matplotlib. It uses Pillow when available to
write PNG, and falls back to SVG when Pillow is not installed.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
RL_DIR = SCRIPT_DIR.parents[1]
DEFAULT_LOG = RL_DIR / "outputs/runs/grpo_rl0511_8gpu_judge/train.log"
DEFAULT_OUTPUT = SCRIPT_DIR / "experiment1_traj_score_success.png"

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
STEP_RE = re.compile(r"\bstep:(\d+)\s+-")
METRIC_RE = re.compile(
    r"(?P<key>[A-Za-z0-9_./-]+):"
    r"(?P<value>(?:np\.[A-Za-z0-9_]+\()?"
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\)?)"
)
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class SeriesSpec:
    label: str
    keys: tuple[str, ...]
    color: tuple[int, int, int]


DEFAULT_SERIES = (
    SeriesSpec(
        label="traj score",
        keys=("traj/score/mean", "streamweave/trajectory_score/mean", "traj/score_mean"),
        color=(36, 99, 235),
    ),
    SeriesSpec(
        label="traj success",
        keys=("traj/success/mean", "streamweave/success_score/mean"),
        color=(220, 38, 38),
    ),
)


def main() -> int:
    args = parse_args()
    log_path = args.log.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    specs = build_series_specs(args.metric)
    rows = parse_log(log_path)
    series = collect_series(rows, specs)
    if not any(values for values in series.values()):
        available = sorted({key for _step, metrics in rows for key in metrics})
        raise SystemExit(
            "No requested metrics found in log.\n"
            f"Requested: {', '.join(spec.label for spec in specs)}\n"
            f"Available metric examples: {', '.join(available[:40])}"
        )

    if output_path.suffix.lower() == ".svg":
        write_svg(output_path, series, args.title, args.smooth_window)
    else:
        try:
            write_png(output_path, series, args.title, args.smooth_window)
        except ModuleNotFoundError:
            fallback = output_path.with_suffix(".svg")
            write_svg(fallback, series, args.title, args.smooth_window)
            print(f"Pillow is not installed; wrote SVG fallback: {fallback}")
            return 0

    print(f"Wrote plot: {output_path}")
    for label, points in series.items():
        if points:
            print(f"{label}: {len(points)} points, step {points[0][0]}..{points[-1][0]}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG,
        help=f"Training log to parse. Default: {DEFAULT_LOG}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output image path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        help=(
            "Metric key to plot. Repeatable. Defaults to traj/score/mean and "
            "traj/success/mean with StreamWeave aliases."
        ),
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Moving-average window for the bold line. Use 1 to disable. Default: 5.",
    )
    parser.add_argument(
        "--title",
        default="Experiment 1 trajectory metrics",
        help="Plot title.",
    )
    return parser.parse_args()


def build_series_specs(metrics: list[str]) -> tuple[SeriesSpec, ...]:
    if not metrics:
        return DEFAULT_SERIES
    palette = (
        (36, 99, 235),
        (220, 38, 38),
        (5, 150, 105),
        (147, 51, 234),
        (234, 88, 12),
        (8, 145, 178),
    )
    return tuple(
        SeriesSpec(label=metric, keys=(metric,), color=palette[index % len(palette)])
        for index, metric in enumerate(metrics)
    )


def parse_log(path: Path) -> list[tuple[int, dict[str, float]]]:
    if not path.exists():
        raise SystemExit(f"Log file does not exist: {path}")

    rows: list[tuple[int, dict[str, float]]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = ANSI_RE.sub("", raw_line.replace("\r", "\n"))
            for part in line.splitlines():
                step_match = STEP_RE.search(part)
                if not step_match:
                    continue
                step = int(step_match.group(1))
                metrics: dict[str, float] = {}
                for metric_match in METRIC_RE.finditer(part):
                    key = metric_match.group("key")
                    value = parse_float(metric_match.group("value"))
                    if value is not None:
                        metrics[key] = value
                if metrics:
                    rows.append((step, metrics))
    return dedupe_steps(rows)


def parse_float(text: str) -> float | None:
    match = NUMBER_RE.search(text)
    if not match:
        return None
    try:
        value = float(match.group(0))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def dedupe_steps(rows: Iterable[tuple[int, dict[str, float]]]) -> list[tuple[int, dict[str, float]]]:
    by_step: dict[int, dict[str, float]] = {}
    for step, metrics in rows:
        by_step.setdefault(step, {}).update(metrics)
    return [(step, by_step[step]) for step in sorted(by_step)]


def collect_series(
    rows: list[tuple[int, dict[str, float]]],
    specs: tuple[SeriesSpec, ...],
) -> dict[str, list[tuple[int, float]]]:
    out: dict[str, list[tuple[int, float]]] = {spec.label: [] for spec in specs}
    for step, metrics in rows:
        for spec in specs:
            for key in spec.keys:
                if key in metrics:
                    out[spec.label].append((step, metrics[key]))
                    break
    return out


def moving_average(points: list[tuple[int, float]], window: int) -> list[tuple[int, float]]:
    window = max(int(window), 1)
    if window <= 1 or len(points) <= 1:
        return list(points)
    smoothed: list[tuple[int, float]] = []
    values: list[float] = []
    for step, value in points:
        values.append(value)
        chunk = values[max(0, len(values) - window) :]
        smoothed.append((step, mean(chunk)))
    return smoothed


def write_png(
    path: Path,
    series: dict[str, list[tuple[int, float]]],
    title: str,
    smooth_window: int,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1400, 820
    margin_left, margin_right = 110, 40
    margin_top, margin_bottom = 90, 95
    bg = (255, 255, 255)
    fg = (17, 24, 39)
    grid = (229, 231, 235)

    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    plot_box = (margin_left, margin_top, width - margin_right, height - margin_bottom)
    draw_axes(draw, plot_box, fg, grid, font)
    specs_by_label = {spec.label: spec for spec in DEFAULT_SERIES}
    draw_series(draw, series, plot_box, specs_by_label, smooth_window)
    draw_labels(draw, title, series, specs_by_label, width, height, font, fg)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def write_svg(
    path: Path,
    series: dict[str, list[tuple[int, float]]],
    title: str,
    smooth_window: int,
) -> None:
    width, height = 1400, 820
    margin_left, margin_right = 110, 40
    margin_top, margin_bottom = 90, 95
    plot_box = (margin_left, margin_top, width - margin_right, height - margin_bottom)
    specs_by_label = {spec.label: spec for spec in DEFAULT_SERIES}
    x_min, x_max, y_min, y_max = bounds(series)

    def xy(step: int, value: float) -> tuple[float, float]:
        x0, y0, x1, y1 = plot_box
        x = x0 + (step - x_min) / max(x_max - x_min, 1) * (x1 - x0)
        y = y1 - (value - y_min) / max(y_max - y_min, 1e-9) * (y1 - y0)
        return x, y

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="36" text-anchor="middle" font-family="monospace" font-size="24" fill="#111827">{escape_xml(title)}</text>',
    ]
    x0, y0, x1, y1 = plot_box
    for index in range(6):
        y = y0 + index / 5 * (y1 - y0)
        value = y_max - index / 5 * (y_max - y_min)
        lines.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x1}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        lines.append(
            f'<text x="{x0 - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="monospace" font-size="14" fill="#111827">{value:.3f}</text>'
        )
    lines.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#111827"/>')
    lines.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#111827"/>')

    legend_y = 62
    for idx, (label, points) in enumerate(series.items()):
        if not points:
            continue
        color = specs_by_label.get(label, SeriesSpec(label, (label,), (0, 0, 0))).color
        color_hex = rgb_hex(color)
        raw_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in (xy(step, val) for step, val in points))
        smooth = moving_average(points, smooth_window)
        smooth_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in (xy(step, val) for step, val in smooth))
        lines.append(f'<polyline points="{raw_points}" fill="none" stroke="{color_hex}" stroke-width="1.3" opacity="0.25"/>')
        lines.append(f'<polyline points="{smooth_points}" fill="none" stroke="{color_hex}" stroke-width="3"/>')
        lx = x0 + idx * 220
        lines.append(f'<line x1="{lx}" y1="{legend_y}" x2="{lx + 28}" y2="{legend_y}" stroke="{color_hex}" stroke-width="4"/>')
        lines.append(
            f'<text x="{lx + 36}" y="{legend_y + 5}" font-family="monospace" font-size="16" fill="#111827">{escape_xml(label)}</text>'
        )

    lines.append(
        f'<text x="{(x0 + x1) / 2:.2f}" y="{height - 34}" text-anchor="middle" font-family="monospace" font-size="16" fill="#111827">training/global_step</text>'
    )
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_axes(draw, plot_box, fg, grid, font) -> None:
    x0, y0, x1, y1 = plot_box
    for index in range(6):
        y = y0 + index / 5 * (y1 - y0)
        draw.line([(x0, y), (x1, y)], fill=grid, width=1)
    draw.line([(x0, y1), (x1, y1)], fill=fg, width=2)
    draw.line([(x0, y0), (x0, y1)], fill=fg, width=2)


def draw_series(draw, series, plot_box, specs_by_label, smooth_window: int) -> None:
    x_min, x_max, y_min, y_max = bounds(series)

    def xy(step: int, value: float) -> tuple[float, float]:
        x0, y0, x1, y1 = plot_box
        x = x0 + (step - x_min) / max(x_max - x_min, 1) * (x1 - x0)
        y = y1 - (value - y_min) / max(y_max - y_min, 1e-9) * (y1 - y0)
        return x, y

    x0, y0, x1, y1 = plot_box
    for index in range(6):
        y = y0 + index / 5 * (y1 - y0)
        value = y_max - index / 5 * (y_max - y_min)
        draw.text((x0 - 64, y - 7), f"{value:.3f}", fill=(17, 24, 39))

    x_ticks = tick_values(int(x_min), int(x_max), count=8)
    for step in x_ticks:
        x, _ = xy(step, y_min)
        draw.line([(x, y1), (x, y1 + 6)], fill=(17, 24, 39), width=1)
        draw.text((x - 14, y1 + 12), str(step), fill=(17, 24, 39))

    for label, points in series.items():
        if not points:
            continue
        spec = specs_by_label.get(label, SeriesSpec(label, (label,), (0, 0, 0)))
        raw = [xy(step, value) for step, value in points]
        smooth = [xy(step, value) for step, value in moving_average(points, smooth_window)]
        if len(raw) > 1:
            draw.line(raw, fill=(*spec.color, 80), width=1)
            draw.line(smooth, fill=spec.color, width=4)
        else:
            x, y = raw[0]
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=spec.color)


def draw_labels(draw, title, series, specs_by_label, width, height, font, fg) -> None:
    draw.text((width // 2 - len(title) * 4, 28), title, fill=fg)
    draw.text((width // 2 - 76, height - 34), "training/global_step", fill=fg)
    legend_x = 112
    legend_y = 62
    for label, points in series.items():
        if not points:
            continue
        spec = specs_by_label.get(label, SeriesSpec(label, (label,), (0, 0, 0)))
        draw.line([(legend_x, legend_y), (legend_x + 30, legend_y)], fill=spec.color, width=4)
        latest = points[-1][1]
        draw.text((legend_x + 38, legend_y - 7), f"{label} latest={latest:.4f}", fill=fg)
        legend_x += 260


def bounds(series: dict[str, list[tuple[int, float]]]) -> tuple[float, float, float, float]:
    all_points = [point for points in series.values() for point in points]
    if not all_points:
        return 0.0, 1.0, 0.0, 1.0
    steps = [step for step, _value in all_points]
    values = [value for _step, value in all_points]
    x_min, x_max = min(steps), max(steps)
    y_min, y_max = min(values), max(values)
    y_min = min(0.0, y_min)
    y_max = max(1.0, y_max)
    padding = max((y_max - y_min) * 0.05, 0.02)
    return float(x_min), float(x_max), y_min - padding, y_max + padding


def tick_values(start: int, end: int, *, count: int) -> list[int]:
    if end <= start:
        return [start]
    raw_step = max((end - start) / max(count - 1, 1), 1.0)
    scale = 10 ** math.floor(math.log10(raw_step))
    for factor in (1, 2, 5, 10):
        nice_step = int(scale * factor)
        if nice_step >= raw_step:
            break
    else:
        nice_step = int(raw_step)
    first = int(math.ceil(start / nice_step) * nice_step)
    ticks = list(range(first, end + 1, nice_step))
    if start not in ticks:
        ticks.insert(0, start)
    if end not in ticks:
        ticks.append(end)
    return ticks


def rgb_hex(color: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    raise SystemExit(main())
