#!/usr/bin/env python3
"""Verify and summarize a TMol workflow profiling run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
from pathlib import Path


COLORS = {
    "score": "#2563eb",
    "score_grad": "#7c3aed",
    "score_graph": "#0891b2",
    "score_graph_grad": "#0f766e",
    "cart_min": "#ea580c",
    "kin_min": "#dc2626",
    "fast_relax": "#9333ea",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(run_dir: Path) -> None:
    path = run_dir / "manifest.json"
    if not path.is_file():
        return
    manifest = json.loads(path.read_text())
    errors = (
        ["results/summary.csv"]
        if not (run_dir / "results/summary.csv").is_file()
        else []
    )
    for case in manifest["cases"]:
        required = [f"results/{case}.json", f"logs/{case}.log"]
        if manifest["trace"]:
            required += [
                f"traces/{case}.nsys-rep",
                f"traces/{case}.sqlite",
                f"traces/{case}.stats.csv",
            ]
        errors += [name for name in required if not (run_dir / name).is_file()]
    errors = [f"missing: {name}" for name in errors]
    for entry in manifest["files"]:
        artifact = run_dir / entry["path"]
        if not artifact.is_file():
            errors.append(f"missing: {entry['path']}")
        elif artifact.stat().st_size != entry["bytes"]:
            errors.append(f"size changed: {entry['path']}")
        elif _sha256(artifact) != entry["sha256"]:
            errors.append(f"checksum changed: {entry['path']}")
    if errors:
        raise RuntimeError("manifest verification failed:\n" + "\n".join(errors))


def read_timings(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "results" / "summary.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def _short_kernel(name: str) -> str:
    score_terms = re.findall(r"tmol::score::([a-z_]+)", name)
    for term in score_terms:
        if term != "common":
            return f"tmol::score::{term}"
    for namespace in ("tmol::kinematics::", "at::native::", "c10::cuda::"):
        if namespace in name:
            suffix = name.split(namespace, 1)[1].split("<", 1)[0].split("(", 1)[0]
            return namespace + suffix
    return name.split("<", 1)[0].split("(", 1)[0][:100]


def _stats_sections(path: Path) -> dict[str, list[dict[str, str]]]:
    sections: dict[str, list[dict[str, str]]] = {}
    section = None
    header = None
    for line in path.read_text(errors="replace").splitlines():
        if "cuda_gpu_kern_sum.py" in line:
            section = "kernels"
            header = None
        elif "cuda_api_sum.py" in line:
            section = "api"
            header = None
        elif "nvtx_sum.py" in line:
            section = "nvtx"
            header = None
        elif section and line.startswith("Time (%)"):
            header = next(csv.reader([line]))
            sections[section] = []
        elif section and header and line:
            values = next(csv.reader([line]))
            if len(values) == len(header):
                sections[section].append(dict(zip(header, values)))
        elif not line:
            header = None
    return sections


def read_traces(run_dir: Path) -> list[dict[str, str | int | float]]:
    rows = []
    for path in sorted((run_dir / "traces").glob("*.stats.csv")):
        sections = _stats_sections(path)
        kernels = sections.get("kernels", [])
        api = sections.get("api", [])
        nvtx = sections.get("nvtx", [])
        launches = next((row for row in api if row["Name"] == "cudaLaunchKernel"), {})
        iteration = next(
            (row for row in nvtx if "/iteration-" in row.get("Range", "")), {}
        )
        top = max(kernels, key=lambda row: int(row["Total Time (ns)"]))
        nested = [
            row
            for row in nvtx
            if "/iteration-" not in row.get("Range", "")
            and not row.get("Range", "").endswith("/measure")
        ]
        top_range = max(nested, key=lambda row: int(row["Total Time (ns)"]), default={})
        rows.append(
            {
                "case": path.name.removesuffix(".stats.csv"),
                "profiled_iteration_ms": int(iteration.get("Total Time (ns)", 0)) / 1e6,
                "gpu_kernel_ms": sum(int(row["Total Time (ns)"]) for row in kernels)
                / 1e6,
                "kernel_launches": int(launches.get("Num Calls", 0)),
                "launch_api_ms": int(launches.get("Total Time (ns)", 0)) / 1e6,
                "top_nvtx": top_range.get("Range", "").removeprefix(":"),
                "top_nvtx_ms": int(top_range.get("Total Time (ns)", 0)) / 1e6,
                "top_kernel": _short_kernel(top["Name"]),
                "top_kernel_ms": int(top["Total Time (ns)"]) / 1e6,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, timings: list[dict[str, str]]) -> None:
    lines = [
        "# TMol GPU workflow timings",
        "",
        "Steady-state medians exclude system construction, scorer rendering, warm-up, "
        "and CUDA-graph capture. Throughput is poses per synchronized workflow call.",
    ]
    for workflow in COLORS:
        selected = [row for row in timings if row["workflow"] == workflow]
        if not selected:
            continue
        lines += [
            "",
            f"## `{workflow}`",
            "",
            "| System | Batch | Median (ms) | ms / pose | "
            "Poses / s | Peak delta (MiB) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in selected:
            median = float(row["median_ms"])
            poses = int(row["poses"])
            peak = int(row["peak_memory_delta_bytes"]) / 2**20
            lines.append(
                f"| {row['system']} | {row['batch']} | {median:.3f} | "
                f"{median / poses:.3f} | {poses * 1000 / median:.1f} | {peak:.1f} |"
            )
    path.write_text("\n".join(lines) + "\n")


def write_latency_svg(path: Path, timings: list[dict[str, str]]) -> None:
    rows = sorted(
        timings,
        key=lambda row: (
            list(COLORS).index(row["workflow"]),
            row["system"],
            int(row["batch"]),
        ),
    )
    values = [float(row["median_ms"]) for row in rows]
    low = 10 ** math.floor(math.log10(min(values)))
    high = 10 ** math.ceil(math.log10(max(values)))
    if low == high:
        high *= 10
    label_width, plot_width, row_height = 285, 820, 18
    width, height = label_width + plot_width + 105, 70 + row_height * len(rows)

    def x(value: float) -> float:
        ratio = math.log10(value / low) / math.log10(high / low)
        return label_width + plot_width * ratio

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font:12px ui-monospace,SFMono-Regular,monospace;fill:#111827}"
        ".axis{stroke:#d1d5db;stroke-width:1}</style>",
        '<text x="10" y="22" font-size="16" font-weight="600">'
        "Steady-state H200 latency (log scale)</text>",
    ]
    tick = low
    while tick <= high:
        tx = x(tick)
        parts += [
            f'<line class="axis" x1="{tx:.1f}" y1="35" x2="{tx:.1f}" '
            f'y2="{height - 15}"/>',
            f'<text x="{tx:.1f}" y="33" text-anchor="middle">{tick:g} ms</text>',
        ]
        tick *= 10
    for index, row in enumerate(rows):
        y = 52 + index * row_height
        value = float(row["median_ms"])
        label = html.escape(f"{row['workflow']}  {row['system']}  b{row['batch']}")
        parts += [
            f'<text x="8" y="{y + 11}">{label}</text>',
            f'<rect x="{label_width}" y="{y}" '
            f'width="{max(1, x(value) - label_width):.1f}" '
            f'height="12" rx="2" fill="{COLORS[row["workflow"]]}"/>',
            f'<text x="{x(value) + 5:.1f}" y="{y + 11}">{value:.3f}</text>',
        ]
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def comparison_rows(
    timings: list[dict[str, str]], baseline: list[dict[str, str]]
) -> list[dict[str, str | float]]:
    before = {row["case"]: row for row in baseline}
    rows = []
    for row in timings:
        if row["case"] not in before:
            continue
        baseline_ms = float(before[row["case"]]["median_ms"])
        current_ms = float(row["median_ms"])
        rows.append(
            {
                "case": row["case"],
                "baseline_ms": baseline_ms,
                "current_ms": current_ms,
                "speedup": baseline_ms / current_ms,
                "delta_percent": 100 * (current_ms / baseline_ms - 1),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    verify_manifest(args.run_dir)
    output = args.output_dir or args.run_dir / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    timings = read_timings(args.run_dir)
    write_markdown(output / "timings.md", timings)
    write_latency_svg(output / "timings.svg", timings)
    _write_csv(output / "trace_summary.csv", read_traces(args.run_dir))
    if args.baseline:
        verify_manifest(args.baseline)
        _write_csv(
            output / "comparison.csv",
            comparison_rows(timings, read_timings(args.baseline)),
        )
    print(f"verified {args.run_dir}; wrote {output}")


if __name__ == "__main__":
    main()
