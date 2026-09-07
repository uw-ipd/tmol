"""Render the two final 2-by-3 composite benchmark figures."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from common import ROOT

SERIES_STYLES = {
    "PyRosetta CPU · B1": ("#222222", "o"),
    "tmol CPU · B1": ("#0072B2", "s"),
    "tmol GPU · B1": ("#009E73", "^"),
    "tmol GPU · B10": ("#56B4E9", "D"),
    "tmol GPU · B100": ("#E69F00", "P"),
    "tmol GPU · B1000": ("#D55E00", "X"),
}
MODALITIES = {
    "protein": "Protein",
    "protein_ligand": "Protein–ligand",
    "protein_nucleic": "Protein–DNA/RNA",
}
WORKLOADS = {
    "score_gradient": "Score + gradient",
    "fastrelax": "FastRelax",
}
HISTORICAL_VERSIONS = {"0.1.46", "0.1.47"}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "legend.fontsize": 6.7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def save(fig, metric: str) -> None:
    output = ROOT / "figures"
    output.mkdir(parents=True, exist_ok=True)
    stem = output / f"multimodal-workloads-{metric}"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=600)


def plot_group(ax, group: pd.DataFrame, series: str, metric: str, historical: bool):
    color, marker = SERIES_STYLES[series]
    y = "structures_per_second" if metric == "speed" else "peak_working_memory_bytes"
    group = group[group[y].notna()].sort_values("polymer_residues")
    if group.empty:
        return
    values = group[y] if metric == "speed" else group[y] / (1024**3)
    ax.plot(
        group.polymer_residues,
        values,
        color=color,
        marker=marker,
        linestyle="--" if historical else "-",
        markerfacecolor="white" if historical else color,
        markeredgecolor=color,
        markeredgewidth=0.7,
        markersize=3.5,
    )
    if metric == "speed" and group.n_samples.min() > 1:
        ax.fill_between(
            group.polymer_residues,
            group.throughput_q25,
            group.throughput_q75,
            color=color,
            alpha=0.08,
            linewidth=0,
        )


def plot_panel(ax, data: pd.DataFrame, modality: str, workload: str, metric: str):
    subset = data[
        (data.modality == modality)
        & (data.protocol == workload)
        & (data.status == "ok")
        & data.selected_for_plot
    ]
    for series in SERIES_STYLES:
        engine_group = subset[subset.series == series]
        if series == "PyRosetta CPU · B1":
            plot_group(ax, engine_group, series, metric, historical=False)
            continue
        for version, group in engine_group.groupby("engine_version"):
            plot_group(
                ax,
                group,
                series,
                metric,
                historical=str(version) in HISTORICAL_VERSIONS,
            )


def legend_handles():
    device_handles = [
        mpl.lines.Line2D(
            [],
            [],
            color=color,
            marker=marker,
            markersize=4,
            label=series,
        )
        for series, (color, marker) in SERIES_STYLES.items()
    ]
    version_handles = [
        mpl.lines.Line2D([], [], color="#555555", label="tmol 0.1.55"),
        mpl.lines.Line2D(
            [],
            [],
            color="#555555",
            linestyle="--",
            marker="o",
            markerfacecolor="white",
            label="historical tmol (0.1.46; NA 0.1.47)",
        ),
    ]
    return [*device_handles, *version_handles]


def composite_figure(data: pd.DataFrame, metric: str) -> None:
    fig, axes = plt.subplots(
        len(WORKLOADS),
        len(MODALITIES),
        figsize=(14.2, 7.4),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )
    for row_index, (workload, workload_title) in enumerate(WORKLOADS.items()):
        for column_index, (modality, modality_title) in enumerate(MODALITIES.items()):
            ax = axes[row_index, column_index]
            plot_panel(ax, data, modality, workload, metric)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(25, 1600)
            ax.grid(which="major", color="#D9D9D9", linewidth=0.55)
            ax.grid(which="minor", color="#EEEEEE", linewidth=0.35)
            panel = chr(ord("a") + row_index * len(MODALITIES) + column_index)
            ax.text(
                -0.10,
                1.04,
                f"({panel})",
                transform=ax.transAxes,
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
            ax.text(
                0.02,
                0.97,
                workload_title,
                transform=ax.transAxes,
                va="top",
                color="#555555",
            )
            if row_index == 0:
                ax.set_title(modality_title)
            if row_index == len(WORKLOADS) - 1:
                ax.set_xlabel("Polymer residues per structure")
            if column_index == 0:
                ylabel = (
                    "Throughput (structures s$^{-1}$)"
                    if metric == "speed"
                    else "Peak working memory (GiB)"
                )
                ax.set_ylabel(f"{ylabel}\n{workload_title}")

    fig.legend(
        handles=legend_handles(),
        frameon=False,
        ncol=4,
        loc="outside lower center",
    )
    note = (
        "Higher throughput is better"
        if metric == "speed"
        else "CPU: process peak RSS; GPU: peak PyTorch CUDA allocation"
    )
    fig.suptitle(
        f"TMol and PyRosetta multimodal workload scaling · {metric.capitalize()}\n"
        f"{note}",
        fontsize=10,
    )
    save(fig, metric)
    plt.close(fig)


def main() -> None:
    set_style()
    timing = pd.read_csv(ROOT / "results/summary/timing_summary.csv")
    for metric in ("speed", "memory"):
        composite_figure(timing, metric)


if __name__ == "__main__":
    main()
