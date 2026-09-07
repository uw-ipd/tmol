from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from common import ROOT

COLORS = {
    "PyRosetta CPU · B1": "#222222",
    "tmol CPU · B1": "#0072B2",
    "tmol GPU · B1": "#009E73",
    "tmol GPU · B10": "#56B4E9",
    "tmol GPU · B100": "#E69F00",
    "tmol GPU · B1000": "#D55E00",
}
MARKERS = {
    "PyRosetta CPU · B1": "o",
    "tmol CPU · B1": "s",
    "tmol GPU · B1": "^",
    "tmol GPU · B10": "D",
    "tmol GPU · B100": "P",
    "tmol GPU · B1000": "X",
}
ORDER = list(COLORS)
MODALITY_TITLES = {
    "protein": "Protein",
    "protein_ligand": "Protein–ligand",
    "protein_nucleic": "Protein–DNA/RNA",
}
PROTOCOL_TITLES = {
    "score": "Score",
    "score_gradient": "Score + gradient",
    "fastrelax": "FastRelax",
}
WORKLOADS = ("score_gradient", "fastrelax")


def style() -> None:
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
            "lines.linewidth": 1.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def save(fig, stem: str) -> None:
    output = ROOT / "figures"
    output.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
        fig.savefig(output / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)


def line_figure(data: pd.DataFrame, modality: str, protocol: str, version: str) -> None:
    subset = data[
        (data.modality == modality)
        & (data.protocol == protocol)
        & (
            (data.engine == "pyrosetta")
            | ((data.engine == "tmol") & (data.engine_version == version))
        )
        & (data.status == "ok")
        & data.selected_for_plot
    ].copy()
    if subset.empty:
        return
    y = "seconds_per_structure" if protocol == "fastrelax" else "structures_per_second"
    ylabel = (
        "Seconds per structure"
        if protocol == "fastrelax"
        else "Throughput (structures s$^{-1}$)"
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.3), constrained_layout=True)
    for series in ORDER:
        group = subset[subset.series == series].sort_values("polymer_residues")
        if group.empty:
            continue
        ax.plot(
            group.polymer_residues,
            group[y],
            label=series,
            color=COLORS[series],
            marker=MARKERS[series],
            markersize=4.0,
            markeredgewidth=0.5,
            markeredgecolor="white",
        )
        if protocol != "fastrelax" and group.n_samples.min() > 1:
            ax.fill_between(
                group.polymer_residues,
                group.throughput_q25,
                group.throughput_q75,
                color=COLORS[series],
                alpha=0.12,
                linewidth=0,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Polymer residues per structure")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{MODALITY_TITLES[modality]} · {PROTOCOL_TITLES[protocol]} · tmol {version}"
    )
    ax.grid(which="major", color="#D9D9D9", linewidth=0.55)
    ax.grid(which="minor", color="#EEEEEE", linewidth=0.35)
    ax.legend(
        frameon=False,
        ncol=1,
        handlelength=1.8,
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        borderaxespad=0,
    )
    save(fig, f"{modality}-{protocol}-tmol-{version}")
    plt.close(fig)


def workload_metric_figure(
    data: pd.DataFrame, modality: str, version: str, metric: str
) -> None:
    """Plot both requested workflows for one modality and TMol release.

    CPU memory is total process peak RSS. GPU memory is peak memory allocated by
    PyTorch's CUDA allocator. These are intentionally labeled as different
    working-memory domains rather than presented as byte-for-byte equivalents.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.55), constrained_layout=True)
    plotted_any = False
    for ax, protocol in zip(axes, WORKLOADS):
        subset = data[
            (data.modality == modality)
            & (data.protocol == protocol)
            & (
                (data.engine == "pyrosetta")
                | ((data.engine == "tmol") & (data.engine_version == version))
            )
            & (data.status == "ok")
            & data.selected_for_plot
        ].copy()
        y = (
            "structures_per_second"
            if metric == "speed"
            else "peak_working_memory_bytes"
        )
        for series in ORDER:
            group = subset[subset.series == series].sort_values("polymer_residues")
            group = group[group[y].notna()]
            if group.empty:
                continue
            plotted_any = True
            values = group[y] if metric == "speed" else group[y] / (1024**3)
            ax.plot(
                group.polymer_residues,
                values,
                label=series,
                color=COLORS[series],
                marker=MARKERS[series],
                markersize=3.6,
                markeredgewidth=0.45,
                markeredgecolor="white",
            )
            if metric == "speed" and group.n_samples.min() > 1:
                ax.fill_between(
                    group.polymer_residues,
                    group.throughput_q25,
                    group.throughput_q75,
                    color=COLORS[series],
                    alpha=0.12,
                    linewidth=0,
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Polymer residues per structure")
        ax.set_ylabel(
            "Throughput (structures s$^{-1}$)"
            if metric == "speed"
            else "Peak working memory (GiB)"
        )
        ax.set_title(PROTOCOL_TITLES[protocol])
        ax.grid(which="major", color="#D9D9D9", linewidth=0.55)
        ax.grid(which="minor", color="#EEEEEE", linewidth=0.35)
        if metric == "memory" and subset["process_peak_rss_bytes"].isna().any():
            ax.text(
                0.02,
                0.02,
                "CPU RSS unavailable in legacy records",
                transform=ax.transAxes,
                fontsize=6.3,
                color="#666666",
            )
    if not plotted_any:
        plt.close(fig)
        return
    handles = [
        mpl.lines.Line2D(
            [],
            [],
            color=COLORS[series],
            marker=MARKERS[series],
            markersize=4,
            linewidth=1.25,
            label=series,
        )
        for series in ORDER
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        ncol=6,
        loc="outside lower center",
    )
    note = (
        "CPU: process peak RSS; GPU: peak PyTorch CUDA allocation"
        if metric == "memory"
        else "Higher is better"
    )
    fig.suptitle(
        f"{MODALITY_TITLES[modality]} · tmol {version} · {metric.capitalize()}\n{note}",
        fontsize=9,
    )
    save(fig, f"{modality}-workloads-{metric}-tmol-{version}")
    plt.close(fig)


def composite_workload_figure(data: pd.DataFrame, metric: str) -> None:
    """Create the requested 2-workload by 3-modality composite figure."""
    fig, axes = plt.subplots(
        len(WORKLOADS),
        len(MODALITY_TITLES),
        figsize=(14.2, 7.4),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )
    for row_index, protocol in enumerate(WORKLOADS):
        for column_index, modality in enumerate(MODALITY_TITLES):
            ax = axes[row_index, column_index]
            subset = data[
                (data.modality == modality)
                & (data.protocol == protocol)
                & (data.status == "ok")
                & data.selected_for_plot
            ].copy()
            y = (
                "structures_per_second"
                if metric == "speed"
                else "peak_working_memory_bytes"
            )
            for series in ORDER:
                engine_group = subset[subset.series == series]
                if series == "PyRosetta CPU · B1":
                    version_groups = [(None, engine_group)]
                else:
                    version_groups = list(engine_group.groupby("engine_version"))
                for version, group in version_groups:
                    group = group[group[y].notna()].sort_values("polymer_residues")
                    if group.empty:
                        continue
                    values = group[y] if metric == "speed" else group[y] / (1024**3)
                    historical = version in {"0.1.46", "0.1.47"}
                    ax.plot(
                        group.polymer_residues,
                        values,
                        color=COLORS[series],
                        marker=MARKERS[series],
                        linestyle="--" if historical else "-",
                        markerfacecolor="white" if historical else COLORS[series],
                        markeredgecolor=COLORS[series],
                        markeredgewidth=0.7,
                        markersize=3.5,
                        linewidth=1.15,
                    )
                    if metric == "speed" and group.n_samples.min() > 1:
                        ax.fill_between(
                            group.polymer_residues,
                            group.throughput_q25,
                            group.throughput_q75,
                            color=COLORS[series],
                            alpha=0.08,
                            linewidth=0,
                        )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(25, 1600)
            ax.grid(which="major", color="#D9D9D9", linewidth=0.55)
            ax.grid(which="minor", color="#EEEEEE", linewidth=0.35)
            panel = chr(ord("a") + row_index * len(MODALITY_TITLES) + column_index)
            ax.text(
                -0.10,
                1.04,
                f"({panel})",
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                fontsize=8,
                fontweight="bold",
            )
            if row_index == 0:
                ax.set_title(MODALITY_TITLES[modality])
            if row_index == len(WORKLOADS) - 1:
                ax.set_xlabel("Polymer residues per structure")
            if column_index == 0:
                ax.set_ylabel(
                    (
                        "Throughput (structures s$^{-1}$)"
                        if metric == "speed"
                        else "Peak working memory (GiB)"
                    )
                    + f"\n{PROTOCOL_TITLES[protocol]}"
                )
            else:
                ax.text(
                    0.02,
                    0.97,
                    PROTOCOL_TITLES[protocol],
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=7,
                    color="#555555",
                )

    device_handles = [
        mpl.lines.Line2D(
            [],
            [],
            color=COLORS[series],
            marker=MARKERS[series],
            linewidth=1.2,
            markersize=4,
            label=series,
        )
        for series in ORDER
    ]
    version_handles = [
        mpl.lines.Line2D([], [], color="#555555", linestyle="-", label="tmol 0.1.55"),
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
    fig.legend(
        handles=[*device_handles, *version_handles],
        frameon=False,
        ncol=4,
        loc="outside lower center",
    )
    note = (
        "CPU memory is process peak RSS; GPU memory is peak PyTorch CUDA allocation"
        if metric == "memory"
        else "Higher throughput is better"
    )
    fig.suptitle(
        f"TMol multimodal workloads · {metric.capitalize()}\n{note}", fontsize=10
    )
    save(fig, f"multimodal-workloads-{metric}")
    plt.close(fig)


def term_agreement_figure(data: pd.DataFrame, modality: str, version: str) -> None:
    subset = data[(data.modality == modality) & (data.tmol_version == version)].copy()
    if len(subset) < 2:
        return
    pearson = pearsonr(subset.pyrosetta_reu, subset.tmol_score).statistic
    spearman = spearmanr(subset.pyrosetta_reu, subset.tmol_score).statistic
    low = min(subset.pyrosetta_reu.min(), subset.tmol_score.min())
    high = max(subset.pyrosetta_reu.max(), subset.tmol_score.max())
    margin = 0.04 * max(high - low, 1.0)
    fig, ax = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
    terms = sorted(subset.term.unique())
    cmap = plt.get_cmap("tab20")
    for index, term in enumerate(terms):
        group = subset[subset.term == term]
        ax.scatter(
            group.pyrosetta_reu,
            group.tmol_score,
            s=14,
            alpha=0.8,
            linewidth=0.35,
            edgecolor="white",
            color=cmap(index % 20),
            label=term,
        )
    ax.plot(
        [low - margin, high + margin],
        [low - margin, high + margin],
        color="#333333",
        linewidth=0.8,
        linestyle="--",
    )
    ax.set_xlim(low - margin, high + margin)
    ax.set_ylim(low - margin, high + margin)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("PyRosetta weighted energy (REU)")
    ax.set_ylabel("tmol weighted energy (score units)")
    ax.set_title(f"{MODALITY_TITLES[modality]} · matched terms · tmol {version}")
    ax.text(
        0.03,
        0.97,
        f"Pearson $r$ = {pearson:.4f}\nSpearman $\\rho$ = {spearman:.4f}\n$n$ = {len(subset)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
    )
    ax.grid(color="#E6E6E6", linewidth=0.4)
    # Keep the main panel readable; a compact term legend sits outside it.
    ax.legend(
        frameon=False,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0,
        markerscale=0.8,
    )
    save(fig, f"{modality}-term-energy-agreement-tmol-{version}")
    plt.close(fig)


def total_agreement_figure(data: pd.DataFrame, modality: str, version: str) -> None:
    subset = data[(data.modality == modality) & (data.tmol_version == version)].copy()
    if len(subset) < 2:
        return
    pearson = pearsonr(subset.pyrosetta_reu, subset.tmol_score).statistic
    spearman = spearmanr(subset.pyrosetta_reu, subset.tmol_score).statistic
    low = min(subset.pyrosetta_reu.min(), subset.tmol_score.min())
    high = max(subset.pyrosetta_reu.max(), subset.tmol_score.max())
    margin = 0.06 * max(high - low, 1.0)
    fig, ax = plt.subplots(figsize=(3.45, 3.15), constrained_layout=True)
    ax.scatter(
        subset.pyrosetta_reu,
        subset.tmol_score,
        s=22,
        color="#0072B2",
        linewidth=0.5,
        edgecolor="white",
        zorder=3,
    )
    ax.plot(
        [low - margin, high + margin],
        [low - margin, high + margin],
        color="#333333",
        linewidth=0.8,
        linestyle="--",
    )
    ax.set_xlim(low - margin, high + margin)
    ax.set_ylim(low - margin, high + margin)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("PyRosetta total weighted energy (REU)")
    ax.set_ylabel("tmol total weighted energy (score units)")
    ax.set_title(f"{MODALITY_TITLES[modality]} · total energy · tmol {version}")
    ax.text(
        0.03,
        0.97,
        f"Pearson $r$ = {pearson:.4f}\nSpearman $\\rho$ = {spearman:.4f}\n$n$ = {len(subset)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
    )
    ax.grid(color="#E6E6E6", linewidth=0.4)
    save(fig, f"{modality}-total-energy-agreement-tmol-{version}")
    plt.close(fig)


def main() -> None:
    style()
    timing = pd.read_csv(ROOT / "results/summary/timing_summary.csv")
    for metric in ("speed", "memory"):
        composite_workload_figure(timing, metric)


if __name__ == "__main__":
    main()
