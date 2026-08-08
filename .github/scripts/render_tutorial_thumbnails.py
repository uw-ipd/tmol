#!/usr/bin/env python3
"""Render tutorial-card SVGs from the structures used in the notebooks."""

from __future__ import annotations

import html
import math
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "docs/_static/tutorials"
WIDTH = 560
HEIGHT = 320

PROTEIN_NAMES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}
NA_NAMES = {"A", "C", "G", "U", "DA", "DC", "DG", "DT"}
WATER_NAMES = {"HOH", "WAT", "DOD"}


@dataclass(frozen=True)
class Atom:
    group: str
    element: str
    atom_name: str
    residue_name: str
    chain: str
    residue_id: str
    xyz: tuple[float, float, float]


def read_cif_atoms(path: Path) -> list[Atom]:
    """Read model-one atom-site rows from a repository mmCIF fixture."""
    lines = path.read_text(encoding="utf-8").splitlines()
    headers: list[str] = []
    rows: list[list[str]] = []
    in_atom_loop = False
    collecting_headers = False

    for line in lines:
        stripped = line.strip()
        if stripped == "loop_":
            headers = []
            rows = []
            collecting_headers = True
            in_atom_loop = False
            continue
        if collecting_headers and stripped.startswith("_"):
            headers.append(stripped.split()[0])
            in_atom_loop = bool(headers and headers[0].startswith("_atom_site."))
            continue
        if collecting_headers and headers:
            collecting_headers = False
        if not in_atom_loop:
            continue
        if not stripped or stripped == "#":
            break
        values = shlex.split(stripped)
        if len(values) >= len(headers):
            rows.append(values[: len(headers)])

    if not headers or not rows:
        raise ValueError(f"No atom-site loop found in {path}")

    index = {name: position for position, name in enumerate(headers)}

    def value(row: Sequence[str], *names: str) -> str:
        for name in names:
            if name in index:
                return row[index[name]]
        raise KeyError(names)

    atoms = []
    for row in rows:
        if value(row, "_atom_site.pdbx_PDB_model_num") not in {"1", ".", "?"}:
            continue
        atoms.append(
            Atom(
                group=value(row, "_atom_site.group_PDB"),
                element=value(row, "_atom_site.type_symbol"),
                atom_name=value(
                    row, "_atom_site.auth_atom_id", "_atom_site.label_atom_id"
                ),
                residue_name=value(
                    row, "_atom_site.auth_comp_id", "_atom_site.label_comp_id"
                ),
                chain=value(row, "_atom_site.auth_asym_id", "_atom_site.label_asym_id"),
                residue_id=value(row, "_atom_site.auth_seq_id", "_atom_site.label_seq_id"),
                xyz=(
                    float(value(row, "_atom_site.Cartn_x")),
                    float(value(row, "_atom_site.Cartn_y")),
                    float(value(row, "_atom_site.Cartn_z")),
                ),
            )
        )
    return atoms


def project(xyz: tuple[float, float, float]) -> tuple[float, float]:
    """Apply a fixed isometric projection for reproducible thumbnails."""
    x, y, z = xyz
    return (0.80 * x + 0.18 * y - 0.57 * z, -0.30 * x + 0.90 * y + 0.31 * z)


def fit_points(
    points: Sequence[tuple[float, float]],
    box: tuple[float, float, float, float],
    *,
    padding: float = 16,
) -> list[tuple[float, float]]:
    x0, y0, width, height = box
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    span_x = max(max(xs) - min(xs), 1e-6)
    span_y = max(max(ys) - min(ys), 1e-6)
    scale = min((width - 2 * padding) / span_x, (height - 2 * padding) / span_y)
    center_x = 0.5 * (min(xs) + max(xs))
    center_y = 0.5 * (min(ys) + max(ys))
    return [
        (
            x0 + width / 2 + (x - center_x) * scale,
            y0 + height / 2 - (y - center_y) * scale,
        )
        for x, y in points
    ]


def trace_atoms(atoms: Sequence[Atom]) -> dict[tuple[str, str], list[Atom]]:
    traces: dict[tuple[str, str], list[Atom]] = {}
    seen: set[tuple[str, str, str]] = set()
    phosphate_residues = {
        (atom.chain, atom.residue_id)
        for atom in atoms
        if atom.residue_name in NA_NAMES and atom.atom_name == "P"
    }
    for atom in atoms:
        if atom.residue_name in PROTEIN_NAMES and atom.atom_name == "CA":
            kind = "protein"
        elif atom.residue_name in NA_NAMES and atom.atom_name in {"P", "C4'"}:
            kind = "na"
            if atom.atom_name == "C4'" and (
                atom.chain,
                atom.residue_id,
            ) in phosphate_residues:
                continue
        else:
            continue
        residue_key = (kind, atom.chain, atom.residue_id)
        if residue_key in seen:
            continue
        seen.add(residue_key)
        traces.setdefault((kind, atom.chain), []).append(atom)
    return traces


def svg_header(title: str, sources: Iterable[Path]) -> list[str]:
    source_names = ", ".join(path.name for path in sources)
    return [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 320" '
        'role="img" aria-labelledby="title description">',
        f'<title id="title">{html.escape(title)}</title>',
        (
            f'<desc id="description">Static molecular render generated from '
            f'{html.escape(source_names)} used by this tutorial.</desc>'
        ),
        f"<metadata>Generated from tutorial fixtures: {html.escape(source_names)}</metadata>",
        '<rect width="560" height="320" rx="24" fill="#111a2d"/>',
        '<rect x="10" y="10" width="540" height="300" rx="18" '
        'fill="none" stroke="#263a5e" stroke-width="2"/>',
    ]


def path(points: Sequence[tuple[float, float]], color: str, width: float, opacity=1.0):
    coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round" '
        f'opacity="{opacity}"/>'
    )


def circles(
    points: Sequence[tuple[float, float]],
    color: str,
    radius: float,
    *,
    opacity: float = 1.0,
    stroke: str = "none",
) -> list[str]:
    return [
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" '
        f'stroke="{stroke}" stroke-width="1.5" opacity="{opacity}"/>'
        for x, y in points
    ]


def render_structure(
    atoms: Sequence[Atom],
    box: tuple[float, float, float, float],
    *,
    highlight_residues: set[str] | None = None,
    opacity: float = 1.0,
) -> tuple[list[str], list[tuple[float, float]], dict[int, tuple[float, float]]]:
    traces = trace_atoms(atoms)
    trace_atoms_flat = [atom for group in traces.values() for atom in group]
    ligand_atoms = [
        atom
        for atom in atoms
        if atom.residue_name not in PROTEIN_NAMES
        and atom.residue_name not in NA_NAMES
        and atom.residue_name not in WATER_NAMES
        and atom.element != "H"
    ]
    points_3d = [atom.xyz for atom in trace_atoms_flat + ligand_atoms]
    projected = [project(point) for point in points_3d]
    fitted = fit_points(projected, box)
    fitted_by_id = {
        id(atom): fitted[index]
        for index, atom in enumerate(trace_atoms_flat + ligand_atoms)
    }

    elements: list[str] = []
    protein_colors = ["#54c7ec", "#77d6a5", "#f5c45d", "#b99bea"]
    na_colors = ["#ff7b89", "#75d7e8", "#c28cff", "#ffd166"]
    protein_index = 0
    na_index = 0
    for (kind, _chain), group in traces.items():
        group_points = [fitted_by_id[id(atom)] for atom in group]
        if kind == "protein":
            color = protein_colors[protein_index % len(protein_colors)]
            protein_index += 1
            width = 8
        else:
            color = na_colors[na_index % len(na_colors)]
            na_index += 1
            width = 6
        elements.append(path(group_points, color, width, opacity))

    if highlight_residues:
        highlighted = [
            fitted_by_id[id(atom)]
            for atom in trace_atoms_flat
            if atom.residue_id in highlight_residues
        ]
        elements.extend(circles(highlighted, "#ff4d6d", 7, stroke="#fff3f5"))

    ligand_points = [fitted_by_id[id(atom)] for atom in ligand_atoms]
    if ligand_points:
        for first_index, first_atom in enumerate(ligand_atoms):
            for second_index in range(first_index + 1, len(ligand_atoms)):
                second_atom = ligand_atoms[second_index]
                distance = math.dist(first_atom.xyz, second_atom.xyz)
                if distance <= 1.95:
                    elements.append(
                        path(
                            [
                                fitted_by_id[id(first_atom)],
                                fitted_by_id[id(second_atom)],
                            ],
                            "#f4f7fb",
                            3,
                            opacity,
                        )
                    )
        element_colors = {
            "C": "#f5c45d",
            "N": "#5ca9ff",
            "O": "#ff6577",
            "P": "#ff9f43",
            "S": "#ffe066",
        }
        for atom, point in zip(ligand_atoms, ligand_points):
            elements.extend(
                circles(
                    [point],
                    element_colors.get(atom.element, "#d9e2f2"),
                    4.2,
                    opacity=opacity,
                    stroke="#111a2d",
                )
            )
    return elements, fitted, fitted_by_id


def write_svg(filename: str, title: str, sources: Sequence[Path], body: Sequence[str]):
    content = svg_header(title, sources)
    content.extend(body)
    content.append("</svg>")
    (OUTPUT_DIR / filename).write_text("\n".join(content) + "\n", encoding="utf-8")


def distance_heatmap(points: Sequence[tuple[float, float, float]]) -> list[str]:
    sampled = list(points)[:: max(len(points) // 13, 1)][:13]
    cells = []
    origin_x, origin_y, size = 360, 84, 12
    max_distance = max(math.dist(a, b) for a in sampled for b in sampled)
    for row, first in enumerate(sampled):
        for column, second in enumerate(sampled):
            normalized = math.dist(first, second) / max_distance
            red = round(235 - 150 * normalized)
            blue = round(115 + 120 * normalized)
            cells.append(
                f'<rect x="{origin_x + column * size}" y="{origin_y + row * size}" '
                f'width="{size - 1}" height="{size - 1}" fill="rgb({red},90,{blue})"/>'
            )
    return cells


def main() -> None:
    cif = ROOT / "tmol/tests/data/cif"
    ligand_cif = (
        ROOT / "tmol/tests/data/protein_ligand_test/ada.tmol.nomin.cif"
    )
    paths = {
        "ubq": cif / "1UBQ.cif",
        "r21": cif / "1R21.cif",
        "bl8": cif / "1BL8.cif",
        "hdd": cif / "1HDD.cif",
        "eht": cif / "1EHT.cif",
        "ligand": ligand_cif,
    }
    structures = {name: read_cif_atoms(path_) for name, path_ in paths.items()}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    body, _, _ = render_structure(
        structures["ubq"], (42, 26, 476, 268), highlight_residues={"3", "4", "5"}
    )
    write_svg(
        "01_working_with_tmol.svg",
        "Working with TMol — 1UBQ structure and selected residues",
        [paths["ubq"]],
        body,
    )

    body = []
    for index, key in enumerate(("ubq", "r21", "bl8")):
        rendered, _, _ = render_structure(
            structures[key], (22 + index * 180, 48, 156, 224)
        )
        body.extend(rendered)
    write_svg(
        "02_gpu_batching.svg",
        "GPU batching — three tutorial proteins",
        [paths["ubq"], paths["r21"], paths["bl8"]],
        body,
    )

    body, _, _ = render_structure(structures["ubq"], (30, 38, 300, 244))
    ca_points = [
        atom.xyz
        for atom in structures["ubq"]
        if atom.residue_name in PROTEIN_NAMES and atom.atom_name == "CA"
    ]
    body.extend(distance_heatmap(ca_points))
    write_svg(
        "03_scoring_and_analysis.svg",
        "Scoring — 1UBQ and a coordinate-derived residue map",
        [paths["ubq"]],
        body,
    )

    body, _, _ = render_structure(
        structures["ubq"],
        (36, 28, 488, 264),
        highlight_residues={str(index) for index in range(3, 9)},
    )
    write_svg(
        "04_packing_and_mutation_scan.svg",
        "Packing — 1UBQ with the repacked region highlighted",
        [paths["ubq"]],
        body,
    )

    first, _, _ = render_structure(structures["ubq"], (38, 30, 484, 260), opacity=0.82)
    second, _, _ = render_structure(
        structures["ubq"], (45, 24, 484, 260), opacity=0.32
    )
    body = second + first
    body.extend(
        [
            '<circle cx="94" cy="76" r="8" fill="#b99bea" stroke="#f5efff" stroke-width="2"/>',
            '<circle cx="468" cy="244" r="8" fill="#b99bea" stroke="#f5efff" stroke-width="2"/>',
        ]
    )
    write_svg(
        "05_minimization_constraints_kinematics.svg",
        "Minimization — overlaid 1UBQ conformations and restraint anchors",
        [paths["ubq"]],
        body,
    )

    body = []
    for offset, opacity in ((-9, 0.18), (-3, 0.30), (3, 0.50), (9, 0.92)):
        rendered, _, _ = render_structure(
            structures["ubq"], (38 + offset, 28 - offset / 2, 484, 264), opacity=opacity
        )
        body.extend(rendered)
    write_svg(
        "06_fast_relax.svg",
        "FastRelax — stage overlay from the 1UBQ tutorial system",
        [paths["ubq"]],
        body,
    )

    body, _, _ = render_structure(structures["ligand"], (34, 26, 492, 268))
    write_svg(
        "07_ligand_and_params.svg",
        "Ligands — the tutorial protein–ligand complex",
        [paths["ligand"]],
        body,
    )

    body, _, _ = render_structure(structures["hdd"], (22, 30, 328, 260))
    aptamer, _, _ = render_structure(structures["eht"], (356, 52, 180, 216))
    body.extend(aptamer)
    write_svg(
        "08_nucleic_acids.svg",
        "Nucleic acids — homeodomain–DNA and theophylline aptamer",
        [paths["hdd"], paths["eht"]],
        body,
    )

    print(f"Rendered eight tutorial thumbnails in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
