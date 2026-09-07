"""Prepare a frozen, size-stratified multimodal benchmark corpus.

Run one modality per process. Successful structures are written to a partial
manifest; rejected candidates retain an explicit reason in a companion table.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

from common import ROOT, sha256, write_rows

RCSB_FILES = "https://files.rcsb.org/download"
CANONICAL_AA = {
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
WATER_AND_BUFFER = {
    "HOH",
    "DOD",
    "WAT",
    "SO4",
    "PO4",
    "GOL",
    "EDO",
    "PEG",
    "MPD",
    "ACT",
    "FMT",
    "TRS",
    "MES",
}
SUPPORTED_LIGAND_ELEMENTS = {
    "H",
    "B",
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "CL",
    "BR",
    "I",
}
LIGAND_TEMPLATES = (
    "hsp90",
    "cdk2",
    "p38",
    "ada",
    "src",
    "ache",
    "cox1",
    "hivrt",
    "ace",
)
TMOL_PAPER_SOURCE = Path("/mnt/home/kdidi/tmol-paper-sources/v0.1.55")
TMOL_LIGAND_DATA = TMOL_PAPER_SOURCE / "tmol/tests/data/protein_ligand_test"
ROSETTA_LIGAND_DATA = Path("/mnt/home/kdidi/tmol-paper-benchmark/data/ligand_params")


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def download(dataset_id: str, suffix: str, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{dataset_id}.{suffix}"
    if path.is_file() and path.stat().st_size:
        return path
    url = f"{RCSB_FILES}/{dataset_id.upper()}.{suffix}"
    temporary = path.with_suffix(path.suffix + ".tmp")
    request = urllib.request.Request(
        url, headers={"User-Agent": "tmol-multimodal-benchmark/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            temporary.write_bytes(response.read())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def candidate_order(rows: list[dict], target: int) -> list[dict]:
    """Try a full-size-range primary set before size-stratified backups."""
    if len(rows) <= target:
        return rows
    primary_indices = {
        round(index * (len(rows) - 1) / (target - 1)) for index in range(target)
    }
    return [rows[index] for index in sorted(primary_indices)] + [
        row for index, row in enumerate(rows) if index not in primary_indices
    ]


def initialize_pyrosetta(pyrosetta_path: Path):
    sys.path.insert(0, str(pyrosetta_path))
    import pyrosetta

    pyrosetta.init(
        "-beta -mute all -ignore_unrecognized_res true -ignore_waters true "
        "-load_PDB_components false -in:file:obey_ENDMDL true "
        "-multithreading:total_threads 1"
    )
    return pyrosetta


def prepare_polymer(
    row: dict, modality: str, pyrosetta, raw_dir: Path, clean_dir: Path
) -> dict:
    dataset_id = row["dataset_id"]
    raw = download(dataset_id, "pdb", raw_dir)
    pose = pyrosetta.pose_from_file(str(raw))
    for index in range(pose.total_residue(), 0, -1):
        residue = pose.residue(index)
        keep = residue.is_protein() or (
            modality == "protein_nucleic" and (residue.is_DNA() or residue.is_RNA())
        )
        if not keep:
            pose.delete_residue_slow(index)
    protein = sum(
        pose.residue(index).is_protein() for index in range(1, pose.total_residue() + 1)
    )
    dna = sum(
        pose.residue(index).is_DNA() for index in range(1, pose.total_residue() + 1)
    )
    rna = sum(
        pose.residue(index).is_RNA() for index in range(1, pose.total_residue() + 1)
    )
    residues = pose.total_residue()
    if protein == 0:
        raise ValueError("no protein residues survived cleanup")
    if modality == "protein_nucleic" and dna + rna == 0:
        raise ValueError("no DNA/RNA residues survived cleanup")
    if not 30 <= residues <= 1500:
        raise ValueError(f"cleaned polymer size {residues} is outside 30..1500")
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean = clean_dir / f"{dataset_id}.pdb"
    pose.dump_pdb(str(clean))
    return {
        "modality": modality,
        "dataset_id": dataset_id,
        "structure_path": relative(clean),
        "ligand_tmol_params": "",
        "ligand_rosetta_params": "",
        "residues": residues,
        "atoms": pose.total_atoms(),
        "protein_residues": protein,
        "dna_residues": dna,
        "rna_residues": rna,
        "ligand_atoms": 0,
        "source": f"{RCSB_FILES}/{dataset_id.upper()}.pdb",
        "source_sha256": sha256(raw),
        "structure_sha256": sha256(clean),
        "status": "ok",
        "reason": "",
        "historical_version": "0.1.47" if modality == "protein_nucleic" else "0.1.46",
        "fastrelax": "yes",
    }


def residue_groups(structure):
    import biotite.structure as struc

    starts = struc.get_residue_starts(structure)
    ends = [*starts[1:], len(structure)]
    return [structure[start:end] for start, end in zip(starts, ends)]


def prepare_ligand_complex(
    row: dict, raw_dir: Path, clean_dir: Path, params_dir: Path
) -> dict:
    import numpy as np
    import biotite.structure as struc
    import biotite.structure.io
    from tmol.database import ParameterDatabase
    from tmol.ligand import load_params_file, prepare_ligands, write_params_file

    from export_ligand_params import neighbor_radius, replace_neighbor_radius

    dataset_id = row["dataset_id"]
    raw = download(dataset_id, "cif", raw_dir)
    structure = biotite.structure.io.load_structure(
        str(raw), model=1, include_bonds=True
    )
    if isinstance(structure, struc.AtomArrayStack):
        structure = structure[0]
    protein_mask = (~structure.hetero) & np.isin(structure.res_name, list(CANONICAL_AA))
    candidates = []
    for residue in residue_groups(structure):
        if not bool(residue.hetero[0]):
            continue
        name = str(residue.res_name[0]).upper()
        elements = {str(value).strip().upper() for value in residue.element}
        heavy_atoms = sum(
            str(value).strip().upper() not in {"", "H"} for value in residue.element
        )
        if (
            name in WATER_AND_BUFFER
            or heavy_atoms < 6
            or heavy_atoms > 100
            or not elements <= SUPPORTED_LIGAND_ELEMENTS
        ):
            continue
        key = (
            str(residue.chain_id[0]),
            int(residue.res_id[0]),
            str(residue.ins_code[0]),
            name,
        )
        candidates.append((heavy_atoms, len(residue), key))
    if not candidates:
        raise ValueError("no supported drug-like nonpolymer residue found")
    heavy_atoms, ligand_atoms, ligand_key = max(candidates)
    chain, residue_id, insertion, ligand_name = ligand_key
    ligand_mask = (
        structure.hetero
        & (structure.chain_id == chain)
        & (structure.res_id == residue_id)
        & (structure.ins_code == insertion)
        & (structure.res_name == ligand_name)
    )
    filtered = structure[protein_mask | ligand_mask]
    protein_residues = sum(
        not bool(residue.hetero[0]) for residue in residue_groups(filtered)
    )
    if protein_residues < 30:
        raise ValueError(f"only {protein_residues} canonical protein residues")

    clean_dir.mkdir(parents=True, exist_ok=True)
    params_dir.mkdir(parents=True, exist_ok=True)
    clean = clean_dir / f"{dataset_id}.pdb"
    tmol_params = params_dir / f"{dataset_id}.tmol"
    rosetta_params = params_dir / f"{dataset_id}.params"
    prepare_ligands(
        filtered,
        param_db=ParameterDatabase.get_default(),
        params_output=str(tmol_params),
        strict_ligands=True,
        sample_proton_chi=False,
    )
    preparations = load_params_file(str(tmol_params))
    if len(preparations) != 1:
        raise ValueError(f"expected one prepared ligand, found {len(preparations)}")
    biotite.structure.io.save_structure(str(clean), filtered)
    write_params_file(preparations[0], str(rosetta_params), format="rosetta")
    replace_neighbor_radius(
        rosetta_params, neighbor_radius(preparations[0], str(clean))
    )
    residues = protein_residues + 1
    return {
        "modality": "protein_ligand",
        "dataset_id": dataset_id,
        "structure_path": relative(clean),
        "ligand_tmol_params": relative(tmol_params),
        "ligand_rosetta_params": relative(rosetta_params),
        "residues": residues,
        "atoms": len(filtered),
        "protein_residues": protein_residues,
        "dna_residues": 0,
        "rna_residues": 0,
        "ligand_atoms": ligand_atoms,
        "source": f"{RCSB_FILES}/{dataset_id.upper()}.cif",
        "source_sha256": sha256(raw),
        "structure_sha256": sha256(clean),
        "status": "ok",
        "reason": "",
        "historical_version": "0.1.46",
        "fastrelax": "yes",
    }


def pdb_xyz(line: str) -> tuple[float, float, float]:
    return tuple(float(line[start : start + 8]) for start in (30, 38, 46))


def translated_pdb_line(
    line: str, serial: int, translation: tuple[float, float, float]
) -> str:
    xyz = tuple(value + delta for value, delta in zip(pdb_xyz(line), translation))
    padded = line.rstrip("\n").ljust(80)
    return (
        f"{padded[:6]}{serial:5d}{padded[11:21]}Z9999{padded[26:30]}"
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{padded[54:]}\n"
    )


def ligand_template_paths(template: str) -> tuple[Path, Path, Path]:
    suffix = "_complex_nometals.pdb" if template in {"ace", "ada"} else "_complex.pdb"
    return (
        TMOL_LIGAND_DATA / f"{template}{suffix}",
        TMOL_LIGAND_DATA / f"{template}.xtal-lig.mmff94.tmol",
        ROSETTA_LIGAND_DATA / f"{template}.params",
    )


def prepare_synthetic_ligand_complex(
    row: dict,
    template: str,
    pyrosetta,
    raw_dir: Path,
    clean_dir: Path,
    params_dir: Path,
) -> dict:
    """Graft a validated ligand onto a diverse, cleaned protein backbone."""
    protein_row = prepare_polymer(
        row, "protein", pyrosetta, raw_dir, clean_dir / "base"
    )
    protein_path = ROOT / protein_row["structure_path"]
    protein_lines = [
        line
        for line in protein_path.read_text().splitlines(keepends=True)
        if not line.startswith(("END", "CONECT"))
    ]
    protein_atoms = [line for line in protein_lines if line.startswith("ATOM")]
    if not protein_atoms:
        raise ValueError("cleaned protein contains no ATOM records")
    anchor = max((pdb_xyz(line) for line in protein_atoms), key=lambda xyz: xyz[0])
    template_pdb, template_tmol, template_rosetta = ligand_template_paths(template)
    ligand_lines = [
        line
        for line in template_pdb.read_text().splitlines(keepends=True)
        if line.startswith("HETATM")
    ]
    if not ligand_lines:
        raise ValueError(f"{template}: template contains no HETATM records")
    centroid = tuple(
        sum(pdb_xyz(line)[axis] for line in ligand_lines) / len(ligand_lines)
        for axis in range(3)
    )
    target = (anchor[0] + 4.0, anchor[1], anchor[2])
    translation = tuple(value - center for value, center in zip(target, centroid))
    maximum_serial = max(int(line[6:11]) for line in protein_atoms)
    grafted = [
        translated_pdb_line(line, maximum_serial + index + 1, translation)
        for index, line in enumerate(ligand_lines)
    ]
    clean_dir.mkdir(parents=True, exist_ok=True)
    params_dir.mkdir(parents=True, exist_ok=True)
    clean = clean_dir / f"{row['dataset_id']}.pdb"
    clean.write_text("".join([*protein_lines, "TER\n", *grafted, "END\n"]))
    tmol_params = params_dir / f"{template}.tmol"
    rosetta_params = params_dir / f"{template}.params"
    if not tmol_params.exists():
        shutil.copyfile(template_tmol, tmol_params)
    if not rosetta_params.exists():
        shutil.copyfile(template_rosetta, rosetta_params)
    return {
        "modality": "protein_ligand",
        "dataset_id": row["dataset_id"],
        "structure_path": relative(clean),
        "ligand_tmol_params": relative(tmol_params),
        "ligand_rosetta_params": relative(rosetta_params),
        "residues": int(protein_row["protein_residues"]) + 1,
        "atoms": int(protein_row["atoms"]) + len(ligand_lines),
        "protein_residues": protein_row["protein_residues"],
        "dna_residues": 0,
        "rna_residues": 0,
        "ligand_atoms": len(ligand_lines),
        "source": f"synthetic_surface_graft:{protein_row['source']}:{template}",
        "source_sha256": protein_row["source_sha256"],
        "structure_sha256": sha256(clean),
        "ligand_template": template,
        "ligand_template_sha256": sha256(template_pdb),
        "status": "ok",
        "reason": "",
        "historical_version": "0.1.46",
        "fastrelax": "yes",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modality",
        required=True,
        choices=("protein", "protein_ligand", "protein_nucleic"),
    )
    parser.add_argument("--target", type=int, default=128)
    parser.add_argument(
        "--pyrosetta-path",
        type=Path,
        default=Path("/mnt/home/kdidi/projects/pyrosetta-2024.39"),
    )
    args = parser.parse_args()
    frozen = json.loads((ROOT / "metadata/rcsb_candidates.json").read_text())
    candidate_modality = (
        "protein" if args.modality == "protein_ligand" else args.modality
    )
    candidates = candidate_order(frozen["modalities"][candidate_modality], args.target)
    base = ROOT / "data/expanded" / args.modality
    successes = []
    failures = []
    pyrosetta = initialize_pyrosetta(args.pyrosetta_path)
    for candidate_index, candidate in enumerate(candidates):
        if len(successes) >= args.target:
            break
        dataset_id = candidate["dataset_id"]
        try:
            if args.modality == "protein_ligand":
                result = prepare_synthetic_ligand_complex(
                    candidate,
                    LIGAND_TEMPLATES[candidate_index % len(LIGAND_TEMPLATES)],
                    pyrosetta,
                    base / "raw",
                    base / "clean",
                    base / "params",
                )
            else:
                result = prepare_polymer(
                    candidate,
                    args.modality,
                    pyrosetta,
                    base / "raw",
                    base / "clean",
                )
            successes.append(result)
            print(args.modality, dataset_id, result["residues"], "ok", flush=True)
        except Exception as error:
            failures.append(
                {
                    "modality": args.modality,
                    "dataset_id": dataset_id,
                    "reported_polymer_residues": candidate.get(
                        "reported_polymer_residues"
                    ),
                    "status": "excluded",
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            print(
                args.modality,
                dataset_id,
                "excluded",
                failures[-1]["reason"],
                flush=True,
            )
    write_rows(ROOT / f"metadata/manifest-{args.modality}.csv", successes)
    if failures:
        write_rows(ROOT / f"metadata/exclusions-{args.modality}.csv", failures)
    if len(successes) < args.target:
        raise SystemExit(
            f"Prepared only {len(successes)}/{args.target} {args.modality} structures"
        )


if __name__ == "__main__":
    main()
