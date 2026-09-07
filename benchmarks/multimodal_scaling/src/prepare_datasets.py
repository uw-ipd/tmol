from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from common import ROOT, sha256, write_rows

TMOL_SOURCE = Path("/mnt/home/kdidi/tmol-paper-sources/v0.1.55")
PDB_DATA = TMOL_SOURCE / "tmol/tests/data/pdb"
PLI_DATA = TMOL_SOURCE / "tmol/tests/data/protein_ligand_test"

PROTEINS = [
    ("5uoi", "bysize_040_res_5uoi.pdb"),
    ("2mtq", "bysize_075_res_2mtq.pdb"),
    ("5umr", "bysize_100_res_5umr.pdb"),
    ("5yzf", "bysize_150_res_5yzf.pdb"),
    ("5n5g", "bysize_250_res_5n5g.pdb"),
    ("6azu", "bysize_400_res_6azu.pdb"),
    ("5m4a", "bysize_600_res_5m4a.pdb"),
    ("5opj", "bysize_800_res_5opj.pdb"),
    ("1s78", "1s78.pdb"),
]

LIGAND_TARGETS = [
    "hsp90",
    "cdk2",
    "p38",
    "ada",
    "src",
    "ache",
    "cox1",
    "hivrt",
    "ace",
]

# High-resolution representatives returned by the RCSB Search API in increasing
# deposited-polymer size bins. The preparation step rejects any entry that loses
# either its protein or its nucleic-acid component during conservative cleanup.
NUCLEIC_CANDIDATES = [
    "3GO3",
    "3G9Y",
    "5EXH",
    "7KII",
    "1YSA",
    "4LUP",
    "3NDH",
    "3BBB",
    "1ASY",
    "1KX5",
    "6Q1H",
]
PROTEIN_RCSB_CANDIDATES = ["1O7J"]
TMOL_UNSUPPORTED = {
    "3GO3": "contains four DSE residues unsupported by tmol",
    "3G9Y": "RNA residue 29 is missing the required non-leaf O5' atom",
    "1ASY": "contains 16 modified RNA residues unsupported by tmol",
}


def pdb_counts(path: Path) -> tuple[int, int]:
    residues = 0
    atoms = 0
    previous_residue: tuple[str, str, str, str] | None = None
    in_first_model = True
    saw_model = False
    with path.open(errors="replace") as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "MODEL":
                if saw_model:
                    in_first_model = False
                saw_model = True
                continue
            if record == "ENDMDL":
                in_first_model = False
                continue
            if record == "TER":
                previous_residue = None
                continue
            if not in_first_model or record not in {"ATOM", "HETATM"}:
                continue
            if line[16:17] not in {" ", "A"}:
                continue
            atoms += 1
            residue = (line[21:22], line[22:26], line[26:27], line[17:20])
            if residue != previous_residue:
                residues += 1
                previous_residue = residue
    return residues, atoms


def clean_rcsb_entries(pyrosetta_path: Path) -> list[dict[str, object]]:
    sys.path.insert(0, str(pyrosetta_path))
    import pyrosetta

    pyrosetta.init(
        "-beta -mute all -ignore_unrecognized_res true -ignore_waters true "
        "-load_PDB_components false -in:file:obey_ENDMDL true"
    )
    rows: list[dict[str, object]] = []
    raw_dir = ROOT / "data/rcsb/raw"
    clean_dir = ROOT / "data/rcsb/clean"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        *(("protein", pdb_id) for pdb_id in PROTEIN_RCSB_CANDIDATES),
        *(("protein_nucleic", pdb_id) for pdb_id in NUCLEIC_CANDIDATES),
    ]
    for modality, pdb_id in candidates:
        raw_path = raw_dir / f"{pdb_id.lower()}.pdb"
        clean_path = clean_dir / f"{pdb_id.lower()}.pdb"
        source = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        status = "ok"
        reason = ""
        try:
            if not raw_path.exists():
                urllib.request.urlretrieve(source, raw_path)
            pose = pyrosetta.pose_from_file(str(raw_path))
            for index in range(pose.total_residue(), 0, -1):
                residue = pose.residue(index)
                keep = residue.is_protein() or (
                    modality == "protein_nucleic"
                    and (residue.is_DNA() or residue.is_RNA())
                )
                if not keep:
                    pose.delete_residue_slow(index)
            n_protein = sum(
                pose.residue(i).is_protein() for i in range(1, pose.total_residue() + 1)
            )
            n_dna = sum(
                pose.residue(i).is_DNA() for i in range(1, pose.total_residue() + 1)
            )
            n_rna = sum(
                pose.residue(i).is_RNA() for i in range(1, pose.total_residue() + 1)
            )
            if n_protein == 0 or (modality == "protein_nucleic" and n_dna + n_rna == 0):
                raise ValueError(
                    f"cleanup retained protein={n_protein}, DNA={n_dna}, RNA={n_rna}"
                )
            pose.dump_pdb(str(clean_path))
            residues = pose.total_residue()
            atoms = pose.total_atoms()
            if pdb_id in TMOL_UNSUPPORTED:
                status = "excluded"
                reason = TMOL_UNSUPPORTED[pdb_id]
        except Exception as error:
            status = "excluded"
            reason = f"{type(error).__name__}: {error}"
            residues = atoms = n_protein = n_dna = n_rna = 0
        rows.append(
            {
                "modality": modality,
                "dataset_id": pdb_id.lower(),
                "structure_path": str(clean_path),
                "ligand_tmol_params": "",
                "ligand_rosetta_params": "",
                "residues": residues,
                "atoms": atoms,
                "protein_residues": n_protein,
                "dna_residues": n_dna,
                "rna_residues": n_rna,
                "source": source,
                "source_sha256": sha256(raw_path) if raw_path.exists() else "",
                "structure_sha256": sha256(clean_path) if clean_path.exists() else "",
                "status": status,
                "reason": reason,
                "historical_version": (
                    "0.1.47" if modality == "protein_nucleic" else "0.1.46"
                ),
            }
        )
    return rows


def fixture_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset_id, filename in PROTEINS:
        path = PDB_DATA / filename
        residues, atoms = pdb_counts(path)
        rows.append(
            {
                "modality": "protein",
                "dataset_id": dataset_id,
                "structure_path": str(path),
                "ligand_tmol_params": "",
                "ligand_rosetta_params": "",
                "residues": residues,
                "atoms": atoms,
                "protein_residues": residues,
                "dna_residues": 0,
                "rna_residues": 0,
                "source": f"tmol:v0.1.55:tmol/tests/data/pdb/{filename}",
                "source_sha256": sha256(path),
                "structure_sha256": sha256(path),
                "status": "ok",
                "reason": "",
                "historical_version": "0.1.46",
            }
        )

    params_dir = ROOT / "data/ligand_params"
    for target in LIGAND_TARGETS:
        pdb_suffix = (
            "_complex_nometals.pdb" if target in {"ace", "ada"} else "_complex.pdb"
        )
        path = PLI_DATA / f"{target}{pdb_suffix}"
        residues, atoms = pdb_counts(path)
        rows.append(
            {
                "modality": "protein_ligand",
                "dataset_id": target,
                "structure_path": str(path),
                "ligand_tmol_params": str(PLI_DATA / f"{target}.xtal-lig.mmff94.tmol"),
                "ligand_rosetta_params": str(params_dir / f"{target}.params"),
                "residues": residues,
                "atoms": atoms,
                "protein_residues": max(0, residues - 1),
                "dna_residues": 0,
                "rna_residues": 0,
                "source": f"tmol:v0.1.55:tmol/tests/data/protein_ligand_test/{path.name}",
                "source_sha256": sha256(path),
                "structure_sha256": sha256(path),
                "status": "ok",
                "reason": "",
                "historical_version": "0.1.46",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pyrosetta-path",
        type=Path,
        default=Path("/mnt/home/kdidi/projects/pyrosetta-2024.39"),
    )
    args = parser.parse_args()
    rows = fixture_rows() + clean_rcsb_entries(args.pyrosetta_path)
    write_rows(ROOT / "metadata/dataset_manifest.csv", rows)
    included = [row for row in rows if row["status"] == "ok"]
    for modality in ("protein", "protein_ligand", "protein_nucleic"):
        subset = [row for row in included if row["modality"] == modality]
        print(
            modality,
            [(row["dataset_id"], row["residues"], row["atoms"]) for row in subset],
        )


if __name__ == "__main__":
    main()
