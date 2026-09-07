"""Merge independently prepared modality manifests."""

from __future__ import annotations

import csv

from common import ROOT, write_rows


def main() -> None:
    rows = []
    for modality in ("protein", "protein_ligand", "protein_nucleic"):
        path = ROOT / f"metadata/manifest-{modality}.csv"
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    rows.sort(
        key=lambda row: (row["modality"], int(row["residues"]), row["dataset_id"])
    )
    write_rows(ROOT / "metadata/dataset_manifest.csv", rows)
    for modality in ("protein", "protein_ligand", "protein_nucleic"):
        subset = [row for row in rows if row["modality"] == modality]
        print(
            modality,
            len(subset),
            f"{min(int(row['residues']) for row in subset)}..{max(int(row['residues']) for row in subset)} residues",
        )


if __name__ == "__main__":
    main()
