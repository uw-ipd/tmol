#!/usr/bin/env python3
"""Build tmol-jax from shared canonical data and the repository version.

Run in a TMol developer environment (tmol, numpy and build installed).
PyTorch is used only to convert the repository's legacy serialized tables.
The resulting wheel and standalone sdist have no Torch dependency.
"""

from pathlib import Path
import argparse
import shutil
import subprocess
import sys
import tempfile
import tomllib

import numpy as np


def export_parameters(source, destination):
    from tmol.database.scoring import (
        RamaDatabase,
        OmegaBBDepDatabase,
        DunbrackRotamerLibrary,
    )

    shutil.copytree(
        source.parent / "score_functions",
        destination / "score_functions",
        dirs_exist_ok=True,
    )
    for path in source.rglob("*.yaml"):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    for name, cls, field, values in (
        ("rama", RamaDatabase, "rama_tables", ("table",)),
        ("omega_bbdep", OmegaBBDepDatabase, "bbdep_omega_tables", ("mu", "sigma")),
    ):
        tables = getattr(cls.from_file(str(source / "scoring" / f"{name}.zip")), field)
        arrays = {"__n_tables": np.array(len(tables), dtype=np.int32)}
        for i, table in enumerate(tables):
            for field in ("table_id", "bbstep", "bbstart", *values):
                arrays[f"t{i}__{field}"] = np.asarray(getattr(table, field))
        np.savez_compressed(destination / "scoring" / f"{name}.npz", **arrays)
    library = DunbrackRotamerLibrary.from_file(str(source / "scoring/dunbrack.bin"))
    arrays = {
        "__n_rot": np.array(len(library.rotameric_libraries), dtype=np.int32),
        "__n_semi": np.array(len(library.semi_rotameric_libraries), dtype=np.int32),
    }
    for category, tables in (
        ("rot", library.rotameric_libraries),
        ("semi", library.semi_rotameric_libraries),
    ):
        for i, table in enumerate(tables):
            prefix = f"{category}{i}__"
            arrays[prefix + "table_name"] = np.array(table.table_name)
            for field in (
                "rotamers",
                "rotamer_probabilities",
                "rotamer_means",
                "rotamer_stdvs",
                "prob_sorted_rot_inds",
                "backbone_dihedral_start",
                "backbone_dihedral_step",
                "rotamer_alias",
            ):
                arrays[prefix + "data__" + field] = np.asarray(
                    getattr(table.rotameric_data, field)
                )
            arrays[prefix + "data__backbone_is_mirrored"] = np.asarray(
                table.rotameric_data.backbone_is_mirrored
            )
            if category == "semi":
                for field in (
                    "non_rot_chi_start",
                    "non_rot_chi_step",
                    "non_rot_chi_period",
                    "rotameric_chi_rotamers",
                    "nonrotameric_chi_probabilities",
                    "rotamer_boundaries",
                ):
                    arrays[prefix + field] = np.asarray(getattr(table, field))
    np.savez_compressed(destination / "scoring/dunbrack.npz", **arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("dist"))
    parser.add_argument(
        "--export-data",
        type=Path,
        help="Only prepare canonical NumPy/YAML tables for source development",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "tmol/database/default"
    if args.export_data is not None:
        export_parameters(source, args.export_data.resolve())
        return
    with tempfile.TemporaryDirectory(prefix="tmol-jax-build-") as temporary:
        stage = Path(temporary)
        shutil.copytree(
            root / "tmol_jax",
            stage / "tmol_jax",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "default"),
        )
        export_parameters(source, stage / "tmol_jax/database/default")
        shutil.copyfile(root / "packages/jax/pyproject.toml", stage / "pyproject.toml")
        shutil.copyfile(root / "LICENSE", stage / "LICENSE")
        shutil.copyfile(root / "docs/jax.md", stage / "README.md")
        metadata = tomllib.loads((root / "pyproject.toml").read_text())
        (stage / "VERSION").write_text(metadata["project"]["version"] + "\n")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--outdir",
                str(args.outdir.resolve()),
                str(stage),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
