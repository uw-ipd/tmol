"""Append the AIB rama table and the AIB lookup rows to the rama and omega binaries.

AIB is achiral, so its table must be invariant under phi,psi -> -phi,-psi. It is
built from ALA's: at each point the higher of the two mirror-image energies, so a
region is favorable only if it is favorable in both hands. The table is then
offset to ALA's partition function, as the symmetric glycine tables are.

The lookup rows come from rama.yaml and omega_bbdep.yaml; only rows the binary
lacks are appended, so rows other scripts appended to the binaries are kept.

Run from the repository root:

    python -m tmol.support.scoring._add_aib_tables
"""

import argparse
import os

import attr
import cattr
import numpy
import torch
import yaml

from tmol.database.scoring._omega_bbdep import OmegaBBDepDatabase
from tmol.database.scoring._rama import RamaDatabase
from tmol.support.scoring._add_symmetric_gly_tables import (
    check_registration,
    detach,
    mirror,
    save,
)

# source table -> name of its max-symmetrized copy
RAMA_TABLES = {"ALA": "AIB"}


def max_symmetrize(energies: numpy.ndarray) -> numpy.ndarray:
    """The higher of each point's energy and its mirror image's, offset so
    sum(exp(-E)) matches the source table, which keeps it on the source's scale."""
    symm = numpy.maximum(energies, mirror(energies))
    offset = numpy.log(numpy.exp(-symm).sum()) - numpy.log(numpy.exp(-energies).sum())
    return (symm + offset).astype(numpy.float32)


def missing_rows(yaml_path: str, key: str, row_type, present) -> tuple:
    """Lookup rows in the yaml source that the binary does not have."""
    with open(yaml_path, "r") as infile:
        rows = cattr.structure(yaml.safe_load(infile)[key], row_type)
    return tuple(row for row in rows if row not in present)


def add_rama(db: RamaDatabase, yaml_path: str) -> RamaDatabase:
    by_id = {t.table_id: t for t in db.rama_tables}
    added = []
    for source, target in RAMA_TABLES.items():
        if target in by_id:
            raise ValueError(f"{target} is already present")
        table = by_id[source]
        check_registration(table)
        symm = max_symmetrize(numpy.asarray(table.table, dtype=numpy.float64))
        print(f"  {source} -> {target}: {symm.shape}, min {symm.min():.4f}")
        added.append(attr.evolve(table, table_id=target, table=torch.tensor(symm)))
    rows = missing_rows(
        yaml_path,
        "rama_lookup",
        attr.fields(RamaDatabase).rama_lookup.type,
        db.rama_lookup,
    )
    print(f"  lookup rows added: {rows}")
    return attr.evolve(
        db,
        rama_lookup=(*db.rama_lookup, *rows),
        rama_tables=(*db.rama_tables, *added),
    )


def add_omega(db: OmegaBBDepDatabase, yaml_path: str) -> OmegaBBDepDatabase:
    rows = missing_rows(
        yaml_path,
        "omega_bbdep_lookup",
        attr.fields(OmegaBBDepDatabase).bbdep_omega_lookup.type,
        db.bbdep_omega_lookup,
    )
    tables = {t.table_id for t in db.bbdep_omega_tables}
    unknown = [row.table_id for row in rows if row.table_id not in tables]
    if unknown:
        raise ValueError(f"lookup rows name tables the binary lacks: {unknown}")
    print(f"  lookup rows added: {rows}")
    return attr.evolve(db, bbdep_omega_lookup=(*db.bbdep_omega_lookup, *rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_dir", default="tmol/database/default/scoring")
    args = parser.parse_args()

    rama_path = os.path.join(args.db_dir, "rama.zip")
    omega_path = os.path.join(args.db_dir, "omega_bbdep.zip")

    print(f"rama: {rama_path}")
    rama = RamaDatabase.from_file(rama_path)
    rama = attr.evolve(
        rama, rama_tables=tuple(detach(t, "table") for t in rama.rama_tables)
    )
    save(add_rama(rama, os.path.join(args.db_dir, "rama.yaml")), rama_path)

    print(f"omega_bbdep: {omega_path}")
    omega = OmegaBBDepDatabase.from_file(omega_path)
    omega = attr.evolve(
        omega,
        bbdep_omega_tables=tuple(
            detach(t, "mu", "sigma") for t in omega.bbdep_omega_tables
        ),
    )
    save(add_omega(omega, os.path.join(args.db_dir, "omega_bbdep.yaml")), omega_path)


if __name__ == "__main__":
    main()
