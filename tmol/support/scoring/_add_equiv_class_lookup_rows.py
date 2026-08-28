"""Give tautomer and disulfide residue types the backbone tables of their class.

CYD and HIS_POS are separate residue types that share an IO equivalence class
with CYS and HIS, but the rama and bbdep-omega lookups name only the parent. A
residue type with no lookup row gets no rama and, because the kernel skips omega
whenever the table index is negative, no omega either.

Rows are added for every residue type that shares an equivalence class with a
type that has tables, so a type added to the chemical database is covered
without an edit here.

Run from the repository root:

    python -m tmol.support.scoring._add_equiv_class_lookup_rows
"""

import argparse
import os

import attr

from tmol.database.chemical import ChemicalDatabase
from tmol.database.scoring._omega_bbdep import (
    OmegaBBDepDatabase,
    OmegaBBDepMappingParams,
)
from tmol.database.scoring._rama import RamaDatabase, RamaMappingParams
from tmol.support.scoring._add_symmetric_gly_tables import detach, save


def borrowers(chemdb: ChemicalDatabase, covered) -> list:
    """(residue name, class name) for each type that must borrow its class's rows."""
    out = []
    for restype in chemdb.residues:
        if restype.name != restype.base_name:
            continue
        polymer = restype.properties.polymer
        if polymer.polymer_type != "amino_acid" or polymer.sidechain_chirality == "d":
            continue
        equiv = restype.io_equiv_class
        if restype.name in covered or equiv == restype.name or equiv not in covered:
            continue
        out.append((restype.name, equiv))
    return out


def add_rows(lookup, chemdb, make_row):
    """Copy each class's lookup rows onto the types that share its class."""
    covered = {row.res_middle for row in lookup}
    by_source = {}
    for row in lookup:
        by_source.setdefault(row.res_middle, []).append(row)

    added = []
    for name, equiv in borrowers(chemdb, covered):
        for row in by_source[equiv]:
            added.append(make_row(row, name))
            print(f"  {name} / {row.res_upper} -> {row.table_id} (from {equiv})")
    return (*lookup, *added)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_dir", default="tmol/database/default")
    args = parser.parse_args()

    chem_dir = os.path.join(args.db_dir, "chemical")
    scoring_dir = os.path.join(args.db_dir, "scoring")
    chemdb = ChemicalDatabase.from_file(chem_dir)

    rama_path = os.path.join(scoring_dir, "rama.zip")
    print(f"rama: {rama_path}")
    rama = RamaDatabase.from_file(rama_path)
    rama = attr.evolve(
        rama,
        rama_tables=tuple(detach(t, "table") for t in rama.rama_tables),
        rama_lookup=add_rows(
            rama.rama_lookup,
            chemdb,
            lambda row, name: RamaMappingParams(
                table_id=row.table_id,
                res_middle=name,
                res_upper=row.res_upper,
                invert_phi=row.invert_phi,
                invert_psi=row.invert_psi,
            ),
        ),
    )
    save(rama, rama_path)

    omega_path = os.path.join(scoring_dir, "omega_bbdep.zip")
    print(f"omega_bbdep: {omega_path}")
    omega = OmegaBBDepDatabase.from_file(omega_path)
    omega = attr.evolve(
        omega,
        bbdep_omega_tables=tuple(
            detach(t, "mu", "sigma") for t in omega.bbdep_omega_tables
        ),
        bbdep_omega_lookup=add_rows(
            omega.bbdep_omega_lookup,
            chemdb,
            lambda row, name: OmegaBBDepMappingParams(
                table_id=row.table_id,
                res_middle=name,
                res_upper=row.res_upper,
                invert_phi=row.invert_phi,
                invert_psi=row.invert_psi,
            ),
        ),
    )
    save(omega, omega_path)


if __name__ == "__main__":
    main()
