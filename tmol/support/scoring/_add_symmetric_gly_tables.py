"""Append mirror-symmetric glycine backbone tables to the rama and omega binaries.

Glycine is achiral, so its intrinsic backbone preferences must be invariant
under phi,psi -> -phi,-psi. The tables shipped by Rosetta are not: they come
from PDB statistics, which put glycine disproportionately in the region a
left-handed residue would occupy. A structure and its mirror image therefore
score differently unless glycine's tables are symmetrized.

The symmetrized tables are added alongside the originals rather than replacing
them, so the default score function is untouched; ParameterDatabase.
with_symmetric_gly() selects them.

Run from the repository root:

    python -m tmol.support.scoring._add_symmetric_gly_tables
"""

import argparse
import os

import attr
import numpy
import torch

from tmol.database.scoring._omega_bbdep import OmegaBBDepDatabase
from tmol.database.scoring._rama import RamaDatabase

# tables to symmetrize -> name of the symmetric copy
RAMA_TABLES = {"GLY": "GLY_symm", "GLY_prepro": "GLY_prepro_symm"}
OMEGA_TABLES = {"gly": "gly_symm", "prepro": "prepro_gly_symm"}

# a mirrored backbone has omega -> -omega, and omega is near 180, which is its
#    own mirror image; a trans-only potential is therefore the symmetric one
OMEGA_SYMM_MU = 180.0
OMEGA_SYMM_SIGMA = 6.0


def mirror(grid: numpy.ndarray) -> numpy.ndarray:
    """Reflect a periodic phi/psi grid through the origin.

    Bin k is centered at bbstart + k*bbstep with bbstart = -pi, so the bin at
    -phi is -k modulo the number of bins.
    """
    return numpy.roll(numpy.roll(grid[::-1, ::-1], 1, axis=0), 1, axis=1)


def symmetrize_energies(energies: numpy.ndarray) -> numpy.ndarray:
    """Average an energy table with its mirror image, in probability space.

    The tables hold -ln(p) + S, so averaging energies would average logarithms
    and give the geometric rather than the arithmetic mean of the two
    probabilities. Recovering p uses the fact that it sums to one, which fixes
    the offset; the symmetrized table is rebuilt with its own entropy so it
    stays on the same energy scale as the table it replaces.
    """
    p = numpy.exp(-energies.astype(numpy.float64))
    p /= p.sum()
    p = 0.5 * (p + mirror(p))
    p /= p.sum()
    entropy = float((p * numpy.log(p)).sum())
    return (-numpy.log(p) + entropy).astype(numpy.float32)


def add_rama_tables(db: RamaDatabase) -> RamaDatabase:
    """Append a symmetrized copy of each glycine rama table."""
    by_id = {t.table_id: t for t in db.rama_tables}
    added = []
    for source, target in RAMA_TABLES.items():
        if target in by_id:
            raise ValueError(f"{target} is already present")
        table = by_id[source]
        energies = numpy.asarray(table.table, dtype=numpy.float64)
        symm = symmetrize_energies(energies)
        residual = numpy.abs(symm - mirror(symm)).max()
        if residual > 1e-5:
            raise ValueError(f"{target} is not symmetric (residual {residual:g})")
        print(
            f"  {source} -> {target}: max shift {numpy.abs(symm - energies).max():.4f}, "
            f"rms {numpy.sqrt(((symm - energies) ** 2).mean()):.4f} kcal/mol"
        )
        added.append(attr.evolve(table, table_id=target, table=torch.tensor(symm)))
    return attr.evolve(db, rama_tables=(*db.rama_tables, *added))


def add_omega_tables(db: OmegaBBDepDatabase) -> OmegaBBDepDatabase:
    """Append glycine bbdep-omega tables that are flat and centered on trans.

    mu and sigma are stored in degrees. Glycine is achiral, so it has no
    phi/psi-dependent omega preference to begin with: the variation in the
    fitted tables tracks the same handedness bias in the source statistics.
    A uniform trans potential is both the symmetric and the achiral answer.

    The prepro copy is glycine-specific; the table it derives from is shared
    with every other residue and is left alone.
    """
    by_id = {t.table_id: t for t in db.bbdep_omega_tables}
    added = []
    for source, target in OMEGA_TABLES.items():
        if target in by_id:
            raise ValueError(f"{target} is already present")
        shape = tuple(numpy.asarray(by_id[source].mu).shape)
        added.append(
            attr.evolve(
                by_id[source],
                table_id=target,
                mu=torch.full(shape, OMEGA_SYMM_MU, dtype=torch.float32),
                sigma=torch.full(shape, OMEGA_SYMM_SIGMA, dtype=torch.float32),
            )
        )
        print(f"  {source} -> {target}: mu={OMEGA_SYMM_MU}, sigma={OMEGA_SYMM_SIGMA}")
    return attr.evolve(db, bbdep_omega_tables=(*db.bbdep_omega_tables, *added))


def detach(table, *fields):
    """Copy a table's tensors out of the memory-mapped file.

    from_file maps the binary read-only; saving over a path whose pages are
    still mapped faults, so every tensor must be materialized first.
    """
    return attr.evolve(table, **{f: getattr(table, f).clone() for f in fields})


def save(db, path: str) -> None:
    """Write a database beside its source, then move it into place."""
    tmp = path + ".tmp"
    torch.save(attr.evolve(db, uniq_id=db.content_id()), tmp)
    os.replace(tmp, path)


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
    save(add_rama_tables(rama), rama_path)

    print(f"omega_bbdep: {omega_path}")
    omega = OmegaBBDepDatabase.from_file(omega_path)
    omega = attr.evolve(
        omega,
        bbdep_omega_tables=tuple(
            detach(t, "mu", "sigma") for t in omega.bbdep_omega_tables
        ),
    )
    save(add_omega_tables(omega), omega_path)


if __name__ == "__main__":
    main()
