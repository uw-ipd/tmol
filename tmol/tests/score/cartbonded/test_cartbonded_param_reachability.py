"""Every cartbonded param must be reachable by the lookup the kernel performs.

A param whose key no subgraph can produce contributes nothing and fails silently,
which is how the terminal OXT / H1-H3 geometry went unscored. These tests pin the
resolution rules so a stranded param is a test failure rather than a zero.
"""

from tmol.database import ParameterDatabase
from tmol.score.cartbonded.cartbonded_energy_term import CROSS_RES_PREFIX

GROUPS = [
    ("length_parameters", 2),
    ("angle_parameters", 3),
    ("torsion_parameters", 4),
    ("improper_parameters", 4),
    ("hxltorsion_parameters", 4),
]

# rows naming atoms that no fullatom residue type has (centroid, proline virtual
# nitrogen, the other His tautomer's proton, CYD's absent HG)
INERT_ATOMS = {"CEN", "NV", "HG", "HD1", "HE2", "Vrt", "CN"}


def _rows(cartbonded):
    for res, params in cartbonded.residue_params.items():
        for group, natoms in GROUPS:
            for row in getattr(params, group):
                atoms = [
                    getattr(row, f"atm{i}")
                    for i in range(1, natoms + 1)
                    if getattr(row, f"atm{i}", None) is not None
                ]
                yield res, group, atoms


def test_cross_marked_atoms_form_a_trailing_run():
    """The kernel joins one residue's path to the partner's, so the atoms across a
    connection are contiguous at one end."""
    cartbonded = ParameterDatabase.get_default().scoring.cartbonded
    for res, group, atoms in _rows(cartbonded):
        marked = [a.startswith(CROSS_RES_PREFIX) for a in atoms]
        if not any(marked):
            continue
        first = marked.index(True)
        assert all(marked[first:]), f"{res} {group} {atoms}: cross atoms not trailing"


def test_no_cross_marked_atoms_in_improper_params():
    """Impropers are intra-residue; a cross marker there would never match."""
    cartbonded = ParameterDatabase.get_default().scoring.cartbonded
    for res, group, atoms in _rows(cartbonded):
        if group != "improper_parameters":
            continue
        assert not any(
            a.startswith(CROSS_RES_PREFIX) for a in atoms
        ), f"{res} {group} {atoms}"


def test_wildcard_intra_rows_are_realizable(default_database):
    """A wildcard row with no cross marker is intra-residue geometry, so some block
    type must actually contain that bonded path -- otherwise it is dead weight."""
    cartbonded = default_database.scoring.cartbonded
    chemdb = default_database.chemical

    bonds = {}
    for restype in chemdb.residues:
        s = set()
        for b in restype.bonds:
            s.add((b[0], b[1]))
            s.add((b[1], b[0]))
        bonds[restype.name] = s

    def is_path(atoms, bondset):
        return all((atoms[i - 1], atoms[i]) in bondset for i in range(1, len(atoms)))

    unrealizable = []
    for res, group, atoms in _rows(cartbonded):
        if res != "wildcard" or group not in ("length_parameters", "angle_parameters"):
            continue
        if any(a.startswith(CROSS_RES_PREFIX) for a in atoms):
            continue
        if set(atoms) & INERT_ATOMS:
            continue
        if not any(
            is_path(atoms, bs) or is_path(atoms[::-1], bs) for bs in bonds.values()
        ):
            unrealizable.append((group, atoms))
    assert (
        not unrealizable
    ), f"wildcard intra rows no block type can match: {unrealizable}"


def test_cross_rows_are_not_realizable_intra(default_database):
    """A cross-marked row must need a bond that only exists across a connection;
    otherwise it would also match intra-residue geometry and double count."""
    cartbonded = default_database.scoring.cartbonded
    chemdb = default_database.chemical

    bonds = {}
    for restype in chemdb.residues:
        s = set()
        for b in restype.bonds:
            s.add((b[0], b[1]))
            s.add((b[1], b[0]))
        bonds[restype.name] = s

    def is_path(atoms, bondset):
        return all((atoms[i - 1], atoms[i]) in bondset for i in range(1, len(atoms)))

    for res, group, atoms in _rows(cartbonded):
        if not any(a.startswith(CROSS_RES_PREFIX) for a in atoms):
            continue
        bare = [a.lstrip(CROSS_RES_PREFIX) for a in atoms]
        for name, bs in bonds.items():
            assert not (
                is_path(bare, bs) or is_path(bare[::-1], bs)
            ), f"{res} {group} {atoms} is also an intra path in {name}"
