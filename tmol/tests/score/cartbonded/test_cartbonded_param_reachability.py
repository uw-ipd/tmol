"""Every cartbonded param must be reachable by the lookup the kernel performs.

A param whose key no subgraph can produce contributes nothing and fails silently,
which is how the terminal OXT / H1-H3 geometry went unscored. These tests pin the
resolution rules so a stranded param is a test failure rather than a zero.
"""

import torch

from tmol.database import ParameterDatabase
from tmol.io import atom_array_from_cif, pose_stack_from_biotite
from tmol.score.cartbonded import CROSS_RES_PREFIX, CartBondedEnergyTerm
from tmol.tests.data import data_path

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


# groups the kernel resolves by joining one residue's path to the partner's
PATH_JOIN_GROUPS = {"length_parameters", "angle_parameters", "torsion_parameters"}


def test_cross_marked_atoms_form_a_trailing_run():
    """A path-join row's atoms across a connection are contiguous at one end.

    Impropers are exempt: their four atoms are not a bonded path, so the atom
    across the connection sits wherever the geometry puts it.
    """
    cartbonded = ParameterDatabase.get_default().scoring.cartbonded
    for res, group, atoms in _rows(cartbonded):
        if group not in PATH_JOIN_GROUPS:
            continue
        marked = [a.startswith(CROSS_RES_PREFIX) for a in atoms]
        if not any(marked):
            continue
        first = marked.index(True)
        assert all(marked[first:]), f"{res} {group} {atoms}: cross atoms not trailing"


def test_no_cross_marked_atoms_in_torsion_params():
    """A connection-spanning torsion has nothing to match it.

    The kernel joins one residue's path to the partner's only far enough to
    reach a length or an angle, since no torsion row spans a connection. A row
    added here would score zero in silence rather than fail.
    """
    cartbonded = ParameterDatabase.get_default().scoring.cartbonded
    for res, group, atoms in _rows(cartbonded):
        if group != "torsion_parameters":
            continue
        assert not any(
            a.startswith(CROSS_RES_PREFIX) for a in atoms
        ), f"{res} {group} {atoms}: no path join reaches four atoms"


def test_cross_marked_impropers_name_their_centre_locally():
    """A cross-marked improper is centred on one residue's connection atom.

    The enumeration takes that atom and two local neighbours and reaches across
    for the third, so the centre -- atm3, by the improper convention -- must be
    the unmarked side and exactly one atom may carry the marker. Marking the
    centre instead leaves the row unreachable, which scores zero in silence.
    """
    cartbonded = ParameterDatabase.get_default().scoring.cartbonded
    for res, params in cartbonded.residue_params.items():
        for row in params.improper_parameters:
            atoms = [row.atm1, row.atm2, row.atm3, row.atm4]
            marked = [a.startswith(CROSS_RES_PREFIX) for a in atoms]
            if not any(marked):
                continue
            assert sum(marked) == 1, f"{res} improper {atoms}: expected one cross atom"
            assert not marked[2], f"{res} improper {atoms}: centre must be local"


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


def test_real_pose_bonds_and_angles_are_parameterized():
    """Every non-virtual intra-block length and angle of a built pose resolves to a
    parameter; an unparameterized one leaves its atoms free to fly apart in min.
    The 9CF0 sample starts a DNA and an RNA chain with a 5'-phosphate."""
    device = torch.device("cpu")
    structure = atom_array_from_cif(
        data_path("sweep_regressions", "five_prime_phosphate_9cf0.cif.zst")
    )
    pose_stack, context = pose_stack_from_biotite(
        structure, device, return_context=True
    )
    term = CartBondedEnergyTerm(param_db=context.parameter_database, device=device)
    pbt = pose_stack.packed_block_types
    ann = term.setup_packed_block_types(pbt)
    subgraphs = ann.cartbonded_subgraphs.cpu().numpy()
    offsets = ann.cartbonded_subgraph_offsets.cpu().numpy()
    counts = ann.cartbonded_subgraph_type_counts.cpu().numpy()
    type_offsets = ann.cartbonded_subgraph_type_offsets.cpu().numpy()
    param_indices = ann.cartbonded_subgraph_param_indices.cpu().numpy()

    used = {int(i) for i in pose_stack.block_type_ind.flatten() if i >= 0}
    names = {pbt.active_block_types[i].name for i in used}
    assert {"DA:na5primephos", "RU:na5primephos"} <= names
    missing = []
    for i in sorted(used):
        block_type = pbt.active_block_types[i]
        virtual = set(block_type.properties.virtual)
        start = offsets[i] + type_offsets[i][0]
        for j in range(start, start + counts[i][0] + counts[i][1]):
            atoms = [block_type.atoms[a].name for a in subgraphs[j] if a >= 0]
            if param_indices[j] < 0 and not virtual & set(atoms):
                missing.append(f"{block_type.name} {'-'.join(atoms)}")
    assert not missing, f"unparameterized lengths/angles: {missing}"
