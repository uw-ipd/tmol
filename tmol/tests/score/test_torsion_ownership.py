"""Every movable torsion is constrained exactly once.

Five terms can claim a torsion, and they decide by different means: genbonded
by atom type, cartbonded by atom name, and dunbrack, backbone_torsion and
na_torsion by naming the torsion outright. A term that claims by name can reach
a bond whose atom types say it belongs to someone else, which is how a bond ends
up scored twice, so the check has to ask every term rather than reason about
types alone.

The comparison is per rotatable bond, not per four-atom torsion: genbonded
scores every torsion about a bond while the named-torsion terms each score one
dihedral, and the bond is the only level at which "exactly once" is well posed.
"""

import collections

import pytest
import torch

from tmol.chemical import BondType, ResidueTypeSet
from tmol.score.cartbonded._cartbonded_energy_term import CartBondedEnergyTerm
from tmol.score.dunbrack._dunbrack_energy_term import DunbrackEnergyTerm
from tmol.score.na_torsion import scored_torsion_bonds


def _adjacency(block_type):
    adj = collections.defaultdict(set)
    for i, j in block_type.bond_indices:
        adj[int(i)].add(int(j))
        adj[int(j)].add(int(i))
    return adj


def _torsions(adj):
    return [
        (a, b, c, d)
        for a in adj
        for b in adj[a]
        for c in adj[b]
        if c != a
        for d in adj[c]
        if d != b and a < d
    ]


def _named_bonds(block_type):
    """Central bond of each torsion the chemical layer names, by name."""
    named = {}
    for name, quad in (block_type.torsion_to_uaids or {}).items():
        middle = [u[0] if isinstance(u, (tuple, list)) else u for u in quad[1:3]]
        if len(middle) == 2 and all(isinstance(x, int) and x >= 0 for x in middle):
            named[name] = frozenset(middle)
    return named


def claiming_terms(block_type, param_db, cartbonded, dunbrack):
    """Which terms score each intra-block rotatable bond, keyed by bond."""
    gen_db = param_db.scoring.genbonded
    rosetta_typed = gen_db.rosetta_typed
    element_for_atom_type = {a.name: a.element for a in param_db.chemical.atom_types}

    claims = collections.defaultdict(set)
    names = [a.name for a in block_type.atoms]
    types = [a.atom_type for a in block_type.atoms]
    torsions = _torsions(_adjacency(block_type))
    na_bonds = scored_torsion_bonds(block_type, element_for_atom_type)

    # genbonded: its own two skip rules, then a hit in the generic database
    for i, j, k, l in torsions:
        if types[j] in rosetta_typed and types[k] in rosetta_typed:
            continue
        if frozenset((j, k)) in na_bonds:
            continue
        bond_type = block_type.bond_to_type.get((j, k), int(BondType.SINGLE))
        in_ring = block_type.bond_to_ringness.get((j, k), False)
        if (
            gen_db.find_torsion_params(
                types[i], types[j], types[k], types[l], bond_type, in_ring
            )
            is not None
        ):
            claims[frozenset((j, k))].add("genbonded")

    # cartbonded: a residue-specific or wildcard parameter under the atom names
    res_params = set(cartbonded.get_params_for_res(block_type.base_name))
    wildcard = set(cartbonded.get_params_for_res("wildcard"))
    for i, j, k, l in torsions:
        quad = (names[i], names[j], names[k], names[l])
        unique = tuple(
            cartbonded.get_atom_unique_id_name(block_type.base_name, a) for a in quad
        )
        wild = tuple(cartbonded.get_atom_wildcard_id_name(a) for a in quad)
        if (
            unique in res_params
            or unique[::-1] in res_params
            or wild in wildcard
            or wild[::-1] in wildcard
        ):
            claims[frozenset((j, k))].add("cartbonded")

    named = _named_bonds(block_type)

    # dunbrack scores chi1..chi{n_chi}; anything past that is a proton chi it
    #    only samples, and cartbonded's hydroxyl torsion owns
    dunbrack.setup_block_type(block_type)
    attrs = getattr(block_type, "dunbrack_attrs", None)
    if attrs is not None and int(attrs.rotamer_table_set) >= 0:
        for index in range(1, int(attrs.n_chi) + 1):
            bond = named.get("chi%d" % index)
            if bond is not None:
                claims[bond].add("dunbrack")

    if block_type.rama_reference is not None:
        for name in ("phi", "psi"):
            if name in named:
                claims[named[name]].add("backbone_torsion")

    for bond in na_bonds:
        claims[bond].add("na_torsion")

    return claims


@pytest.fixture
def _claim_terms(default_database):
    cpu = torch.device("cpu")
    return (
        CartBondedEnergyTerm(param_db=default_database, device=cpu),
        DunbrackEnergyTerm(param_db=default_database, device=cpu),
    )


def test_no_bond_is_constrained_twice(default_database, _claim_terms):
    """No rotatable bond is scored by two terms at once."""
    cartbonded, dunbrack = _claim_terms
    conflicts = []
    for block_type in ResidueTypeSet.get_default().residue_types:
        names = [a.name for a in block_type.atoms]
        claims = claiming_terms(block_type, default_database, cartbonded, dunbrack)
        for bond, terms in claims.items():
            if len(terms) > 1:
                atoms = tuple(names[i] for i in sorted(bond))
                conflicts.append((block_type.name, atoms, sorted(terms)))
    assert conflicts == []


def test_amide_impropers_do_not_follow_omega(ubq_pdb, default_database, torch_device):
    """The backbone impropers restrain planarity, so omega must not move them.

    They were once imported as proper torsions about the peptide bond, which
    left the amide centres unrestrained and counted omega three times over.
    """
    import numpy

    from tmol.io import pose_stack_from_pdb
    from tmol.score import ScoreFunction
    from tmol.score._score_types import ScoreType

    pose_stack = pose_stack_from_pdb(ubq_pdb, torch_device)
    sfxn = ScoreFunction(default_database, torch_device)
    for score_type in (ScoreType.cart_impropers, ScoreType.omega):
        sfxn.set_weight(score_type, 1.0)
    module = sfxn.render_whole_pose_scoring_module(pose_stack)
    order = [st.name for st in sfxn.all_score_types()]

    def terms(coords):
        values = module(coords, sum_terms=False, apply_weights=False)
        values = values.detach().cpu().numpy()
        return {name: float(values[i].sum()) for i, name in enumerate(order)}

    pbt = pose_stack.packed_block_types
    block_types = pose_stack.block_type_ind[0].cpu().numpy()

    def atom_index(block, name):
        bt = pbt.active_block_types[block_types[block]]
        offset = int(pose_stack.block_coord_offset[0][block].cpu())
        for i, atom in enumerate(bt.atoms):
            if atom.name == name:
                return offset + i
        return None

    lower, upper = 1, 2
    start = int(pose_stack.block_coord_offset[0][upper].cpu())
    coords = pose_stack.coords.detach().cpu().numpy()[0]
    axis = coords[atom_index(upper, "N")] - coords[atom_index(lower, "C")]
    axis = axis / numpy.linalg.norm(axis)
    origin = coords[atom_index(upper, "N")].copy()

    base = terms(pose_stack.coords)
    turned = coords.copy()
    angle = numpy.radians(45.0)
    cross = numpy.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation = (
        numpy.eye(3)
        + numpy.sin(angle) * cross
        + (1 - numpy.cos(angle)) * (cross @ cross)
    )
    # everything downstream turns together, or the next junction is distorted
    for i in range(start, coords.shape[0]):
        turned[i] = origin + rotation @ (coords[i] - origin)
    moved = terms(
        torch.tensor(
            turned, dtype=pose_stack.coords.dtype, device=torch_device
        ).unsqueeze(0)
    )

    assert moved["omega"] != pytest.approx(base["omega"], abs=1e-2)
    assert moved["cart_impropers"] == pytest.approx(base["cart_impropers"], abs=1e-2)


def test_modified_nucleotide_chi_is_left_to_na_torsion(default_database, torch_device):
    """na_torsion scores the glycosidic torsion of a modified base by name.

    Its two atoms are mixed-typed -- the base nitrogen is ligand-typed and C1'
    is not -- so the atom-type rule alone would hand the bond to genbonded too.
    """
    from tmol.io import atom_array_from_cif
    from tmol.ligand import prepare_ligands
    from tmol.score.genbonded import GenBondedEnergyTerm
    from tmol.tests.data import data_path

    element_for_atom_type = {
        a.name: a.element for a in default_database.chemical.atom_types
    }
    rosetta_typed = default_database.scoring.genbonded.rosetta_typed
    genbonded = GenBondedEnergyTerm(
        param_db=default_database, device=torch.device("cpu")
    )

    exercised = []
    for path in sorted(data_path("ncaa_fixtures").glob("na_*.cif")):
        prepared, _ = prepare_ligands(
            atom_array_from_cif(path), param_db=default_database
        )
        for block_type in ResidueTypeSet.from_database(prepared.chemical).residue_types:
            na_bonds = scored_torsion_bonds(block_type, element_for_atom_type)
            types = [a.atom_type for a in block_type.atoms]
            mixed = {
                bond
                for bond in na_bonds
                if not all(types[i] in rosetta_typed for i in bond)
            }
            if not mixed:
                continue
            torsions = _torsions(_adjacency(block_type))
            kept, _params = genbonded.resolve_torsion_params(block_type, torsions)
            for i, j, k, l in kept:
                assert frozenset((j, k)) not in mixed, (
                    "%s: genbonded kept a torsion about a bond na_torsion scores"
                    % block_type.name
                )
            exercised.append(block_type.base_name)

    assert exercised, "no modified nucleotide exercised the na_torsion exclusion"
