"""Generic lookup references must not transfer canonical torsion ownership."""

import attr
import pytest
import torch

from tmol.chemical import ResidueTypeSet
from tmol.score.genbonded import GenBondedEnergyTerm, MAX_HIER_DEPTH

GLOBAL_HIERARCHY_HOPS = {
    "CH1": "CS1",
    "CH2": "CS2",
    "CH3": "CS3",
    "CObb": "CDp",
    "CNH2": "CDp",
    "COO": "CDp",
    "Nbb": "Nad",
    "NH2O": "Nad",
    "Npro": "Nad3",
    "Narg": "Ngu2",
    "NtrR": "Ngu1",
    "Oet2": "Oet",
    "Oet3": "Oet",
    "Hapo": "HC",
}

GLOBAL_HIERARCHY_IMPROPER_REACHABILITY = {
    "CH1": (("CDp", "Oal", "CH1", "HC"), ("CDp", "Oal", "CS1", "HC")),
    "CH2": (("CDp", "Oal", "CH2", "HC"), ("CDp", "Oal", "CS2", "HC")),
    "CH3": (("CDp", "Oal", "CH3", "HC"), ("CDp", "Oal", "CS3", "HC")),
    "CObb": (("CObb", "Oal", "CS", "HC"), ("CDp", "Oal", "CS", "HC")),
    "CNH2": (("CNH2", "Oal", "CS", "HC"), ("CDp", "Oal", "CS", "HC")),
    "COO": (("COO", "Oal", "CS", "HC"), ("CDp", "Oal", "CS", "HC")),
    "Nbb": (("Nbb", "CDp", "CS", "HN"), ("Nad", "CDp", "CS", "HN")),
    "NH2O": (("NH2O", "CDp", "CS", "HN"), ("Nad", "CDp", "CS", "HN")),
    "Npro": (("CDp", "Oad", "Npro", "CS"), ("CDp", "Oad", "Nad3", "CS")),
    "Narg": (
        ("CDp", "Narg", "Ngu2", "Ngu2"),
        ("CDp", "Ngu2", "Ngu2", "Ngu2"),
    ),
    "NtrR": (
        ("CDp", "NtrR", "Ngu2", "Ngu2"),
        ("CDp", "Ngu1", "Ngu2", "Ngu2"),
    ),
    "Oet2": (("CDp", "Oal", "CS", "Oet2"), ("CDp", "Oal", "CS", "Oet")),
    "Oet3": (("CDp", "Oal", "CS", "Oet3"), ("CDp", "Oal", "CS", "Oet")),
    "Hapo": (("CDp", "Oal", "CS", "Hapo"), ("CDp", "Oal", "CS", "HC")),
}


@pytest.mark.parametrize("source,target", GLOBAL_HIERARCHY_HOPS.items())
def test_global_hierarchy_hop_is_exact_and_kernel_reachable(
    default_database, source, target
):
    database = default_database.scoring.genbonded
    expected = [source, *database.hierarchy_for(target)]
    assert database.hierarchy_for(source) == expected
    assert len(expected) <= MAX_HIER_DEPTH

    term = GenBondedEnergyTerm(default_database, torch.device("cpu"))
    indices = term.atom_hierarchy_indices(source)
    decoded = [term._all_type_names[index] if index >= 0 else None for index in indices]
    assert decoded == expected + [None] * (MAX_HIER_DEPTH - len(expected))

    query, selected = GLOBAL_HIERARCHY_IMPROPER_REACHABILITY[source]
    assert database.find_improper_params(*query).atoms == selected


def _lysine(default_database, references):
    raw = next(r for r in default_database.chemical.residues if r.name == "LYS")
    return ResidueTypeSet._refine(
        attr.evolve(
            raw,
            atoms=tuple(
                attr.evolve(a, genbonded_type=references.get(a.name)) for a in raw.atoms
            ),
        )
    )


def test_lookup_reference_preserves_canonical_chi_ownership(default_database):
    term = GenBondedEnergyTerm(default_database, torch.device("cpu"))
    bt = _lysine(default_database, {"CD": "CS2", "CE": "CS2", "NZ": "Nam"})
    quad = tuple(bt.atom_to_idx[name] for name in ("CG", "CD", "CE", "NZ"))
    # This axis would acquire a nonzero generic torsion if reference types
    # determined ownership, adding to the existing LYS chi4 Dunbrack term.
    entry = term.gen_database.find_torsion_params(
        *(term.get_atom_chem_type(bt, i) for i in quad), 1, False
    )
    assert entry is not None and any((entry.k1, entry.k2, entry.k3, entry.k4))
    assert term.resolve_torsion_params(bt, [quad])[0] == []


@pytest.mark.parametrize("physical_center", ["Nbb", "Nad"])
def test_intra_improper_uses_physical_center_ownership(
    default_database, physical_center
):
    from types import SimpleNamespace
    from tmol.database.chemical import Atom

    bt = SimpleNamespace(
        atoms=(
            Atom("N", physical_center, "Nad"),
            Atom("C", "CNH2", "CDp"),
            Atom("CA", "CH2", "CS2"),
            Atom("H", "Hpol", "HN"),
        )
    )
    term = GenBondedEnergyTerm(default_database, torch.device("cpu"))
    kept, params = term.resolve_improper_params(bt, [(0, 1, 2, 3)])
    assert kept == ([(0, 1, 2, 3)] if physical_center == "Nad" else [])
    if kept:
        assert params.tolist() == [[80.0, 0.0]]


@pytest.mark.parametrize("reference", ["missing_type", "HN", "C*", ""])
def test_invalid_reference_is_rejected(default_database, reference):
    term = GenBondedEnergyTerm(default_database, torch.device("cpu"))
    bt = _lysine(default_database, {"CE": reference})
    for _ in range(2):
        with pytest.raises(ValueError, match="LYS atom CE: invalid genbonded_type"):
            term.setup_block_type(bt)
        assert not hasattr(bt, "genbonded_intra_subgraphs")


def test_a_connection_through_a_ring_reaches_the_ring_bond_parameters():
    """A cut through a ring leaves a ring bond, and it must bin as one.

    Generic bonded lookup bins a bond by order *and* ring membership, and the
    two sets are disjoint, so a connection whose ring membership is lost can
    never match a rule written for a ring bond -- it falls through to the
    any-bond rule, which for a CS-CS single bond is a hundredfold stiffer.
    """
    from tmol.database.scoring._genbonded import _BOND_CHAR_COVERED_BINS, _bond_btidx

    ring_bins = _BOND_CHAR_COVERED_BINS["@"]
    for bond_type in (1, 2, 3, 4):
        assert _bond_btidx(bond_type, True) in ring_bins
        assert _bond_btidx(bond_type, False) not in ring_bins


def test_connection_ring_membership_survives_the_residue_type(default_database):
    """RefinedResidueType carries the flag the bond bins are built from."""
    import attr
    import cattr

    from tmol.chemical import RefinedResidueType

    base = next(r for r in default_database.chemical.residues if r.name == "ALA")
    assert base.connections, "ALA is expected to carry polymer connections"
    ringed = attr.evolve(
        base,
        connections=tuple(
            attr.evolve(c, in_ring=(i == 0)) for i, c in enumerate(base.connections)
        ),
    )
    refined = cattr.structure(cattr.unstructure(ringed), RefinedResidueType)
    assert [bool(x) for x in refined.connection_bond_in_ring] == [
        i == 0 for i in range(len(base.connections))
    ]
    # And the flag is what the serialized form round-trips, so a prepared
    # ligand's cut through a ring survives being written and read back.
    assert cattr.unstructure(ringed)["connections"][0]["in_ring"] is True
