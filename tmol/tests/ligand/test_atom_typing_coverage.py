"""Direct coverage of the Rosetta atom-type classifier across diverse chemistry.

:func:`tmol.ligand.atom_typing.assign_tmol_atom_types` is the self-contained
typing entry point used by the ligand pipeline. It prepares the molecule,
builds the Rosetta typing state, and routes every atom through the element
classifiers plus the polar-carbon, amide, ring-nitrogen and conjugated-bond
correction passes. Feeding it a wide spread of functional groups exercises the
per-element classification branches (amines/amides/aromatic N, carboxylate /
ester / phosphate O, thiol / sulfone S, halogens, strained rings, ring
amidines, conjugated systems) that the happy-path e2e tests never reach, while
asserting the classifier stays robust (one finite type per atom) for all of
them.
"""

from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

# name -> SMILES, grouped by the classification branch each is meant to hit.
_MOLECULES: dict[str, str] = {
    # Nitrogen: amines, amides, aromatic/heteroaromatic, charged, ring amidines
    "primary_amine": "CCN",
    "secondary_amine": "CCNC",
    "tertiary_amine": "CCN(C)C",
    "amide": "CC(=O)N",
    "n_methyl_amide": "CC(=O)NC",
    "nitrile": "CC#N",
    "pyridine": "c1ccncc1",
    "pyrrole": "c1cc[nH]c1",
    "imidazole": "c1c[nH]cn1",
    "nitro": "O=[N+]([O-])C",
    "guanidine": "NC(=N)N",
    "amidine": "CC(=N)N",
    "aminopyridine": "Nc1ccccn1",
    "adenine": "Nc1ncnc2[nH]cnc12",
    "uracil": "O=c1cc[nH]c(=O)[nH]1",
    "cytosine": "Nc1cc[nH]c(=O)n1",
    "sulfonamide": "CS(=O)(=O)N",
    "indole": "c1ccc2[nH]ccc2c1",
    "aziridine": "C1CN1",
    "azide_like": "CN=[N+]=[N-]",
    # Oxygen: hydroxyl, ether, carbonyl, acids/esters, on-P/on-N, aromatic
    "hydroxyl": "CCO",
    "ether": "COC",
    "aldehyde": "CC=O",
    "ketone": "CC(=O)C",
    "carboxylic_acid": "CC(=O)O",
    "carboxylate": "CC(=O)[O-]",
    "ester": "CC(=O)OC",
    "phosphate": "OP(=O)(O)O",
    "phosphonate": "CP(=O)(O)O",
    "sulfate": "OS(=O)(=O)O",
    "furan": "c1ccoc1",
    "water": "O",
    "epoxide": "C1CO1",
    "n_oxide": "C[N+](C)(C)[O-]",
    # Sulfur
    "thiol": "CS",
    "thioether": "CSC",
    "sulfoxide": "CS(=O)C",
    "sulfone": "CS(=O)(=O)C",
    "thiophene": "c1ccsc1",
    "disulfide": "CSSC",
    # Carbon / halogen / strained / conjugated
    "trifluoromethyl": "CC(F)(F)F",
    "dichloromethane": "ClCCl",
    "dibromomethane": "BrCBr",
    "diiodomethane": "ICI",
    "cyclopropane": "C1CC1",
    "alkyne": "CC#C",
    "alkene": "C=C",
    "benzene": "c1ccccc1",
    "biphenyl": "c1ccc(-c2ccccc2)cc1",
    "styrene": "C=Cc1ccccc1",
    "acrylamide": "C=CC(=O)N",
    "phenol": "Oc1ccccc1",
    "aniline": "Nc1ccccc1",
    "benzaldehyde": "O=Cc1ccccc1",
    "pyrimidine": "c1cncnc1",
    "purine": "c1ncc2[nH]cnc2n1",
    "thione": "CC(=S)C",
    "nitroso": "CN=O",
    "phosphine": "P",
    "hydroxylamine": "NO",
    "oxime": "CC=NO",
    "thiocarbamate": "CSC(=O)N",
    # Aromatic-ring / non-aromatic-ring junction: exercises the conjugated
    # bond-order ring(N=C)-N-H exception probe across an aryl-to-alkene-ring bond.
    "phenylcyclohexene": "C1=C(CCCC1)c1ccccc1",
    # Boron isn't in the element->classifier table -> unknown-element fallback.
    "boronic_acid": "OB(O)O",
}


def _embed_3d(smiles: str) -> Chem.Mol:
    """Parse a SMILES and produce an explicit-H 3D conformer for typing."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"unparseable SMILES {smiles!r}"
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    if AllChem.EmbedMolecule(mol, params) != 0:
        # Fall back to random-coordinate embedding for awkward cases.
        if AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=1) != 0:
            AllChem.Compute2DCoords(mol)
    return mol


@pytest.mark.parametrize("name", sorted(_MOLECULES))
def test_assign_tmol_atom_types_is_robust(name: str) -> None:
    """Every atom of a diverse molecule receives exactly one non-empty type."""
    from tmol.ligand.atom_typing import assign_tmol_atom_types

    mol = _embed_3d(_MOLECULES[name])
    assignments = assign_tmol_atom_types(mol)

    assert len(assignments) == mol.GetNumAtoms()
    for a in assignments:
        assert a.atom_type and isinstance(a.atom_type, str)
        assert a.element
    # Indices form a complete, unique cover of the molecule's atoms.
    assert {a.index for a in assignments} == set(range(mol.GetNumAtoms()))


def test_assign_tmol_atom_types_spot_checks() -> None:
    """A few canonical functional groups map to their expected element types."""
    from tmol.ligand.atom_typing import assign_tmol_atom_types

    def types_for(smiles: str, element: str) -> set[str]:
        mol = _embed_3d(smiles)
        return {
            a.atom_type for a in assign_tmol_atom_types(mol) if a.element == element
        }

    # Carboxylate oxygens are charged carboxyl oxygens (Oat).
    assert "Oat" in types_for("CC(=O)[O-]", "O")
    # A hydroxyl oxygen is typed as the hydroxyl oxygen (Ohx).
    assert "Ohx" in types_for("CCO", "O")
    # Aromatic ring carbons get an aromatic carbon type.
    benzene_c = types_for("c1ccccc1", "C")
    assert any(t.startswith("CR") or t.startswith("aro") for t in benzene_c)


def test_assign_missing_hybridization_geometry_and_aromatic() -> None:
    """Hybridization back-fill covers the geometry and aromatic-N branches."""
    from tmol.ligand.atom_typing import (
        HYB_SP2,
        HYB_SP3,
        _assign_missing_hybridization,
    )

    # Neopentane: central carbon has degree 4 -> improper dihedral (sp3) path,
    # terminal carbons/hydrogens exercise the degree<3 fast path.
    mol = _embed_3d("CC(C)(C)C")
    hyb = {a.GetIdx(): 0 for a in mol.GetAtoms()}
    _assign_missing_hybridization(mol, hyb, set())
    central = next(
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6
        and sum(1 for n in a.GetNeighbors() if n.GetAtomicNum() == 6) == 4
    )
    assert hyb[central] in (HYB_SP2, HYB_SP3)
    assert all(v in (HYB_SP2, HYB_SP3) for v in hyb.values())

    # Aniline: the amine N attached to an aromatic ring -> aromatic-N branch.
    aniline = _embed_3d("Nc1ccccc1")
    n_idx = next(a.GetIdx() for a in aniline.GetAtoms() if a.GetAtomicNum() == 7)
    aro_ring = {a.GetIdx() for a in aniline.GetAtoms() if a.GetIsAromatic()}
    hyb_n = {n_idx: 0}
    _assign_missing_hybridization(aniline, hyb_n, aro_ring)
    assert hyb_n[n_idx] in (HYB_SP2, HYB_SP3)


def test_bond_is_planar() -> None:
    """Planarity check covers the conformer, terminal-neighbor and angle paths."""
    from tmol.ligand.atom_typing import _bond_is_planar

    # No conformer -> defaults to planar (True).
    flat = Chem.MolFromSmiles("C=CC=C")
    assert _bond_is_planar(flat, 1, 2) is True

    mol = _embed_3d("C=CC=C")
    # Central bond: both atoms carry heavy neighbors -> dihedral evaluation.
    assert isinstance(_bond_is_planar(mol, 1, 2), bool)
    # Terminal bond: atom 0 has no other heavy neighbor -> early True.
    assert _bond_is_planar(mol, 0, 1) is True


class TestClassifierHelpers:
    """Direct tests for the per-element classifier helpers' state-free paths.

    In the full pipeline these helpers always receive a precomputed
    ``RosettaTypingState``, so their ``state is None`` fallback branches (which
    recompute neighbor counts / hybridization on demand) are only reachable by
    calling them directly.
    """

    def _mol(self, smiles: str) -> Chem.Mol:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        return Chem.AddHs(mol)

    def test_neighbor_counts_state_none(self) -> None:
        from tmol.ligand.atom_typing import _neighbor_counts

        # Atoms collectively bonded to C, H, O, N and S exercise every branch.
        mol = self._mol("CC(=O)NCS")
        for atom in mol.GetAtoms():
            nC, nH, nO, nN, nS, ntot = _neighbor_counts(atom)
            assert ntot == atom.GetDegree()
            assert nC + nH + nO + nN + nS <= ntot

    def test_get_hyb_state_none(self) -> None:
        from tmol.ligand.atom_typing import _get_hyb

        for smiles in ("c1ccccc1", "CCO", "CC#N"):
            for atom in self._mol(smiles).GetAtoms():
                assert _get_hyb(atom) in (1, 2, 3, 5, 9)

    def test_has_sp2_oxygen_neighbor_state_none(self) -> None:
        from tmol.ligand.atom_typing import _has_sp2_oxygen_neighbor

        ketone = self._mol("CC(=O)C")
        carbonyl_c = next(
            a
            for a in ketone.GetAtoms()
            if a.GetAtomicNum() == 6
            and any(
                n.GetAtomicNum() == 8
                and ketone.GetBondBetweenAtoms(
                    a.GetIdx(), n.GetIdx()
                ).GetBondTypeAsDouble()
                == 2.0
                for n in a.GetNeighbors()
            )
        )
        assert _has_sp2_oxygen_neighbor(carbonyl_c) is True
        methyl = next(
            a
            for a in ketone.GetAtoms()
            if a.GetAtomicNum() == 6 and a.GetTotalNumHs() == 0 and a is not carbonyl_c
        )
        assert _has_sp2_oxygen_neighbor(methyl) is False

    def test_classify_h_on_phosphorus_is_hg(self) -> None:
        from tmol.ligand.atom_typing import _classify_H

        ph3 = self._mol("P")
        h = next(a for a in ph3.GetAtoms() if a.GetAtomicNum() == 1)
        assert _classify_H(h, ph3) == "HG"

    def test_classify_s_terminal_thione(self) -> None:
        from tmol.ligand.atom_typing import _classify_S

        thione = self._mol("CC(=S)C")
        s = next(a for a in thione.GetAtoms() if a.GetAtomicNum() == 16)
        assert _classify_S(s, thione) == "SG2"

    def test_classify_o_no_carbon_branches(self) -> None:
        from tmol.ligand.atom_typing import _classify_O_no_carbon

        # Phosphate -OH: sp3 O with an H and a P neighbor -> Ohx.
        po4 = self._mol("OP(=O)(O)O")
        oh_on_p = next(
            a
            for a in po4.GetAtoms()
            if a.GetAtomicNum() == 8
            and any(n.GetAtomicNum() == 1 for n in a.GetNeighbors())
            and any(n.GetAtomicNum() == 15 for n in a.GetNeighbors())
        )
        assert _classify_O_no_carbon(oh_on_p, 3, 1, 0, 2) == "Ohx"

        # Hydroxylamine O: sp3 O with an H and an N (no P) -> OG31.
        no = self._mol("NO")
        o = next(a for a in no.GetAtoms() if a.GetAtomicNum() == 8)
        assert _classify_O_no_carbon(o, 3, 1, 1, 2) == "OG31"
        # The same oxygen with no hydrogens routes to the bare sp3 OG3.
        assert _classify_O_no_carbon(o, 3, 0, 1, 1) == "OG3"

        # Nitroso O: sp2 O bonded only to one N -> Ont.
        nitroso = self._mol("CN=O")
        no_o = next(a for a in nitroso.GetAtoms() if a.GetAtomicNum() == 8)
        assert _classify_O_no_carbon(no_o, 2, 0, 1, 1) == "Ont"

        # Bare oxygen reached with a non-sp2/sp3 hybridization -> default OG2.
        assert _classify_O_no_carbon(no_o, 1, 0, 1, 1) == "OG2"

    def test_classify_n_sp2_carbon_only_branches(self) -> None:
        from tmol.ligand.atom_typing import _classify_N_sp2

        # Tertiary amide N (3 carbons incl. carbonyl) -> Nad3.
        tert_amide = self._mol("CC(=O)N(C)C")
        n = next(a for a in tert_amide.GetAtoms() if a.GetAtomicNum() == 7)
        assert _classify_N_sp2(n, 3, 0) == "Nad3"

        # N,N-dimethylaniline N (3 carbons, one aromatic, not amide) -> Nad3.
        aniline = self._mol("CN(C)c1ccccc1")
        n = next(a for a in aniline.GetAtoms() if a.GetAtomicNum() == 7)
        assert _classify_N_sp2(n, 3, 0) == "Nad3"

        # Pyrrole N-H sits in a Hückel-aromatic 5-ring -> Nin.
        pyrrole = _embed_3d("c1cc[nH]c1")
        n = next(a for a in pyrrole.GetAtoms() if a.GetAtomicNum() == 7)
        assert _classify_N_sp2(n, 2, 1) == "Nin"

    def test_classify_o_carbon_branches(self) -> None:
        from tmol.ligand.atom_typing import _classify_O, _classify_O_sp2

        # Dialkyl ether: sp3 O bonded to two carbons, non-aromatic -> Oet.
        ether = _embed_3d("COC")
        o = next(a for a in ether.GetAtoms() if a.GetAtomicNum() == 8)
        assert _classify_O(o, ether) == "Oet"

        # Lone oxygen with no bonds routes through the sp2 classifier's
        # empty-neighbor guard -> OG2.
        lone = Chem.MolFromSmiles("[O]")
        o = next(a for a in lone.GetAtoms() if a.GetAtomicNum() == 8)
        assert _classify_O_sp2(o, 0) == "OG2"


def test_assign_tmol_atom_types_returns_state() -> None:
    """The ``return_state`` flag exposes the shared Rosetta typing perception."""
    from tmol.ligand.atom_typing import (
        RosettaTypingState,
        assign_tmol_atom_types,
    )

    mol = _embed_3d("c1ccncc1")
    assignments, state = assign_tmol_atom_types(mol, return_state=True)
    assert len(assignments) == mol.GetNumAtoms()
    assert isinstance(state, RosettaTypingState)
