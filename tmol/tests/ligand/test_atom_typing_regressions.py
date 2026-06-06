import math
from types import SimpleNamespace

import numpy as np
from rdkit import Chem

from tmol.ligand.atom_typing import (
    AtomTypeAssignment,
    HYB_AROMATIC,
    HYB_SP2,
    _build_rosetta_typing_state,
    _correct_amide_bond_orders,
    _classify_N,
    _classify_O,
    _classify_N_sp2,
    _classify_P,
    _classify_S,
    _correct_conjugated_single_bond_orders,
    _correct_ring_nitrogen,
    _modify_polar_c,
    _get_hyb,
    sanitize_tolerant,
)
from tmol.ligand.rdkit_mol import ligand_atom_array_to_rdkit_mol
from tmol.ligand.residue_builder import build_residue_type


def test_get_hyb_distinguishes_ar_vs_aro_subtypes():
    mol = Chem.MolFromSmiles("N")
    atom = mol.GetAtomWithIdx(0)

    atom.SetProp("_tmol_source_subtype", "ar")
    assert _get_hyb(atom) == HYB_AROMATIC

    atom.SetProp("_tmol_source_subtype", "aro")
    assert _get_hyb(atom) == HYB_SP2


def test_ring_nitrogen_correction_forces_nim():
    mol = Chem.MolFromSmiles("N1=NC=CC=C1")
    n_idx = next(
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 7 and atom.GetDegree() == 2
    )

    assignments = [
        AtomTypeAssignment(
            atom_name="N1",
            atom_type="NG2",
            element="N",
            index=n_idx,
        )
    ]

    state = _build_rosetta_typing_state(mol)
    corrected = _correct_ring_nitrogen(assignments, mol, state)
    assert corrected[0].atom_type == "Nim"


def test_classify_n_sp2_no_nameerror_on_aromatic_context():
    # Pyridine-like aromatic N (no attached H) to exercise the nH==0 branch.
    mol = Chem.MolFromSmiles("n1ccccc1")
    atom = mol.GetAtomWithIdx(0)

    state = _build_rosetta_typing_state(mol)
    atom_type = _classify_N_sp2(atom, nC=2, nH=0, state=state)
    assert atom_type == "Nim"


def test_state_marks_strained_and_aromatic_ring_atoms():
    strained = Chem.MolFromSmiles("C1CC1")
    s_state = _build_rosetta_typing_state(strained)
    assert len(s_state.atms_strained) == 3

    aromatic = Chem.MolFromSmiles("n1ccccc1")
    a_state = _build_rosetta_typing_state(aromatic)
    assert len(a_state.atms_aro) == 6


def test_amide_bond_correction_promotes_nad_cdp_single_bond():
    mol = Chem.MolFromSmiles("NC(=O)C")
    n_idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    c_idx = next(
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6
        and any(
            b.GetOtherAtom(a).GetAtomicNum() == 8 and b.GetBondTypeAsDouble() == 2.0
            for b in a.GetBonds()
        )
        and any(b.GetOtherAtom(a).GetAtomicNum() == 7 for b in a.GetBonds())
    )

    assignments = [
        AtomTypeAssignment("N1", "Nad", "N", n_idx),
        AtomTypeAssignment("C1", "CDp", "C", c_idx),
    ]
    bond = mol.GetBondBetweenAtoms(n_idx, c_idx)
    assert bond.GetBondType() == Chem.BondType.SINGLE

    _correct_amide_bond_orders(assignments, mol)
    assert bond.GetBondType() == Chem.BondType.DOUBLE


def test_classify_n_hyb8_amide_primary_and_tertiary():
    primary = Chem.AddHs(Chem.MolFromSmiles("NC=O"))
    n_primary = next(a for a in primary.GetAtoms() if a.GetAtomicNum() == 7)
    n_primary.SetProp("_tmol_source_subtype", "am")
    p_state = _build_rosetta_typing_state(primary)
    assert _classify_N(n_primary, primary, p_state) == "Nad"

    tertiary = Chem.MolFromSmiles("CN(C)C=O")
    n_tertiary = next(a for a in tertiary.GetAtoms() if a.GetAtomicNum() == 7)
    n_tertiary.SetProp("_tmol_source_subtype", "am")
    t_state = _build_rosetta_typing_state(tertiary)
    assert _classify_N(n_tertiary, tertiary, t_state) == "Nad3"


def test_classify_o2_oxime_guard_requires_sp2_n():
    oxime = Chem.AddHs(Chem.MolFromSmiles("C=NO"))
    o_atom = next(a for a in oxime.GetAtoms() if a.GetAtomicNum() == 8)
    o_atom.SetProp("_tmol_source_subtype", "2")
    o_state = _build_rosetta_typing_state(oxime)
    assert _classify_O(o_atom, oxime, o_state) == "OG31"

    hydroxylamine = Chem.AddHs(Chem.MolFromSmiles("CNO"))
    o_atom2 = next(a for a in hydroxylamine.GetAtoms() if a.GetAtomicNum() == 8)
    o_atom2.SetProp("_tmol_source_subtype", "2")
    o_state2 = _build_rosetta_typing_state(hydroxylamine)
    assert _classify_O(o_atom2, hydroxylamine, o_state2) != "OG31"


def test_classify_o3_aromatic_ring_oxygen_maps_to_ofu():
    mol = Chem.MolFromSmiles("O1C=CC=C1")
    atom = next(a for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    atom.SetProp("_tmol_source_subtype", "3")
    state = _build_rosetta_typing_state(mol)
    assert _classify_O(atom, mol, state) == "Ofu"


def test_classify_o3_nonaromatic_ring_oxygen_maps_to_oet():
    mol = Chem.MolFromSmiles("O1CC=CC1")
    atom = next(a for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    atom.SetProp("_tmol_source_subtype", "3")
    state = _build_rosetta_typing_state(mol)
    assert atom.GetIdx() not in state.atms_aro
    assert _classify_O(atom, mol, state) == "Oet"


def test_six_member_mixed_sp2_sp3_ring_is_not_aromatic():
    mol = Chem.MolFromSmiles("O1CC=CCC1")
    o_atom = next(a for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    o_atom.SetProp("_tmol_source_subtype", "3")
    state = _build_rosetta_typing_state(mol)
    assert o_atom.GetIdx() not in state.atms_aro
    assert _classify_O(o_atom, mol, state) == "Oet"


def test_missing_hybridization_assignment_for_nh_subtype():
    aniline = Chem.AddHs(Chem.MolFromSmiles("Nc1ccccc1"))
    n1 = next(a for a in aniline.GetAtoms() if a.GetAtomicNum() == 7)
    n1.SetProp("_tmol_source_subtype", "nh")
    s1 = _build_rosetta_typing_state(aniline)
    assert _get_hyb(n1, s1) == HYB_SP2

    mixed = Chem.AddHs(Chem.MolFromSmiles("CNc1ccccc1"))
    n2 = next(a for a in mixed.GetAtoms() if a.GetAtomicNum() == 7)
    n2.SetProp("_tmol_source_subtype", "nh")
    s2 = _build_rosetta_typing_state(mixed)
    assert _get_hyb(n2, s2) == 3


def test_p_and_s_follow_hyb5_classification():
    p_mol = Chem.MolFromSmiles("P")
    p_atom = p_mol.GetAtomWithIdx(0)
    p_atom.SetProp("_tmol_source_subtype", "o2")
    p_state = _build_rosetta_typing_state(p_mol)
    assert _classify_P(p_atom, p_mol, p_state) == "PG5"

    s5_mol = Chem.MolFromSmiles("CS(=O)(=O)C")
    s5_atom = next(a for a in s5_mol.GetAtoms() if a.GetAtomicNum() == 16)
    s5_atom.SetProp("_tmol_source_subtype", "o2")
    s5_state = _build_rosetta_typing_state(s5_mol)
    assert _classify_S(s5_atom, s5_mol, s5_state) == "SG5"

    # Thioether (C-S-C) maps to Ssl, not SG3, regardless of source subtype.
    thioether = Chem.MolFromSmiles("CSC")
    thioether_s = next(a for a in thioether.GetAtoms() if a.GetAtomicNum() == 16)
    thioether_s.SetProp("_tmol_source_subtype", "3")
    thioether_state = _build_rosetta_typing_state(thioether)
    assert _classify_S(thioether_s, thioether, thioether_state) == "Ssl"

    s3_mol = Chem.MolFromSmiles("CS(C)=O")
    s3_atom = next(a for a in s3_mol.GetAtoms() if a.GetAtomicNum() == 16)
    s3_atom.SetProp("_tmol_source_subtype", "3")
    s3_state = _build_rosetta_typing_state(s3_mol)
    assert _classify_S(s3_atom, s3_mol, s3_state) == "SG3"

    disulfide = Chem.MolFromSmiles("CSSC")
    s_atom = next(
        a for a in disulfide.GetAtoms() if a.GetAtomicNum() == 16 and a.GetDegree() == 2
    )
    ds_state = _build_rosetta_typing_state(disulfide)
    assert _classify_S(s_atom, disulfide, ds_state) in {"Ssl", "SR"}


def test_modify_polar_c_promotes_cdp():
    mol = Chem.MolFromSmiles("NC(=O)C")
    carbonyl_c = next(
        a
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6
        and any(
            b.GetOtherAtom(a).GetAtomicNum() == 8 and b.GetBondTypeAsDouble() == 2.0
            for b in a.GetBonds()
        )
    )
    n_atom = next(a for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    o_atom = next(a for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    c_atom = next(
        a
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6 and a.GetIdx() != carbonyl_c.GetIdx()
    )

    assignments = [
        AtomTypeAssignment("C1", "CD", "C", carbonyl_c.GetIdx()),
        AtomTypeAssignment("N1", "Nad", "N", n_atom.GetIdx()),
        AtomTypeAssignment("O1", "Oal", "O", o_atom.GetIdx()),
        AtomTypeAssignment("C2", "CS3", "C", c_atom.GetIdx()),
    ]
    out = _modify_polar_c(assignments, mol)
    by_idx = {a.index: a.atom_type for a in out}
    assert by_idx[carbonyl_c.GetIdx()] == "CDp"


def test_long_ring_aromatic_planarity_gate():
    mol = Chem.MolFromSmiles("C1CCCCCC1")
    conf = Chem.Conformer(mol.GetNumAtoms())
    # Planar heptagon in xy-plane.
    for i in range(mol.GetNumAtoms()):
        angle = 2.0 * math.pi * i / mol.GetNumAtoms()
        conf.SetAtomPosition(
            i, (1.5 * float(math.cos(angle)), 1.5 * float(math.sin(angle)), 0.0)
        )
    mol.RemoveAllConformers()
    mol.AddConformer(conf)
    planar = _build_rosetta_typing_state(mol)
    assert len(planar.atms_aro) == 7

    conf2 = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        angle = 2.0 * math.pi * i / mol.GetNumAtoms()
        z = 0.8 if i == 0 else 0.0
        conf2.SetAtomPosition(
            i, (1.5 * float(math.cos(angle)), 1.5 * float(math.sin(angle)), z)
        )
    mol2 = Chem.Mol(mol)
    mol2.RemoveAllConformers()
    mol2.AddConformer(conf2)
    non_planar = _build_rosetta_typing_state(mol2)
    assert len(non_planar.atms_aro) == 0


def test_classify_n_pl3_ring_hetero_tertiary_maps_to_nad3():
    mol = Chem.MolFromSmiles("CN1N=CC=CC1")
    atom = next(
        a for a in mol.GetAtoms() if a.GetAtomicNum() == 7 and a.GetDegree() == 3
    )
    atom.SetProp("_tmol_source_subtype", "pl3")
    state = _build_rosetta_typing_state(mol)
    assert _classify_N(atom, mol, state) == "Nad3"


def test_classify_n2_bridge_in_pyrazole_maps_to_nim_not_nad3():
    """cox2/p38: N.2 bridge N between N and C is Nim, not Nad3."""
    mol = Chem.MolFromSmiles("CN1N=CC=CC1")
    atom = next(
        a
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 7
        and a.GetDegree() == 2
        and any(n.GetAtomicNum() == 7 for n in a.GetNeighbors())
    )
    atom.SetProp("_tmol_source_subtype", "2")
    state = _build_rosetta_typing_state(mol)
    assert _classify_N(atom, mol, state) == "Nim"


def test_protonated_pl3_with_hetero_neighbor_maps_to_nam2():
    mol = Chem.AddHs(Chem.MolFromSmiles("CS[NH2+]C"))
    n_atom = next(a for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    n_atom.SetProp("_tmol_source_subtype", "pl3")
    state = _build_rosetta_typing_state(mol)
    assert _classify_N(n_atom, mol, state) == "Nam2"


def test_conjugated_single_bond_promotion_for_conjugating_classes():
    mol = Chem.MolFromSmiles("NNC=C")
    n0 = mol.GetAtomWithIdx(0)
    n1 = mol.GetAtomWithIdx(1)
    c2 = mol.GetAtomWithIdx(2)
    c3 = mol.GetAtomWithIdx(3)
    n0.SetProp("_tmol_source_subtype", "pl3")
    n1.SetProp("_tmol_source_subtype", "2")
    c2.SetProp("_tmol_source_subtype", "2")
    c3.SetProp("_tmol_source_subtype", "2")

    bond = mol.GetBondBetweenAtoms(0, 1)
    assert bond.GetBondType() == Chem.BondType.SINGLE

    state = _build_rosetta_typing_state(mol)
    assignments = [
        AtomTypeAssignment("N1", "Nad3", "N", 0),
        AtomTypeAssignment("N2", "Nim", "N", 1),
        AtomTypeAssignment("C1", "CD", "C", 2),
        AtomTypeAssignment("C2", "CD", "C", 3),
    ]
    _correct_conjugated_single_bond_orders(assignments, mol, state)
    assert bond.GetBondType() == Chem.BondType.DOUBLE


def test_classify_o2_uses_rosetta_first_bond_behavior():
    # Rosetta classify_O() inspects only the first bonded atom for O.2.
    # If that first neighbor is non-carbon, O.2 falls back to OG2.
    mol = Chem.MolFromSmiles("NOC(=N)N")
    o_atom = next(a for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    o_atom.SetProp("_tmol_source_subtype", "2")
    state = _build_rosetta_typing_state(mol)
    assert _classify_O(o_atom, mol, state) == "OG2"


def test_five_member_ring_sp3_oxygen_aromatic_exception():
    # Rosetta's 5-membered-ring exception allows one sp3 O/S ring atom
    # to still count as aromatic for atom typing.
    mol = Chem.MolFromSmiles("O1C=CC=C1")
    o_atom = next(a for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    o_atom.SetProp("_tmol_source_subtype", "3")
    state = _build_rosetta_typing_state(mol)
    assert o_atom.GetIdx() in state.atms_aro
    assert _classify_O(o_atom, mol, state) == "Ofu"


def test_classify_n_hetero_accepts_sp2_oxygen_neighbor_without_double_bond():
    mol = Chem.RWMol()
    n_idx = mol.AddAtom(Chem.Atom(7))
    c_idx = mol.AddAtom(Chem.Atom(6))
    o_sp2_idx = mol.AddAtom(Chem.Atom(8))
    c2_idx = mol.AddAtom(Chem.Atom(6))
    o_het_idx = mol.AddAtom(Chem.Atom(8))
    h_idx = mol.AddAtom(Chem.Atom(1))
    mol.AddBond(n_idx, c_idx, Chem.BondType.SINGLE)
    mol.AddBond(c_idx, o_sp2_idx, Chem.BondType.SINGLE)
    mol.AddBond(c_idx, c2_idx, Chem.BondType.DOUBLE)
    mol.AddBond(n_idx, o_het_idx, Chem.BondType.SINGLE)
    mol.AddBond(n_idx, h_idx, Chem.BondType.SINGLE)
    mol = mol.GetMol()

    mol.GetAtomWithIdx(n_idx).SetProp("_tmol_source_subtype", "pl3")
    mol.GetAtomWithIdx(c_idx).SetProp("_tmol_source_subtype", "2")
    mol.GetAtomWithIdx(o_sp2_idx).SetProp("_tmol_source_subtype", "2")
    mol.GetAtomWithIdx(c2_idx).SetProp("_tmol_source_subtype", "2")
    mol.GetAtomWithIdx(o_het_idx).SetProp("_tmol_source_subtype", "3")

    state = _build_rosetta_typing_state(mol)
    n_atom = mol.GetAtomWithIdx(n_idx)
    assert _classify_N(n_atom, mol, state) == "Nad"


def test_classify_n_sp2_nonaromatic_nh_maps_to_ng21_not_nin():
    mol = Chem.RWMol()
    n_idx = mol.AddAtom(Chem.Atom(7))
    c_sp2_idx = mol.AddAtom(Chem.Atom(6))
    c_sp3_idx = mol.AddAtom(Chem.Atom(6))
    c_aux_idx = mol.AddAtom(Chem.Atom(6))
    h_idx = mol.AddAtom(Chem.Atom(1))
    mol.AddBond(n_idx, c_sp2_idx, Chem.BondType.SINGLE)
    mol.AddBond(n_idx, c_sp3_idx, Chem.BondType.SINGLE)
    mol.AddBond(n_idx, h_idx, Chem.BondType.SINGLE)
    mol.AddBond(c_sp2_idx, c_aux_idx, Chem.BondType.DOUBLE)
    mol = mol.GetMol()

    mol.GetAtomWithIdx(n_idx).SetProp("_tmol_source_subtype", "pl3")
    mol.GetAtomWithIdx(c_sp2_idx).SetProp("_tmol_source_subtype", "2")
    mol.GetAtomWithIdx(c_sp3_idx).SetProp("_tmol_source_subtype", "3")
    mol.GetAtomWithIdx(c_aux_idx).SetProp("_tmol_source_subtype", "2")

    state = _build_rosetta_typing_state(mol)
    n_atom = mol.GetAtomWithIdx(n_idx)
    assert n_atom.GetIdx() not in state.atms_aro
    assert _classify_N(n_atom, mol, state) == "NG21"


def test_classify_n2_aromatic_tertiary_with_n_neighbor_maps_to_nim():
    mol = Chem.RWMol()
    n0 = mol.AddAtom(Chem.Atom(7))
    c1 = mol.AddAtom(Chem.Atom(6))
    c2 = mol.AddAtom(Chem.Atom(6))
    c3 = mol.AddAtom(Chem.Atom(6))
    c4 = mol.AddAtom(Chem.Atom(6))
    c5 = mol.AddAtom(Chem.Atom(6))
    n6 = mol.AddAtom(Chem.Atom(7))
    # six-member conjugated ring containing N0
    mol.AddBond(n0, c1, Chem.BondType.SINGLE)
    mol.AddBond(c1, c2, Chem.BondType.DOUBLE)
    mol.AddBond(c2, c3, Chem.BondType.SINGLE)
    mol.AddBond(c3, c4, Chem.BondType.DOUBLE)
    mol.AddBond(c4, c5, Chem.BondType.SINGLE)
    mol.AddBond(c5, n0, Chem.BondType.DOUBLE)
    # exocyclic N neighbor on the ring N
    mol.AddBond(n0, n6, Chem.BondType.SINGLE)
    mol = mol.GetMol()
    Chem.rdmolops.FastFindRings(mol)

    mol.GetAtomWithIdx(n0).SetProp("_tmol_source_subtype", "2")
    mol.GetAtomWithIdx(n6).SetProp("_tmol_source_subtype", "pl3")
    for idx in (c1, c2, c3, c4, c5):
        mol.GetAtomWithIdx(idx).SetProp("_tmol_source_subtype", "2")

    state = _build_rosetta_typing_state(mol)
    assert n0 in state.atms_aro
    assert _classify_N(mol.GetAtomWithIdx(n0), mol, state) == "Nim"


def test_conjugated_single_bond_promotion_does_not_require_planarity_by_default():
    mol = Chem.MolFromSmiles("NNC=C")
    conf = Chem.Conformer(mol.GetNumAtoms())
    # Deliberately twisted around N-N to be far from planar.
    conf.SetAtomPosition(0, (0.0, 0.0, 0.0))
    conf.SetAtomPosition(1, (1.3, 0.0, 0.0))
    conf.SetAtomPosition(2, (2.0, 1.2, 0.0))
    conf.SetAtomPosition(3, (3.2, 1.4, 1.5))
    mol.RemoveAllConformers()
    mol.AddConformer(conf)

    n0 = mol.GetAtomWithIdx(0)
    n1 = mol.GetAtomWithIdx(1)
    c2 = mol.GetAtomWithIdx(2)
    c3 = mol.GetAtomWithIdx(3)
    n0.SetProp("_tmol_source_subtype", "pl3")
    n1.SetProp("_tmol_source_subtype", "2")
    c2.SetProp("_tmol_source_subtype", "2")
    c3.SetProp("_tmol_source_subtype", "2")

    bond = mol.GetBondBetweenAtoms(0, 1)
    assert bond.GetBondType() == Chem.BondType.SINGLE

    state = _build_rosetta_typing_state(mol)
    assignments = [
        AtomTypeAssignment("N1", "Nad3", "N", 0),
        AtomTypeAssignment("N2", "Nim", "N", 1),
        AtomTypeAssignment("C1", "CD", "C", 2),
        AtomTypeAssignment("C2", "CD", "C", 3),
    ]
    _correct_conjugated_single_bond_orders(assignments, mol, state)
    assert bond.GetBondType() == Chem.BondType.DOUBLE


def test_sanitize_tolerant_handles_nonring_aromatic_placeholders():
    mol = Chem.MolFromSmiles("CC")
    rw = Chem.RWMol(mol)
    b = rw.GetBondBetweenAtoms(0, 1)
    b.SetBondType(Chem.BondType.AROMATIC)
    b.SetIsAromatic(True)
    rw.GetAtomWithIdx(0).SetIsAromatic(True)
    rw.GetAtomWithIdx(1).SetIsAromatic(True)
    mol = rw.GetMol()

    sanitize_tolerant(mol)
    bond = mol.GetBondBetweenAtoms(0, 1)
    assert bond.GetBondType() == Chem.BondType.SINGLE
    assert not bond.GetIsAromatic()


def test_classify_n_pl3_protonated_maps_to_nam2():
    mol = Chem.AddHs(Chem.MolFromSmiles("C[NH2+]C"))
    n_atom = next(a for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    n_atom.SetProp("_tmol_source_subtype", "pl3")
    state = _build_rosetta_typing_state(mol)
    assert _classify_N(n_atom, mol, state) == "Nam2"


def test_classify_n_am_protonated_tertiary_maps_to_nam2():
    mol = Chem.AddHs(Chem.MolFromSmiles("C[NH+](C)C=O"))
    n_atom = next(a for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    n_atom.SetProp("_tmol_source_subtype", "am")
    state = _build_rosetta_typing_state(mol)
    assert _classify_N(n_atom, mol, state) == "Nam2"


def test_classify_n2_protonated_with_n_neighbor_maps_to_nam2():
    mol = Chem.RWMol()
    n0 = mol.AddAtom(Chem.Atom(7))
    c1 = mol.AddAtom(Chem.Atom(6))
    c2 = mol.AddAtom(Chem.Atom(6))
    n3 = mol.AddAtom(Chem.Atom(7))
    h4 = mol.AddAtom(Chem.Atom(1))
    mol.AddBond(n0, c1, Chem.BondType.SINGLE)
    mol.AddBond(n0, c2, Chem.BondType.SINGLE)
    mol.AddBond(n0, n3, Chem.BondType.SINGLE)
    mol.AddBond(n0, h4, Chem.BondType.SINGLE)
    mol = mol.GetMol()
    mol.GetAtomWithIdx(n0).SetProp("_tmol_source_subtype", "2")
    mol.GetAtomWithIdx(c1).SetProp("_tmol_source_subtype", "3")
    mol.GetAtomWithIdx(c2).SetProp("_tmol_source_subtype", "3")
    mol.GetAtomWithIdx(n3).SetProp("_tmol_source_subtype", "2")

    state = _build_rosetta_typing_state(mol)
    assert _classify_N(mol.GetAtomWithIdx(n0), mol, state) == "Nam2"


def test_classify_n2_protonated_formal_charge_maps_to_nam2():
    mol = Chem.RWMol()
    n0 = mol.AddAtom(Chem.Atom(7))
    c1 = mol.AddAtom(Chem.Atom(6))
    c2 = mol.AddAtom(Chem.Atom(6))
    c3 = mol.AddAtom(Chem.Atom(6))
    h4 = mol.AddAtom(Chem.Atom(1))
    mol.AddBond(n0, c1, Chem.BondType.SINGLE)
    mol.AddBond(n0, c2, Chem.BondType.SINGLE)
    mol.AddBond(n0, c3, Chem.BondType.SINGLE)
    mol.AddBond(n0, h4, Chem.BondType.SINGLE)
    mol = mol.GetMol()

    n_atom = mol.GetAtomWithIdx(n0)
    n_atom.SetFormalCharge(1)
    n_atom.SetProp("_tmol_source_subtype", "2")
    state = _build_rosetta_typing_state(mol)
    assert _classify_N(n_atom, mol, state) == "Nam2"


def test_ligand_atom_array_allows_passthrough_unknown_bond_type(monkeypatch):
    class _FakeBondTable:
        def __init__(self, rows):
            self._rows = np.array(rows, dtype=int)

        def get_bond_count(self):
            return len(self._rows)

        def as_array(self):
            return self._rows

    class _FakeAtomArray:
        def __init__(self):
            self.bonds = _FakeBondTable([[0, 1, 99], [1, 2, 2]])
            self.element = np.array(["C", "C", "C"], dtype=object)

        def __len__(self):
            return len(self.element)

    monkeypatch.setattr(
        "tmol.ligand.rdkit_mol.to_mol",
        lambda _aa: Chem.MolFromSmiles("CCC"),
    )
    ligand_info = SimpleNamespace(res_name="LG1", atom_array=_FakeAtomArray())
    mol = ligand_atom_array_to_rdkit_mol(ligand_info)
    assert mol.GetNumAtoms() == 3


def test_conjugated_single_bond_skips_biaryl_like_ring_pivot():
    mol = Chem.MolFromSmiles("c1ccccc1-c2ccccc2")
    bond = next(
        b
        for b in mol.GetBonds()
        if not b.GetIsAromatic()
        and b.GetBeginAtom().GetIsAromatic()
        and b.GetEndAtom().GetIsAromatic()
    )
    a_idx = bond.GetBeginAtomIdx()
    b_idx = bond.GetEndAtomIdx()
    assert bond.GetBondType() == Chem.BondType.SINGLE

    state = _build_rosetta_typing_state(mol)
    assignments = [
        AtomTypeAssignment("CR1", "CR", "C", a_idx),
        AtomTypeAssignment("CR2", "CR", "C", b_idx),
    ]
    _correct_conjugated_single_bond_orders(assignments, mol, state)
    assert bond.GetBondType() == Chem.BondType.SINGLE


def test_large_ring_bonds_keep_ring_flag_in_residue_type():
    mol = Chem.MolFromSmiles("C1CCCCCCCC1")
    atom_types = [
        AtomTypeAssignment(atom_name=f"C{i + 1}", atom_type="CS3", element="C", index=i)
        for i in range(mol.GetNumAtoms())
    ]
    restype = build_residue_type(mol, "LG1", atom_types)
    assert all(not b[3] for b in restype.bonds)


def test_sanitize_tolerant_preserves_existing_double_bond_without_aromatic_rewrite():
    mol = Chem.MolFromSmiles("CC=O")
    bond = next(
        b
        for b in mol.GetBonds()
        if {b.GetBeginAtom().GetAtomicNum(), b.GetEndAtom().GetAtomicNum()} == {6, 8}
    )
    assert bond.GetBondType() == Chem.BondType.DOUBLE
    sanitize_tolerant(mol)
    assert bond.GetBondType() == Chem.BondType.DOUBLE
