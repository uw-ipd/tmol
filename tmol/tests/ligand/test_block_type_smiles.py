"""SMILES from residue types and from PoseStack residues."""

import pytest
from rdkit import Chem

from tmol.chemical import ResidueTypeSet
from tmol.database import ParameterDatabase
from tmol.io import (
    atom_array_from_file,
    extended_pose_stack_from_sequences,
    pose_stack_from_atom_array,
)
from tmol.ligand import (
    prepare_ligand_from_smiles,
    smiles_from_block_type,
    smiles_from_pose_stack,
)
from tmol.tests.data import data_path


@pytest.fixture(scope="module")
def default_types():
    return (
        {rt.name: rt for rt in ResidueTypeSet.get_default().residue_types},
        ParameterDatabase.get_default().chemical,
    )


@pytest.mark.parametrize(
    "name, expected",
    [
        ("ALA", "*N[C@@H](C)C(*)=O"),
        ("DALA", "*N[C@H](C)C(*)=O"),
        ("ALA:cterm:nterm", "C[C@H]([NH3+])C(=O)[O-]"),
        ("ARG", "*N[C@@H](CCCNC(N)=[NH2+])C(*)=O"),
        ("ASP", "*N[C@@H](CC(=O)[O-])C(*)=O"),
        ("CYS_DEP", "*N[C@@H](C[S-])C(*)=O"),
        ("CYD", "*N[C@@H](CS*)C(*)=O"),
        ("HIS", "*N[C@@H](Cc1c[nH]cn1)C(*)=O"),
        ("HIS_POS", "*N[C@@H](Cc1c[nH]c[nH+]1)C(*)=O"),
        ("HIS_DEP", "*N[C@@H](Cc1cnc[n-]1)C(*)=O"),
        ("PRO:nterm", "*C(=O)[C@@H]1CCC[NH2+]1"),
        ("TYR_DEP", "*N[C@@H](Cc1ccc([O-])cc1)C(*)=O"),
        ("DA", "*O[C@H]1C[C@H](n2cnc3c(N)ncnc32)O[C@@H]1COP(*)(=O)[O-]"),
        (
            "DA:na5primephos",
            "*O[C@H]1C[C@H](n2cnc3c(N)ncnc32)O[C@@H]1COP(=O)([O-])[O-]",
        ),
    ],
)
def test_canonical_types(default_types, name, expected):
    types, chem_db = default_types
    assert smiles_from_block_type(types[name], chem_db) == expected


def test_every_default_type_converts(default_types):
    types, chem_db = default_types
    for name, rt in types.items():
        if name == "VRT":
            continue
        assert Chem.MolFromSmiles(smiles_from_block_type(rt, chem_db)) is not None


def test_type_without_real_atoms_raises(default_types):
    types, chem_db = default_types
    with pytest.raises(ValueError):
        smiles_from_block_type(types["VRT"], chem_db)


def test_atom_map_numbers_block_type_atoms(default_types):
    types, chem_db = default_types
    rt = types["SER"]
    element = {t.name: t.element for t in chem_db.atom_types}
    mol = Chem.MolFromSmiles(smiles_from_block_type(rt, chem_db, atom_map=True))
    mapped = {
        a.GetAtomMapNum(): a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomMapNum()
    }
    expected = {
        i + 1: element[a.atom_type]
        for i, a in enumerate(rt.atoms)
        if element[a.atom_type] != "H"
    }
    assert mapped == expected


@pytest.mark.parametrize(
    "smiles",
    [
        "Nc1nc2c(ncn2[C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])[O-])[C@@H](O)[C@H]2O)"
        "c(=O)[nH]1",
        "CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C=O)C(C)(C)CCC1",
        "C[C@H](N)Cc1ccc([N+](=O)[O-])cc1",
        "C[S@@](=O)c1ccccc1",
        "C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1CC[C@@H]2O",
    ],
    ids=["gdp", "retinal", "nitro", "sulfoxide", "estradiol"],
)
def test_prepared_ligand_round_trips(smiles):
    db, _ = prepare_ligand_from_smiles(smiles, protonate=False, res_name="LIG", seed=1)
    rt = next(
        r
        for r in ResidueTypeSet.from_database(db.chemical).residue_types
        if r.name == "LIG"
    )
    assert smiles_from_block_type(rt, db.chemical) == Chem.CanonSmiles(smiles)


def test_pose_matches_type(torch_device):
    pose_stack = extended_pose_stack_from_sequences("ACDHKPSY", device=torch_device)
    chem_db = pose_stack.packed_block_types.chem_db
    for block in range(pose_stack.max_n_blocks):
        assert smiles_from_pose_stack(pose_stack, 0, block) == smiles_from_block_type(
            pose_stack.block_type(0, block), chem_db
        )


def test_covalent_ligand_attachment(torch_device):
    structure = atom_array_from_file(
        data_path("covalent_fixtures", "lys_biotin_1bdo.cif")
    )
    pose_stack = pose_stack_from_atom_array(
        structure, torch_device, prepare_ligands=True, ligand_seed=1, no_optH=True
    )
    smiles = {
        pose_stack.block_type(0, b).name3: smiles_from_pose_stack(pose_stack, 0, b)
        for b in range(pose_stack.max_n_blocks)
        if pose_stack.is_real_block(0, b)
    }
    biotinyl = Chem.MolFromSmiles(smiles["BTN"])
    assert smiles["BTN"].count("*") == 1
    acid = Chem.ReplaceSubstructs(
        biotinyl, Chem.MolFromSmarts("[#0]"), Chem.MolFromSmiles("O")
    )[0]
    biotin = "OC(=O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@H]12"
    assert Chem.MolToSmiles(acid) == Chem.CanonSmiles(biotin)


def test_metal_coordination_is_left_out(torch_device):
    structure = atom_array_from_file(
        data_path("metal_fixtures", "zn_tetrahedral_3ks3.cif.zst")
    )
    pose_stack = pose_stack_from_atom_array(
        structure, torch_device, prepare_ligands=True, no_optH=True
    )
    partners = pose_stack.inter_residue_connections64[0].tolist()
    zn = next(
        b
        for b in range(pose_stack.max_n_blocks)
        if pose_stack.is_real_block(0, b) and pose_stack.block_type(0, b).metal_sites
    )
    assert smiles_from_pose_stack(pose_stack, 0, zn) == "[Zn]"
    donors = {p for p, _ in partners[zn] if p >= 0}
    assert len(donors) == 3
    for donor in donors:
        bonded = [p for p, _ in partners[donor] if p >= 0 and p != zn]
        assert smiles_from_pose_stack(pose_stack, 0, donor).count("*") == len(bonded)
