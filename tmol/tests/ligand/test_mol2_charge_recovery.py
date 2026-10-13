"""Prepared MOL2 charge totals constrain missing aromatic formal charges."""

import numpy as np
import pytest
from rdkit import Chem

from tmol.io import pose_stack_from_biotite
from tmol.ligand import prepare_ligand_from_mol2
from tmol.ligand._detect import nonstandard_residue_info_from_mol2_block
from tmol.ligand._rdkit_mol import ligand_atom_array_to_rdkit_mol
from tmol.tests.data import data_path
from tmol.tests.ligand.test_ligand_entry_paths import _score_and_minimize_ligand

DATA = data_path() / "sweep_regressions"


@pytest.mark.parametrize("code", ["3djf", "1tou", "4yt6"])
def test_prepared_charge_recovers_source_resonance_and_scores(code, torch_device):
    path = DATA / f"aromatic_charge_{code}.mol2"
    text = path.read_text()
    info = nonstandard_residue_info_from_mol2_block(text, res_name="LIG")
    actual = ligand_atom_array_to_rdkit_mol(info, keep_hydrogens=True)
    reference = next(
        iter(Chem.SDMolSupplier(str(path.with_suffix(".sdf")), removeHs=False))
    )

    def resonance(molecule):
        return {
            Chem.MolToSmiles(form, isomericSmiles=False)
            for form in Chem.ResonanceMolSupplier(Chem.RemoveHs(molecule))
        }

    assert resonance(actual) & resonance(reference)
    assert Chem.GetFormalCharge(actual) == Chem.GetFormalCharge(reference)
    atoms = [
        line.split()
        for line in text.split("@<TRIPOS>ATOM\n")[1].split("@<TRIPOS>")[0].splitlines()
        if line.strip()
    ]
    bonds = [
        line.split()
        for line in text.split("@<TRIPOS>BOND\n")[1].split("@<TRIPOS>")[0].splitlines()
        if line.strip()
    ]
    assert info.atom_names == tuple(row[1] for row in atoms)
    np.testing.assert_allclose(
        info.coords, [[float(v) for v in row[2:5]] for row in atoms], atol=1e-5
    )
    assert {
        frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        for bond in actual.GetBonds()
    } == {frozenset((int(row[1]) - 1, int(row[2]) - 1)) for row in bonds}
    assert actual.GetNumAtoms() == len(atoms)  # Includes every supplied hydrogen.
    assert not any(atom.GetNumRadicalElectrons() for atom in actual.GetAtoms())
    database, _ = prepare_ligand_from_mol2(path, res_name="LIG", mode="keep")
    pose = pose_stack_from_biotite(
        info.atom_array,
        torch_device,
        param_db=database,
        no_optH=True,
        trust_hydrogen_names=True,
    )
    block = pose.packed_block_types.active_block_types[int(pose.block_type_ind[0, 0])]
    np.testing.assert_allclose(
        pose.coords[0, [block.atom_to_idx[name] for name in info.atom_names]].cpu(),
        info.coords,
        atol=1e-5,
    )
    _score_and_minimize_ligand(pose, database)


@pytest.mark.parametrize(
    "unusable", ["untrusted", "fractional", "nonfinite", "incomplete", "conflicting"]
)
def test_missing_charge_is_not_guessed_without_consistent_constraints(unusable):
    text = (DATA / "aromatic_charge_3djf.mol2").read_text()
    if unusable == "untrusted":
        text = text.replace("USER_CHARGES", "GASTEIGER")
    elif unusable == "conflicting":
        # All stated neutral atoms contradict this ligand's prepared +1 total.
        count = len(text.split("@<TRIPOS>ATOM\n")[1].split("@<TRIPOS>")[0].splitlines())
        text += "@<TRIPOS>UNITY_ATOM_ATTR\n" + "".join(
            f"{i + 1} 1\ncharge 0\n" for i in range(count)
        )
    else:
        lines = text.splitlines()
        index = lines.index("@<TRIPOS>ATOM") + 1
        fields = lines[index].split()
        if unusable == "incomplete":
            fields.pop()
        else:
            fields[-1] = (
                "nan" if unusable == "nonfinite" else str(float(fields[-1]) + 0.25)
            )
        lines[index] = " ".join(fields)
        text = "\n".join(lines) + "\n"
    with pytest.raises(ValueError, match="sanitizable chemical graph"):
        nonstandard_residue_info_from_mol2_block(text)
