"""Generate mapped conjugate geometry through the ordinary ligand pipeline."""

import numpy as np
from rdkit import Chem

from tmol.ligand._atom_typing import sanitize_tolerant
from tmol.ligand._detect import nonstandard_residue_info_from_smiles_via_mol2
from tmol.ligand._generated_geometry import correct_generated_geometry
from tmol.ligand._rdkit_mol import ligand_atom_array_to_rdkit_mol


def generated_conjugate_coordinates(mol, seed):
    """Return generated coordinates in the protonated input molecule's order.

    Heavy atoms carry source-index map numbers. Equivalent hydrogens are mapped
    by their common parent. Mapped chemistry, including specified stereochemistry,
    must survive generation; carboxylate resonance representations may differ.
    Experimental coordinates are never used as equilibrium targets.
    """
    source = Chem.RemoveHs(mol)
    info = nonstandard_residue_info_from_smiles_via_mol2(
        Chem.MolToSmiles(source, ignoreAtomMapNumbers=True),
        res_name="CONJ",
        protonate=False,
        seed=seed,
    )
    generated = ligand_atom_array_to_rdkit_mol(info, keep_hydrogens=True)
    sanitize_tolerant(generated)
    correct_generated_geometry(generated)
    heavy = [a.GetIdx() for a in generated.GetAtoms() if a.GetAtomicNum() > 1]
    mapping = dict(zip(info.source_atom_order or (), heavy, strict=True))
    expected = {a.GetAtomMapNum() - 1 for a in source.GetAtoms()}
    if set(mapping) != expected:
        raise ValueError("Generated conjugate geometry changed mapped heavy atoms")
    for original, index in mapping.items():
        generated.GetAtomWithIdx(index).SetAtomMapNum(original + 1)
    target = Chem.MolToSmiles(Chem.RemoveHs(generated))
    if not any(
        Chem.MolToSmiles(resonance) == target
        for resonance in Chem.ResonanceMolSupplier(source, maxStructs=128)
    ):
        raise ValueError(
            "Generated conjugate geometry changed mapped chemistry or assigned "
            "unspecified stereochemistry; supply a resolved stereochemical reference"
        )

    positions = generated.GetConformer().GetPositions()
    coords = np.full((mol.GetNumAtoms(), 3), np.nan)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        index = mapping[atom.GetAtomMapNum() - 1]
        own_h = sorted(n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1)
        other_h = sorted(
            n.GetIdx()
            for n in generated.GetAtomWithIdx(index).GetNeighbors()
            if n.GetAtomicNum() == 1
        )
        if len(own_h) != len(other_h):
            raise ValueError("Generated conjugate geometry changed attached hydrogens")
        coords[atom.GetIdx()] = positions[index]
        coords[own_h] = positions[other_h]
    if not np.isfinite(coords).all():
        raise ValueError("Generated conjugate geometry has missing coordinates")
    return coords
