"""Public API for tmol ligand preparation.

This package-level module intentionally stays thin and re-exports stable
entry points from focused implementation modules. The single supported
chemistry source is a SMILES string (derived from a CIF/atom-array ligand by
:mod:`tmol.ligand.structure_to_smiles`), which feeds the SMILES -> params
pipeline.
"""

from tmol.database.chemical import RawResidueType  # noqa: F401  re-exported
from tmol.ligand.detect import (
    NonStandardResidueInfo,
    detect_nonstandard_residues,
    nonstandard_residue_info_from_smiles_via_mol2,
)
from tmol.ligand.preparation import (
    prepare_ligand_from_cif,
    prepare_ligand_from_smiles,
    prepare_ligands,
    prepare_single_ligand,
)
from tmol.ligand.registry import LigandPreparation
from tmol.ligand.structure_to_smiles import (
    ligand_smiles_candidates_from_atom_array,
    ligand_smiles_from_atom_array,
)

__all__ = [
    "LigandPreparation",
    "NonStandardResidueInfo",
    "RawResidueType",
    "prepare_ligands",
    "prepare_single_ligand",
    "prepare_ligand_from_cif",
    "prepare_ligand_from_smiles",
    "detect_nonstandard_residues",
    "nonstandard_residue_info_from_smiles_via_mol2",
    "ligand_smiles_from_atom_array",
    "ligand_smiles_candidates_from_atom_array",
]
