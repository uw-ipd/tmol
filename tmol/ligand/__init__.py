"""Public API for tmol ligand preparation.

Stable entry points for the unified CIF/AtomArray/SMILES/mol2 → params pipeline.
"""

from tmol.database.chemical import RawResidueType  # noqa: F401  re-exported
from tmol.ligand.detect import (
    NonStandardResidueInfo,
    detect_nonstandard_residues,
    nonstandard_residue_info_from_smiles_via_mol2,
)
from tmol.ligand.params_file import inject_params_file
from tmol.ligand.params_io import write_params_from_mol2
from tmol.ligand.preparation import (
    LigandPreparationError,
    prepare_ligand_from_cif,
    prepare_ligand_from_mol2,
    prepare_ligand_from_smiles,
    prepare_ligands,
    prepare_single_ligand,
)
from tmol.ligand.registry import LigandPreparation, clear_cache
from tmol.ligand.structure_to_smiles import (
    ligand_smiles_candidates_from_atom_array,
    ligand_smiles_from_atom_array,
)

__all__ = [
    "LigandPreparation",
    "LigandPreparationError",
    "NonStandardResidueInfo",
    "RawResidueType",
    "clear_cache",
    "detect_nonstandard_residues",
    "inject_params_file",
    "ligand_smiles_candidates_from_atom_array",
    "ligand_smiles_from_atom_array",
    "nonstandard_residue_info_from_smiles_via_mol2",
    "prepare_ligand_from_cif",
    "prepare_ligand_from_mol2",
    "prepare_ligand_from_smiles",
    "prepare_ligands",
    "prepare_single_ligand",
    "write_params_from_mol2",
]
