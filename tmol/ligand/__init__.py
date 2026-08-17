_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "HYB_SP": ("atom_typing", "HYB_SP"),
    "HYB_SP2": ("atom_typing", "HYB_SP2"),
    "HYB_SP3": ("atom_typing", "HYB_SP3"),
    "HYB_AMIDE": ("atom_typing", "HYB_AMIDE"),
    "HYB_AROMATIC": ("atom_typing", "HYB_AROMATIC"),
    "ELEMENT_SYMBOLS": ("atom_typing", "ELEMENT_SYMBOLS"),
    "AtomTypeAssignment": ("atom_typing", "AtomTypeAssignment"),
    "RosettaTypingState": ("atom_typing", "RosettaTypingState"),
    "sanitize_tolerant": ("atom_typing", "sanitize_tolerant"),
    "kekulize_tolerant": ("atom_typing", "kekulize_tolerant"),
    "should_kekulize_for_typing": ("atom_typing", "should_kekulize_for_typing"),
    "assign_tmol_atom_types": ("atom_typing", "assign_tmol_atom_types"),
    "get_hbond_properties": ("chemistry_tables", "get_hbond_properties"),
    "get_polar_classes": ("chemistry_tables", "get_polar_classes"),
    "get_sp2_atom_types": ("chemistry_tables", "get_sp2_atom_types"),
    "MAX_CONFS": ("chi_topology", "MAX_CONFS"),
    "build_chi_topology": ("chi_topology", "build_chi_topology"),
    "W_EXACT": ("conformer_generation", "W_EXACT"),
    "W_BOUND": ("conformer_generation", "W_BOUND"),
    "W_PLANE": ("conformer_generation", "W_PLANE"),
    "W_CHIRAL": ("conformer_generation", "W_CHIRAL"),
    "CHIRAL_MARGIN": ("conformer_generation", "CHIRAL_MARGIN"),
    "REFINE_ITERS": ("conformer_generation", "REFINE_ITERS"),
    "STAGE_A_ITERS": ("conformer_generation", "STAGE_A_ITERS"),
    "STAGE_A_ANNEAL": ("conformer_generation", "STAGE_A_ANNEAL"),
    "N_RESTART": ("conformer_generation", "N_RESTART"),
    "TORCH_DTYPE": ("conformer_generation", "TORCH_DTYPE"),
    "generate_conformer": ("conformer_generation", "generate_conformer"),
    "SKIP_RESIDUES": ("detect", "SKIP_RESIDUES"),
    "get_chem_comp_type": ("detect", "get_chem_comp_type"),
    "nonstandard_residue_info_from_mol2": (
        "detect",
        "nonstandard_residue_info_from_mol2",
    ),
    "nonstandard_residue_info_from_mol2_block": (
        "detect",
        "nonstandard_residue_info_from_mol2_block",
    ),
    "print_header": ("dimorphite_dl", "print_header"),
    "main": ("dimorphite_dl", "main"),
    "MyParser": ("dimorphite_dl", "MyParser"),
    "ArgParseFuncs": ("dimorphite_dl", "ArgParseFuncs"),
    "UtilFuncs": ("dimorphite_dl", "UtilFuncs"),
    "LoadSMIFile": ("dimorphite_dl", "LoadSMIFile"),
    "Protonate": ("dimorphite_dl", "Protonate"),
    "ProtSubstructFuncs": ("dimorphite_dl", "ProtSubstructFuncs"),
    "ProtectUnprotectFuncs": ("dimorphite_dl", "ProtectUnprotectFuncs"),
    "TestFuncs": ("dimorphite_dl", "TestFuncs"),
    "protonate_mol_variants": ("dimorphite_dl", "protonate_mol_variants"),
    "MAX_FRAGMENT_CONNECTIONS": ("fragmentation", "MAX_FRAGMENT_CONNECTIONS"),
    "MIN_FRAGMENT_HEAVY_ATOMS": ("fragmentation", "MIN_FRAGMENT_HEAVY_ATOMS"),
    "FragmentConnection": ("fragmentation", "FragmentConnection"),
    "LigandFragmentDefinition": ("fragmentation", "LigandFragmentDefinition"),
    "fragment_ids_from_atom_array": ("fragmentation", "fragment_ids_from_atom_array"),
    "build_ligand_fragment_definition": (
        "fragmentation",
        "build_ligand_fragment_definition",
    ),
    "expand_fragmented_ligands": ("fragmentation", "expand_fragmented_ligands"),
    "apply_fragment_connections": ("fragmentation", "apply_fragment_connections"),
    "logger": ("generated_geometry", "logger"),
    "planarize_conjugated_nh2": ("generated_geometry", "planarize_conjugated_nh2"),
    "correct_generated_geometry": ("generated_geometry", "correct_generated_geometry"),
    "disambiguate_mol2_atom_name": ("mol2_names", "disambiguate_mol2_atom_name"),
    "apply_disambiguated_mol2_names": ("mol2_names", "apply_disambiguated_mol2_names"),
    "authoritative_charges_by_index": ("mol3d", "authoritative_charges_by_index"),
    "OpenBabelUnavailableError": ("openbabel_compat", "OpenBabelUnavailableError"),
    "strip_nontetrahedral_stereo": ("openbabel_compat", "strip_nontetrahedral_stereo"),
    "normalize_azide": ("openbabel_compat", "normalize_azide"),
    "source_atom_order_from_mapped_smiles": (
        "openbabel_compat",
        "source_atom_order_from_mapped_smiles",
    ),
    "obabel_read_mol2": ("openbabel_compat", "obabel_read_mol2"),
    "obabel_read_mol2_block": ("openbabel_compat", "obabel_read_mol2_block"),
    "obabel_smiles_to_mol2_block": ("openbabel_compat", "obabel_smiles_to_mol2_block"),
    "obabel_smiles_to_mol2": ("openbabel_compat", "obabel_smiles_to_mol2"),
    "TMOL_FORMAT_VERSION": ("params_file", "TMOL_FORMAT_VERSION"),
    "load_params_file": ("params_file", "load_params_file"),
    "inject_params_files": ("params_file", "inject_params_files"),
    "read_params_file": ("params_io", "read_params_file"),
    "write_params_file": ("params_io", "write_params_file"),
    "unused_ligand_name": ("preparation", "unused_ligand_name"),
    "normalize_non_ring_aromatic_bonds": (
        "rdkit_mol",
        "normalize_non_ring_aromatic_bonds",
    ),
    "normalize_cumulated_azide": ("rdkit_mol", "normalize_cumulated_azide"),
    "source_subtype": ("rdkit_mol", "source_subtype"),
    "source_carried_kekule": ("rdkit_mol", "source_carried_kekule"),
    "source_has_aromatic_annotations": ("rdkit_mol", "source_has_aromatic_annotations"),
    "rdkit_mol_from_ligand_atom_array": (
        "rdkit_mol",
        "rdkit_mol_from_ligand_atom_array",
    ),
    "ligand_atom_array_to_rdkit_mol": ("rdkit_mol", "ligand_atom_array_to_rdkit_mol"),
    "collect_new_atom_types": ("registry", "collect_new_atom_types"),
    "inject_ligand_preparations": ("registry", "inject_ligand_preparations"),
    "rebuild_canonical_ordering": ("registry", "rebuild_canonical_ordering"),
    "build_residue_type": ("residue_builder", "build_residue_type"),
    "apply_geometry_bond_corrections": (
        "structure_to_smiles",
        "apply_geometry_bond_corrections",
    ),
    "_METAL_SYMBOLS": ("detect", "_METAL_SYMBOLS"),
    "_build_cartbonded_params": ("registry", "_build_cartbonded_params"),
    "_build_rosetta_typing_state": ("atom_typing", "_build_rosetta_typing_state"),
    "_classify_N": ("atom_typing", "_classify_N"),
    "_classify_N_sp2": ("atom_typing", "_classify_N_sp2"),
    "_classify_O": ("atom_typing", "_classify_O"),
    "_classify_P": ("atom_typing", "_classify_P"),
    "_classify_S": ("atom_typing", "_classify_S"),
    "_correct_amide_bond_orders": ("atom_typing", "_correct_amide_bond_orders"),
    "_correct_conjugated_single_bond_orders": (
        "atom_typing",
        "_correct_conjugated_single_bond_orders",
    ),
    "_correct_ring_nitrogen": ("atom_typing", "_correct_ring_nitrogen"),
    "_get_hyb": ("atom_typing", "_get_hyb"),
    "_modify_polar_c": ("atom_typing", "_modify_polar_c"),
    "_prepare_ligand_via_smiles": ("preparation", "_prepare_ligand_via_smiles"),
    "_residue_names_with_cross_residue_bonds": (
        "detect",
        "_residue_names_with_cross_residue_bonds",
    ),
    "_strip_metals": ("detect", "_strip_metals"),
    "FRAGMENT_ID_ANNOTATION": ("fragmentation", "FRAGMENT_ID_ANNOTATION"),
    "FragmentedLigandPoseMapping": ("fragmentation", "FragmentedLigandPoseMapping"),
    "LigandPreparation": ("registry", "LigandPreparation"),
    "LigandPreparationError": ("preparation", "LigandPreparationError"),
    "NonStandardResidueInfo": ("detect", "NonStandardResidueInfo"),
    "_BOND_TOK_TO_TYPE": ("params_io", "_BOND_TOK_TO_TYPE"),
    "_assign_missing_hybridization": ("atom_typing", "_assign_missing_hybridization"),
    "_bond_is_planar": ("atom_typing", "_bond_is_planar"),
    "_charge_model_is_authoritative": ("detect", "_charge_model_is_authoritative"),
    "_classify_H": ("atom_typing", "_classify_H"),
    "_classify_O_no_carbon": ("atom_typing", "_classify_O_no_carbon"),
    "_classify_O_sp2": ("atom_typing", "_classify_O_sp2"),
    "_dimorphite_protonate_smiles": ("detect", "_dimorphite_protonate_smiles"),
    "_fragment_atom_tree": ("fragmentation", "_fragment_atom_tree"),
    "_has_sp2_oxygen_neighbor": ("atom_typing", "_has_sp2_oxygen_neighbor"),
    "_import_openbabel": ("openbabel_compat", "_import_openbabel"),
    "_infer_res_name_from_mol2": ("detect", "_infer_res_name_from_mol2"),
    "_ligand_info_from_cif": ("preparation", "_ligand_info_from_cif"),
    "_mol2_charge_model_from_text": ("detect", "_mol2_charge_model_from_text"),
    "_mol2_single_bond_ids": ("detect", "_mol2_single_bond_ids"),
    "_neighbor_counts": ("atom_typing", "_neighbor_counts"),
    "_normalize_radical_oxygens": ("detect", "_normalize_radical_oxygens"),
    "_obmol_to_rdkit_mol": ("openbabel_compat", "_obmol_to_rdkit_mol"),
    "_rdkit_bond_to_biotite_type": ("detect", "_rdkit_bond_to_biotite_type"),
    "_residue_covers_cif_heavy_atoms": (
        "preparation",
        "_residue_covers_cif_heavy_atoms",
    ),
    "_source_subtype_from_mol2_atom_type": (
        "detect",
        "_source_subtype_from_mol2_atom_type",
    ),
    "_validate_bonded_cut_layout": ("fragmentation", "_validate_bonded_cut_layout"),
    "_validate_scoring_cut_layout": ("fragmentation", "_validate_scoring_cut_layout"),
    "detect_nonstandard_residues": ("detect", "detect_nonstandard_residues"),
    "inject_params_file": ("params_file", "inject_params_file"),
    "ligand_smiles_from_atom_array": (
        "structure_to_smiles",
        "ligand_smiles_from_atom_array",
    ),
    "nonstandard_residue_info_from_smiles_via_mol2": (
        "detect",
        "nonstandard_residue_info_from_smiles_via_mol2",
    ),
    "prepare_ligand_from_cif": ("preparation", "prepare_ligand_from_cif"),
    "prepare_ligand_from_mol2": ("preparation", "prepare_ligand_from_mol2"),
    "prepare_ligand_from_smiles": ("preparation", "prepare_ligand_from_smiles"),
    "prepare_ligands": ("preparation", "prepare_ligands"),
    "prepare_ligands_from_smiles": ("preparation", "prepare_ligands_from_smiles"),
    "prepare_single_ligand": ("preparation", "prepare_single_ligand"),
    "recombine_fragmented_ligands": ("fragmentation", "recombine_fragmented_ligands"),
    "write_params_from_mol2": ("params_io", "write_params_from_mol2"),
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        import importlib

        mod_leaf, attr = _LAZY_ATTRS[name]
        mod = importlib.import_module(f".{mod_leaf}", package=__name__)
        # Re-cache every name from this module so that Python's import
        # machinery (which sets globals()[mod_leaf] = MODULE as a side-effect)
        # does not overwrite previously resolved function/class references.
        for _n, (_m, _a) in _LAZY_ATTRS.items():
            if _m == mod_leaf:
                try:
                    globals()[_n] = getattr(mod, _a)
                except AttributeError:
                    pass
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


"""Public API for tmol ligand preparation.

Stable entry points for the unified CIF/AtomArray/SMILES/mol2 → params pipeline.
"""

from tmol.database.chemical import RawResidueType  # noqa: F401, E402  re-exported
from tmol.ligand.detect import (  # noqa: E402
    NonStandardResidueInfo,
    detect_nonstandard_residues,
    nonstandard_residue_info_from_smiles_via_mol2,
)

from tmol.ligand.params_file import inject_params_file  # noqa: E402
from tmol.ligand.params_io import write_params_from_mol2  # noqa: E402
from tmol.ligand.fragmentation import (  # noqa: E402
    FRAGMENT_ID_ANNOTATION,
    FragmentedLigandPoseMapping,
    LigandFragmentBlockMapping,
    recombine_fragmented_ligands,
)
from tmol.ligand.preparation import (  # noqa: E402
    LigandPreparationError,
    prepare_ligand_from_cif,
    prepare_ligand_from_mol2,
    prepare_ligand_from_smiles,
    prepare_ligands,
    prepare_ligands_from_smiles,
    prepare_single_ligand,
)
from tmol.ligand.registry import LigandPreparation  # noqa: E402
from tmol.ligand.structure_to_smiles import (  # noqa: E402
    ligand_smiles_from_atom_array,
)

__all__ = [
    "LigandPreparation",
    "LigandPreparationError",
    "FRAGMENT_ID_ANNOTATION",
    "FragmentedLigandPoseMapping",
    "LigandFragmentBlockMapping",
    "NonStandardResidueInfo",
    "RawResidueType",
    "detect_nonstandard_residues",
    "inject_params_file",
    "ligand_smiles_from_atom_array",
    "nonstandard_residue_info_from_smiles_via_mol2",
    "prepare_ligand_from_cif",
    "prepare_ligand_from_mol2",
    "prepare_ligand_from_smiles",
    "prepare_ligands",
    "prepare_ligands_from_smiles",
    "prepare_single_ligand",
    "recombine_fragmented_ligands",
    "write_params_from_mol2",
]
