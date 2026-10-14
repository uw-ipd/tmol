import importlib
import subprocess
import sys

import pytest

# The public API: the classes and functions that each package's __init__.py must
# provide. Anything else an __init__.py defines is also available from it, but
# these names are the supported interface.
API = {
    "tmol": [
        "has_cuda",
        "__version__",
        "switchable_view",
        "view",
        "selection_gallery",
        "extended_pose_stack_from_sequences",
        "fast_relax",
        "CanonicalOrdering",
        "beta2016_score_function",
        "ParameterDatabase",
        "PatchedChemicalDatabase",
        "pose_stack_from_biotite",
        "biotite_from_pose_stack",
        "build_context_from_biotite",
        "atom_array_from_file",
        "atom_array_from_cif",
        "pose_stack_from_atom37",
        "atom37_slot_map_for_ordering",
        "pose_stack_from_atom37_and_topology",
        "prepare_atom37_pose_builder",
        "CanonicalForm",
        "pose_stack_from_canonical_form",
        "canonical_form_from_pose_stack",
        "create_pose_stack_from_sequences",
        "canonical_form_from_atom37",
        "pose_stack_from_canonical_aa_atom37",
        "atomworks_from_pose_stack",
        "canonical_form_from_biotite",
        "canonical_ordering_for_biotite",
        "packed_block_types_for_biotite",
        "pose_stack_from_canonical_form_and_context",
        "PoseBuildContext",
        "FoldForest",
        "MoveMap",
        "CartesianMoveMap",
        "EdgeType",
        "LigandFragmentDefinition",
        "recombine_fragmented_ligands",
        "run_cart_min",
        "run_kin_min",
        "PackerPalette",
        "PackerTask",
        "pack_rotamers",
        "PoseStackBuilder",
        "ConstraintSet",
        "PackedBlockTypes",
        "PoseStack",
        "PDBInfo",
        "ScoreFunction",
        "ScoreType",
        "calculate_fragment_interactions",
    ],
    "tmol.chemical": [
        "ResidueTypeSet",
        "RefinedResidueType",
    ],
    "tmol.database.scoring": [
        "CartBondedDatabase",
        "ElecDatabase",
        "GenBondedDatabase",
        "HBondDatabase",
        "LJLKDatabase",
        "MetalCoordinationDatabase",
        "NaTorsionDatabase",
        "OmegaBBDepDatabase",
        "RamaDatabase",
        "DunbrackRotamerLibrary",
    ],
    "tmol.io": [
        "pose_stack_from_pdb",
        "default_packed_block_types",
        "default_canonical_ordering",
        "write_pose_stack_pdb",
        "atom_records_from_pose_stack",
        "HisSpecialCaseIndices",
        "his_taut_variant_NE2_protonated",
        "his_taut_variant_ND1_protonated",
        "his_taut_variant_both_ND1_and_NE2_protonated",
    ],
    "tmol.kinematics": [
        "set_named_torsions",
        "JumpDOFTypes",
        "NodeType",
        "PoseStackKinematicsModule",
        "inverseKin",
    ],
    "tmol.ligand": [
        "load_params_file",
        "FRAGMENT_ID_ANNOTATION",
        "inject_params_file",
        "nonstandard_residue_info_from_mol2",
        "prepare_single_ligand",
        "write_params_file",
    ],
    "tmol.numeric": [
        "coord_dihedrals",
    ],
    "tmol.ops": [
        "res_mask_to_coord_mask",
        "compute_block_adjacency",
        "compute_block_centroids_and_furthest_dist",
        "build_sidechain_coord_mask",
        "build_coord_mask_for_mask_and_interacting_atoms",
        "calculate_block_pair_ddg",
    ],
    "tmol.pack": [
        "SetPackerTask",
        "build_rotamers",
        "FixedAAChiSampler",
        "IncludeCurrentSampler",
        "NaChiRotamerSampler",
        "ConjugatedSCSampler",
        "OptHSampler",
        "RotamerSet",
        "FallbackSampler",
        "ChiSampler",
        "SidechainSampler",
        "create_dunbrack_sampler_from_database",
        "DunbrackChiSampler",
    ],
    "tmol.pose": [
        "get_named_torsions",
    ],
    "tmol.score": [
        "BackboneTorsionEnergyTerm",
        "CartBondedEnergyTerm",
        "create_mainchain_coordinate_constraints",
        "ConstraintEnergyTerm",
        "DisulfideEnergyTerm",
        "DunbrackEnergyTerm",
        "ElecEnergyTerm",
        "GenBondedEnergyTerm",
        "HBondEnergyTerm",
        "LJLKEnergyTerm",
        "LKBallEnergyTerm",
        "NaTorsionEnergyTerm",
        "RefEnergyTerm",
        "ScoreTermFactory",
    ],
    "tmol.score.na_torsion": [
        "NORTH_PUCKERS",
        "SYN_RANGE",
        "pucker_weights",
        "sugar_ring_atoms",
    ],
    "tmol.utility": [
        "resolve_device",
    ],
}

# Packages whose __init__.py declares an explicit __all__.
PUBLIC_PACKAGES = (
    "tmol",
    "tmol.chemical",
    "tmol.io",
    "tmol.kinematics",
    "tmol.ligand",
    "tmol.numeric",
    "tmol.ops",
    "tmol.pack",
    "tmol.pose",
    "tmol.score",
    "tmol.score.na_torsion",
    "tmol.utility",
)


@pytest.mark.parametrize(
    "module_name,name", [(m, n) for m, names in API.items() for n in names]
)
def test_api_name_is_available(module_name, name):
    module = importlib.import_module(module_name)
    assert hasattr(module, name)


@pytest.mark.parametrize("module_name", PUBLIC_PACKAGES)
def test_public_package_all_is_explicit_and_resolvable(module_name):
    module = importlib.import_module(module_name)

    exported = module.__all__
    assert isinstance(exported, list)
    assert len(exported) == len(set(exported))
    assert all(not name.startswith("_") for name in exported)
    assert all(hasattr(module, name) for name in exported)
    assert {n for n in API.get(module_name, []) if not n.startswith("_")} <= set(
        exported
    )


def test_ligand_all_excludes_vendored_and_internal_helpers():
    from tmol import ligand

    assert "ArgParseFuncs" not in ligand.__all__
    assert "Protonate" not in ligand.__all__
    assert "main" not in ligand.__all__
    assert "print_header" not in ligand.__all__


def test_documented_fragment_interaction_helper_is_public():
    import tmol

    assert "calculate_fragment_interactions" in tmol.__all__


def test_top_level_names_are_exported_only_from_tmol():
    import tmol

    top_level = {n for n in API["tmol"] if not n.startswith("_")}
    assert set(tmol.__all__) == top_level | {"include_paths"}
    for name in top_level:
        assert hasattr(tmol, name)


def test_extern_import_is_independent_of_vendored_tests():
    # Check a fresh process: pytest may already have collected the vendored tests.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib, sys, unittest
import tmol.extern
assert 'tmol.extern.test_toposort' not in sys.modules
assert list(tmol.extern.toposort({'a': set()})) == [{'a'}]
tests = importlib.import_module('tmol.extern.test_toposort')
result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromModule(tests))
raise SystemExit(not result.wasSuccessful())
""",
        ],
        check=True,
    )
