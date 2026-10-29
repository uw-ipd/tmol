import importlib

import pytest

PUBLIC_PACKAGES = (
    "tmol",
    "tmol.chemical",
    "tmol.database",
    "tmol.io",
    "tmol.kinematics",
    "tmol.ligand",
    "tmol.numeric",
    "tmol.ops",
    "tmol.optimization",
    "tmol.pack",
    "tmol.pack.rotamer",
    "tmol.pack.rotamer.dunbrack",
    "tmol.pose",
    "tmol.relax",
    "tmol.score",
    "tmol.score.constraint",
    "tmol.score.na_torsion",
    "tmol.score.terms",
    "tmol.types",
    "tmol.utility",
)


@pytest.mark.parametrize("module_name", PUBLIC_PACKAGES)
def test_public_package_all_is_explicit_and_resolvable(module_name):
    module = importlib.import_module(module_name)

    exported = module.__all__
    assert isinstance(exported, list)
    assert len(exported) == len(set(exported))
    assert all(not name.startswith("_") for name in exported)
    assert all(hasattr(module, name) for name in exported)


def test_ligand_all_excludes_vendored_and_internal_helpers():
    from tmol import ligand

    assert "ArgParseFuncs" not in ligand.__all__
    assert "Protonate" not in ligand.__all__
    assert "main" not in ligand.__all__
    assert "print_header" not in ligand.__all__


def test_ligand_all_contains_user_facing_names():
    from tmol import ligand

    expected = {
        # errors, return types and format constant users need to name
        "LigandPreparationError",
        "NonStandardResidueInfo",
        "TMOL_FORMAT_VERSION",
        # SMILES helpers paired with the public pose-level ones
        "smiles_from_block_type",
        "smiles_from_pose_stack",
        "ligand_smiles_from_atom_array",
        "nonstandard_residue_info_from_mol2",
        "nonstandard_residue_info_from_smiles_via_mol2",
    }
    assert expected <= set(ligand.__all__)
    for name in expected:
        assert getattr(ligand, name) is not None
    assert "LigandPreparation" not in ligand.__all__


def test_documented_fragment_interaction_helper_is_public():
    import tmol

    # tutorial_api_usage.csv makes it a top-level API name.
    assert "calculate_fragment_interactions" in tmol.__all__


def test_default_relax_minimizers_are_public():
    from tmol import relax
    from tmol.relax._fast_relax import (
        DefaultCartesianMinimizer,
        DefaultKinematicMinimizer,
    )

    assert relax.DefaultCartesianMinimizer is DefaultCartesianMinimizer
    assert relax.DefaultKinematicMinimizer is DefaultKinematicMinimizer
    assert {"DefaultCartesianMinimizer", "DefaultKinematicMinimizer"} <= set(
        relax.__all__
    )
