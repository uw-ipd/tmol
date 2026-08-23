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
