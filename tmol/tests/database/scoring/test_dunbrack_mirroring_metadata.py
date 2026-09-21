"""Reflection semantics belong to library data and survive persistence/renaming."""

from pathlib import Path

import attr
import pytest
import torch

from tmol.database.scoring._dunbrack_libraries import (
    DunbrackRotamerLibrary,
    DunMappingParams,
)
from tmol.database.scoring._mirrored_dunbrack import mirror_rotameric_data
from tmol.score.dunbrack import DunbrackParamResolver

LEGACY_BINARY = Path(__file__).parents[3] / "database/default/scoring/dunbrack.bin"


def test_legacy_binary_defaults_to_original_backbone_orientation():
    database = DunbrackRotamerLibrary.from_file(LEGACY_BINARY)
    libraries = (*database.rotameric_libraries, *database.semi_rotameric_libraries)
    assert libraries
    assert all(lib.rotameric_data.backbone_is_mirrored is False for lib in libraries)
    assert all(lib.rotameric_data.backbone_source_start is None for lib in libraries)


@pytest.mark.parametrize("semi", [False, True])
@pytest.mark.parametrize("custom_origin", [False, True])
def test_double_reflection_restores_all_library_data(
    default_database, semi, custom_origin
):
    db = default_database.scoring.dun
    library = (db.semi_rotameric_libraries if semi else db.rotameric_libraries)[0]
    original = library.rotameric_data
    if custom_origin:
        original = attr.evolve(
            original,
            backbone_dihedral_start=original.backbone_dihedral_start
            + torch.tensor([1.25, -2.0]),
        )
    mirrored = mirror_rotameric_data(original)
    restored = mirror_rotameric_data(mirrored)
    assert mirrored.backbone_is_mirrored is True
    assert restored.backbone_is_mirrored is False
    for field in attr.fields(type(original)):
        if field.name == "backbone_is_mirrored":
            continue
        torch.testing.assert_close(
            getattr(restored, field.name), getattr(original, field.name), rtol=0, atol=0
        )


@pytest.mark.parametrize("custom_origin", [False, True])
def test_reflection_follows_renamed_libraries_through_serialization(
    default_database, tmp_path, torch_device, custom_origin
):
    db = default_database.scoring.dun
    ordinary = db.rotameric_libraries[0]
    if custom_origin:
        ordinary = attr.evolve(
            ordinary,
            rotameric_data=attr.evolve(
                ordinary.rotameric_data,
                backbone_dihedral_start=ordinary.rotameric_data.backbone_dihedral_start
                + torch.tensor([1.25, -2.0]),
            ),
        )
    reflected = attr.evolve(
        ordinary,
        table_name="private_rotamers",
        rotameric_data=mirror_rotameric_data(ordinary.rotameric_data),
    )
    # An unmirrored name beginning with d must not imply reflection either.
    original = attr.evolve(ordinary, table_name="different_original")
    semi = db.semi_rotameric_libraries[0]
    subset = DunbrackRotamerLibrary(
        dun_lookup=(
            DunMappingParams("private_rotamers", "DARG"),
            DunMappingParams("different_original", "ARG"),
            DunMappingParams(semi.table_name, "PHE"),
        ),
        rotameric_libraries=(reflected, original),
        semi_rotameric_libraries=(semi,),
    )
    path = tmp_path / "libraries.bin"
    torch.save(subset, path)
    loaded = DunbrackRotamerLibrary.from_file(path)
    assert [
        lib.rotameric_data.backbone_is_mirrored for lib in loaded.rotameric_libraries
    ] == [True, False]
    torch.testing.assert_close(
        loaded.rotameric_libraries[0].rotameric_data.backbone_source_start,
        ordinary.rotameric_data.backbone_dihedral_start,
        rtol=0,
        atol=0,
    )
    resolver = DunbrackParamResolver.from_database(loaded, torch_device)
    assert resolver.sampling_db.rotameric_bb_is_mirrored.tolist() == [
        True,
        False,
        False,
    ]
    starts = resolver.sampling_db.rotameric_bb_source_start
    torch.testing.assert_close(starts[0], starts[1], rtol=0, atol=0)
    torch.testing.assert_close(
        starts[0],
        ordinary.rotameric_data.backbone_dihedral_start.to(torch_device)
        * (torch.pi / 180),
        rtol=0,
        atol=0,
    )


def test_mirrored_mean_coefficients_use_the_reflected_angular_branch(default_database):
    # Exercise a mean crossing +120 degrees: wrapping both orientations toward
    # +180 adds a full turn to only some reflected grid points before fitting.
    ordinary = default_database.scoring.dun.rotameric_libraries[0]
    means = ordinary.rotameric_data.rotamer_means.clone()
    means[:, :, ::2, :] = 110
    means[:, :, 1::2, :] = 130
    source = attr.evolve(
        ordinary,
        rotameric_data=attr.evolve(ordinary.rotameric_data, rotamer_means=means),
    )
    mirrored = attr.evolve(
        source, rotameric_data=mirror_rotameric_data(source.rotameric_data)
    )
    left, _, _ = DunbrackParamResolver._calculate_rot_mean_coeffs(
        [source], torch.device("cpu")
    )
    right, _, _ = DunbrackParamResolver._calculate_rot_mean_coeffs(
        [mirrored], torch.device("cpu")
    )
    # Independent periodic index permutation for the default -180/10 grids.
    rows = (-torch.arange(left.shape[1])) % left.shape[1]
    columns = (-torch.arange(left.shape[2])) % left.shape[2]
    torch.testing.assert_close(
        right, -left[:, rows][:, :, columns], rtol=1e-6, atol=2e-6
    )
