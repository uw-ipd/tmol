"""Generate requested D libraries without inferring coverage from table names."""

import attr
import pytest

from tmol.database.scoring._dunbrack_libraries import DunMappingParams
from tmol.database.scoring._mirrored_dunbrack import with_mirrored_libraries


@pytest.fixture
def original_libraries(default_database):
    database = default_database.scoring.dun
    rot = tuple(
        lib
        for lib in database.rotameric_libraries
        if not lib.rotameric_data.backbone_is_mirrored
    )
    semi = tuple(
        lib
        for lib in database.semi_rotameric_libraries
        if not lib.rotameric_data.backbone_is_mirrored
    )
    names = {lib.table_name for lib in (*rot, *semi)}
    return attr.evolve(
        database,
        rotameric_libraries=rot,
        semi_rotameric_libraries=semi,
        dun_lookup=tuple(
            row for row in database.dun_lookup if row.dun_table_name in names
        ),
    )


def tables(database):
    return {
        lib.table_name: lib
        for lib in (*database.rotameric_libraries, *database.semi_rotameric_libraries)
    }


def lookup(database):
    names = [row.residue_name for row in database.dun_lookup]
    assert len(set(names)) == len(
        names
    ), "A residue received competing library mappings"
    return {row.residue_name: row.dun_table_name for row in database.dun_lookup}


@pytest.mark.parametrize("requested", [{}, {"ALA": "DALA"}])
def test_no_required_library_is_a_no_op(original_libraries, requested):
    assert with_mirrored_libraries(original_libraries, requested) is original_libraries


def test_generate_only_the_requested_library(original_libraries):
    result = with_mirrored_libraries(original_libraries, {"ARG": "DARG"})
    before, after = tables(original_libraries), tables(result)
    assert set(after) - set(before) == {lookup(result)["DARG"]}
    assert all(after[name] is lib for name, lib in before.items())
    assert after[lookup(result)["DARG"]].rotameric_data.backbone_is_mirrored
    assert with_mirrored_libraries(result, {"ARG": "DARG"}) is result


def test_original_table_name_starting_with_d_does_not_disable_generation(
    original_libraries,
):
    old_name = lookup(original_libraries)["ARG"]
    renamed = attr.evolve(
        original_libraries,
        rotameric_libraries=tuple(
            (
                attr.evolve(lib, table_name="different_original")
                if lib.table_name == old_name
                else lib
            )
            for lib in original_libraries.rotameric_libraries
        ),
        dun_lookup=tuple(
            (
                attr.evolve(row, dun_table_name="different_original")
                if row.dun_table_name == old_name
                else row
            )
            for row in original_libraries.dun_lookup
        ),
    )
    result = with_mirrored_libraries(renamed, {"ARG": "DARG"})
    assert lookup(result)["DARG"] == "ddifferent_original"


def test_incremental_generation_preserves_existing_libraries(original_libraries):
    first = with_mirrored_libraries(original_libraries, {"ARG": "DARG"})
    second = with_mirrored_libraries(first, {"ARG": "DARG", "PHE": "DPHE"})
    before, after = tables(first), tables(second)
    assert set(after) - set(before) == {lookup(second)["DPHE"]}
    assert all(after[name] is lib for name, lib in before.items())
    assert "DPHE" not in lookup(first)


def test_explicit_target_mapping_is_authoritative(original_libraries):
    chosen = lookup(original_libraries)["LEU"]
    source = attr.evolve(
        original_libraries,
        dun_lookup=(*original_libraries.dun_lookup, DunMappingParams(chosen, "DARG")),
    )
    result = with_mirrored_libraries(source, {"ARG": "DARG", "PHE": "DPHE"})
    assert lookup(result)["DARG"] == chosen
    assert set(tables(result)) - set(tables(source)) == {lookup(result)["DPHE"]}


def test_generated_name_collision_is_explicit(original_libraries):
    name = "d" + lookup(original_libraries)["ARG"]
    colliding = attr.evolve(original_libraries.rotameric_libraries[0], table_name=name)
    source = attr.evolve(
        original_libraries,
        rotameric_libraries=(*original_libraries.rotameric_libraries, colliding),
    )
    with pytest.raises(ValueError, match="collision"):
        with_mirrored_libraries(source, {"ARG": "DARG"})


def test_missing_source_table_fails_at_generation(original_libraries):
    source = attr.evolve(
        original_libraries,
        dun_lookup=(
            *original_libraries.dun_lookup,
            DunMappingParams("absent", "PRIVATE"),
        ),
    )
    with pytest.raises(ValueError, match="absent"):
        with_mirrored_libraries(source, {"PRIVATE": "DPRIVATE"})


def test_selective_libraries_preserve_native_sampling(
    original_libraries, default_database, torch_device
):
    import torch
    from tmol.pack.rotamer.dunbrack import DunbrackChiSampler
    from tmol.score.dunbrack import DunbrackParamResolver
    from tmol.tests.pack.rotamer.dunbrack.test_dunbrack_chi_sampler import (
        _table_indices,
    )

    selective = with_mirrored_libraries(original_libraries, {"ARG": "DARG"})

    def integer(values):
        return torch.tensor(values, dtype=torch.int32, device=torch_device)

    def sample(library):
        resolver = DunbrackParamResolver.from_database(library, torch_device)
        indices = _table_indices(resolver, ("ARG", "DARG", "PHE"), torch_device)
        # PHE moves to a different table index when unused D rotameric tables
        # disappear. Both families and the retained reflected table must work.
        return DunbrackChiSampler(resolver).launch_rotamer_building(
            torch.zeros((1, 3), device=torch_device),
            integer([2]),
            integer([0]),
            torch.full((2, 4), -1, dtype=torch.int32, device=torch_device),
            integer([[0, table] for table in indices]),
            torch.zeros((3, 4), dtype=torch.int32, device=torch_device),
            torch.zeros((3, 4, 1), device=torch_device),
            torch.zeros((3, 4), dtype=torch.int32, device=torch_device),
            torch.full((3,), 0.98, device=torch_device),
            integer([4, 4, 2]),
        )

    before, after = sample(default_database.scoring.dun), sample(selective)
    for first, second in zip(before, after):
        torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_conflicting_target_requests_are_rejected(original_libraries):
    with pytest.raises(ValueError, match="Conflicting"):
        with_mirrored_libraries(
            original_libraries, {"ARG": "PRIVATE", "PHE": "PRIVATE"}
        )
