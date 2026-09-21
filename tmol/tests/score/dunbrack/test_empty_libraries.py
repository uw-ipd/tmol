"""Private chemistry databases need not carry both Dunbrack library families."""

import attr
import pytest
import torch

from tmol.io import pose_stack_from_pdb
from tmol.score.dunbrack import DunbrackEnergyTerm, DunbrackParamResolver
from tmol.tests.score.dunbrack.test_parameter_identity import setup, render, evaluate


def select_libraries(database, kind):
    selected_residue = {"rotameric": "LEU", "semirotameric": "PHE", "empty": None}[kind]
    names = {
        row.dun_table_name
        for row in database.dun_lookup
        if row.residue_name == selected_residue
    }
    return attr.evolve(
        database,
        dun_lookup=tuple(
            row for row in database.dun_lookup if row.dun_table_name in names
        ),
        rotameric_libraries=tuple(
            lib for lib in database.rotameric_libraries if lib.table_name in names
        ),
        semi_rotameric_libraries=tuple(
            lib for lib in database.semi_rotameric_libraries if lib.table_name in names
        ),
    )


@pytest.mark.parametrize("kind", ["rotameric", "semirotameric", "empty"])
@pytest.mark.parametrize("block_pairs", [False, True])
def test_empty_families_preserve_scores_and_gradients(
    default_database, ubq_pdb, torch_device, kind, block_pairs
):
    full = default_database.scoring.dun
    selected = select_libraries(full, kind)
    reference = attr.evolve(full, dun_lookup=selected.dun_lookup)
    pose = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=12)
    results = []
    for library in (reference, selected):
        database = attr.evolve(
            default_database,
            scoring=attr.evolve(default_database.scoring, dun=library),
        )
        term = DunbrackEnergyTerm(database, torch_device)
        setup(term, pose)
        results.append(evaluate(render(term, pose, block_pairs), pose))
    for before, after in zip(*results):
        torch.testing.assert_close(before, after, atol=1e-10, rtol=0)
    if kind == "empty":
        assert all(torch.count_nonzero(value) == 0 for value in results[-1])
    else:
        assert torch.count_nonzero(results[-1][0]) > 0


def test_no_libraries_still_supports_explicit_chi_sampling(
    default_database, torch_device
):
    from tmol.pack.rotamer.dunbrack import DunbrackChiSampler

    database = select_libraries(default_database.scoring.dun, "empty")
    resolver = DunbrackParamResolver.from_database(database, torch_device)
    # Empty families have no phantom offset rows or table-sized payload.
    for view in (resolver.scoring_db, resolver.scoring_db_aux, resolver.sampling_db):
        for field in attr.fields(type(view)):
            tensor = getattr(view, field.name)
            assert tensor.numel() == 0, field.name

    def integer(values):
        return torch.tensor(values, dtype=torch.int32, device=torch_device)

    values = torch.tensor([[[-0.7, 0.3, 1.8]]], device=torch_device)
    counts, _, _, chi = DunbrackChiSampler(resolver).launch_rotamer_building(
        torch.zeros((1, 3), device=torch_device),
        integer([0]),
        integer([0]),
        torch.empty((0, 4), dtype=torch.int32, device=torch_device),
        integer([[0, -1]]),
        integer([[0]]),
        values,
        integer([[3]]),
        torch.ones(1, device=torch_device),
        integer([1]),
    )
    assert counts.tolist() == [3]
    torch.testing.assert_close(chi[:, 0], values.flatten(), atol=0, rtol=0)
