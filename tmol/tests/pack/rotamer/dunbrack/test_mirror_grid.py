"""Native reflected lookup agrees at every default grid point and adjacent ULP."""

import pytest
import torch

from tmol.score.dunbrack import DunbrackParamResolver
from tmol.tests.pack.rotamer.dunbrack.test_dunbrack_chi_sampler import get_compiled


@pytest.mark.parametrize("stage", ["probability", "chi"])
def test_reflected_grid_points_neighbors_and_missing_axes(
    default_database, torch_device, stage
):
    resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    data = resolver.sampling_db
    libraries = (
        *default_database.scoring.dun.rotameric_libraries,
        *default_database.scoring.dun.semi_rotameric_libraries,
    )
    table_by_name = {lib.table_name: i for i, lib in enumerate(libraries)}
    compiled = get_compiled()
    checked = 0
    for table, library in enumerate(libraries):
        if library.rotameric_data.backbone_is_mirrored:
            continue
        mirrored = table_by_name["d" + library.table_name]
        start = data.rotameric_bb_start[table]
        step = data.rotameric_bb_step[table]
        bins = library.rotameric_data.rotamer_probabilities.shape[1:3]
        axes = [
            start[i] + torch.arange(n + 1, device=torch_device) * step[i]
            for i, n in enumerate(bins)
        ]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 2)
        # Exercise exact boundaries plus the representable float immediately
        # on either side. Negate the completed input, so reflection is exact.
        points = torch.cat(
            (
                torch.nextafter(grid, torch.full_like(grid, -torch.inf)),
                grid,
                torch.nextafter(grid, torch.full_like(grid, torch.inf)),
                torch.tensor(
                    [[torch.nan, 0.7], [-0.9, torch.nan], [torch.nan, torch.nan]],
                    device=torch_device,
                ),
            )
        )
        n_points = len(points)
        n_rotamers = len(library.rotameric_data.rotamers)
        n_chi = library.rotameric_data.rotamer_means.shape[-1]
        indices = torch.arange(n_points, dtype=torch.int32, device=torch_device)
        rotamer_to_point = indices.repeat_interleave(n_rotamers)
        offsets = indices * n_rotamers

        def evaluate(which_table, backbone):
            builds = torch.stack(
                (indices, torch.full_like(indices, which_table)), dim=1
            )
            if stage == "probability":
                result = torch.empty(n_points * n_rotamers, device=torch_device)
                compiled.interpolate_probabilities_for_possible_rotamers(
                    data.rotameric_prob_tables,
                    data.rotprob_table_sizes,
                    data.rotprob_table_strides,
                    data.rotameric_bb_start,
                    data.rotameric_bb_step,
                    data.rotameric_bb_periodicity,
                    data.rotameric_bb_is_mirrored,
                    data.n_rotamers_for_tableset_offsets,
                    data.sorted_rotamer_2_rotamer,
                    builds,
                    rotamer_to_point,
                    offsets,
                    backbone.flatten(),
                    result,
                )
            else:
                result = torch.empty(
                    (n_points * n_rotamers, n_chi), device=torch_device
                )
                compiled.sample_chi_for_rotamers(
                    data.rotameric_mean_tables,
                    data.rotameric_sdev_tables,
                    data.rotmean_table_sizes,
                    data.rotmean_table_strides,
                    data.rotameric_meansdev_tableset_offsets,
                    data.rotameric_bb_start,
                    data.rotameric_bb_step,
                    data.rotameric_bb_periodicity,
                    data.rotameric_bb_is_mirrored,
                    data.sorted_rotamer_2_rotamer,
                    data.nchi_for_table_set,
                    builds,
                    torch.zeros(
                        (n_points, n_chi), dtype=torch.int32, device=torch_device
                    ),
                    torch.zeros((n_points, n_chi, 1), device=torch_device),
                    torch.full_like(indices, n_chi),
                    backbone.flatten(),
                    offsets,
                    rotamer_to_point,
                    torch.ones_like(indices),
                    data.n_rotamers_for_tableset_offsets,
                    torch.ones(
                        (n_points, n_chi), dtype=torch.int32, device=torch_device
                    ),
                    result,
                )
            return result

        left, right = evaluate(table, points), evaluate(mirrored, -points)
        assert torch.isfinite(left).all() and torch.isfinite(right).all()
        if stage == "chi":
            # Means may differ by full turns; compare the actual angular state.
            left = torch.stack((torch.cos(left), torch.sin(left)))
            right = torch.stack((torch.cos(right), -torch.sin(right)))
        torch.testing.assert_close(
            left, right, atol=2e-5, rtol=1e-5, msg=library.table_name
        )
        checked += 1
    assert checked == 18
