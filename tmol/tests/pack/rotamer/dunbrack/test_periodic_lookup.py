"""Both sampling stages must wrap the periodic endpoint before table lookup."""

import pytest
import torch

from tmol.score.dunbrack import DunbrackParamResolver
from tmol.tests.pack.rotamer.dunbrack.test_dunbrack_chi_sampler import (
    _table_indices,
    get_compiled,
)


def evaluate_boundary(resolver, axis, angle, stage):
    dtype = torch.float32
    device = resolver.device
    data = resolver.sampling_db
    (phe,) = _table_indices(resolver, ("PHE",), device)

    def real(tensor):
        return tensor.to(dtype)

    def integer(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    # Guard row/column 36 so a broken implementation reads a different valid
    # rotamer, never unallocated memory. The actual source grids have 36 bins.
    assert data.sorted_rotamer_2_rotamer.shape[:2] == (36, 36)
    sorted_lookup = torch.zeros(
        (37, 37, data.sorted_rotamer_2_rotamer.shape[2]),
        dtype=torch.int64,
        device=device,
    )
    sorted_lookup[36, :, :] = 1
    sorted_lookup[:, 36, :] = 1
    backbone = torch.zeros(2, dtype=dtype, device=device)
    # Use the stored grid endpoint, not a higher-precision approximation to it.
    backbone[axis] = (
        real(data.rotameric_bb_start)[phe, axis]
        + angle * real(data.rotameric_bb_periodicity)[phe, axis]
    )
    compiled = get_compiled()
    if stage == "probability":
        result = torch.zeros(1, dtype=dtype, device=device)
        compiled.interpolate_probabilities_for_possible_rotamers(
            real(data.rotameric_prob_tables),
            data.rotprob_table_sizes,
            data.rotprob_table_strides,
            real(data.rotameric_bb_start),
            real(data.rotameric_bb_step),
            real(data.rotameric_bb_periodicity),
            data.rotameric_bb_is_mirrored,
            data.n_rotamers_for_tableset_offsets,
            sorted_lookup,
            integer([[0, phe]]),
            integer([0]),
            integer([0]),
            backbone,
            result,
        )
    else:
        nchi = int(data.nchi_for_table_set[phe])
        result = torch.zeros((1, nchi), dtype=dtype, device=device)
        compiled.sample_chi_for_rotamers(
            real(data.rotameric_mean_tables),
            real(data.rotameric_sdev_tables),
            data.rotmean_table_sizes,
            data.rotmean_table_strides,
            data.rotameric_meansdev_tableset_offsets,
            real(data.rotameric_bb_start),
            real(data.rotameric_bb_step),
            real(data.rotameric_bb_periodicity),
            data.rotameric_bb_is_mirrored,
            sorted_lookup,
            data.nchi_for_table_set,
            integer([[0, phe]]),
            torch.zeros((1, nchi), dtype=torch.int32, device=device),
            torch.zeros((1, nchi, 1), dtype=dtype, device=device),
            integer([nchi]),
            backbone,
            integer([0]),
            integer([0]),
            integer([1]),
            data.n_rotamers_for_tableset_offsets,
            torch.ones((1, nchi), dtype=torch.int32, device=device),
            result,
        )
    return result


@pytest.mark.parametrize("axis", [0, 1], ids=["phi", "psi"])
@pytest.mark.parametrize("stage", ["probability", "chi"])
def test_periodic_endpoints_select_the_same_rotamer(
    default_database, torch_device, axis, stage
):
    resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    first = evaluate_boundary(resolver, axis, 0, stage)
    last = evaluate_boundary(resolver, axis, 1, stage)
    assert torch.isfinite(first).all() and torch.isfinite(last).all()
    # Probabilities and chi means should read precisely the same table/point.
    torch.testing.assert_close(first, last, rtol=0, atol=0)
