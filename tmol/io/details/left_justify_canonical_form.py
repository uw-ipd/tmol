from typing import Optional

import numpy
import torch

from tmol.pose.pdb_info import DEFAULT_ATOM_B_FACTOR, DEFAULT_ATOM_OCCUPANCY
from tmol.score.common.stack_condense import (
    condense_torch_inds,
)
from tmol.types.array import NDArray
from tmol.types.torch import Tensor


def left_justify_canonical_form(
    chain_id: Tensor[torch.int32][:, :],
    res_types: Tensor[torch.int32][:, :],
    coords: Tensor[torch.float32][:, :, :, 3],
    atom_is_present: Optional[Tensor[torch.bool][:, :, :]] = None,
    disulfides: Optional[Tensor[torch.int64][:, 3]] = None,
    res_not_connected: Optional[Tensor[torch.bool][:, :, 2]] = None,
    res_labels: Optional[NDArray[int][:, :]] = None,
    res_ins_codes: Optional[NDArray[str][:, :]] = None,
    chain_labels: Optional[NDArray[str][:, :]] = None,
    atom_occupancy: Optional[NDArray[float][:, :, :]] = None,
    atom_b_factor: Optional[NDArray[float][:, :, :]] = None,
):
    old_res_types_real = res_types != -1
    cinds = condense_torch_inds(old_res_types_real, res_types.device)
    good_cinds = cinds[cinds >= 0].view(-1)
    nz_cinds = torch.nonzero(cinds >= 0, as_tuple=False)
    np_nz_cinds = nz_cinds.cpu().numpy()

    def lj(x, fill):
        # kinda surprised I don't already have this??
        selected_values = torch.full_like(x, fill)
        if len(x.shape) > 2:
            selected_values[nz_cinds[:, 0], nz_cinds[:, 1], :] = x[nz_cinds[:, 0], good_cinds, :]
        else:
            selected_values[nz_cinds[:, 0], nz_cinds[:, 1]] = x[nz_cinds[:, 0], good_cinds]
        return selected_values

    np_good_cinds = good_cinds.cpu().numpy()

    def lj_np(x, fill):
        new_x = numpy.full_like(x, fill)
        if len(x.shape) > 2:
            new_x[np_nz_cinds[:, 0], np_nz_cinds[:, 1], :] = x[np_nz_cinds[:, 0], np_good_cinds, :]
        else:
            new_x[np_nz_cinds[:, 0], np_nz_cinds[:, 1]] = x[np_nz_cinds[:, 0], np_good_cinds]
        return new_x

    chain_id = lj(chain_id, -1)
    res_types = lj(res_types, -1)

    coords = lj(coords, float("nan"))
    if atom_is_present is not None:
        atom_is_present = lj(atom_is_present, 0)

    if disulfides is not None:
        old_2_new = torch.full(res_types.shape, -1, dtype=torch.int64, device=res_types.device)
        old_2_new[old_res_types_real] = torch.nonzero(res_types != -1)[:, 1]
        dslf_pose_ind = disulfides[:, 0]
        dslf_res1_ind = disulfides[:, 1]
        dslf_res2_ind = disulfides[:, 2]
        disulfides = torch.cat(
            [
                dslf_pose_ind.unsqueeze(1),
                old_2_new[dslf_pose_ind, dslf_res1_ind].unsqueeze(1),
                old_2_new[dslf_pose_ind, dslf_res2_ind].unsqueeze(1),
            ],
            dim=1,
        )
    if res_not_connected is not None:
        res_not_connected = lj(res_not_connected, False)

    res_labels = lj_np(res_labels, 0) if res_labels is not None else None
    res_ins_codes = lj_np(res_ins_codes, "") if res_ins_codes is not None else None
    chain_labels = lj_np(chain_labels, "") if chain_labels is not None else None
    atom_occupancy = lj_np(atom_occupancy, DEFAULT_ATOM_OCCUPANCY) if atom_occupancy is not None else None
    atom_b_factor = lj_np(atom_b_factor, DEFAULT_ATOM_B_FACTOR) if atom_b_factor is not None else None

    return (
        chain_id,
        res_types,
        coords,
        atom_is_present,
        disulfides,
        res_not_connected,
        res_labels,
        res_ins_codes,
        chain_labels,
        atom_occupancy,
        atom_b_factor,
    )
