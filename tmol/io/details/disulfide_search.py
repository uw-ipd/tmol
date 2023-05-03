import numpy
import torch
import numba

from tmol.types.array import NDArray
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args
from tmol.io.canonical_ordering import (
    ordered_canonical_aa_types,
    ordered_canonical_aa_atoms,
)

cys_co_aa_ind = ordered_canonical_aa_types.index("CYS")
sg_atom_for_co_cys = ordered_canonical_aa_atoms["CYS"].index("SG")


@validate_args
def find_disulfides(
    res_types: Tensor[torch.int32][:, :],
    coords: Tensor[torch.float32][:, :, :, 3],
    atom_is_present: Tensor[torch.int32][:, :, :],
    cutoff_dis: float = 2.5,
):
    res_types_n = res_types.cpu().numpy()
    coords_n = coords.cpu().numpy()
    atom_is_present_n = atom_is_present.cpu().numpy()

    cys_pose_ind, cys_res_ind = numpy.nonzero(
        numpy.logical_and(
            res_types_n == cys_co_aa_ind,
            atom_is_present_n[:, :, sg_atom_for_co_cys] != 0,
        )
    )
    restype_variant_inds = numpy.full_like(res_types_n, 0)

    found_disulfides = find_disulf_numba(
        coords_n,
        cys_pose_ind,
        cys_res_ind,
        sg_atom_for_co_cys,
        cutoff_dis,
        restype_variant_inds,
    )

    def ti32(x):
        return torch.tensor(x, dtype=torch.int32, device=coords.device)

    return ti32(found_disulfides), ti32(restype_variant_inds)


@numba.jit(nopython=True)
def find_disulf_numba(
    coords: NDArray[numpy.float32][:, :, :, 3],
    cys_pose_ind: NDArray[numpy.int64][:],
    cys_res_ind: NDArray[numpy.int64][:],
    sg_atom_for_co_cys: int,
    cutoff_dis: float,
    restype_variant_inds: NDArray[numpy.int32][:, :],
):
    # TEMP: Implement this in numpy/numba on the CPU to start

    # algorithm for CYD matching:
    # greedy
    # process the cys pairs in order from n->c
    # for cys i,
    #    take the closest as-of-yet unpaired SG to i's SG w/i 2.5A
    #    mark the two as now paired

    n_cys = cys_pose_ind.shape[0]
    n_poses = coords.shape[0]
    found_dslf = numpy.zeros((n_cys, 3), dtype=numpy.int32)
    already_paired = numpy.zeros((n_cys,), dtype=numpy.int32)
    n_found_dslf = 0
    cutoff_dis2 = cutoff_dis * cutoff_dis

    for i in range(cys_pose_ind.shape[0]):
        i_pose = cys_pose_ind[i]
        i_res = cys_res_ind[i]
        closest_match = -1
        closest_match_res = -1
        closest_dis2 = -1.0
        i_coord = coords[i_pose, i_res, sg_atom_for_co_cys]
        for j in range(i + 1, cys_pose_ind.shape[0]):
            j_pose = cys_pose_ind[j]
            if j_pose != i_pose:
                break
            j_res = cys_res_ind[j]
            if already_paired[j]:
                continue
            dis2 = numpy.sum(
                numpy.square(i_coord - coords[j_pose, j_res, sg_atom_for_co_cys])
            )
            if dis2 < cutoff_dis2 and (closest_dis2 < 0 or dis2 < closest_dis2):
                closest_dis2 = dis2
                closest_match = j
                closest_match_res = j_res
        if closest_match >= 0:
            already_paired[i] = 1
            already_paired[closest_match] = 1
            found_dslf[n_found_dslf, 0] = i_pose
            found_dslf[n_found_dslf, 1] = i_res
            found_dslf[n_found_dslf, 2] = closest_match_res
            # mark these two as disulfides
            restype_variant_inds[i_pose, i_res] = 1
            restype_variant_inds[i_pose, closest_match_res] = 1
            n_found_dslf += 1
    found_dslf = found_dslf[:n_found_dslf]
    return found_dslf
