import numpy
import torch
import numba

from typing import Optional
from tmol.types.array import NDArray
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args
from tmol.io.canonical_ordering import CanonicalOrdering


@validate_args
def find_disulfides(
    canonical_ordering: CanonicalOrdering,
    res_types: Tensor[torch.int32][:, :],
    coords: Tensor[torch.float32][:, :, :, 3],
    disulfides: Optional[Tensor[torch.int64][:, 3]] = None,
    find_additional_disulfides: Optional[bool] = True,
    cutoff_dis: float = 2.5,
):
    if canonical_ordering.cys_inds.cys_co_aa_ind == -1:
        # nothing to do: CYS is not a valid residue type
        # in the ChemicalDatabase
        return (
            torch.zeros((0, 3), dtype=torch.int64, device=res_types.device),
            torch.zeros_like(res_types),
        )

    cys_res = res_types == canonical_ordering.cys_inds.cys_co_aa_ind
    restype_variants = torch.full_like(res_types, 0)
    if disulfides is not None:
        # mark the disulfide-bonded residues
        restype_variants[disulfides[:, 0], disulfides[:, 1]] = 1
        restype_variants[disulfides[:, 0], disulfides[:, 2]] = 1
        unpaired_cys_present = torch.sum(
            torch.logical_and(cys_res, restype_variants != 1)
        )
        # if all the cys in the PoseStack are paired, then we do not
        # need to run disulfide detection;
        if unpaired_cys_present == 0 or not find_additional_disulfides:
            return disulfides, restype_variants

    # If we arrive here:
    # we either have some disulfides and remaining unpaired CYS
    # or we have not received any disulfides from the user
    # we now proceed to detect paired cysteines

    # first we ask: are there even any cys residues? If not, avoid
    # sending coordinates back to the CPU and just move on
    cys_pose_ind, cys_res_ind = torch.nonzero(cys_res, as_tuple=True)
    if cys_pose_ind.shape[0] == 0 or not find_additional_disulfides:
        return (
            torch.zeros((0, 3), dtype=torch.int64, device=res_types.device),
            restype_variants,
        )

    res_types_n = res_types.cpu().numpy()
    coords_n = coords.detach().cpu().numpy()
    cys_pose_ind_n = cys_pose_ind.cpu().numpy()
    cys_res_ind_n = cys_res_ind.cpu().numpy()

    n_cys = cys_pose_ind.shape[0]
    found_disulfides = numpy.zeros((n_cys, 3), dtype=numpy.int64)

    if disulfides is not None and disulfides.shape[0] != 0:
        input_disulfides_n = disulfides.cpu().numpy()
        n_input_dslf = input_disulfides_n.shape[0]
        found_disulfides[:n_input_dslf] = input_disulfides_n
        restype_variants = restype_variants.cpu().numpy()
    else:
        n_input_dslf = 0
        restype_variants = numpy.full_like(res_types_n, 0)

    found_disulfides = find_disulf_numba(
        coords_n,
        n_input_dslf,
        found_disulfides,
        cys_pose_ind_n,
        cys_res_ind_n,
        canonical_ordering.cys_inds.sg_atom_for_co_cys,
        cutoff_dis,
        restype_variants,
    )

    return (
        torch.tensor(found_disulfides, dtype=torch.int64, device=coords.device),
        torch.tensor(restype_variants, dtype=torch.int32, device=coords.device),
    )


@numba.jit(nopython=True)
def find_disulf_numba(
    coords: NDArray[numpy.float32][:, :, :, 3],
    n_input_dslf: int,
    found_dslf: NDArray[numpy.int64][:, 3],
    cys_pose_ind: NDArray[numpy.int64][:],
    cys_res_ind: NDArray[numpy.int64][:],
    sg_atom_for_co_cys: int,
    cutoff_dis: float,
    restype_variants: NDArray[numpy.int32][:, :],
):
    # TEMP: Implement this in numpy/numba on the CPU to start

    # algorithm for CYD matching:
    # greedy
    # process the cys pairs in order from n->c
    # for cys i,
    #    take the closest as-of-yet unpaired SG to i's SG w/i 2.5A
    #    mark the two as now paired

    n_cys = cys_pose_ind.shape[0]
    # oops -- doesn't work in nopython mode
    # already_paired = restype_variants[cys_pose_ind, cys_res_ind] == 1
    already_paired = numpy.zeros(n_cys, dtype=numpy.int32)
    for i in range(n_cys):
        already_paired[i] = restype_variants[cys_pose_ind[i], cys_res_ind[i]]
    n_found_dslf = n_input_dslf
    cutoff_dis2 = cutoff_dis * cutoff_dis

    for i in range(n_cys):
        if already_paired[i]:
            continue
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
            restype_variants[i_pose, i_res] = 1
            restype_variants[i_pose, closest_match_res] = 1
            n_found_dslf += 1
    found_dslf = found_dslf[:n_found_dslf]
    return found_dslf
