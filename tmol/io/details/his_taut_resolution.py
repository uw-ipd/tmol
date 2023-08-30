import numpy
import torch
import numba
import toolz.functoolz

from typing import Tuple
from tmol.types.array import NDArray
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args
from tmol.io.canonical_ordering import (
    ordered_canonical_aa_types,
    ordered_canonical_aa_atoms,
    max_n_canonical_atoms,
)
from tmol.utility.auto_number import AutoNumber


# set some constants based on the fixed, canonical ordering of atoms in the canonical AAs
# defined in
his_co_aa_ind = ordered_canonical_aa_types.index("HIS")
his_ND1_in_co = ordered_canonical_aa_atoms["HIS"].index("ND1")
his_NE2_in_co = ordered_canonical_aa_atoms["HIS"].index("NE2")
his_HD1_in_co = ordered_canonical_aa_atoms["HIS"].index("HD1")
his_HE2_in_co = ordered_canonical_aa_atoms["HIS"].index("HE2")
his_HN_in_co = ordered_canonical_aa_atoms["HIS"].index("HN")
his_NH_in_co = ordered_canonical_aa_atoms["HIS"].index("NH")
his_NN_in_co = ordered_canonical_aa_atoms["HIS"].index("NN")
his_CG_in_co = ordered_canonical_aa_atoms["HIS"].index("CG")

his_taut_variant_NE2_protonated = 0
his_taut_variant_ND1_protonated = 1


class HisTautomerResolution(AutoNumber):
    his_taut_missing_atoms = ()
    his_taut_HD1 = ()
    his_taut_HE2 = ()
    his_taut_NH_is_ND1 = ()
    his_taut_NN_is_ND1 = ()
    his_taut_HD1_HE2 = ()  # future
    his_taut_unresolved = ()  # future


@validate_args
def resolve_his_tautomerization(
    res_types: Tensor[torch.int32][:, :],
    res_type_variants: Tensor[torch.int32][:, :],
    coords: Tensor[torch.float32][:, :, :, 3],
    atom_is_present: Tensor[torch.int32][:, :, :],
) -> Tuple[
    Tensor[torch.int32][:, :],
    Tensor[torch.int32][:, :],
    Tensor[torch.float32][:, :, :, 3],
    Tensor[torch.int32][:, :, :],
]:
    # short circuit
    # return (
    #     torch.zeros_like(res_types),
    #     res_type_variants,
    #     coords,
    #     atom_is_present
    # )
    from tmol.io.details.compiled.compiled import resolve_his_taut

    # TEMP! Do it in numpy/numba for now
    # This will be slower than if the tensors were
    # to remain resident on the GPU and should be
    # replaced with better code.
    # res_types_n = res_types.cpu().numpy()
    # res_type_variants_n = res_type_variants.cpu().numpy()
    # coords_n = coords.detach().cpu().numpy()
    # atom_is_present_n = atom_is_present.cpu().numpy()

    his_pose_ind, his_res_ind = torch.nonzero(res_types == his_co_aa_ind, as_tuple=True)
    # his_remapping_dst_index = numpy.tile(
    #     numpy.repeat(numpy.arange(max_n_canonical_atoms, dtype=numpy.int32)),
    #     (res_types.shape[0], res_types.shape[1], 1, 1),
    # ).reshape(res_types.shape[0], res_types.shape[1], max_n_canonical_atoms, 3)
    his_remapping_dst_index = torch.tile(
        torch.arange(max_n_canonical_atoms, dtype=torch.int64, device=res_types.device),
        (res_types.shape[0], res_types.shape[1], 1),
    ).reshape(res_types.shape[0], res_types.shape[1], max_n_canonical_atoms)

    # his_taut, his_remapping_dst_index = resolve_his_tautomerization_numba(
    #     res_types_n,
    #     res_type_variants_n,
    #     his_pose_ind,
    #     his_res_ind,
    #     coords_n,
    #     atom_is_present_n,
    #     his_remapping_dst_index,
    # )
    his_taut = resolve_his_taut(
        coords,
        res_types,
        res_type_variants,
        his_pose_ind,
        his_res_ind,
        atom_is_present,
        _his_atom_inds_tensor(coords.device),
        his_remapping_dst_index,
    )

    # his_remapping_dst_index = torch.tensor(
    #     his_remapping_dst_index, dtype=torch.int64, device=res_types.device
    # )
    # print("his_remapping_dst_index.unsqueeze(4)", his_remapping_dst_index.unsqueeze(3).shape)
    his_remapping_dst_index = his_remapping_dst_index.unsqueeze(3).expand(-1, -1, -1, 3)
    # print("his_remapping_dst_index", his_remapping_dst_index.shape)
    # resolved_restype_variants = torch.tensor(
    #     res_type_variants_n, dtype=torch.int32, device=res_types.device
    # )
    resolved_coords = torch.gather(coords, dim=2, index=his_remapping_dst_index)
    resolved_atom_is_present = torch.gather(
        atom_is_present, dim=2, index=his_remapping_dst_index[:, :, :, 0]
    )

    # Now send the data back to the device
    return (
        torch.tensor(his_taut, dtype=torch.int32, device=coords.device),
        res_type_variants,
        resolved_coords,
        resolved_atom_is_present,
    )


@numba.jit(nopython=True)
def resolve_his_tautomerization_numba(
    res_types: NDArray[numpy.int32][:, :],
    res_type_variants: NDArray[numpy.int32][:, :],
    his_pose_ind: NDArray[numpy.int32][:],
    his_res_ind: NDArray[numpy.int32][:],
    coords: NDArray[numpy.float32][:, :, :, 3],
    atom_is_present: NDArray[numpy.float32][:, :, :],
    his_remapping_dst_index: NDArray[numpy.int32][:, :, :, 3],
):
    """Resolve which of four cases we have for HIS's tautomerization state:
    a. HIS HD1 is provided (and HE2 is not) and we are in tautomerization state HD1,
       HisTautomerResolution.his_taut_HD1
    b. HIS HE2 is provided (and HD1 is not) and we are in tautomerization state HE2
    c. HIS HN, ND1 and NE2 are provided (and neither HD1 nor HE2 is) and we select the
       tautomerization state based on which nitrogen HN is closest to
    d. HIS HN, NH, and NN are provided and we select the tautomerization state based
       on the distances between both NH and NN to CG; the one closer to CG is taken
       as ND1.
    e. Neither HD1, HE2, nor HN are proviced, so we make the arbitrary choice to use
       HE2
    """
    # We will remap atoms using an index copy operation so that we can
    # properly track derivatives through the backprop step. For
    # derivative tracking, we will have torch do the copy operation;
    # but we can compute the indices (what gets copied where) in numpy.
    # This all should get translated down into C++/CUDA to make it fast.
    # To use torch.gather, we need to put the index of the source atom
    # in all three of the x, y, and z slots.

    his_taut = numpy.zeros_like(res_types)
    for i in range(his_pose_ind.shape[0]):
        ip = his_pose_ind[i]
        ir = his_res_ind[i]
        ND1_present = atom_is_present[ip, ir, his_ND1_in_co]
        NE2_present = atom_is_present[ip, ir, his_NE2_in_co]
        HD1_present = atom_is_present[ip, ir, his_HD1_in_co]
        HE2_present = atom_is_present[ip, ir, his_HE2_in_co]
        HN_present = atom_is_present[ip, ir, his_HN_in_co]
        NH_present = atom_is_present[ip, ir, his_NH_in_co]
        NN_present = atom_is_present[ip, ir, his_NN_in_co]
        CG_present = atom_is_present[ip, ir, his_CG_in_co]

        state = HisTautomerResolution.his_taut_unresolved.value

        if HD1_present and not HE2_present:
            state = HisTautomerResolution.his_taut_HD1.value
            res_type_variants[ip, ir] = his_taut_variant_ND1_protonated
        elif HE2_present and not HD1_present:
            state = HisTautomerResolution.his_taut_HE2.value
            res_type_variants[ip, ir] = his_taut_variant_NE2_protonated
        elif (
            HN_present
            and not HD1_present
            and not HE2_present
            and ND1_present
            and NE2_present
        ):
            dis2_ND1 = numpy.sum(
                numpy.square(
                    coords[ip, ir, his_ND1_in_co] - coords[ip, ir, his_HN_in_co]
                )
            )
            dis2_NE2 = numpy.sum(
                numpy.square(
                    coords[ip, ir, his_NE2_in_co] - coords[ip, ir, his_HN_in_co]
                )
            )
            if dis2_ND1 < dis2_NE2:
                state = HisTautomerResolution.his_taut_HD1.value
                res_type_variants[ip, ir] = his_taut_variant_ND1_protonated
                his_remapping_dst_index[ip, ir, his_HD1_in_co, :] = his_HN_in_co
            else:
                state = HisTautomerResolution.his_taut_HE2.value
                res_type_variants[ip, ir] = his_taut_variant_NE2_protonated
                his_remapping_dst_index[ip, ir, his_HE2_in_co, :] = his_HN_in_co
        elif NH_present and NN_present and HN_present and CG_present:
            dis2_NH = numpy.sum(
                numpy.square(
                    coords[ip, ir, his_NH_in_co] - coords[ip, ir, his_CG_in_co]
                )
            )
            dis2_NN = numpy.sum(
                numpy.square(
                    coords[ip, ir, his_NN_in_co] - coords[ip, ir, his_CG_in_co]
                )
            )
            if dis2_NH < dis2_NN:
                state = HisTautomerResolution.his_taut_NH_is_ND1.value
                his_remapping_dst_index[ip, ir, his_ND1_in_co, :] = his_NH_in_co
                his_remapping_dst_index[ip, ir, his_HD1_in_co, :] = his_HN_in_co
                his_remapping_dst_index[ip, ir, his_NE2_in_co, :] = his_NN_in_co
                res_type_variants[ip, ir] = his_taut_variant_ND1_protonated
            else:
                state = HisTautomerResolution.his_taut_NN_is_ND1.value
                his_remapping_dst_index[ip, ir, his_ND1_in_co, :] = his_NN_in_co
                his_remapping_dst_index[ip, ir, his_HE2_in_co, :] = his_HN_in_co
                his_remapping_dst_index[ip, ir, his_NE2_in_co, :] = his_NH_in_co
                res_type_variants[ip, ir] = his_taut_variant_NE2_protonated
        elif not HD1_present and not HE2_present and not HN_present:
            # arbitrary choice: go with his_taut_HE2
            state = HisTautomerResolution.his_taut_HE2.value
        his_taut[ip, ir] = int(state)

    return his_taut, his_remapping_dst_index


@toolz.functoolz.memoize
def _his_atom_inds_tensor(device: torch.device):
    return torch.tensor(
        [
            [
                his_ND1_in_co,
                his_NE2_in_co,
                his_HD1_in_co,
                his_HE2_in_co,
                his_HN_in_co,
                his_NH_in_co,
                his_NN_in_co,
                his_CG_in_co,
            ],
        ],
        dtype=torch.int32,
        device=device,
    )
