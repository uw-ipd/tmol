import torch
import toolz.functoolz

from typing import Tuple
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args
from tmol.io.canonical_ordering import CanonicalOrdering, HisSpecialCaseIndices
from tmol.utility.auto_number import AutoNumber


# Special case variant-type handling for HIS
# The I/O code knows that spcase variant 0 for HIS is HIS-E
# and the spcase variant 1 for HIS is HIS-D
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
    canonical_ordering: CanonicalOrdering,
    res_types: Tensor[torch.int32][:, :],
    res_type_variants: Tensor[torch.int32][:, :],
    coords: Tensor[torch.float32][:, :, :, 3],
    atom_is_present: Tensor[torch.bool][:, :, :],
) -> Tuple[
    Tensor[torch.int32][:, :],
    Tensor[torch.int32][:, :],
    Tensor[torch.float32][:, :, :, 3],
    Tensor[torch.bool][:, :, :],
]:
    if canonical_ordering.his_inds.his_co_aa_ind == -1:
        return (torch.zeros_like(res_types), res_type_variants, coords, atom_is_present)
    from tmol.io.details.compiled.compiled import resolve_his_taut

    his_inds = canonical_ordering.his_inds

    his_pose_ind, his_res_ind = torch.nonzero(
        res_types == his_inds.his_co_aa_ind, as_tuple=True
    )
    his_remapping_dst_index = torch.tile(
        torch.arange(
            canonical_ordering.max_n_canonical_atoms,
            dtype=torch.int64,
            device=res_types.device,
        ),
        (res_types.shape[0], res_types.shape[1], 1),
    ).reshape(
        res_types.shape[0], res_types.shape[1], canonical_ordering.max_n_canonical_atoms
    )

    his_taut = resolve_his_taut(
        coords,
        res_types,
        res_type_variants,
        his_pose_ind,
        his_res_ind,
        atom_is_present,
        _his_atom_inds_tensor(canonical_ordering.his_inds, coords.device),
        his_remapping_dst_index,
    )

    his_remapping_dst_index = his_remapping_dst_index.unsqueeze(3).expand(-1, -1, -1, 3)
    resolved_coords = torch.gather(coords, dim=2, index=his_remapping_dst_index)
    resolved_atom_is_present = torch.gather(
        atom_is_present, dim=2, index=his_remapping_dst_index[:, :, :, 0]
    )

    return (
        his_taut.to(dtype=torch.int32, device=coords.device),
        res_type_variants,
        resolved_coords,
        resolved_atom_is_present,
    )


@toolz.functoolz.memoize(key=lambda args, kwargs: (hash(args[0]), args[1]))
def _his_atom_inds_tensor(his_inds: HisSpecialCaseIndices, device: torch.device):
    return torch.tensor(
        [
            [
                his_inds.his_ND1_in_co,
                his_inds.his_NE2_in_co,
                his_inds.his_HD1_in_co,
                his_inds.his_HE2_in_co,
                his_inds.his_HN_in_co,
                his_inds.his_NH_in_co,
                his_inds.his_NN_in_co,
                his_inds.his_CG_in_co,
            ],
        ],
        dtype=torch.int32,
        device=device,
    )
