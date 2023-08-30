import torch
import toolz.functoolz

from typing import Tuple
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

    his_pose_ind, his_res_ind = torch.nonzero(res_types == his_co_aa_ind, as_tuple=True)
    his_remapping_dst_index = torch.tile(
        torch.arange(max_n_canonical_atoms, dtype=torch.int64, device=res_types.device),
        (res_types.shape[0], res_types.shape[1], 1),
    ).reshape(res_types.shape[0], res_types.shape[1], max_n_canonical_atoms)

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

    his_remapping_dst_index = his_remapping_dst_index.unsqueeze(3).expand(-1, -1, -1, 3)
    resolved_coords = torch.gather(coords, dim=2, index=his_remapping_dst_index)
    resolved_atom_is_present = torch.gather(
        atom_is_present, dim=2, index=his_remapping_dst_index[:, :, :, 0]
    )

    return (
        torch.tensor(his_taut, dtype=torch.int32, device=coords.device),
        res_type_variants,
        resolved_coords,
        resolved_atom_is_present,
    )


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
