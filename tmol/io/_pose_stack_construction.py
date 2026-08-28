import torch
import numpy

from tmol.types import (
    Tensor,
    NDArray,
    validate_args,
)
from typing import Optional
from tmol.pose import (
    PDBInfo,
    DEFAULT_ATOM_OCCUPANCY,
    DEFAULT_ATOM_B_FACTOR,
    PoseStack,
    PackedBlockTypes,
)
from tmol.io import CanonicalOrdering


@validate_args
def pose_stack_from_canonical_form(  # noqa: C901
    canonical_ordering: CanonicalOrdering,
    pbt: PackedBlockTypes,
    chain_id: Tensor[torch.int32][:, :],
    res_types: Tensor[torch.int32][:, :],
    coords: Tensor[torch.float32][:, :, :, 3],
    res_labels: Optional[NDArray[int][:, :]],
    res_ins_codes: Optional[NDArray[object][:, :]],
    chain_labels: Optional[NDArray[object][:, :]],
    atom_occupancy: Optional[NDArray[numpy.float32][:, :, :]] = None,
    atom_b_factor: Optional[NDArray[numpy.float32][:, :, :]] = None,
    disulfides: Optional[Tensor[torch.int64][:, 3]] = None,
    res_not_connected: Optional[Tensor[torch.bool][:, :, 2]] = None,
    *,
    find_additional_disulfides: Optional[bool] = True,
    return_chain_ind: bool = False,
    return_atom_mapping: bool = False,
    return_block_has_missing_atoms: bool = False,
):
    """Build a pose stack from tensors in canonical atom ordering.

    Residue variants are selected from the atoms present in ``coords``. A NaN
    coordinate marks an absent input atom; TMol builds missing leaf atoms, but
    requires non-leaf atoms unless ``return_block_has_missing_atoms`` is set.

    Args:
        canonical_ordering: Residue and atom ordering used by the input tensors.
        pbt: Packed residue types and score-term annotations for the new pose.
        chain_id: Chain index for each ``[pose, residue]``; residues in a chain
            must be consecutive.
        res_types: Canonical residue-type index for each ``[pose, residue]``;
            ``-1`` marks padding.
        coords: Coordinates shaped ``[pose, residue, canonical_atom, xyz]``.
        res_labels: Optional source residue numbers shaped ``[pose, residue]``.
        res_ins_codes: Optional source insertion codes shaped ``[pose, residue]``.
        chain_labels: Optional source chain labels shaped ``[pose, residue]``.
        atom_occupancy: Optional source occupancies in canonical atom ordering.
        atom_b_factor: Optional source B-factors in canonical atom ordering.
        disulfides: Explicit ``[pose, cys1, cys2]`` rows. If omitted, nearby
            cysteine SG atoms are paired geometrically.
        res_not_connected: Flags shaped ``[pose, residue, direction]`` for
            polymer connections absent before or after each residue.
        find_additional_disulfides: Detect geometrically plausible disulfides
            not included in ``disulfides``.
        return_chain_ind: Include the left-justified ``chain_ind`` tensor.
        return_atom_mapping: Include ``can_atom_mapping`` and
            ``ps_atom_mapping`` tensors between canonical and pose atom order.
        return_block_has_missing_atoms: Include a ``[pose, residue]`` mask for
            blocks missing non-leaf input atoms instead of rejecting them.

    Returns:
        The pose stack. If any return flag is set, returns ``(pose_stack,
        metadata)`` with the requested tensors in ``metadata``.
    """

    from tmol.io.details import left_justify_canonical_form
    from tmol.io.details import find_disulfides
    from tmol.io.details import resolve_his_tautomerization
    from tmol.io.details import (
        assign_block_types,
        take_block_type_atoms_from_canonical,
    )
    from tmol.io.details import build_missing_leaf_atoms

    assert chain_id.device == res_types.device
    assert chain_id.device == coords.device

    assert chain_id.shape[0] == res_types.shape[0]
    assert chain_id.shape[1] == res_types.shape[1]
    assert chain_id.shape[0] == coords.shape[0]
    assert chain_id.shape[1] == coords.shape[1]
    assert coords.shape[2] == canonical_ordering.max_n_canonical_atoms
    assert res_labels is None or res_labels.shape[0] == chain_id.shape[0]
    assert res_labels is None or res_labels.shape[1] == chain_id.shape[1]
    assert res_ins_codes is None or res_ins_codes.shape[0] == chain_id.shape[0]
    assert res_ins_codes is None or res_ins_codes.shape[1] == chain_id.shape[1]
    assert chain_labels is None or chain_labels.shape[0] == chain_id.shape[0]
    assert chain_labels is None or chain_labels.shape[1] == chain_id.shape[1]
    assert atom_occupancy is None or atom_occupancy.shape[0] == chain_id.shape[0]
    assert atom_occupancy is None or atom_occupancy.shape[1] == chain_id.shape[1]
    assert atom_b_factor is None or atom_b_factor.shape[0] == chain_id.shape[0]
    assert atom_b_factor is None or atom_b_factor.shape[1] == chain_id.shape[1]
    assert res_not_connected is None or res_not_connected.shape[0] == chain_id.shape[0]
    assert res_not_connected is None or res_not_connected.shape[1] == chain_id.shape[1]

    # step 1: record which atoms the user has given us by looking for NaNs
    #         in the input coordinate tensor.
    # step 2: remove any "virtual residues," marked with a res-type ind of -1
    #         by shifting all of the residues in each Pose "to the left"
    # step 3: resolve disulfides
    # step 4: resolve his tautomer
    # step 5: resolve termini variants, assign block-types to each input
    #         residue, and populate the inter-block connectivity tensors
    # step 6: select the atoms from the canonically-ordered input tensors
    #         (the coords and atom_is_present tensors) that belong to the
    #         now-assigned block types, discarding/ignoring
    #         any others that may have been provided
    # step 7: if any atoms missing, build them
    # step 8: construct PoseStack object
    # step 9: construct the forward/reverse atom mapping indices if required

    # 1: look for NaNs in the input coordinates tensor
    atom_is_present = torch.all(torch.logical_not(torch.isnan(coords)), dim=3)

    # 2
    # "left justify" the input canonical-form residues: residues that are
    # given with a "-1" residue-type should be excised from the center of
    # their Poses to ensure that the polymeric-bond-detection logic
    # downstream will work properly. This effectively means "shifting left"
    # all the other residues in the Pose to fill the vacated slots.
    # A single residue slot is already left-justified: each pose either has its
    # residue in slot zero or is empty. Avoid the GPU compaction and host index
    # copies for the common batched-ligand scoring case.
    if res_types.shape[1] != 1:
        (
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
        ) = left_justify_canonical_form(
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

    if res_types.shape[1] == 0:
        raise ValueError(
            "pose_stack_from_canonical_form: no recognized residues found in input. "
            "Check that residue names match entries in the CanonicalOrdering and that "
            "the structure was parsed via biotite (HETATM records are supported) rather "
            "than the internal PDB parser (ATOM records only)."
        )

    # 3
    if res_types.shape[1] == 1 and disulfides is None:
        found_disulfides = torch.zeros(
            (0, 3), dtype=torch.int64, device=res_types.device
        )
        res_type_variants = torch.zeros_like(res_types)
    else:
        found_disulfides, res_type_variants = find_disulfides(
            canonical_ordering,
            res_types,
            coords,
            disulfides,
            find_additional_disulfides,
        )

    # 4
    (
        his_taut,
        res_type_variants,
        resolved_coords,
        resolved_atom_is_present,
    ) = resolve_his_tautomerization(
        canonical_ordering, res_types, res_type_variants, coords, atom_is_present
    )

    # 5
    (
        block_types64,
        inter_residue_connections64,
        inter_block_bondsep,
    ) = assign_block_types(
        canonical_ordering,
        pbt,
        resolved_atom_is_present,
        chain_id,
        res_types,
        res_type_variants,
        found_disulfides,
        res_not_connected,
    )

    # 6
    (
        block_coords,
        missing_atoms,
        real_atoms,
        real_canonical_atom_inds,
        atom_occupancy,
        atom_b_factor,
    ) = take_block_type_atoms_from_canonical(
        pbt, block_types64, coords, atom_is_present, atom_occupancy, atom_b_factor
    )

    # 7
    inter_residue_connections = inter_residue_connections64.to(torch.int32)
    (
        pose_stack_coords,
        block_coord_offset,
        real_block_atoms,
        pose_at_is_real,
        block_has_missing_atoms,
    ) = build_missing_leaf_atoms(
        pbt,
        block_types64,
        real_atoms,
        block_coords,
        missing_atoms,
        inter_residue_connections,
        fail_on_missing_nonleaf_atoms=not return_block_has_missing_atoms,
    )

    def i64(x):
        return x.to(torch.int64)

    def i32(x):
        return x.to(torch.int32)

    # 8
    if atom_occupancy is not None or atom_b_factor is not None:
        real_block_atoms = real_block_atoms.cpu().numpy()
        pose_at_is_real = pose_at_is_real.cpu().numpy()
    atom_occupancy_pose_layout = numpy.full(
        pose_stack_coords.shape[:2], DEFAULT_ATOM_OCCUPANCY, dtype=numpy.float32
    )
    if atom_occupancy is not None:
        atom_occupancy_pose_layout[pose_at_is_real] = atom_occupancy[real_block_atoms]
    atom_b_factor_pose_layout = numpy.full(
        pose_stack_coords.shape[:2], DEFAULT_ATOM_B_FACTOR, dtype=numpy.float32
    )
    if atom_b_factor is not None:
        atom_b_factor_pose_layout[pose_at_is_real] = atom_b_factor[real_block_atoms]

    pdb_info = PDBInfo(
        residue_labels=res_labels,
        residue_insertion_codes=res_ins_codes,
        chain_labels=chain_labels,
        atom_occupancy=atom_occupancy_pose_layout,
        atom_b_factor=atom_b_factor_pose_layout,
    )

    block_coord_offset64 = i64(block_coord_offset)
    ps = PoseStack(
        packed_block_types=pbt,
        coords=pose_stack_coords,
        block_coord_offset=block_coord_offset,
        block_coord_offset64=block_coord_offset64,
        inter_residue_connections=inter_residue_connections,
        inter_residue_connections64=inter_residue_connections64,
        inter_block_bondsep=inter_block_bondsep,
        inter_block_bondsep64=i64(inter_block_bondsep),
        block_type_ind=i32(block_types64),
        block_type_ind64=block_types64,
        chain_id=chain_id,
        chain_id64=i64(chain_id),
        pdb_info=pdb_info,
        constraint_set=None,
        device=pbt.device,
    )

    # 9
    if return_atom_mapping:
        (
            nz_block_layout_pose_ind,
            nz_block_layout_block_ind,
            nz_block_at_ind,
        ) = torch.nonzero(real_atoms, as_tuple=True)
        pose_atom_ind = (
            block_coord_offset64[nz_block_layout_pose_ind, nz_block_layout_block_ind]
            + nz_block_at_ind
        )

        def _u1(x):
            return x.unsqueeze(1)

        can_atom_mapping = torch.cat(
            (
                _u1(nz_block_layout_pose_ind),
                _u1(nz_block_layout_block_ind),
                _u1(real_canonical_atom_inds),
            ),
            dim=1,
        )
        ps_atom_mapping = torch.cat(
            (
                _u1(nz_block_layout_pose_ind),
                _u1(pose_atom_ind),
            ),
            dim=1,
        )

    # return the optional arguments in a dictionary
    opt_return_vals = {}
    if return_chain_ind:
        opt_return_vals["chain_ind"] = chain_id
    if return_atom_mapping:
        opt_return_vals["can_atom_mapping"] = can_atom_mapping
        opt_return_vals["ps_atom_mapping"] = ps_atom_mapping
    if return_block_has_missing_atoms:
        opt_return_vals["block_has_missing_atoms"] = block_has_missing_atoms

    if len(opt_return_vals) > 0:
        return ps, opt_return_vals
    return ps
