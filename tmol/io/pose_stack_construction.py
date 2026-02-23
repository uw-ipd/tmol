import torch
import numpy

from tmol.types.torch import Tensor
from tmol.types.array import NDArray
from typing import Optional
from tmol.types.functional import validate_args
from tmol.pose.pdb_info import PDBInfo
from tmol.pose.pose_stack import PoseStack
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.io.canonical_ordering import CanonicalOrdering


@validate_args
def pose_stack_from_canonical_form(
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
    """Create a PoseStack, resolving which block type is requested by the
    presence and absence of the provided atoms for each residue type.
    There are five required arguments and several optional arguments.

    Arguments:
    canonical_ordering: an object that describes the set of atoms that each
        residue type (aka block type) and all of its interchangable variants
        contain and the order in which those atoms should appear in the
        coords tensor
    packed_block_types: the object that holds score-term annotations needed
        by the score terms and which is intended to be shared between
        multiple PoseStacks for efficiency; the PoseStack this function
        creates will hold this packed_block_types object
    chain_id: an n-pose x max-n-residue tensor with an index for which chain
        each residue belongs to. Residues belonging to the same chain must be
        consecutive.
    res_types: an n-pose x max-n-residue tensor with the canonically
        ordered amino acid index for each residue. A sentinel value of "-1"
        should be used to indicate that a given position does not contain
        a residue (perhaps because it belongs to a pose with fewer than
        max-n-residue residues; each pose in the PoseStack is allowed to
        have fewer than max-n-residue residues).
    coords: an n-pose x max-n-residue x max-n-atoms-per-residue tensor
        providing the coordinates of some or all of the atoms. The order
        in which atoms should appear in this tensor is given by the
        CanonicalOrdering object. Any atoms whose coordinates are not
        being provided to this function must have their coordinates marked
        as NaN. Any atom with a coordinate of NaN will be taken as "not
        present" and all others (including any coordinates at the origin, e.g.)
        will be treated as if they "are present." Note: "present" here
        means "the coordinate is being provided" and not "this atom
        should be modeled;" conversely, there is no way to say "do not
        include a particular atom in tmol calculations."

        Currently, all heavy atoms must be provided to tmol except
        leaf atoms. A "leaf atom" is one that has no atoms that use it
        as a parent or grand parent when describing their icoors.
        Hydrogen atoms are all leaf atoms. Backbone carbonyl oxygens
        are also leaf atoms. Even though hydrogen atoms are optional,
        the hydroxyl hydrogens on SER, THR, and TYR are recommended
        as tmol will build them suboptimally: at a dihedral of 180
        regardless of the presence of nearby hydrogen-bond acceptors

    disulfides: an optional n-total-disulfides x 3 tensor. If this argument
        is not provided, then the coordinates of SG atoms on CYS residues
        will be used to determine which pairs are closest to each other and
        within a cutoff distance of 2.5A and declare those SGs as forming
        disulfide bonds. This means that SGs slightly longer than 2.5A
        will not be detected. If you should know which pairs of cysteines
        form disulfide bonds, then you can provide their pairing:
        [ [pose_ind, cys1_ind, cys2_ind], ...].

    find_additional_disulfides: an optional boolean argument to control whether
        to look for disulfide bonds between pairs of CYS residues that are
        not listed in the "disulfides" argument. By default, this is True,
        but if you want to skip disulfide detection or want to prevent
        unpaired CYS from being locked into disulfides, then set this flag
        to False

    res_not_connected: an optional input used to indicate that a given (polymeric)
        residue is not connected to either its previous or next residue; for
        termini residues, they will not be built with their termini-variant
        types. The purpose is to allow the user to include a subset of the
        residues in a protein where a series of "gap" residues can be omitted
        between i and i+1 without those two residues being treated as if they
        are chemically bonded. This will keep the Ramachandran term from scoring
        nonsense dihdral angles and will keep the cart-bonded term from scoring
        nonsense bond lengths and angles.


<<<<<<< HEAD
    Optional return values:
        If any of the following flags are provided, this function will return a tuple
        instead of just a pose stack, with the first argument being the pose stack and
        the second argument being a dictionary with keys corresponding to the requested
        values.

        return_chain_ind: return the chain-index tensor as "chain_ind" that has been
            "left-justified" from the chain

        return_atom_mapping: return the mapping for atoms in the canonical-form tensor
            to their PoseStack index; this could be used to update the coordinates
            in a PoseStack without rebuilding it (as long as the chemical identity
            is meant to be unchanged) or to perhaps remap derivatives to or from
            pose stack ordering. If requested, the atom mapping will returned as two tensors
            under the keys "ps_atom_mapping" and "can_atom_mapping" by this function:
                ps, opt_vals = pose_stack_from_canonical_form(
                    ...,
                    return_atom_mapping=True
                )
                t1 = opt_vals["can_atom_mapping"]
                t2 = opt_vals["ps_atom_mapping"]
                can_ord_coords[
                    t1[:, 0], t1[:, 1], t1[:, 2]
                ] = ps.coords[
                    t2[:, 0], t2[:, 1]
                ]
            where can_atom_mapping is a tensor nats x 3 where
            - position [i, 0] is the pose index
            - position [i, 1] is the residue index, and
            - position [i, 2] is the canonical-ordering atom index
            and ps_atom_mapping is a tensor nats x 2 where
            - position [i, 0] is the pose index, and
            - position [i, 1] is the pose-ordered atom index

        return_block_has_missing_atoms: returns a [n_pose x max_n_res] bool
            tensor in the dictionary under the key "block_has_missing_atoms" with
            elements being true iff any non-leaf atoms were missing (NaN). To be used
            with a packer to build these missing atoms. If this argument is False, an
            exception will be thrown when these missing atoms are encountered.
||||||| constructed merge base
=======
    return_block_has_missing_atoms: returns a [n_pose x max_n_res] bool tensor with
        elements being true iff any non-leaf atoms were missing (NaN). To be used
        with a packer to build these missing atoms. If this argument is False, an
        exception will be thrown when these missing atoms are encountered.
>>>>>>> Default is now to fail on missing non-leaf atoms (like before), with an option to return a tensor of the missing atoms. Add check to C++ to ensure we aren't deriving locations from NaN atoms.
    """

    from tmol.io.details.left_justify_canonical_form import left_justify_canonical_form
    from tmol.io.details.disulfide_search import find_disulfides
    from tmol.io.details.his_taut_resolution import resolve_his_tautomerization
    from tmol.io.details.select_from_canonical import (
        assign_block_types,
        take_block_type_atoms_from_canonical,
    )
    from tmol.io.details.build_missing_leaf_atoms import build_missing_leaf_atoms

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

    # 3
    found_disulfides, res_type_variants = find_disulfides(
        canonical_ordering, res_types, coords, disulfides, find_additional_disulfides
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
    if atom_occupancy is not None:
        atom_occupancy_pose_layout = numpy.full(
            pose_stack_coords.shape[:2], 1.0, dtype=numpy.float32
        )
        atom_occupancy_pose_layout[pose_at_is_real] = atom_occupancy[real_block_atoms]
        atom_occupancy = atom_occupancy_pose_layout
    if atom_b_factor is not None:
        atom_b_factor_pose_layout = numpy.full(
            pose_stack_coords.shape[:2], 0.0, dtype=numpy.float32
        )
        atom_b_factor_pose_layout[pose_at_is_real] = atom_b_factor[real_block_atoms]
        atom_b_factor = atom_b_factor_pose_layout

    pdb_info = PDBInfo(
        residue_labels=res_labels,
        residue_insertion_codes=res_ins_codes,
        chain_labels=chain_labels,
        atom_occupancy=atom_occupancy,
        atom_b_factor=atom_b_factor,
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
    else:
        return ps
