import torch
from tmol.types.torch import Tensor
from typing import Optional
from tmol.pose.pose_stack import PoseStack
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.io.canonical_ordering import CanonicalOrdering


def pose_stack_from_canonical_form(
    canonical_ordering: CanonicalOrdering,
    pbt: PackedBlockTypes,
    chain_id: Tensor[torch.int32][:, :],
    res_types: Tensor[torch.int32][:, :],
    coords: Tensor[torch.float32][:, :, :, 3],
    *,
    atom_is_present: Optional[Tensor[torch.bool][:, :, :]] = None,
    disulfides: Optional[Tensor[torch.int64][:, 3]] = None,
    find_additional_disulfides: Optional[bool] = True,
    res_not_connected: Optional[Tensor[torch.bool][:, :, 2]] = None,
    return_chain_ind: bool = False,
    return_atom_mapping: bool = False,
):
    """Create a PoseStack given atom coordinates in canonical ordering

    Arguments:
    chain_id: an n-pose x max-n-residue tensor with an index for which chain
        each residue belongs to. Residues belonging to the same chain must be
        consecutive.
    res_types: an n-pose x max-n-residue tensor with the canonically
        ordered amino acid index for each residue. A sentinel value of "-1"
        should be used to indicate that a given position does not contain
        a residue (perhaps because it belongs to a pose with fewer than
        max-n-residue residues).
    coords: an n-pose x max-n-residue x max-n-atoms-per-residue tensor
        providing the coordinates of some or all of the atoms. The order
        in which atoms should appear in this tensor is given in
        tmol.io.canonical_ordering.ordered_canonical_aa_atoms (see also
        tmol.io.canonical_ordering.ordered_canonical_aa_atoms_v2 for
        the older PDB v2 atom naming scheme). It is recommended that
        atoms whose coordinates are not being provided to this function
        have their coordinates marked as NaN to ensure that tmol is not
        reading from positions in this array that it should not be and
        thus delivering inaccurate energies.
    atom_is_present: an n-pose x max-n-residue x max-n-atoms-per-residue
        tensor answering the yes/no question of whether an atom's
        coordinate is being provided; 0 for no, 1 for yes. If this
        tensor is not provided, then any coordinates in the "coords"
        tensor with a coordinate of NaN will be taken as "not present"
        and all others (including any coordinates at the origin, e.g.)
        will be treated as if they "are present." Note: "present" here
        means "the coordinate is being provided" and not "this atom
        should be modeled;" conversely, there is no way to say "do not
        include a particular atom in tmol calculations."

        Currently, all heavy atoms must be provided to tmol except
        leaf atoms. A "leaf atom" is one that has no atoms that use it
        as a parent or grand parent when describing their icoors.
        Hydrogen atoms are all leaf atoms. Backbone carbonyl oxygens
        are also leaf atoms. Even thoguh hydrogen atoms are optional,
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

    return_chain_ind: return the chain-index tensor that has been "left-justified"
        from the chain

    return_atom_mapping: return the mapping for atoms in the canonical-form tensor
        to their PoseStack index; this could be used to update the coordinates
        in a PoseStack without rebuilding it (as long as the chemical identity
        is meant to be unchanged) or to perhaps remap derivatives to or from
        pose stack ordering. If requested, the atom mapping will be the last two
        arguments returned by this function, as two tensors:
            ps, t1, t2 = pose_stack_from_canonical_form(
                ...,
                return_atom_mapping=True
            )
            can_ord_coords[
                t1[:, 0], t1[:, 1], t1[:, 2]
            ] = ps.coords[
                t2[:, 0], t2[:, 1]
            ]
        where t1 is a tensor nats x 3 where
        - position [i, 0] is the pose index
        - position [i, 1] is the residue index, and
        - position [i, 2] is the canonical-ordering atom index
        and t2 is a tensor nats x 2 where
        - position [i, 0] is the pose index, and
        - position [i, 1] is the pose-ordered atom index

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

    # step 1: retrieve the global packed_block_types object with the 66
    #         canonical residue types
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

    if atom_is_present is None:
        atom_is_present = torch.all(torch.logical_not(torch.isnan(coords)), dim=3)
    else:
        # SANITY: don't give tmol NaNs
        if torch.any(
            torch.isnan(coords[atom_is_present.unsqueeze(3).expand(-1, -1, -1, 3) == 1])
        ):
            msg = "ERROR: NaN coordinate given in PoseStack construction for one or more atoms marked as present"
            raise ValueError(msg)

    # 1
    # this will return the same object each time to minimize the number
    # of calls to the setup_packed_block_types annotation functions
    # pbt, atom_type_resolver = default_canonical_packed_block_types(chain_id.device)

    # 2
    # "left justify" the input canonical-form residues: residues that are
    # given with a "-1" residue-type should be excised from the center of
    # their Poses to ensure that the polymeric-bond-detection logic
    # downstream will work properly. This effectively means "shifting left"
    # all the other residues in the Pose to fill the vacated slots.
    # print("2")
    (
        chain_id,
        res_types,
        coords,
        atom_is_present,
        disulfides,
        res_not_connected,
    ) = left_justify_canonical_form(
        chain_id, res_types, coords, atom_is_present, disulfides, res_not_connected
    )

    # 3
    # print("3")
    found_disulfides, res_type_variants = find_disulfides(
        canonical_ordering, res_types, coords, disulfides, find_additional_disulfides
    )

    # 4
    # print("4")
    (
        his_taut,
        res_type_variants,
        resolved_coords,
        resolved_atom_is_present,
    ) = resolve_his_tautomerization(
        canonical_ordering, res_types, res_type_variants, coords, atom_is_present
    )

    # 5
    # print("5")
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
    # print("6")
    (
        block_coords,
        missing_atoms,
        real_atoms,
        real_canonical_atom_inds,
    ) = take_block_type_atoms_from_canonical(
        pbt, block_types64, coords, atom_is_present
    )

    # 7
    # print("7")
    inter_residue_connections = inter_residue_connections64.to(torch.int32)
    pose_stack_coords, block_coord_offset = build_missing_leaf_atoms(
        pbt,
        block_types64,
        real_atoms,
        block_coords,
        missing_atoms,
        inter_residue_connections,
    )

    def i64(x):
        return x.to(torch.int64)

    def i32(x):
        return x.to(torch.int32)

    # 8
    # print("8")
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

    if return_chain_ind:
        if return_atom_mapping:
            return (
                ps,
                chain_id,
                can_atom_mapping,
                ps_atom_mapping,
            )
        else:
            return (ps, chain_id)
    else:
        if return_atom_mapping:
            return (ps, can_atom_mapping, ps_atom_mapping)
        else:
            return ps
