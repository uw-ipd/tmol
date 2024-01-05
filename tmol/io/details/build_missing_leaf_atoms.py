import numpy
import torch

from tmol.types.torch import Tensor
from tmol.types.functional import validate_args
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.score.chemical_database import (
    AtomTypeParamResolver,
)
from tmol.io.details.compiled.compiled import gen_pose_leaf_atoms


@validate_args
def build_missing_leaf_atoms(
    packed_block_types: PackedBlockTypes,
    atom_type_resolver: AtomTypeParamResolver,
    block_types64: Tensor[torch.int64][:, :],
    real_block_atoms: Tensor[torch.bool][:, :, :],
    block_coords: Tensor[torch.float32][:, :, :, 3],
    block_atom_missing: Tensor[torch.int32][:, :, :],
    inter_residue_connections: Tensor[torch.int32][:, :, :, 2],
):
    """Convert the block layout into the condensed layout used by PoseStack and
    build any missing "leaf" atoms at the same time. This is a fully differentiable
    process, and so gradients accumulated for any leaf atoms that were absent
    from the input structure will be distributed across the atoms that define the
    geometry of those atoms. A leaf atom is an atom that is not a parent to any other
    atom; these include hydrogens and carbonyl/carboxyl oxygens.
    """

    # ok,
    # we're going to call gen_pose_leaf_atoms,
    # but first we need to prepare the input tensors
    # that are going to use
    pbt = packed_block_types
    device = pbt.device
    n_poses = block_coords.shape[0]
    max_n_blocks = block_coords.shape[1]

    # make sure we have all the data we need
    _annotate_packed_block_types_atom_is_leaf_atom(pbt, atom_type_resolver)
    _annotate_packed_block_types_w_leaf_atom_icoors(pbt)

    n_atoms = torch.zeros((n_poses, max_n_blocks), dtype=torch.int32, device=device)
    real_blocks = block_types64 != -1
    n_atoms[real_blocks] = packed_block_types.n_atoms[block_types64[real_blocks]]

    n_ats_inccumsum = torch.cumsum(n_atoms, dim=1, dtype=torch.int32)
    max_n_ats = torch.max(n_ats_inccumsum[:, -1])
    block_coord_offset = torch.cat(
        (
            torch.zeros((n_poses, 1), dtype=torch.int32, device=device),
            n_ats_inccumsum[:, :-1],
        ),
        dim=1,
    )
    pose_at_is_real = (
        torch.arange(max_n_ats, dtype=torch.int64, device=device).repeat(n_poses, 1)
        < n_ats_inccumsum[:, -1:]
    )

    pose_like_coords = torch.zeros(
        (n_poses, max_n_ats, 3), dtype=torch.float32, device=device
    )
    pose_like_coords[pose_at_is_real] = block_coords[real_block_atoms]

    # SHORT CIRCUIT
    # return (pose_like_coords, block_coord_offset)

    block_at_is_leaf = torch.zeros(
        (n_poses, max_n_blocks, pbt.max_n_atoms), dtype=torch.int32, device=device
    )
    block_at_is_leaf[real_blocks] = pbt.is_leaf_atom[block_types64[real_blocks]]

    # multiplication of booleans-as-integers is equivalent to a logical "and"
    block_leaf_atom_is_missing = block_at_is_leaf * block_atom_missing
    pose_stack_atom_is_missing = torch.zeros(
        (n_poses, max_n_ats), dtype=torch.int32, device=device
    )
    pose_stack_atom_is_missing[pose_at_is_real] = block_leaf_atom_is_missing[
        real_block_atoms
    ]

    # ok, we're ready
    new_pose_coords = gen_pose_leaf_atoms(
        pose_like_coords,
        block_leaf_atom_is_missing,
        pose_stack_atom_is_missing,
        block_coord_offset,
        block_types64.to(torch.int32),
        inter_residue_connections,
        pbt.n_atoms,
        pbt.atom_downstream_of_conn,
        pbt.build_missing_leaf_atom_icoor_atom_ancestor_uaids,
        pbt.build_missing_leaf_atom_icoor_geom,
        pbt.build_missing_leaf_atom_icoor_atom_ancestor_uaids_backup,
        pbt.build_missing_leaf_atom_icoor_geom_backup,
    )

    return new_pose_coords, block_coord_offset


@validate_args
def _annotate_packed_block_types_atom_is_leaf_atom(
    pbt: PackedBlockTypes, atom_type_resolver: AtomTypeParamResolver
):
    if hasattr(pbt, "is_leaf_atom"):
        return

    # annotate the block types, then concatenate

    def ti32(x):
        return torch.tensor(x, dtype=torch.int32, device=pbt.device)

    is_leaf_atom = torch.zeros(
        (pbt.n_types, pbt.max_n_atoms), dtype=torch.int32, device=pbt.device
    )

    for i, block_type in enumerate(pbt.active_block_types):
        is_parent = numpy.zeros(block_type.n_atoms, dtype=bool)
        icoor_is_parent = numpy.zeros(block_type.n_icoors, dtype=bool)
        icoor_is_parent[block_type.icoors_ancestors[:, 0]] = True

        icoorind_to_atomind = numpy.full(block_type.n_icoors, -1, dtype=numpy.int32)
        icoorind_to_atomind[block_type.at_to_icoor_ind] = numpy.arange(
            block_type.n_atoms, dtype=numpy.int32
        )

        is_parent[icoorind_to_atomind[icoorind_to_atomind != -1]] = icoor_is_parent[
            icoorind_to_atomind != -1
        ]

        # we also need to turn off "is leaf" for any atom that is the
        # fourth one defining a named dihedral:
        fourth_torsion_atoms = block_type.ordered_torsions[:, 3, 0]
        real_fourth_torsion_atoms = fourth_torsion_atoms[fourth_torsion_atoms != -1]

        is_parent[real_fourth_torsion_atoms] = True

        is_leaf = numpy.logical_not(is_parent)

        setattr(block_type, "is_leaf_atom", is_leaf)
        is_leaf_atom[i, : block_type.n_atoms] = ti32(is_leaf)

    setattr(pbt, "is_leaf_atom", is_leaf_atom)


@validate_args
def _annotate_packed_block_types_w_leaf_atom_icoors(pbt: PackedBlockTypes):
    if hasattr(pbt, "build_missing_leaf_atom_icoor_atom_ancestor_uaids"):
        assert hasattr(pbt, "build_missing_leaf_atom_icoor_geom")
        assert hasattr(pbt, "build_missing_leaf_atom_icoor_atom_ancestor_uaids_backup")
        assert hasattr(pbt, "build_missing_leaf_atom_icoor_geom_backup")
        return

    assert hasattr(pbt, "is_leaf_atom")
    assert hasattr(pbt, "atom_is_hydrogen")
    icoor_atom_ancestor_uaids = numpy.full(
        (pbt.n_types, pbt.max_n_atoms, 3, 3), -1, dtype=numpy.int32
    )
    icoor_geom = numpy.full((pbt.n_types, pbt.max_n_atoms, 3), -1, dtype=numpy.float32)
    icoor_atom_ancestor_uaids_backup = numpy.full(
        (pbt.n_types, pbt.max_n_atoms, 3, 3), -1, dtype=numpy.int32
    )
    icoor_geom_backup = numpy.full(
        (pbt.n_types, pbt.max_n_atoms, 3), -1, dtype=numpy.float32
    )
    atom_is_hydrogen_cpu = pbt.atom_is_hydrogen.cpu()
    for i, bt in enumerate(pbt.active_block_types):
        geom, uaids, geom_bu, uaids_bu = _determine_leaf_atom_icoors_for_block_type(
            bt, atom_is_hydrogen_cpu[i, :]
        )

        icoor_geom[i, : bt.n_atoms] = geom
        icoor_atom_ancestor_uaids[i, : bt.n_atoms] = uaids
        icoor_geom_backup[i, : bt.n_atoms] = geom_bu
        icoor_atom_ancestor_uaids_backup[i, : bt.n_atoms] = uaids_bu

    icoor_geom = torch.tensor(icoor_geom, dtype=torch.float32, device=pbt.device)
    icoor_atom_ancestor_uaids = torch.tensor(
        icoor_atom_ancestor_uaids, dtype=torch.int32, device=pbt.device
    )
    icoor_geom_backup = torch.tensor(
        icoor_geom_backup, dtype=torch.float32, device=pbt.device
    )
    icoor_atom_ancestor_uaids_backup = torch.tensor(
        icoor_atom_ancestor_uaids_backup, dtype=torch.int32, device=pbt.device
    )

    setattr(
        pbt,
        "build_missing_leaf_atom_icoor_atom_ancestor_uaids",
        icoor_atom_ancestor_uaids,
    )
    setattr(pbt, "build_missing_leaf_atom_icoor_geom", icoor_geom)
    setattr(
        pbt,
        "build_missing_leaf_atom_icoor_atom_ancestor_uaids_backup",
        icoor_atom_ancestor_uaids_backup,
    )
    setattr(pbt, "build_missing_leaf_atom_icoor_geom_backup", icoor_geom_backup)


def _determine_leaf_atom_icoors_for_block_type(bt, atom_is_hydrogen):
    bt_icoor_uaids = numpy.full((bt.n_atoms, 3, 3), -1, dtype=numpy.int32)
    bt_icoor_geom = numpy.full((bt.n_atoms, 3), 0, dtype=numpy.float32)
    bt_icoor_uaids_backup = numpy.full((bt.n_atoms, 3, 3), -1, dtype=numpy.int32)
    bt_icoor_geom_backup = numpy.full((bt.n_atoms, 3), 0, dtype=numpy.float32)
    for j, at in enumerate(bt.atoms):
        atname = at.name
        j_icoor_ind = bt.icoors_index[atname]
        j_icoor = bt.icoors[j_icoor_ind]

        def uaid_for_at(icoor_at_name):
            if icoor_at_name == "up":
                return (-1, bt.up_connection_ind, 0)
            elif icoor_at_name == "down":
                return (-1, bt.down_connection_ind, 0)
            else:
                return (bt.atom_to_idx[icoor_at_name], -1, -1)

        def icoor_at_is_leaf(icoor_at_name):
            if icoor_at_name not in bt.atom_to_idx:
                return 0
            return bt.is_leaf_atom[bt.atom_to_idx[icoor_at_name]]

        def icoor_at_is_h(icoor_at_name):
            if icoor_at_name not in bt.atom_to_idx:
                return 0
            return atom_is_hydrogen[bt.atom_to_idx[icoor_at_name]]

        def icoor_at_is_inter_res(icoor_at_name):
            return icoor_at_name not in bt.atom_to_idx

        # ok, let's turn p, gp, and ggp into uaids
        # if ggp is a leaf, then we need to recurse backwards through the ggps
        # and accumulate the phi offsets

        p_uaid = uaid_for_at(j_icoor.parent)
        gp_uaid = uaid_for_at(j_icoor.grand_parent)

        phi = j_icoor.phi
        theta = numpy.pi - j_icoor.theta
        d = j_icoor.d

        if icoor_at_is_h(j_icoor.great_grand_parent):
            # use phi offsets from the non-leaf ggp* ancestor of the ggp
            # as the default strategy for building coords for hydrogen atoms
            while icoor_at_is_leaf(j_icoor.great_grand_parent):
                ggp_ind = bt.icoors_index[j_icoor.great_grand_parent]
                j_icoor = bt.icoors[ggp_ind]
                phi += j_icoor.phi
        else:
            # if the ggp is not a hydrogen, even if it's a leaf atom, then try
            # and build the coordinate for this atom based on its position first
            # before falling back on the non-leaf ggp* ancestor. This "general"
            # code is specifically for building the OXT atom on a cterm residue
            # when the O atom is provided; the phi should be 180 off O and not
            # 260 off N (and some unknown offset of O).
            pass
        ggp_uaid = uaid_for_at(j_icoor.great_grand_parent)

        ggp_ind_backup = None
        phi_backup = phi
        if icoor_at_is_inter_res(j_icoor.great_grand_parent):
            # Logic for when the great-grand parent atom is in another residue
            # and is absent. This "general" logic is specifically for building
            # the H atom on a residue where i-1 does not exist or is not
            # chemically bonded to residue i.
            while icoor_at_is_inter_res(j_icoor.great_grand_parent):
                ggp_ind_backup = bt.icoors_index[j_icoor.great_grand_parent]
                j_icoor = bt.icoors[ggp_ind_backup]
                phi_backup += j_icoor.phi
        elif not icoor_at_is_h(j_icoor.great_grand_parent):
            # Logic for handling when the heavy-atom great-grand parent,
            # which itself is a leaf atom, is absent. This "general" logic
            # is specifically for building the OXT atom on a cterm residue
            # when the O atom is given but OXT is not.
            while icoor_at_is_leaf(j_icoor.great_grand_parent):
                ggp_ind_backup = bt.icoors_index[j_icoor.great_grand_parent]
                j_icoor = bt.icoors[ggp_ind_backup]
                phi_backup += j_icoor.phi

        bt_icoor_uaids[j, 0] = numpy.array(p_uaid, dtype=numpy.int32)
        bt_icoor_uaids[j, 1] = numpy.array(gp_uaid, dtype=numpy.int32)
        bt_icoor_uaids[j, 2] = numpy.array(ggp_uaid, dtype=numpy.int32)

        bt_icoor_geom[j, 0] = phi
        bt_icoor_geom[j, 1] = theta
        bt_icoor_geom[j, 2] = d
        if ggp_ind_backup is not None:
            ggp_uaid_backup = uaid_for_at(j_icoor.great_grand_parent)
            bt_icoor_uaids_backup[j, 0] = numpy.array(p_uaid, dtype=numpy.int32)
            bt_icoor_uaids_backup[j, 1] = numpy.array(gp_uaid, dtype=numpy.int32)
            bt_icoor_uaids_backup[j, 2] = numpy.array(
                ggp_uaid_backup, dtype=numpy.int32
            )

            bt_icoor_geom_backup[j, 0] = phi_backup
            bt_icoor_geom_backup[j, 1] = theta
            bt_icoor_geom_backup[j, 2] = d
    return bt_icoor_geom, bt_icoor_uaids, bt_icoor_geom_backup, bt_icoor_uaids_backup
