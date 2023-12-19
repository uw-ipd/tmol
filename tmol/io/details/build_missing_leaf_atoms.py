import numpy
import torch
import attr

from tmol.types.torch import Tensor
from tmol.types.array import NDArray
from tmol.types.functional import validate_args
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.chemical.restypes import RefinedResidueType

# from tmol.score.chemical_database import (
#     AtomTypeParamResolver,
# )
from tmol.io.details.compiled.compiled import gen_pose_leaf_atoms


@validate_args
def build_missing_leaf_atoms(
    packed_block_types: PackedBlockTypes,
    block_types64: Tensor[torch.int64][:, :],
    real_block_atoms: Tensor[torch.bool][:, :, :],
    block_coords: Tensor[torch.float32][:, :, :, 3],
    block_atom_missing: Tensor[torch.bool][:, :, :],
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
    _annotate_packed_block_types_atom_is_leaf_atom(pbt)
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
        (n_poses, max_n_blocks, pbt.max_n_atoms), dtype=torch.bool, device=device
    )
    block_at_is_leaf[real_blocks] = pbt.is_leaf_atom[block_types64[real_blocks]]

    # error checking: have we been asked to build an atom that cannot be built?
    non_leaf_atom_is_missing = torch.logical_and(
        block_atom_missing, torch.logical_not(block_at_is_leaf)
    )
    if torch.any(non_leaf_atom_is_missing):
        err_msg = []
        leaf_atom_missing_inds = torch.nonzero(non_leaf_atom_is_missing)
        for i in range(leaf_atom_missing_inds.shape[0]):
            i_bt_ind = block_types64[
                leaf_atom_missing_inds[i, 0], leaf_atom_missing_inds[i, 1]
            ]
            i_bt = packed_block_types.active_block_types[i_bt_ind]
            err_msg.append(
                " ".join(
                    [
                        "Error: missing non-leaf atom",
                        i_bt.atoms[leaf_atom_missing_inds[i, 2]].name,
                        "on residue",
                        str(leaf_atom_missing_inds[i, 1].item()),
                        i_bt.name,
                        "on pose",
                        str(leaf_atom_missing_inds[i, 0].item()),
                        "real res?",
                        str(
                            real_blocks[
                                leaf_atom_missing_inds[i, 0],
                                leaf_atom_missing_inds[i, 1],
                            ].item()
                        ),
                    ]
                )
            )
        # TO DO: useful error message
        raise ValueError("\n".join(err_msg))

    block_leaf_atom_is_missing = torch.logical_and(block_at_is_leaf, block_atom_missing)
    pose_stack_atom_is_missing = torch.zeros(
        (n_poses, max_n_ats), dtype=torch.bool, device=device
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
        pbt.build_missing_leaf_atom_icoor_ann.anc_uaids,
        pbt.build_missing_leaf_atom_icoor_ann.geom,
        pbt.build_missing_leaf_atom_icoor_ann.anc_uaids_backup,
        pbt.build_missing_leaf_atom_icoor_ann.geom_backup,
    )

    return new_pose_coords, block_coord_offset


@attr.s(auto_attribs=True, slots=True, frozen=True)
class BlockTypeLeafAtomsAnnotation:
    is_leaf: NDArray[bool][:]


@validate_args
def _annotate_packed_block_types_atom_is_leaf_atom(pbt: PackedBlockTypes):
    if hasattr(pbt, "is_leaf_atom"):
        return

    # annotate the block types, then concatenate
    is_leaf_atom = torch.zeros((pbt.n_types, pbt.max_n_atoms), dtype=torch.bool)

    for i, block_type in enumerate(pbt.active_block_types):
        _annotate_block_type_atom_is_leaf_atom(block_type, pbt.atom_is_hydrogen[i, :])
        is_leaf_atom[i, : block_type.n_atoms] = torch.from_numpy(
            block_type.leaf_atom_ann.is_leaf
        )

    setattr(pbt, "is_leaf_atom", is_leaf_atom.to(device=pbt.device))


@validate_args
def _annotate_block_type_atom_is_leaf_atom(
    block_type: RefinedResidueType, is_hydrogen: Tensor[torch.int32][:]
):
    if hasattr(block_type, "leaf_atom_ann"):
        return
    is_hydrogen = is_hydrogen.cpu().numpy()[
        : block_type.n_atoms
    ]  # ugh, should this just live in the block type?!
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

    # We also need to turn off "is leaf" for any atom that is the
    # fourth one defining a named dihedral, unless, however, that atom
    # is a hydroxyl / free methyl (e.g. chi4 would be on methionine)
    # What atoms / dihedrals would this affect? OD1 / chi2 on ASP and
    # OE1 / chi3 on GLU are not parent atoms and so would be called
    # leaf atoms. Perhaps we could allow these atoms to be missing and
    # build them with a garbage chi2 / chi3 of 180 the way we build
    # missing hydroxyls? It seems like a poor decision.
    fourth_torsion_atoms = block_type.ordered_torsions[:, 3, 0]
    real_fourth_torsion_atoms = fourth_torsion_atoms[fourth_torsion_atoms != -1]

    is_fourth_chi_atom = numpy.zeros(block_type.n_atoms, dtype=bool)
    is_fourth_chi_atom[real_fourth_torsion_atoms] = True
    is_fourth_chi_atom[is_hydrogen == 1] = False
    is_parent = numpy.logical_or(is_parent, is_fourth_chi_atom)

    is_leaf = numpy.logical_not(is_parent)

    # TEMP? TO DO?
    # special case: we cannot build missing atoms if there are not enough
    # atoms to define a coordinate frame
    if block_type.n_atoms == 3:
        is_leaf[:] = False

    leaf_annotation = BlockTypeLeafAtomsAnnotation(is_leaf)

    setattr(block_type, "leaf_atom_ann", leaf_annotation)


@attr.s(auto_attribs=True, slots=True, frozen=True)
class BlockTypeLeafAtomICoorAnnotation:
    geom: NDArray[numpy.float32][:, 3]
    anc_uaids: NDArray[numpy.int32][:, 3, 3]
    geom_backup: NDArray[numpy.float32][:, 3]
    anc_uaids_backup: NDArray[numpy.int32][:, 3, 3]


@attr.s(auto_attribs=True, slots=True, frozen=True)
class PackedBlockTypesLeafAtomICoorAnnotation:
    geom: Tensor[torch.float32][:, :, 3]
    anc_uaids: Tensor[torch.int32][:, :, 3, 3]
    geom_backup: Tensor[torch.float32][:, :, 3]
    anc_uaids_backup: Tensor[torch.int32][:, :, 3, 3]


@validate_args
def _annotate_packed_block_types_w_leaf_atom_icoors(pbt: PackedBlockTypes):
    if hasattr(pbt, "build_missing_leaf_atom_icoor_ann"):
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
        _determine_leaf_atom_icoors_for_block_type(bt, atom_is_hydrogen_cpu[i, :])

        ann = bt.leaf_atom_icoor_ann
        icoor_geom[i, : bt.n_atoms] = ann.geom
        icoor_atom_ancestor_uaids[i, : bt.n_atoms] = ann.anc_uaids
        icoor_geom_backup[i, : bt.n_atoms] = ann.geom_backup
        icoor_atom_ancestor_uaids_backup[i, : bt.n_atoms] = ann.anc_uaids_backup

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

    ann = PackedBlockTypesLeafAtomICoorAnnotation(
        geom=icoor_geom,
        anc_uaids=icoor_atom_ancestor_uaids,
        geom_backup=icoor_geom_backup,
        anc_uaids_backup=icoor_atom_ancestor_uaids_backup,
    )

    setattr(
        pbt,
        "build_missing_leaf_atom_icoor_ann",
        ann,
    )


def _uaid_for_at(bt, icoor_at_name):
    if icoor_at_name == "up":
        return (-1, bt.up_connection_ind, 0)
    elif icoor_at_name == "down":
        return (-1, bt.down_connection_ind, 0)
    else:
        return (bt.atom_to_idx[icoor_at_name], -1, -1)


def _icoor_at_is_leaf(bt, icoor_at_name):
    if icoor_at_name not in bt.atom_to_idx:
        return 0
    return bt.leaf_atom_ann.is_leaf[bt.atom_to_idx[icoor_at_name]]


def _icoor_at_is_h(bt, atom_is_hydrogen, icoor_at_name):
    if icoor_at_name not in bt.atom_to_idx:
        return 0
    return atom_is_hydrogen[bt.atom_to_idx[icoor_at_name]]


def _icoor_at_is_inter_res(bt, icoor_at_name):
    return icoor_at_name not in bt.atom_to_idx


def _determine_leaf_atom_icoors_for_block_type(bt, atom_is_hydrogen):
    if hasattr(bt, "leaf_atom_icoor_ann"):
        return

    icoor_uaids = numpy.full((bt.n_atoms, 3, 3), -1, dtype=numpy.int32)
    icoor_geom = numpy.full((bt.n_atoms, 3), 0, dtype=numpy.float32)
    icoor_uaids_backup = numpy.full((bt.n_atoms, 3, 3), -1, dtype=numpy.int32)
    icoor_geom_backup = numpy.full((bt.n_atoms, 3), 0, dtype=numpy.float32)

    # We cannot build hydrogen atoms on a water molecule, for example
    # because there isn't a (meaningful) coordinate frame we can create
    # from only a single xyz coordinate. So, for now, we will skip
    # water.
    if bt.n_atoms <= 3:
        ann = BlockTypeLeafAtomICoorAnnotation(
            geom=icoor_geom,
            anc_uaids=icoor_uaids,
            geom_backup=icoor_geom_backup,
            anc_uaids_backup=icoor_uaids_backup,
        )
        setattr(bt, "leaf_atom_icoor_ann", ann)
        return
    for j, at in enumerate(bt.atoms):
        atname = at.name
        j_icoor_ind = bt.icoors_index[atname]
        j_icoor = bt.icoors[j_icoor_ind]

        # ok, let's turn p, gp, and ggp into uaids
        # if ggp is a leaf, then we need to recurse backwards through the ggps
        # and accumulate the phi offsets

        p_uaid = _uaid_for_at(bt, j_icoor.parent)
        gp_uaid = _uaid_for_at(bt, j_icoor.grand_parent)

        phi = j_icoor.phi
        theta = numpy.pi - j_icoor.theta
        d = j_icoor.d

        if _icoor_at_is_h(bt, atom_is_hydrogen, j_icoor.great_grand_parent):
            # use phi offsets from the non-leaf ggp* ancestor of the ggp
            # as the default strategy for building coords for hydrogen atoms
            # Note that this algorithm only works if we do not have two atoms
            # that are each other's ggps / aren't their own ggps, which can
            # and will happen if, e.g., the residue has only three atoms total.
            # that is why we handle that case
            seen = numpy.zeros(bt.n_icoors, dtype=bool)
            seen[j_icoor_ind] = True
            while _icoor_at_is_leaf(bt, j_icoor.great_grand_parent):
                ggp_ind = bt.icoors_index[j_icoor.great_grand_parent]
                if seen[ggp_ind]:
                    # infinite loop. This should never happen.
                    print(
                        bt.name,
                        "ggp_ind",
                        ggp_ind,
                        bt.icoors[ggp_ind].name,
                        "from",
                        j,
                        bt.atoms[j].name,
                    )
                    raise RuntimeError(
                        "Infinite loop detected in icoor ancestor traversal for residue type "
                        + bt.name
                    )
                else:
                    seen[ggp_ind] = True
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
        ggp_uaid = _uaid_for_at(bt, j_icoor.great_grand_parent)

        ggp_ind_backup = None
        phi_backup = phi
        if _icoor_at_is_inter_res(bt, j_icoor.great_grand_parent):
            # Logic for when the great-grand parent atom is in another residue
            # and is absent. This "general" logic is specifically for building
            # the H atom on a residue where i-1 does not exist or is not
            # chemically bonded to residue i.
            while _icoor_at_is_inter_res(bt, j_icoor.great_grand_parent):
                ggp_ind_backup = bt.icoors_index[j_icoor.great_grand_parent]
                j_icoor = bt.icoors[ggp_ind_backup]
                phi_backup += j_icoor.phi
        elif not _icoor_at_is_h(bt, atom_is_hydrogen, j_icoor.great_grand_parent):
            # Logic for handling when the heavy-atom great-grand parent,
            # which itself is a leaf atom, is absent. This "general" logic
            # is specifically for building the OXT atom on a cterm residue
            # when the O atom is given but OXT is not.
            while _icoor_at_is_leaf(bt, j_icoor.great_grand_parent):
                ggp_ind_backup = bt.icoors_index[j_icoor.great_grand_parent]
                j_icoor = bt.icoors[ggp_ind_backup]
                phi_backup += j_icoor.phi

        icoor_uaids[j, 0] = numpy.array(p_uaid, dtype=numpy.int32)
        icoor_uaids[j, 1] = numpy.array(gp_uaid, dtype=numpy.int32)
        icoor_uaids[j, 2] = numpy.array(ggp_uaid, dtype=numpy.int32)

        icoor_geom[j, 0] = phi
        icoor_geom[j, 1] = theta
        icoor_geom[j, 2] = d
        if ggp_ind_backup is not None:
            ggp_uaid_backup = _uaid_for_at(bt, j_icoor.great_grand_parent)
            icoor_uaids_backup[j, 0] = numpy.array(p_uaid, dtype=numpy.int32)
            icoor_uaids_backup[j, 1] = numpy.array(gp_uaid, dtype=numpy.int32)
            icoor_uaids_backup[j, 2] = numpy.array(ggp_uaid_backup, dtype=numpy.int32)

            icoor_geom_backup[j, 0] = phi_backup
            icoor_geom_backup[j, 1] = theta
            icoor_geom_backup[j, 2] = d

    ann = BlockTypeLeafAtomICoorAnnotation(
        geom=icoor_geom,
        anc_uaids=icoor_uaids,
        geom_backup=icoor_geom_backup,
        anc_uaids_backup=icoor_uaids_backup,
    )
    setattr(bt, "leaf_atom_icoor_ann", ann)
    # return bt_icoor_geom, bt_icoor_uaids, bt_icoor_geom_backup, bt_icoor_uaids_backup
