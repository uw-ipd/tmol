import numpy
import numba
import toolz
import torch
import attr

from typing import List, Tuple

from tmol.types.array import NDArray
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args

from tmol.utility.tensor.common_operations import exclusive_cumsum1d, stretch
from tmol.database.chemical import ChemicalDatabase
from tmol.kinematics.datatypes import KinForest, NodeType
from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.pack.packer_task import PackerTask
from tmol.pack.rotamer.rotamer_set import RotamerSet
from tmol.pack.rotamer.chi_sampler import ChiSampler
from tmol.numeric.dihedrals import coord_dihedrals

from tmol.pack.rotamer.single_residue_kinforest import (
    construct_single_residue_kinforest,
    coalesce_single_residue_kinforests,
)
from tmol.pack.rotamer.mainchain_fingerprint import (
    annotate_residue_type_with_sampler_fingerprints,
    find_unique_fingerprints,
)


def _build_chi4_atom_table(pbt):
    """Return (n_types, max_n_chi, 4) int32 array of RTO atom indices for each chi.

    Entries are -1 where the residue type has fewer than max_n_chi chi angles.
    Built once and cached on pbt as _chi4_atom_table.
    """
    if hasattr(pbt, "_chi4_atom_table"):
        return pbt._chi4_atom_table

    n_types = pbt.n_types
    max_n_chi = (
        max(
            sum(1 for k in rt.torsion_to_uaids if k.startswith("chi"))
            for rt in pbt.active_block_types
        )
        or 1
    )

    table = numpy.full((n_types, max_n_chi, 4), -1, dtype=numpy.int32)
    for ti, rt in enumerate(pbt.active_block_types):
        chi_names = sorted(k for k in rt.torsion_to_uaids if k.startswith("chi"))
        for ci, chi_name in enumerate(chi_names):
            uaids = rt.torsion_to_uaids[chi_name]
            table[ti, ci] = [int(u[0]) for u in uaids]

    # cache to avoid rebuilding on subsequent calls
    object.__setattr__(pbt, "_chi4_atom_table", table)
    return table


def _build_chi_phi_c_corrections(pbt):
    """Precompute phi_c correction per (block_type, chi_index), cached on pbt.

    For each chi angle the relationship between what is written to phi_c and
    what forward-kin produces as the actual dihedral is:

        chi_measured = phi_c + offset

    where offset = chi_ideal - phi_c_ideal is constant for a given residue
    type and chi index.  This holds for both jump-parent atoms (chi1, where
    the jump frame introduces an offset) and bond-parent atoms (chi2+, where
    a non-zero torsion ICOOR introduces the offset).

    Returns (n_types, max_n_chi) float32 array; 0.0 where not applicable.
    """
    if hasattr(pbt, "_chi_phi_c_corrections"):
        return pbt._chi_phi_c_corrections

    chi4_table = _build_chi4_atom_table(pbt)  # (n_types, max_n_chi, 4)
    kfidx = pbt.rotamer_kinforest.kinforest_idx  # (n_types, max_n_atoms)
    dofs_ideal = pbt.rotamer_kinforest.dofs_ideal  # (n_types, max_n_atoms, 9)
    parent_arr = pbt.rotamer_kinforest.parent  # (n_types, max_n_atoms) KFO order
    doftype_arr = pbt.rotamer_kinforest.doftype  # (n_types, max_n_atoms) KFO order

    n_types, max_n_chi = chi4_table.shape[:2]
    corrections = numpy.zeros((n_types, max_n_chi), dtype=numpy.float32)

    for ti, rt in enumerate(pbt.active_block_types):
        # ideal coords in restype atom order
        ideal_coords = torch.tensor(
            rt.ideal_coords[rt.at_to_icoor_ind], dtype=torch.float64
        )
        chi_names = sorted(k for k in rt.torsion_to_uaids if k.startswith("chi"))
        for ci in range(len(chi_names)):
            four_rto = chi4_table[ti, ci]
            if any(a < 0 for a in four_rto):
                continue
            cda_kfo = int(kfidx[ti, four_rto[2]])
            # For root children (jump parents), the correction is pose-dependent
            # and cannot be precomputed; leave it as 0 and let
            # correct_phi_c_for_jump_parents handle it at build time.
            parent_kfo = int(parent_arr[ti, cda_kfo])
            if doftype_arr[ti, parent_kfo] == NodeType.jump:
                continue
            xyzs = ideal_coords[four_rto]
            chi_ideal = float(
                coord_dihedrals(xyzs[0:1], xyzs[1:2], xyzs[2:3], xyzs[3:4])[0]
            )
            phi_c_ideal = float(dofs_ideal[ti, cda_kfo, 3])
            corrections[ti, ci] = chi_ideal - phi_c_ideal

    object.__setattr__(pbt, "_chi_phi_c_corrections", corrections)
    return corrections


def correct_phi_c_for_jump_parents(
    pbt,
    conformer_samples,
    new_ind_for_sampler_rotamer,
    block_type_ind_for_conformer_torch,
    n_atoms_offset_for_conformer_torch,
    conformer_kinforest,
    nodes,
    scans,
    gens,
    conf_dofs_kto,
):
    """For chi-defining atoms whose kinforest parent is a jump atom, the phi_c
    written by assign_chi_dofs_from_samples does not directly map to the
    chi dihedral angle measured from coordinates.  This function:
      1. Does a trial forward pass with the current DOFs.
      2. For each such atom, measures the actual dihedral from the trial coords.
      3. Adds (intended - measured) to conf_dofs_kto[atom_kto, 3] so the
         final forward pass produces the correct geometry.
    """
    # trial forward pass to get coords in RTO
    n_rots = block_type_ind_for_conformer_torch.shape[0]
    n_at_total = (
        n_atoms_offset_for_conformer_torch[-1].item()
        + pbt.n_atoms[block_type_ind_for_conformer_torch[-1]].item()
    )
    trial_coords_rto = calculate_rotamer_coords(
        pbt, n_rots, n_at_total, conformer_kinforest, nodes, scans, gens, conf_dofs_kto
    )

    doftype_cpu = conformer_kinforest.doftype.cpu().numpy()
    parent_cpu = conformer_kinforest.parent.cpu().numpy()
    kfidx = pbt.rotamer_kinforest.kinforest_idx  # (n_types, max_n_atoms) numpy int32
    bt_ind_np = block_type_ind_for_conformer_torch.cpu().numpy()
    at_off_np = n_atoms_offset_for_conformer_torch.cpu().numpy()
    chi4_table = _build_chi4_atom_table(pbt)  # (n_types, max_n_chi, 4)
    coords_np = trial_coords_rto.cpu().double().numpy()

    for i, sample_data in enumerate(conformer_samples):
        sample_dict = sample_data[2]
        if "chi_for_rotamers" not in sample_dict:
            continue
        chi_intended = sample_dict["chi_for_rotamers"].cpu()  # (n_samp_rots, max_n_chi)
        chi_atoms_rto = sample_dict[
            "chi_defining_atom_for_rotamer"
        ].cpu()  # (n_samp_rots, max_n_chi)
        if chi_intended.shape[0] == 0:
            continue

        _, max_n_chi = chi_atoms_rto.shape
        conf_inds = new_ind_for_sampler_rotamer[i].cpu().numpy()  # (n_samp,)

        # global rot index and per-rot metadata for every (samp_rot, chi) pair
        g_rot = conf_inds  # (n_samp,)
        bt_idx = bt_ind_np[g_rot]  # (n_samp,)
        at_off = at_off_np[g_rot]  # (n_samp,)

        # chi-defining atom (RTO) for every (samp_rot, chi): (n_samp, max_n_chi)
        cda_rto = chi_atoms_rto.numpy()

        # KTO index of the chi-defining atom: (n_samp, max_n_chi)
        # kinforest_idx[bt, atom_rto] + at_off + 1 (virtual root offset)
        cda_kto = (
            kfidx[
                bt_idx[:, None], numpy.clip(cda_rto, 0, None)
            ]  # clip -1 before indexing
            + at_off[:, None]
            + 1
        )
        cda_kto[cda_rto < 0] = 0  # will be masked out below

        # parent doftype for each chi-defining atom: (n_samp, max_n_chi)
        parent_kto = parent_cpu[cda_kto]
        is_jump_parent = doftype_cpu[parent_kto] == NodeType.jump  # (n_samp, max_n_chi)
        valid = (cda_rto >= 0) & is_jump_parent  # (n_samp, max_n_chi)

        if not valid.any():
            continue

        # 4-atom RTO indices for every (samp_rot, chi): (n_samp, max_n_chi, 4)
        four_atoms_rto = chi4_table[bt_idx[:, None], numpy.arange(max_n_chi)[None, :]]

        # absolute RTO indices into trial_coords_rto: (n_samp, max_n_chi, 4)
        four_atoms_abs = four_atoms_rto + at_off[:, None, None]
        four_atoms_abs[four_atoms_rto < 0] = 0  # clip invalid entries

        # flatten to the valid (samp_rot, chi) pairs
        valid_flat = valid.reshape(-1)
        four_abs_flat = four_atoms_abs.reshape(-1, 4)[valid_flat]  # (n_valid, 4)
        intended_flat = chi_intended.numpy().reshape(-1)[valid_flat]  # (n_valid,)
        cda_kto_flat = cda_kto.reshape(-1)[valid_flat]  # (n_valid,)

        # gather coordinates: (n_valid, 4, 3)
        xyzs = torch.tensor(
            coords_np[four_abs_flat], dtype=torch.float64, device=torch.device("cpu")
        )

        meas_rad = coord_dihedrals(xyzs[:, 0], xyzs[:, 1], xyzs[:, 2], xyzs[:, 3])

        delta = torch.tensor(intended_flat, dtype=torch.float64) - meas_rad
        delta = (delta + numpy.pi) % (2 * numpy.pi) - numpy.pi

        conf_dofs_kto[torch.tensor(cda_kto_flat, dtype=torch.int64), 3] += delta.to(
            conf_dofs_kto.dtype
        ).to(conf_dofs_kto.device)


def exc_cumsum_from_inc_cumsum(cumsum):
    return numpy.concatenate((numpy.zeros(1, dtype=numpy.int64), cumsum[:-1]))


# from tmol.system.restype import RefinedResidueType


# residue type annotations:
#  - icoors
#  - parent index (i.e. tree)
#  - list of atoms whose icoors are controlled by dunbrack
#  - bonds that are part of named torsions
#  - mainchain atoms
#  - rottable set for this RT
#
# condensed residue type annotations:
#  - (all data in residue-type annotations)
#
#
# NOTE: sidechain samplers should be listed in the residue-level packer task
# NOTE: all samplers should be given the same starting data and then a mask
#       to indicate which residues they should operate on


@validate_args
def rebuild_poses_if_necessary(
    poses: PoseStack, task: PackerTask
):  # -> Tuple[PoseStack, Tuple[ChiSampler, ...]]:
    """Examine the BlockTypes that the packer will entertain for the input PoseStack
    and, if there are BlockTypes that the PoseStack is not currently using,
    build a new PoseStack including thos BlockTypes in its PackedBlockTypes
    datastore. Also return the set of ChiSamplers that are collectively
    held in the PackerTask.

    Note: the ChiSamplers are put into a set, as are the BlockTypes. Both
    require the classes to implement stable __hash__ and __eq__ methods,
    so that the order in which samplers and block types are used/listed
    is consistent between runs (and will not change when the addresses
    that these objects are allocated in changes between runs).

    This code, in its reliance on the id() function, is currently "unstable"
    in that it can produce different results between executions, even if the
    same random seed is provided
    """

    all_restypes = {}
    samplers = set([])

    for one_pose_blts in task.blts:
        for blt in one_pose_blts:
            for sampler in blt.conformer_samplers:
                samplers.add(sampler)
            for bt in blt.considered_block_types:
                if id(bt) not in all_restypes:
                    all_restypes[id(bt)] = bt

    samplers = tuple(samplers)

    # rebuild the poses, perhaps, if there are residue types in the task
    # that are absent from the poses' PBT

    pose_rts = set([id(bt) for bt in poses.packed_block_types.active_block_types])
    needs_rebuilding = False
    for bt_id in all_restypes:
        if bt_id not in pose_rts:
            needs_rebuilding = True
            break

    if needs_rebuilding:
        # make sure all the pose's residue types are also included
        for i in range(poses.n_poses):
            for j in range(poses.max_n_blocks):
                if not poses.is_real_block(i, j):
                    continue
                bt = poses.block_type(i, j)
                if id(bt) not in all_restypes:
                    all_restypes[id(bt)] = bt

        pbt = PackedBlockTypes.from_restype_list(
            poses.packed_block_types.chem_db,
            poses.packed_block_types.restype_set,
            [bt for bt_id, bt in all_restypes.items()],
            poses.packed_block_types.device,
        )

        # rebuild the PoseStack with a new packed_block_types
        poses = PoseStackBuilder.rebuild_with_new_packed_block_types(
            poses, packed_block_types=pbt
        )

    return poses, samplers


def annotate_restype(
    restype: RefinedResidueType,
    samplers: Tuple[ChiSampler, ...],
    chem_db: ChemicalDatabase,
):
    construct_single_residue_kinforest(restype)
    annotate_residue_type_with_sampler_fingerprints(restype, samplers, chem_db)


def annotate_packed_block_types(pbt: PackedBlockTypes):
    coalesce_single_residue_kinforests(pbt)
    find_unique_fingerprints(pbt)


def annotate_everything(
    chem_db: ChemicalDatabase, samplers: Tuple[ChiSampler, ...], pbt: PackedBlockTypes
):
    # annotate residue types and packed block types
    # give the samplers a chance first to annotate the
    # residue types, and pbt
    # then we will call our own annotation functions
    for sampler in samplers:
        for rt in pbt.active_block_types:
            sampler.annotate_residue_type(rt)
        sampler.annotate_packed_block_types(pbt)

    for rt in pbt.active_block_types:
        annotate_restype(rt, samplers, chem_db)
    annotate_packed_block_types(pbt)


@numba.jit(nopython=True)
def update_nodes(
    nodes_orig, genStartsStack, n_nodes_offset_for_rot, n_atoms_offset_for_rot
):
    """Merge the 1-residue-kinforest nodes data so that all the rotamers can be
    built in a single generational-segmented-scan call. This has the structure
    of load-balanced search operation.
    """

    n_gens = genStartsStack.shape[1]
    n_nodes = nodes_orig.shape[0]
    n_rotamers = n_atoms_offset_for_rot.shape[0]
    nodes = numpy.zeros(n_nodes, dtype=numpy.int32)
    count = 0
    for i in range(n_gens - 1):
        for j in range(n_rotamers):
            for k in range(genStartsStack[j, i, 0], genStartsStack[j, i + 1, 0]):
                nodes[count] = nodes_orig[n_nodes_offset_for_rot[j] + k]
                if nodes[count] != 0:
                    nodes[count] += n_atoms_offset_for_rot[j]
                count += 1
    return nodes


@numba.jit(nopython=True)
def update_scan_starts(
    n_scans, atomStartsOffsets, scanStartsStack, genStartsStack, ngenStack
):
    n_gens = genStartsStack.shape[1]
    n_rotamers = genStartsStack.shape[0]
    scanStarts = numpy.zeros((n_scans,), dtype=numpy.int32)
    count = 0
    for i in range(n_gens - 1):
        for j in range(n_rotamers):
            for k in range(ngenStack[i, j]):
                scanStarts[count] = (
                    atomStartsOffsets[i, j]
                    + scanStartsStack[j, genStartsStack[j, i, 1] + k]
                )
                count += 1
    return scanStarts


@validate_args
def construct_scans_for_conformers(
    pbt: PackedBlockTypes,
    block_type_ind_for_conf: NDArray[numpy.int32][:],
    n_atoms_for_conf: Tensor[torch.int32][:],
    n_atoms_offset_for_conf: NDArray[numpy.int64][:],
):
    scanStartsStack = pbt.rotamer_kinforest.scans[block_type_ind_for_conf]
    genStartsStack = pbt.rotamer_kinforest.gens[block_type_ind_for_conf]

    atomStartsStack = numpy.swapaxes(genStartsStack[:, :, 0], 0, 1)
    natomsPerGen = atomStartsStack[1:, :] - atomStartsStack[:-1, :]
    natomsPerGen[natomsPerGen < 0] = 0

    cumsumAtomStarts = numpy.cumsum(natomsPerGen, axis=1)
    atomStartsOffsets = numpy.concatenate(
        (
            numpy.zeros(natomsPerGen.shape[0], dtype=numpy.int64).reshape(-1, 1),
            cumsumAtomStarts[:, :-1],
        ),
        axis=1,
    )

    ngenStack = numpy.swapaxes(
        pbt.rotamer_kinforest.n_scans_per_gen[block_type_ind_for_conf], 0, 1
    )
    ngenStack[ngenStack < 0] = 0
    ngenStackCumsum = numpy.cumsum(ngenStack.reshape(-1), axis=0)

    # jitted function that operates on the CPU; need to figure
    # out how to replace this with a GPU-compatible version
    scanStarts = update_scan_starts(
        ngenStackCumsum[-1],
        atomStartsOffsets,
        scanStartsStack,
        genStartsStack,
        ngenStack,
    )

    nodes_orig = pbt.rotamer_kinforest.nodes[block_type_ind_for_conf].ravel()
    nodes_orig = nodes_orig[nodes_orig >= 0]

    n_nodes_for_conf = pbt.rotamer_kinforest.n_nodes[block_type_ind_for_conf]
    first_node_for_conf = numpy.cumsum(n_nodes_for_conf)
    n_nodes_offset_for_conf = exc_cumsum_from_inc_cumsum(first_node_for_conf)

    nodes = update_nodes(
        nodes_orig, genStartsStack, n_nodes_offset_for_conf, n_atoms_offset_for_conf
    )

    gen_starts = numpy.sum(genStartsStack, axis=0)

    return nodes, scanStarts, gen_starts


@numba.jit(nopython=True)
def load_from_rotamers(
    arr: NDArray[numpy.int32][:, :],
    n_atoms_total: int,
    n_atoms_for_rot: NDArray[numpy.int32][:],
):
    compact_arr = numpy.zeros((n_atoms_total + 1,), dtype=numpy.int32)
    count = 1
    for i in range(n_atoms_for_rot.shape[0]):
        for j in range(n_atoms_for_rot[i]):
            compact_arr[count] = arr[i][j]
            count += 1
    return compact_arr


@numba.jit(nopython=True)
def load_from_rotamers_w_offsets(
    arr: NDArray[numpy.int32][:, :],
    n_atoms_total: int,
    n_atoms_for_rot: NDArray[numpy.int32][:],
    n_atoms_offset_for_rot: NDArray[numpy.int32][:],
):
    compact_arr = numpy.zeros((n_atoms_total + 1,), dtype=numpy.int32)
    count = 1
    for i in range(n_atoms_for_rot.shape[0]):
        for j in range(n_atoms_for_rot[i]):
            compact_arr[count] = arr[i][j] + n_atoms_offset_for_rot[i]
            count += 1
    return compact_arr


@numba.jit(nopython=True)
def load_rotamer_parents(
    parents: NDArray[numpy.int32][:, :],
    n_atoms_total: int,
    n_atoms_for_rot: NDArray[numpy.int32][:],
    n_atoms_offset_for_rot: NDArray[numpy.int32][:],
):
    compact_arr = numpy.zeros((n_atoms_total + 1,), dtype=numpy.int32)
    count = 1
    for i in range(n_atoms_for_rot.shape[0]):
        for j in range(n_atoms_for_rot[i]):
            # offset by 1 for the root node of each rotamer
            # because the parent array has -1 for root nodes'
            # parents and we want to make it 0
            compact_arr[count] = parents[i][j] + (
                n_atoms_offset_for_rot[i] if j != 0 else 1
            )
            count += 1
    return compact_arr


@validate_args
def construct_kinforest_for_conformers(
    pbt: PackedBlockTypes,
    conf_block_type_ind: NDArray[numpy.int32][:],
    n_atoms_total: int,
    n_atoms_for_conf: Tensor[torch.int32][:],
    block_offset_for_conf: NDArray[numpy.int64][:],
    device: torch.device,
):
    """Construct a KinForest for a set of conformers by stringing
    together the kinforest data for individual conformers.
    The "block_ofset_for_conf" array is used to construct
    the "id" tensor in the KinForest, which maps to the atom
    indices; thus it should contain the atom-index offsets
    for the first atom in each rotamer in the coords tensor
    that will be used to construct the kinforest_coords tensor.
    """

    n_atoms_for_conf = n_atoms_for_conf.cpu().numpy()

    # append a 1 for the root node and then treat
    # the resulting (inclusive) scan as if it
    # represents offsets
    temp = numpy.concatenate((numpy.ones(1, dtype=numpy.int32), n_atoms_for_conf))
    n_atoms_offset_for_conf = numpy.cumsum(temp)

    def nab(func, arr):
        return func(arr[conf_block_type_ind], n_atoms_total, n_atoms_for_conf)

    def nab2(func, arr, conf_offset):
        return func(
            arr[conf_block_type_ind], n_atoms_total, n_atoms_for_conf, conf_offset
        )

    def _t(arr):
        return torch.tensor(arr, dtype=torch.int32, device=device)

    id = _t(
        nab2(
            load_from_rotamers_w_offsets,
            pbt.rotamer_kinforest.id,
            block_offset_for_conf,
        )
    )
    id[0] = -1
    doftype = _t(nab(load_from_rotamers, pbt.rotamer_kinforest.doftype))
    parent = _t(
        nab2(
            load_rotamer_parents, pbt.rotamer_kinforest.parent, n_atoms_offset_for_conf
        )
    )
    frame_x = _t(
        nab2(
            load_from_rotamers_w_offsets,
            pbt.rotamer_kinforest.frame_x,
            n_atoms_offset_for_conf,
        )
    )
    frame_y = _t(
        nab2(
            load_from_rotamers_w_offsets,
            pbt.rotamer_kinforest.frame_y,
            n_atoms_offset_for_conf,
        )
    )
    frame_z = _t(
        nab2(
            load_from_rotamers_w_offsets,
            pbt.rotamer_kinforest.frame_z,
            n_atoms_offset_for_conf,
        )
    )

    return KinForest(
        id=id,
        doftype=doftype,
        parent=parent,
        frame_x=frame_x,
        frame_y=frame_y,
        frame_z=frame_z,
    )


def measure_dofs_from_orig_coords(
    coords: Tensor[torch.float32][:, :, :], kinforest: KinForest
):
    from tmol.kinematics.compiled.compiled_inverse_kin import inverse_kin

    kinforest_coords = coords.view(-1, 3)[kinforest.id.to(torch.int64)]
    kinforest_coords[0, :] = 0  # reset root

    dofs_orig = inverse_kin(
        kinforest_coords,
        kinforest.parent,
        kinforest.frame_x,
        kinforest.frame_y,
        kinforest.frame_z,
        kinforest.doftype,
    )
    return dofs_orig


def measure_pose_dofs(poses):
    # measure the DOFs for the original residues
    # -- first build kinforests for the original residues,
    # -- then call measure_dofs_from_orig_coords

    pbt = poses.packed_block_types
    pbti = poses.block_type_ind.view(-1)
    real_poses_blocks = pbti != -1
    orig_res_block_type_ind = pbti[real_poses_blocks]

    # coordinate layout: n-poses x max-n-atoms-per-pose x 3
    # offsets provided by the pose stack
    n_poses = poses.coords.shape[0]
    max_n_atoms_per_pose = poses.max_n_pose_atoms
    max_n_blocks_per_pose = poses.max_n_blocks
    per_pose_offset = max_n_atoms_per_pose * stretch(
        torch.arange(n_poses, dtype=torch.int64, device=poses.device),
        max_n_blocks_per_pose,
    )
    orig_atom_offset_for_poses_blocks = (
        (
            poses.block_coord_offset.flatten()[real_poses_blocks].to(torch.int64)
            + per_pose_offset[real_poses_blocks]
        )
        .cpu()
        .numpy()
    )

    n_atoms_for_orig = pbt.n_atoms[orig_res_block_type_ind.to(torch.int64)]
    n_atoms_offset_for_orig = torch.cumsum(n_atoms_for_orig, dim=0)
    n_atoms_offset_for_orig = n_atoms_offset_for_orig.cpu().numpy()
    n_orig_atoms_total = n_atoms_offset_for_orig[-1]

    orig_kinforest = construct_kinforest_for_conformers(
        poses.packed_block_types,
        orig_res_block_type_ind.cpu().numpy(),
        int(n_orig_atoms_total),
        n_atoms_for_orig,
        orig_atom_offset_for_poses_blocks,
        poses.device,
    )

    # orig_dofs returned in kinforest order
    return orig_kinforest, measure_dofs_from_orig_coords(poses.coords, orig_kinforest)


def merge_conformer_samples(
    conformer_samples,
) -> Tuple[
    Tensor[torch.int64][:],
    Tensor[torch.int64][:],
    Tensor[torch.int64][:],
    List[Tensor[torch.bool][:]],
    List[Tensor[torch.int64][:]],
]:
    """Merge the lists of conformers as described by different conformer samplers.

    The conformer_samples variable is a list of tuples:
     - elem 0: Tensor[int][:] <-- the number of rotamers for each pose for each block for each block type
        where each buildable block type for each real residue is given a global index
     - elem 1: Tensor[int][:] <-- the global block-type index for each rotamer
     - elem 2+: Extra data that the chi sampler needs to preserve, where the first dimension
       is rotamer index based on elem 1's rotamer indices; the mapping from orig rotamer indices
       to merged rotamer indices will be constructed by this routine
    """
    # deprecated notes:
    # chi_samples
    # 0. n_rots_for_rt
    # 1. rt_for_rotamer
    # 2. chi_defining_atom_for_rotamer
    # 3. chi_for_rotamers

    # everything needs to be on the same device
    torch.set_printoptions(threshold=10000)
    for samples in conformer_samples:
        assert samples[0].device == samples[1].device
        assert conformer_samples[0][0].device == samples[0].device

    device = conformer_samples[0][0].device

    # pre-merge offsets for each gbt in the set of conformers from the same sampler
    gbt_n_rot_offsets = []  # formerly rt_nrot_offsets
    for samples in conformer_samples:
        gbt_n_rot_offsets.append(exclusive_cumsum1d(samples[0]).to(torch.int64))

    all_gbt_for_conformer_unsorted = torch.cat(
        [samples[1] for samples in conformer_samples]
    )
    max_n_conformers_per_gbt_per_sampler = max(
        torch.max(samples[0]).item() for samples in conformer_samples
    )

    # create an "index" for each conformer on each GBT
    # so that we can sort these indices and come up with an ordering
    # of all of the conformers that will group all of the conformers
    # belonging to a single GBT into a contiguous segment;
    # This is accomplished by "spreading out" all of the rotamers for a single GBT
    # by the maximum possible number of rotamers that could be built for any one
    # GBT (i.e. n-samplers x max-n-confs-per-gbt-per-sampler x gbt-index),
    # then finding which block of conformers for the given sampler
    # (i.e. max-n-confs-per-gbt-per-sampler * sampler-index),
    # and finally, incrementing each individual sample by its position in the
    # list of rotamers for that GBT, which is readily computed as
    # arange(sampler_n_rots) - gbt_n_rot_offsets[gbt_index]
    # and note that gbt_index is what's stored in samples[1]
    n_conformer_samplers = len(conformer_samples)
    sort_index_for_conformer = torch.cat(
        [
            samples[1].to(torch.int64)
            * n_conformer_samplers
            * max_n_conformers_per_gbt_per_sampler
            + i * max_n_conformers_per_gbt_per_sampler
            + torch.arange(samples[1].shape[0], dtype=torch.int64, device=device)
            - gbt_n_rot_offsets[i][samples[1].to(torch.int64)]
            for i, samples in enumerate(conformer_samples)
        ]
    )

    sampler_for_conformer_unsorted = torch.cat(
        [
            torch.full((samples[1].shape[0],), i, dtype=torch.int64, device=device)
            for i, samples in enumerate(conformer_samples)
        ]
    )
    argsort_ind_for_conformer = torch.argsort(sort_index_for_conformer)

    # testing: remove this, probably
    sort_ind_for_conformer_sorted = sort_index_for_conformer[argsort_ind_for_conformer]
    uniq_sort_ind_for_conformer = torch.unique(sort_ind_for_conformer_sorted)
    assert (
        uniq_sort_ind_for_conformer.shape[0] == sort_ind_for_conformer_sorted.shape[0]
    )

    sampler_for_conformer = sampler_for_conformer_unsorted[argsort_ind_for_conformer]

    # list of boolean tensors for each of the samplers: did you build the given rotamer
    conformer_built_by_sampler = [
        sampler_for_conformer == i for i in range(n_conformer_samplers)
    ]
    # list of index tensors reporting the final index of the conformers built by the samplers
    new_ind_for_sampler_rotamer = [
        torch.nonzero(built_by_sampler, as_tuple=True)[0]
        for built_by_sampler in conformer_built_by_sampler
    ]

    all_gbt_for_conformer_sorted = all_gbt_for_conformer_unsorted[
        argsort_ind_for_conformer
    ]

    # ok, now we need to figure out how many rotamers each gbt is getting.
    n_rots_for_gbt = toolz.reduce(
        torch.add, [samples[0] for samples in conformer_samples]
    )

    return (
        n_rots_for_gbt,
        sampler_for_conformer,
        all_gbt_for_conformer_sorted,
        conformer_built_by_sampler,
        new_ind_for_sampler_rotamer,
    )


def calculate_rotamer_coords(
    pbt: PackedBlockTypes,
    n_rots: int,
    n_atoms_total: int,
    rot_kinforest: KinForest,
    nodes: NDArray[numpy.int32][:],
    scans: NDArray[numpy.int32][:],
    gens: NDArray[numpy.int32][:],
    rot_dofs_kto: Tensor[torch.float32][:, 9],
):
    from tmol.kinematics.compiled.compiled_ops import forward_only_op

    def _p(t):
        return torch.nn.Parameter(t, requires_grad=False)

    def _t(t):
        return torch.tensor(t, dtype=torch.int32, device=pbt.device)

    def _tcpu(t):
        return torch.tensor(t, dtype=torch.int32, device=torch.device("cpu"))

    kinforest_stack = _p(
        torch.stack(
            [
                rot_kinforest.id,
                rot_kinforest.doftype,
                rot_kinforest.parent,
                rot_kinforest.frame_x,
                rot_kinforest.frame_y,
                rot_kinforest.frame_z,
            ],
            dim=1,
        ).to(pbt.device)
    )

    new_coords_kto = forward_only_op(
        rot_dofs_kto, _p(_t(nodes)), _p(_t(scans)), _p(_tcpu(gens)), kinforest_stack
    )

    new_coords_rto = torch.zeros(
        (n_atoms_total, 3), dtype=torch.float32, device=pbt.device
    )

    new_coords_rto[rot_kinforest.id[1:].to(torch.int64)] = new_coords_kto[1:]
    return new_coords_rto


def get_rotamer_origin_data(task: PackerTask, gbt_for_rot: Tensor[torch.int32][:]):
    n_poses = len(task.blts)
    pose_for_gbt = torch.tensor(
        [
            i
            for i, one_pose_blts in enumerate(task.blts)
            for blts in one_pose_blts
            for blt in blts.considered_block_types
        ],
        dtype=torch.int32,
        device=gbt_for_rot.device,
    )

    block_ind_for_rt = torch.tensor(
        [
            j
            for one_pose_blts in task.blts
            for j, blts in enumerate(one_pose_blts)
            for blt in blts.considered_block_types
        ],
        dtype=torch.int32,
        device=gbt_for_rot.device,
    )
    max_n_blocks = max(len(one_pose_blts) for one_pose_blts in task.blts)

    gbt_for_rot64 = gbt_for_rot.to(torch.int64)
    pose_for_rot = pose_for_gbt[gbt_for_rot64].to(torch.int64)
    n_rots_for_pose = torch.bincount(pose_for_rot, minlength=len(task.blts))
    rot_offset_for_pose = exclusive_cumsum1d(n_rots_for_pose)
    block_ind_for_rot = block_ind_for_rt[gbt_for_rot64]
    block_ind_for_rt_global = max_n_blocks * pose_for_gbt + block_ind_for_rt
    block_ind_for_rot_global = block_ind_for_rt_global[gbt_for_rot64]
    n_rots_for_block = torch.bincount(
        block_ind_for_rot_global, minlength=n_poses * max_n_blocks
    ).reshape(n_poses, max_n_blocks)
    rot_offset_for_block = exclusive_cumsum1d(n_rots_for_block.flatten()).reshape(
        n_poses, max_n_blocks
    )

    return (
        n_rots_for_pose,
        rot_offset_for_pose,
        n_rots_for_block,
        rot_offset_for_block,
        pose_for_rot,
        block_ind_for_rot,
    )


def build_rotamers(poses: PoseStack, task: PackerTask, chem_db: ChemicalDatabase):
    # step 1: replace the existing PBT in the Pose w/ a new one in case
    #     there will possibly be new block types in the repacked Pose;
    #     but since PoseStack should not be altered after construction,
    #     what this really means is build an entirely new PoseStack
    # step 2: let the dunbrack library annotate the block types and the packed block types
    # step 3: get the block-type index for the "global block types" (gbts)
    # step 4: use the conformer samplers to decide how many conformers they will build for
    #         each bt/block/pose
    # step 5: merge the conformer samples from different samplers, so that different
    #         conformers for the same bt/block/pose will be in contiguous ranges in the
    #         rotamer set, and keeping track of the mapping back to the original set of
    #         conformer samples
    # step 6: allocate a n_atoms_total x 3 tensor for rotamer coordinates and
    #         create the tensor of offsets
    # step 7: build a kintree for all of the rotamers; initialize the DOFs to ideal
    # step 8: build a kintree for the PoseStack residues
    # step 9: measure the DOFs of the PoseStack residues
    # step 9a: take starting coordinates from residue roots
    # step 9b: take internal dofs from mainchain-fingerprint atoms
    # step 10: ask the samplers to set the DOFs for everything else
    # step 11: refold

    # Step 1
    poses, samplers = rebuild_poses_if_necessary(poses, task)
    pbt = poses.packed_block_types

    # Step 2
    annotate_everything(chem_db, samplers, pbt)

    # Step 3
    # create a list of the name of every considered block type at every block in every
    # pose so that we can then create an integer version of that same data;
    # the "global block type" (gbt) if you will. The order in which these block-
    # types appear will be used as an index for talking about which rotamers are
    # built where. This cannot be efficient. Perhaps worth thinking hard about the
    # PackerTask's structure.
    gbt_names = [
        bt.name
        for one_pose_blts in task.blts
        for blt in one_pose_blts
        for bt in blt.considered_block_types
    ]
    gbt_block_type_ind = pbt.restype_index.get_indexer(gbt_names).astype(numpy.int32)

    # Step 4
    conformer_samples = [
        sampler.create_samples_for_poses(poses, task) for sampler in samplers
    ]

    # Step 5
    (
        n_rots_for_gbt,
        sampler_for_conformer,
        gbt_for_conformer,
        conformer_built_by_sampler,
        new_ind_for_sampler_rotamer,
    ) = merge_conformer_samples(conformer_samples)

    def _t(t, dtype):
        return torch.tensor(t, dtype=dtype, device=pbt.device)

    gbt_for_conformer_np = gbt_for_conformer.cpu().numpy()

    gbt_for_conformer_torch = gbt_for_conformer.to(dtype=torch.int64, device=pbt.device)

    n_conformers = sampler_for_conformer.shape[0]

    block_type_ind_for_conformer = gbt_block_type_ind[gbt_for_conformer_np]
    block_type_ind_for_conformer_torch = _t(block_type_ind_for_conformer, torch.int64)

    n_atoms_for_conformer = pbt.n_atoms[block_type_ind_for_conformer_torch]
    n_atoms_offset_for_conformer = torch.cumsum(n_atoms_for_conformer, dim=0)
    n_atoms_offset_for_conformer = n_atoms_offset_for_conformer.cpu().numpy()
    n_atoms_total = n_atoms_offset_for_conformer[-1].item()
    n_atoms_offset_for_conformer = exc_cumsum_from_inc_cumsum(
        n_atoms_offset_for_conformer
    )
    n_atoms_offset_for_conformer_torch = _t(n_atoms_offset_for_conformer, torch.int64)

    # Step 7
    conformer_kinforest = construct_kinforest_for_conformers(
        pbt,
        block_type_ind_for_conformer,
        n_atoms_total,
        n_atoms_for_conformer.to(dtype=torch.int32),
        n_atoms_offset_for_conformer,
        pbt.device,
    )

    nodes, scans, gens = construct_scans_for_conformers(
        pbt,
        block_type_ind_for_conformer,
        n_atoms_for_conformer,
        n_atoms_offset_for_conformer,
    )

    # Step 8 & 9
    orig_kinforest, orig_dofs_kto = measure_pose_dofs(poses)

    # Step 9a
    conf_dofs_kto = torch.zeros(
        (n_atoms_total + 1, 9), dtype=torch.float32, device=pbt.device
    )
    conf_dofs_kto[1:] = torch.tensor(
        pbt.rotamer_kinforest.dofs_ideal[block_type_ind_for_conformer].reshape((-1, 9))[
            pbt.atom_is_real.cpu().numpy()[block_type_ind_for_conformer].reshape(-1)
            != 0
        ],
        dtype=torch.float32,
        device=pbt.device,
    )

    for i, sampler in enumerate(samplers):
        sampler.fill_dofs_for_samples(
            poses,
            task,
            orig_kinforest,
            orig_dofs_kto,
            gbt_for_conformer_torch,
            block_type_ind_for_conformer_torch,
            n_atoms_offset_for_conformer_torch,
            conformer_built_by_sampler[i],
            new_ind_for_sampler_rotamer[i],
            conformer_samples[i][0],
            conformer_samples[i][1],
            conformer_samples[i][2],
            conf_dofs_kto,
        )

    correct_phi_c_for_jump_parents(
        pbt,
        conformer_samples,
        new_ind_for_sampler_rotamer,
        block_type_ind_for_conformer_torch,
        n_atoms_offset_for_conformer_torch,
        conformer_kinforest,
        nodes,
        scans,
        gens,
        conf_dofs_kto,
    )

    rotamer_coords = calculate_rotamer_coords(
        pbt,
        n_conformers,
        n_atoms_total,
        conformer_kinforest,
        nodes,
        scans,
        gens,
        conf_dofs_kto,
    )

    (
        n_rots_for_pose,
        rot_offset_for_pose,
        n_rots_for_block,
        rot_offset_for_block,
        pose_for_rot,
        block_ind_for_rot,
    ) = get_rotamer_origin_data(task, gbt_for_conformer_torch)

    return (
        poses,
        RotamerSet(
            max_n_rots_per_pose=torch.max(n_rots_for_pose).item(),
            n_rots_for_pose=n_rots_for_pose,
            rot_offset_for_pose=rot_offset_for_pose,
            n_rots_for_block=n_rots_for_block,
            rot_offset_for_block=rot_offset_for_block,
            pose_for_rot=pose_for_rot,
            block_type_ind_for_rot=block_type_ind_for_conformer_torch,
            block_ind_for_rot=block_ind_for_rot,
            coord_offset_for_rot=n_atoms_offset_for_conformer_torch.to(torch.int32),
            coords=rotamer_coords,
        ),
    )
