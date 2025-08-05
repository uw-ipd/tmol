import numpy
import numba
import toolz
import torch
import attr

from typing import List, Tuple

from tmol.types.array import NDArray
from tmol.types.attrs import ValidateAttrs
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args

from tmol.utility.tensor.common_operations import exclusive_cumsum1d, stretch
from tmol.database.chemical import ChemicalDatabase
from tmol.kinematics.datatypes import KinForest
from tmol.kinematics.compiled.compiled_ops import forward_only_op
from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.pack.packer_task import PackerTask
from tmol.pack.rotamer.chi_sampler import ChiSampler

from tmol.pack.rotamer.single_residue_kinforest import (
    construct_single_residue_kinforest,
    coalesce_single_residue_kinforests,
)
from tmol.pack.rotamer.mainchain_fingerprint import (
    annotate_residue_type_with_sampler_fingerprints,
    find_unique_fingerprints,
)


def exc_cumsum_from_inc_cumsum(cumsum):
    return numpy.concatenate((numpy.zeros(1, dtype=numpy.int64), cumsum[:-1]))


@attr.s(auto_attribs=True, slots=True, frozen=True)
class RotamerSet(ValidateAttrs):
    n_rots_for_pose: Tensor[torch.int64][:]
    rot_offset_for_pose: Tensor[torch.int64][:]
    n_rots_for_block: Tensor[torch.int64][:, :]
    rot_offset_for_block: Tensor[torch.int64][:, :]
    pose_for_rot: Tensor[torch.int64][:]
    block_type_ind_for_rot: Tensor[torch.int64][:]
    block_ind_for_rot: Tensor[torch.int32][:]
    coord_offset_for_rot: Tensor[torch.int32][:]
    coords: Tensor[torch.float32][:, :]


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
            [brt for bt_id, bt in all_restypes.items()],
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
    # print("nodes_orig")
    # print(nodes_orig)

    n_nodes_for_conf = pbt.rotamer_kinforest.n_nodes[block_type_ind_for_conf]
    first_node_for_conf = numpy.cumsum(n_nodes_for_conf)
    n_nodes_offset_for_conf = exc_cumsum_from_inc_cumsum(first_node_for_conf)
    # print("n_nodes_offset_for_rot")
    # print(n_nodes_offset_for_rot)
    # print("n_atoms_offset_for_rot")
    # print(n_atoms_offset_for_rot)

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

    # print("coords")
    # print(coords.shape)
    # print("kinforest.id")
    # print(kinforest.id)

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

    # print("orig_atom_offset_for_poses_blocks", orig_atom_offset_for_poses_blocks.dtype)
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
        # print("samples", samples[0].shape, samples[1].shape)
        # print("samples[0]")
        # print(samples[0])
        # print("samples[1]")
        # print(samples[1])

    device = conformer_samples[0][0].device

    # pre-merge offsets for each gbt in the set of conformers from the same sampler
    gbt_n_rot_offsets = []  # formerly rt_nrot_offsets
    for samples in conformer_samples:
        gbt_n_rot_offsets.append(exclusive_cumsum1d(samples[0]).to(torch.int64))

    all_gbt_for_conformer_unsorted = torch.cat(
        [samples[1] for samples in conformer_samples]
    )
    n_conformers_total = all_gbt_for_conformer_unsorted.shape[0]  # formerly n_rotamers
    max_n_conformers_per_gbt_per_sampler = max(
        torch.max(samples[0]).item() for samples in conformer_samples
    )

    # for i, samples in enumerate(conformer_samples):
    #     conf_counter_for_gbt = (
    #         torch.arange(samples[1].shape[0], dtype=torch.int64, device=device)
    #         - rt_nrot_offsets[i][samples[1].to(torch.int64)]
    #     )
    #     numpy.testing.assert_array_less(
    #         rot_counter_for_rt.cpu().numpy(), max_n_rotamers_per_rt
    #     )

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
    # temp
    # torch.set_printoptions(threshold=10000)
    # print("sort_index_for_conformer")
    # print(sort_index_for_conformer)

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
    #  print("sampler_for_conformer")
    #  print(sampler_for_conformer)

    # list of boolean tensors for each of the samplers: did you build the given rotamer
    conformer_built_by_sampler = [
        sampler_for_conformer == i for i in range(n_conformer_samplers)
    ]
    # list of index tensors reporting the final index of the conformers built by the samplers
    new_ind_for_sampler_rotamer = [
        torch.nonzero(built_by_sampler, as_tuple=True)[0]
        for built_by_sampler in conformer_built_by_sampler
    ]

    # for i, new_inds in enumerate(new_ind_for_sampler_rotamer):
    #     print("i", i, "new_inds")
    #     print(new_inds)

    all_gbt_for_conformer_sorted = all_gbt_for_conformer_unsorted[
        argsort_ind_for_conformer
    ]
    # print("all_gbt_for_conformer_sorted")
    # print(all_gbt_for_conformer_sorted)

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

    # temp
    # n_atoms = 12765
    # print("rot_dofs_kto[:50]", rot_dofs_kto[:50])
    # print("rot_dofs_kto[(n_atoms-50):(n_atoms+50)]", rot_dofs_kto[(n_atoms-50):(n_atoms+50)])

    new_coords_kto = forward_only_op(
        rot_dofs_kto, _p(_t(nodes)), _p(_t(scans)), _p(_tcpu(gens)), kinforest_stack
    )

    new_coords_rto = torch.zeros(
        (n_atoms_total, 3), dtype=torch.float32, device=pbt.device
    )
    # torch.set_printoptions(threshold=100000)
    # print("id")
    # print(rot_kinforest.id)

    new_coords_rto[rot_kinforest.id[1:].to(torch.int64)] = new_coords_kto[1:]
    # new_coords_rto = new_coords_rto.view(n_rots, pbt.max_n_atoms, 3)
    # print("new_coords_rto.shape", new_coords_rto.shape)
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


# def build_rotamers(poses: PoseStack, task: PackerTask, chem_db: ChemicalDatabase):
#     # step 0: replace the existing PBT in the Pose w/ a new one in case
#     #     there will possibly be new block types in the repacked Pose;
#     #     but since PoseStack should not be altered after construction,
#     #     what this really means is build an entirely new PoseStack
#     # step 1: let the dunbrack library annotate the block types
#     # step 2: let the dunbrack library annotate the packed block types
#     # step 3: flatten poses
#     # step 4: use the chi sampler to get the chi samples for all poses
#     # step 5: count the number of rotamers per pose
#     # step 5a: including rotamers that the dunbrack sampler does not provide (e.g. gly)
#     # step 6: allocate a n_poses x max_n_rotamers x max_n_atoms x 3 tensor
#     # step 7: create (n_poses * max_n_rotamers * max_n_atoms) x 3 view of coord tensor
#     # step 8: create parent indexing based on start-position offset + residue-type tree data
#     # step 9: build kinforest
#     # step 10: take starting coordinates from residue roots
#     # step 10a: take internal dofs from mainchain atoms
#     # step 10b: take internal dofs for other atoms from rt icoors
#     # step 11: refold
#
#     poses, samplers = rebuild_poses_if_necessary(poses, task)
#     pbt = poses.packed_block_types
#     annotate_everything(chem_db, samplers, pbt)
#
#     rt_names = [
#         rt.name
#         for one_pose_blts in task.blts
#         for blt in one_pose_blts
#         for rt in blt.allowed_blocktypes
#     ]
#     # rt_block_type_ind: a mapping from the list of all block types at all
#     # residues across all poses to the PBT-block-type index. We will use the
#     # rt_block_type_ind as a way to refer to a particular block in a particular pose
#     # as well as a particular block-type for that block.
#     rt_block_type_ind = pbt.restype_index.get_indexer(rt_names).astype(numpy.int32)
#
#     chi_samples = [sampler.sample_chi_for_poses(poses, task) for sampler in samplers]
#     merged_samples = merge_chi_samples(chi_samples)
#     n_rots_for_rt, sampler_for_rotamer, rt_for_rotamer, chi_atoms, chi = merged_samples
#
#     # fd NOTE: THIS CODE FAILS IF n_rots_for_rt CONTAINS 0s
#     assert 0 not in n_rots_for_rt
#
#     n_rots = chi_atoms.shape[0]
#     rt_for_rot = torch.zeros(n_rots, dtype=torch.int64, device=poses.device)
#     n_rots_for_rt_cumsum = torch.cumsum(n_rots_for_rt, dim=0)
#     rt_for_rot[n_rots_for_rt_cumsum[:-1]] = 1
#     rt_for_rot = torch.cumsum(rt_for_rot, dim=0).cpu().numpy()
#
#     block_type_ind_for_rot = rt_block_type_ind[rt_for_rot]
#     block_type_ind_for_rot_torch = torch.tensor(
#         block_type_ind_for_rot, dtype=torch.int64, device=pbt.device
#     )
#     n_atoms_for_rot = pbt.n_atoms[block_type_ind_for_rot_torch]
#     n_atoms_offset_for_rot = torch.cumsum(n_atoms_for_rot, dim=0)
#     n_atoms_offset_for_rot = n_atoms_offset_for_rot.cpu().numpy()
#     n_atoms_total = n_atoms_offset_for_rot[-1]
#     n_atoms_offset_for_rot = exc_cumsum_from_inc_cumsum(n_atoms_offset_for_rot)
#
#     rot_kinforest = construct_kinforest_for_rotamers(
#         pbt,
#         block_type_ind_for_rot,
#         int(n_atoms_total),
#         torch.tensor(n_atoms_for_rot, dtype=torch.int32),
#         numpy.arange(n_rots, dtype=numpy.int32) * pbt.max_n_atoms,
#         pbt.device,
#     )
#
#     nodes, scans, gens = construct_scans_for_rotamers(
#         pbt, block_type_ind_for_rot, n_atoms_for_rot, n_atoms_offset_for_rot
#     )
#
#     orig_kinforest, orig_dofs_kto = measure_pose_dofs(poses)
#
#     n_rotamer_atoms = torch.sum(n_atoms_for_rot).item()
#
#     rot_dofs_kto = torch.zeros(
#         (n_rotamer_atoms + 1, 9), dtype=torch.float32, device=pbt.device
#     )
#
#     rot_dofs_kto[1:] = torch.tensor(
#         pbt.rotamer_kinforest.dofs_ideal[block_type_ind_for_rot].reshape((-1, 9))[
#             pbt.atom_is_real.cpu().numpy()[block_type_ind_for_rot].reshape(-1) != 0
#         ],
#         dtype=torch.float32,
#         device=pbt.device,
#     )
#
#     rt_for_rot_torch = torch.tensor(rt_for_rot, dtype=torch.int64, device=pbt.device)
#
#     copy_dofs_from_orig_to_rotamers(
#         poses,
#         task,
#         samplers,
#         rt_for_rot_torch,
#         block_type_ind_for_rot_torch,
#         sampler_for_rotamer,
#         torch.tensor(n_atoms_offset_for_rot, dtype=torch.int32, device=pbt.device),
#         orig_dofs_kto,
#         rot_dofs_kto,
#     )
#
#     assign_dofs_from_samples(
#         pbt,
#         rt_for_rot_torch,
#         block_type_ind_for_rot_torch,
#         chi_atoms,
#         chi,
#         rot_dofs_kto,
#     )
#
#     rotamer_coords = calculate_rotamer_coords(
#         pbt, n_rots, rot_kinforest, nodes, scans, gens, rot_dofs_kto
#     )
#
#     (
#         n_rots_for_pose,
#         rot_offset_for_pose,
#         n_rots_for_block,
#         rot_offset_for_block,
#         pose_for_rot,
#         block_ind_for_rot,
#     ) = get_rotamer_origin_data(task, rt_for_rot_torch)
#
#     return (
#         poses,
#         RotamerSet(
#             n_rots_for_pose=n_rots_for_pose,
#             rot_offset_for_pose=rot_offset_for_pose,
#             n_rots_for_block=n_rots_for_block,
#             rot_offset_for_block=rot_offset_for_block,
#             pose_for_rot=pose_for_rot,
#             block_type_ind_for_rot=block_type_ind_for_rot_torch,
#             block_ind_for_rot=block_ind_for_rot,
#             coords=rotamer_coords,
#         ),
#     )


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

    # torch.set_printoptions(threshold=10000)
    # print("n_rots_for_gbt")
    # print(n_rots_for_gbt)

    def _t(t, dtype):
        return torch.tensor(t, dtype=dtype, device=pbt.device)

    gbt_for_conformer_np = gbt_for_conformer.cpu().numpy()

    gbt_for_conformer_torch = _t(gbt_for_conformer, torch.int64)

    # apl: I hope to have fixed that
    # fd NOTE: THIS CODE FAILS IF n_rots_for_gbt CONTAINS 0s
    # assert 0 not in n_rots_for_gbt

    n_conformers = sampler_for_conformer.shape[0]
    # gbt_for_rot = torch.zeros(n_conformers, dtype=torch.int64, device=poses.device)
    n_rots_for_gbt_cumsum = torch.cumsum(n_rots_for_gbt, dim=0)
    # gbt_for_rot[n_rots_for_gbt_cumsum[:-1]] = 1
    # gbt_for_rot = torch.cumsum(gbt_for_rot, dim=0).cpu().numpy()

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
        torch.tensor(n_atoms_for_conformer, dtype=torch.int32),
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

    rt_for_conformer_torch = torch.tensor(
        block_type_ind_for_conformer, dtype=torch.int64, device=pbt.device
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

    # copy_dofs_from_orig_to_rotamers(
    #     poses,
    #     task,
    #     samplers,
    #     rt_for_rot_torch,
    #     block_type_ind_for_rot_torch,
    #     sampler_for_rotamer,
    #     torch.tensor(n_atoms_offset_for_rot, dtype=torch.int32, device=pbt.device),
    #     orig_dofs_kto,
    #     rot_dofs_kto,
    # )

    print("conf_dofs_kto")
    print(
        conf_dofs_kto[
            torch.tensor(
                # [    0,     1,     2,     3,     4,     5,     6,     7,     8,     9,
                #     10,    11,    12,    13,    14,    15,    16,    17,    18],
                [
                    14952,
                    14953,
                    14954,
                    14955,
                    14956,
                    14957,
                    14958,
                    14959,
                    14960,
                    14961,
                    14962,
                    14963,
                    14964,
                    14965,
                    14956,
                    14957,
                ],
                dtype=torch.int64,
                device=pbt.device,
            ),
            :,
        ]
    )
    is_pro_rot = torch.logical_and(
        conformer_kinforest.id >= 14952, conformer_kinforest.id < 14966
    )
    print("id:")
    print(torch.nonzero(is_pro_rot))
    print("conformer_kinforest.parent")
    print(conformer_kinforest.parent[14953:14967])
    print("conformer_kinforest.frame_x")
    print(conformer_kinforest.frame_x[14953:14967])
    print("conformer_kinforest.frame_y")
    print(conformer_kinforest.frame_y[14953:14967])
    print("conformer_kinforest.frame_z")
    print(conformer_kinforest.frame_z[14953:14967])

    # assign_dofs_from_samples(
    #     pbt,
    #     rt_for_rot_torch,
    #     block_type_ind_for_rot_torch,
    #     chi_atoms,
    #     chi,
    #     rot_dofs_kto,
    # )
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
