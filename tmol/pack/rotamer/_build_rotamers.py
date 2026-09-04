import numpy
import numba
import toolz
import torch

from typing import List, Tuple

from tmol.types import (
    NDArray,
    Tensor,
    validate_args,
)
from tmol.utility.tensor import exclusive_cumsum1d, stretch
from tmol.database.chemical import ChemicalDatabase
from tmol.kinematics import KinForest
from tmol.chemical import RefinedResidueType
from tmol.pose import (
    PackedBlockTypes,
    PoseStack,
)
from tmol.pack.rotamer import (
    RotamerSet,
    ChiSampler,
    construct_single_residue_kinforest,
    coalesce_single_residue_kinforests,
    annotate_residue_type_with_sampler_fingerprints,
    find_unique_fingerprints,
)
from tmol.pack import SetPackerTask
from tmol.numeric import coord_dihedrals


def _turns_its_fourth_atom(rt, third, fourth):
    """Whether writing phi_c on a chi's third atom moves its fourth.

    A chi that closes a ring measures an atom upstream of the one carrying its
    degree of freedom, so nothing the packer writes can put that dihedral at a
    requested value; the write would only spin whatever hangs off the third
    atom instead.
    """
    kf = rt.rotamer_kinforest
    target = int(kf.kinforest_idx[third])
    node = int(kf.kinforest_idx[fourth])
    while True:
        parent = int(kf.parent[node])
        if parent == target:
            return True
        if parent < 0 or parent == node:
            return False
        node = parent


def _build_chi4_by_defining_atom(pbt):
    """(n_types, max_n_atoms, 4) RTO atom indices of the chi each atom defines.

    Keyed by the chi-defining atom rather than by a chi's position, because a
    residue type's chi are not necessarily numbered without gaps and samplers
    do not agree on how to lay them out in a column. A chi whose fourth atom
    the third does not move is left out: it cannot be set.
    """
    if hasattr(pbt, "_chi4_by_defining_atom"):
        return pbt._chi4_by_defining_atom

    table = numpy.full((pbt.n_types, pbt.max_n_atoms, 4), -1, dtype=numpy.int32)
    for ti, rt in enumerate(pbt.active_block_types):
        for name, uaids in rt.torsion_to_uaids.items():
            if not name.startswith("chi"):
                continue
            four = [int(u[0]) for u in uaids]
            if any(a < 0 for a in four):
                continue
            if not _turns_its_fourth_atom(rt, four[2], four[3]):
                continue
            table[ti, four[2]] = four

    object.__setattr__(pbt, "_chi4_by_defining_atom", table)
    return table


def _build_ring_chi_phi_c_corrections(pbt):
    """(n_types, max_n_atoms) phi_c offset for chi that cannot be measured.

    A chi that closes a ring measures an atom upstream of the one carrying its
    degree of freedom, so no trial pass can tell what to add. Its offset is the
    difference between the chi and the phi_c of the ideal geometry, which turns
    what hangs off the third atom by however far the ring has moved from ideal.
    Zero everywhere else, where the measured correction is exact.
    """
    if hasattr(pbt, "_ring_chi_phi_c_corrections"):
        return pbt._ring_chi_phi_c_corrections

    corrections = numpy.zeros((pbt.n_types, pbt.max_n_atoms), dtype=numpy.float32)
    for ti, rt in enumerate(pbt.active_block_types):
        ideal = torch.tensor(rt.ideal_coords[rt.at_to_icoor_ind], dtype=torch.float64)
        for name, uaids in rt.torsion_to_uaids.items():
            if not name.startswith("chi"):
                continue
            four = [int(u[0]) for u in uaids]
            if any(a < 0 for a in four) or _turns_its_fourth_atom(rt, four[2], four[3]):
                continue
            xyz = ideal[four]
            chi_ideal = float(
                coord_dihedrals(xyz[0:1], xyz[1:2], xyz[2:3], xyz[3:4])[0]
            )
            kfo = int(rt.rotamer_kinforest.kinforest_idx[four[2]])
            corrections[ti, four[2]] = chi_ideal - float(
                rt.rotamer_kinforest.dofs_ideal[kfo, 3]
            )

    object.__setattr__(pbt, "_ring_chi_phi_c_corrections", corrections)
    return corrections


def _chi4_and_kfo_device_tables(pbt, device):
    """Device copies of the per-atom chi table and the RTO -> KFO atom map."""
    if not hasattr(pbt, "_chi4_kfo_device_tables"):
        tables = (
            torch.as_tensor(_build_chi4_by_defining_atom(pbt), device=device).to(
                torch.int64
            ),
            torch.as_tensor(pbt.rotamer_kinforest.kinforest_idx, device=device).to(
                torch.int64
            ),
        )
        object.__setattr__(pbt, "_chi4_kfo_device_tables", tables)
    return pbt._chi4_kfo_device_tables


def correct_phi_c_from_measured_chi(
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
    """Make every chi dihedral come out at the value its sampler asked for.

    The phi_c written by assign_chi_dofs_from_samples is offset from the chi
    dihedral by an amount that depends on the rotamer's own geometry, so it is
    measured rather than derived:
      1. Does a trial forward pass with the current DOFs.
      2. Measures each chi's actual dihedral from the trial coords.
      3. Adds (intended - measured) to conf_dofs_kto[atom_kto, 3] so the
         final forward pass produces the intended dihedral.

    A dihedral is invariant under rotation of its own upstream frame, so one
    trial pass corrects every chi at once.
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

    chi4_by_atom, kfidx = _chi4_and_kfo_device_tables(pbt, conf_dofs_kto.device)
    phi_c = conf_dofs_kto.select(1, 3)

    for i, sample_data in enumerate(conformer_samples):
        sample_dict = sample_data[2]
        if "chi_for_rotamers" not in sample_dict:
            continue
        chi_intended = sample_dict["chi_for_rotamers"]  # (n_samp_rots, max_n_chi)
        cda_rto = sample_dict["chi_defining_atom_for_rotamer"]
        if chi_intended.shape[0] == 0:
            continue

        # per-rotamer metadata for every (samp_rot, chi) pair
        g_rot = new_ind_for_sampler_rotamer[i]
        bt_idx = block_type_ind_for_conformer_torch[g_rot][:, None]
        rot_off = n_atoms_offset_for_conformer_torch[g_rot][:, None]

        safe_rto = cda_rto.clamp_min(0).to(torch.int64)
        cda_kto = kfidx[bt_idx, safe_rto] + rot_off + 1  # +1 for the virtual root

        # the chi each defining atom turns: (n_samp, max_n_chi, 4)
        four_rto = chi4_by_atom[bt_idx, safe_rto]
        valid = ((cda_rto >= 0) & (four_rto >= 0).all(dim=-1)).reshape(-1)

        # invalid entries read atom 0 and contribute a zero, so nothing here
        # depends on a count the host would have to be told
        four_abs = torch.where(four_rto >= 0, four_rto + rot_off.unsqueeze(-1), 0)
        xyz = trial_coords_rto[four_abs.reshape(-1, 4)].to(torch.float64)

        measured = coord_dihedrals(xyz[:, 0], xyz[:, 1], xyz[:, 2], xyz[:, 3])
        delta = chi_intended.reshape(-1).to(torch.float64) - measured
        delta = (delta + numpy.pi) % (2 * numpy.pi) - numpy.pi

        phi_c.index_add_(
            0,
            torch.where(valid, cda_kto.reshape(-1), 0),
            torch.where(valid, delta, torch.zeros_like(delta)).to(phi_c.dtype),
        )


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
    from tmol.kinematics.compiled import inverse_kin

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
    from tmol.kinematics.compiled import forward_only_op

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


def get_rotamer_origin_data(task: SetPackerTask, gbt_for_rot: Tensor[torch.int32][:]):
    n_poses = task.per_block_orig_block_type.shape[0]
    pose_for_gbt = task.cons_bt_pose.to(torch.int32)

    block_ind_for_rt = task.cons_bt_block.to(torch.int32)
    max_n_blocks = task.per_block_orig_block_type.shape[1]

    gbt_for_rot64 = gbt_for_rot.to(torch.int64)
    pose_for_rot = pose_for_gbt[gbt_for_rot64].to(torch.int64)
    n_rots_for_pose = torch.bincount(pose_for_rot, minlength=n_poses)
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


def build_rotamers(poses: PoseStack, task: SetPackerTask, chem_db: ChemicalDatabase):
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
    samplers = tuple(task.conformer_samplers)
    pbt = poses.packed_block_types

    # Step 2
    annotate_everything(chem_db, samplers, pbt)

    # Step 3
    # the "global block type" (gbt) is the list of all considered block types
    # at all positions in all poses.
    gbt_block_type_ind = task.cons_bt_block_type.cpu().numpy().astype(numpy.int32)

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

    conf_dofs_kto[1:] = (
        pbt.rotamer_kinforest.dofs_ideal[block_type_ind_for_conformer]
        .reshape((-1, 9))[
            pbt.atom_is_real[block_type_ind_for_conformer].reshape(-1) != 0
        ]
        .to(dtype=torch.float32, device=pbt.device)
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

    correct_phi_c_from_measured_chi(
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
