import attr
import numpy
import torch

from tmol.types.functional import validate_args
from tmol.types.array import NDArray

from tmol.kinematics.datatypes import NodeType, KinForest
from tmol.kinematics.scan_ordering import (
    KinForestScanOrdering,
    annotate_block_type_with_residue_kinforest_data,
)
from tmol.kinematics.compiled.compiled_inverse_kin import inverse_kin
from tmol.utility.ndarray.common_operations import invert_mapping

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RotamerKintree:
    kinforest_idx: NDArray[numpy.int32][
        :
    ]  # mapping from restype order to kinforest order

    # the following arrays are all in kinforest order
    id: NDArray[numpy.int32][:]  # mapping from kinforest order to restype order
    doftype: NDArray[numpy.int32][:]
    parent: NDArray[numpy.int32][:]
    frame_x: NDArray[numpy.int32][:]
    frame_y: NDArray[numpy.int32][:]
    frame_z: NDArray[numpy.int32][:]
    nodes: NDArray[numpy.int32][:]
    scans: NDArray[numpy.int32][:]
    gens: NDArray[numpy.int32][:]
    n_scans_per_gen: NDArray[numpy.int32][:]
    dofs_ideal: NDArray[numpy.int32][:]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class PackedRotamerKintree:
    kinforest_idx: NDArray[numpy.int32][
        :, :
    ]  # mapping from restype order to kinforest order

    # the following arrays are all in kinforest order
    id: NDArray[numpy.int32][:, :]  # mapping from kinforest order to restype order
    doftype: NDArray[numpy.int32][:, :]
    parent: NDArray[numpy.int32][:, :]
    frame_x: NDArray[numpy.int32][:, :]
    frame_y: NDArray[numpy.int32][:, :]
    frame_z: NDArray[numpy.int32][:, :]
    n_nodes: NDArray[numpy.int32][:]
    nodes: NDArray[numpy.int32][:, :]
    scans: NDArray[numpy.int32][:, :]
    gens: NDArray[numpy.int32][:, :]
    n_scans_per_gen: NDArray[numpy.int32][:, :]
    dofs_ideal: NDArray[numpy.int32][:, :]


@validate_args
def construct_single_residue_kinforest(restype: RefinedResidueType):
    """Create a kinforest for a single residue and its associated
    scan ordering data.

    The kinforest data structure on its own is incomplete and
    before it can be stored will need to be left-padded with
    0s. In particular, the id, doftype, parent, frame_x, _y
    and _z data all require the 0th position to be occupied
    by the "root" atom

    Also create the backbone fingerprint
    """
    if hasattr(restype, "rotamer_kinforest"):
        return

    annotate_block_type_with_residue_kinforest_data(restype)
    rkd = restype.residue_kinforest_data
    n_atoms = restype.n_atoms

    kfo_2_to = rkd.bfto_2_orig.astype(numpy.int64)  # KFO → TO (0-indexed)
    preds = rkd.preds.astype(numpy.int64)  # BFS predecessors in TO; -9999 for root
    dof_type_to = rkd.dof_type  # NodeType per atom in TO

    # TO → KFO mapping (used for kinforest_idx output field)
    to_2_kfo = invert_mapping(kfo_2_to, n_atoms).astype(numpy.int64)

    # KFO parent indices, 0-indexed (root points to itself)
    kfo_parents = numpy.zeros(n_atoms, dtype=numpy.int64)
    kfo_parents[1:] = to_2_kfo[preds[kfo_2_to[1:]]]

    # DOF types in KFO order
    dof_type_kfo = dof_type_to[kfo_2_to].astype(numpy.int32)

    # --- Vectorized frame defaults (correct for all bond atoms) ---
    # frame_x = self, frame_y = parent, frame_z = grandparent
    frame_x = numpy.arange(n_atoms, dtype=numpy.int64)
    frame_y = kfo_parents.copy()
    frame_z = kfo_parents[kfo_parents]

    # --- Fix jump root (KFO index 0) and its direct children ---
    # In BFS order, root is at KFO 0 and its first child is at KFO 1 (= c1).
    # c2 = first child of c1 if c1 has any children, else second child of root.
    c1 = 1  # first BFS child of root (all children are non-jump)
    c1_children = numpy.where(kfo_parents == c1)[0]
    if len(c1_children) > 0:
        c2 = int(c1_children[0])
    else:
        # c1 has no children; use the second child of root
        root_children = numpy.where((kfo_parents == 0) & (numpy.arange(n_atoms) > 0))[0]
        c2 = int(root_children[1])  # root_children[0] is c1=1

    # Root (KFO 0): frame_x=c1, frame_y=self(=0), frame_z=c2
    frame_x[0] = c1
    frame_y[0] = 0
    frame_z[0] = c2

    # c1 (KFO 1): default frame_z = kfo_parents[kfo_parents[1]] = kfo_parents[0] = 0
    # (root itself), but it should be c2.
    frame_z[c1] = c2

    # Other direct children of root (KFO > 1, parent == 0):
    # default frame_z = kfo_parents[root] = root = 0, but it should be c1.
    other_root_children = numpy.where((kfo_parents == 0) & (numpy.arange(n_atoms) > 1))[
        0
    ]
    frame_z[other_root_children] = c1

    # --- Build KinForest with global root at position 0 (required by inverse_kin) ---
    def _t(x):
        return torch.tensor(numpy.asarray(x, dtype=numpy.int32))

    # kfo_parents[0] == 0 is a self-loop in KFO space (the jump root is its own parent).
    # After the +1 shift to kinforest positions this gives parent[1] = 1, which
    # get_scans treats as a disconnected second root (segfault).
    # Fix: use -1 for the jump root so that -1+1 = 0 (points to the global root).
    kfo_parents_kf = kfo_parents.copy()
    kfo_parents_kf[0] = -1

    kinforest = KinForest(
        id=_t(numpy.concatenate([[-1], kfo_2_to])),
        doftype=_t(numpy.concatenate([[NodeType.root], dof_type_kfo])),
        parent=_t(numpy.concatenate([[0], kfo_parents_kf + 1])),
        frame_x=_t(numpy.concatenate([[0], frame_x + 1])),
        frame_y=_t(numpy.concatenate([[0], frame_y + 1])),
        frame_z=_t(numpy.concatenate([[0], frame_z + 1])),
    )

    forward_scan_paths = KinForestScanOrdering.calculate_from_kinforest(
        kinforest
    ).forward_scan_paths

    nodes = forward_scan_paths.nodes.numpy()
    scans = forward_scan_paths.scans.numpy()
    gens = forward_scan_paths.gens.numpy()

    n_scans_per_gen = gens[1:, 1] - gens[:-1, 1]

    ideal_coords = torch.cat(
        (
            torch.zeros((1, 3), dtype=torch.float32),
            torch.tensor(
                restype.ideal_coords[restype.at_to_icoor_ind][kfo_2_to],
                dtype=torch.float32,
            ),
        )
    )

    dofs_ideal = inverse_kin(
        ideal_coords,
        kinforest.parent,
        kinforest.frame_x,
        kinforest.frame_y,
        kinforest.frame_z,
        kinforest.doftype,
    )
    dofs_ideal = dofs_ideal.numpy()

    rotamer_kinforest = RotamerKintree(
        kinforest_idx=to_2_kfo.astype(numpy.int32),
        id=kfo_2_to.astype(numpy.int32),
        doftype=dof_type_kfo,
        parent=kfo_parents_kf.astype(
            numpy.int32
        ),  # root is -1; load_rotamer_parents adds 1
        frame_x=frame_x.astype(numpy.int32),
        frame_y=frame_y.astype(numpy.int32),
        frame_z=frame_z.astype(numpy.int32),
        nodes=nodes,
        scans=scans,
        gens=gens,
        n_scans_per_gen=n_scans_per_gen,
        dofs_ideal=dofs_ideal[1:],
    )
    setattr(restype, "rotamer_kinforest", rotamer_kinforest)


@validate_args
def coalesce_single_residue_kinforests(pbt: PackedBlockTypes):
    if hasattr(pbt, "rotamer_kinforest"):
        return

    max_n_nodes = max(len(rt.rotamer_kinforest.nodes) for rt in pbt.active_block_types)
    max_n_scans = max(
        rt.rotamer_kinforest.scans.shape[0] for rt in pbt.active_block_types
    )
    max_n_gens = max(
        rt.rotamer_kinforest.gens.shape[0] for rt in pbt.active_block_types
    )
    max_n_atoms = max(rt.rotamer_kinforest.id.shape[0] for rt in pbt.active_block_types)

    rt_n_nodes = numpy.zeros((pbt.n_types,), dtype=numpy.int32)
    rt_nodes = numpy.full((pbt.n_types, max_n_nodes), -1, dtype=numpy.int32)
    rt_scans = numpy.full((pbt.n_types, max_n_scans), -1, dtype=numpy.int32)
    rt_gens = numpy.full((pbt.n_types, max_n_gens, 2), 0, dtype=numpy.int32)
    rt_n_scans_per_gen = numpy.full(
        (pbt.n_types, max_n_gens - 1), -1, dtype=numpy.int32
    )

    rt_kinforest_idx = numpy.full((pbt.n_types, max_n_atoms), -1, dtype=numpy.int32)
    rt_id = numpy.full((pbt.n_types, max_n_atoms), -1, dtype=numpy.int32)
    rt_doftype = numpy.full((pbt.n_types, max_n_atoms), -1, dtype=numpy.int32)
    rt_parent = numpy.full((pbt.n_types, max_n_atoms), -1, dtype=numpy.int32)
    rt_frame_x = numpy.full((pbt.n_types, max_n_atoms), -1, dtype=numpy.int32)
    rt_frame_y = numpy.full((pbt.n_types, max_n_atoms), -1, dtype=numpy.int32)
    rt_frame_z = numpy.full((pbt.n_types, max_n_atoms), -1, dtype=numpy.int32)
    rt_dofs_ideal = numpy.zeros((pbt.n_types, max_n_atoms, 9), dtype=numpy.float32)

    for i, rt in enumerate(pbt.active_block_types):
        rt_n_nodes[i] = rt.rotamer_kinforest.nodes.shape[0]
        rt_nodes[i, : rt.rotamer_kinforest.nodes.shape[0]] = rt.rotamer_kinforest.nodes
        rt_scans[i, : rt.rotamer_kinforest.scans.shape[0]] = rt.rotamer_kinforest.scans
        rt_gens[i, : rt.rotamer_kinforest.gens.shape[0], :] = rt.rotamer_kinforest.gens
        # fill forward
        rt_gens[i, rt.rotamer_kinforest.gens.shape[0] :, :] = rt.rotamer_kinforest.gens[
            -1, :
        ]
        rt_n_scans_per_gen[i, : rt.rotamer_kinforest.n_scans_per_gen.shape[0]] = (
            rt.rotamer_kinforest.n_scans_per_gen
        )
        rt_kinforest_idx[i, : rt.n_atoms] = rt.rotamer_kinforest.kinforest_idx
        rt_id[i, : rt.n_atoms] = rt.rotamer_kinforest.id
        rt_doftype[i, : rt.n_atoms] = rt.rotamer_kinforest.doftype
        rt_parent[i, : rt.n_atoms] = rt.rotamer_kinforest.parent
        rt_frame_x[i, : rt.n_atoms] = rt.rotamer_kinforest.frame_x
        rt_frame_y[i, : rt.n_atoms] = rt.rotamer_kinforest.frame_y
        rt_frame_z[i, : rt.n_atoms] = rt.rotamer_kinforest.frame_z
        rt_dofs_ideal[i, : rt.n_atoms] = rt.rotamer_kinforest.dofs_ideal

    packed_rotamer_kinforest = PackedRotamerKintree(
        kinforest_idx=rt_kinforest_idx,
        id=rt_id,
        doftype=rt_doftype,
        parent=rt_parent,
        frame_x=rt_frame_x,
        frame_y=rt_frame_y,
        frame_z=rt_frame_z,
        n_nodes=rt_n_nodes,
        nodes=rt_nodes,
        scans=rt_scans,
        gens=rt_gens,
        n_scans_per_gen=rt_n_scans_per_gen,
        dofs_ideal=rt_dofs_ideal,
    )
    setattr(pbt, "rotamer_kinforest", packed_rotamer_kinforest)
