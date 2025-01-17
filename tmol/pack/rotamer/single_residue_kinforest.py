import attr
import numpy
import torch

from tmol.types.functional import validate_args
from tmol.types.array import NDArray

from tmol.kinematics.scan_ordering import KinForestScanOrdering
from tmol.kinematics.builder import _KinematicBuilder
from tmol.kinematics.compiled.compiled_inverse_kin import inverse_kin

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

    torsion_pairs = numpy.array(
        [uaids[1:3] for tor, uaids in restype.torsion_to_uaids.items()]
    )
    if torsion_pairs.shape[0] > 0:
        torsion_pairs = torsion_pairs[:, :, 0]
        all_real = numpy.all(torsion_pairs >= 0, axis=1)
        torsion_pairs = torsion_pairs[all_real, :]

        kinforest = (
            _KinematicBuilder()
            .append_connected_components(
                numpy.zeros((1,), dtype=numpy.int32),
                *_KinematicBuilder.define_trees_with_prioritized_bonds(
                    roots=numpy.zeros((1,), dtype=numpy.int32),
                    potential_bonds=restype.bond_indices,
                    prioritized_bonds=torsion_pairs,
                    # all_bonds=restype.bond_indices,
                    # n_atoms_total=restype.n_atoms,
                ),
                to_jump_nodes=numpy.array([], dtype=numpy.int32),
            )
            .kinforest
        )
    else:
        # print("bonds")
        # print(restype.bond_indices.shape)
        # print(restype.bond_indices.dtype)
        kinforest = (
            _KinematicBuilder()
            .append_connected_components(
                numpy.zeros((1,), dtype=numpy.int32),
                *_KinematicBuilder.bonds_to_forest(
                    roots=numpy.array([0], dtype=numpy.int32),
                    bonds=restype.bond_indices,
                ),
            )
            .kinforest
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
                restype.ideal_coords[restype.at_to_icoor_ind][kinforest.id[1:]],
                dtype=torch.float32,
            ),
        )
    )
    # print("ideal coords")
    # print(ideal_coords)

    dofs_ideal = inverse_kin(
        ideal_coords,
        kinforest.parent,
        kinforest.frame_x,
        kinforest.frame_y,
        kinforest.frame_z,
        kinforest.doftype,
    )
    dofs_ideal = dofs_ideal.numpy()
    # print("dofs ideal")
    # print(dofs_ideal[:,:4])

    kinforest_idx = numpy.zeros((restype.n_atoms,), dtype=numpy.int32)
    kinforest_idx[kinforest.id.numpy()[1:]] = numpy.arange(
        restype.n_atoms, dtype=numpy.int32
    )

    rotamer_kinforest = RotamerKintree(
        kinforest_idx=kinforest_idx,
        id=kinforest.id.numpy()[1:],
        doftype=kinforest.doftype.numpy()[1:],
        parent=kinforest.parent.numpy()[1:] - 1,
        frame_x=kinforest.frame_x.numpy()[1:] - 1,
        frame_y=kinforest.frame_y.numpy()[1:] - 1,
        frame_z=kinforest.frame_z.numpy()[1:] - 1,
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
