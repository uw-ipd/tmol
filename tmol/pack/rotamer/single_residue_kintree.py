import numpy

from tmol.types.functional import validate_args

from tmol.kinematics.scan_ordering import get_scans, KinTreeScanOrdering
from tmol.kinematics.builder import KinematicBuilder

from tmol.system.restypes import RefinedResidueType
from tmol.system.pose import PackedBlockTypes


@validate_args
def construct_single_residue_kintree(restype: RefinedResidueType):
    """Create a kintree for a single residue and its associated
    scan ordering data.

    The kintree data structure on its own is incomplete and
    before it can be stored will need to be left-padded with
    0s. In particular, the id, doftype, parent, frame_x, _y
    and _z data all require the 0th position to be occupied
    by the "root" atom

    Also create the backbone fingerprint
    """
    if hasattr(restype, "kintree_id"):
        assert hasattr(restype, "kintree_doftype")
        assert hasattr(restype, "kintree_parent")
        assert hasattr(restype, "kintree_frame_x")
        assert hasattr(restype, "kintree_frame_y")
        assert hasattr(restype, "kintree_frame_z")
        assert hasattr(restype, "kintree_nodes")
        assert hasattr(restype, "kintree_scans")
        assert hasattr(restype, "kintree_gens")
        assert hasattr(restype, "kintree_n_scans_per_gen")
        return
    icoor_parents = restype.icoors_ancestors[:, 0]

    torsion_pairs = numpy.array(
        [uaids[1:3] for tor, uaids in restype.torsion_to_uaids.items()]
    )
    if torsion_pairs.shape[0] > 0:
        torsion_pairs = torsion_pairs[:, :, 0]
        all_real = numpy.all(torsion_pairs >= 0, axis=1)
        torsion_pairs = torsion_pairs[all_real, :]

        kintree = (
            KinematicBuilder()
            .append_connected_component(
                *KinematicBuilder.component_for_prioritized_bonds(
                    roots=0,
                    mandatory_bonds=torsion_pairs,
                    all_bonds=restype.bond_indices,
                )
            )
            .kintree
        )
    else:
        kintree = (
            KinematicBuilder()
            .append_connected_component(
                *KinematicBuilder.bonds_to_connected_component(
                    roots=0, bonds=restype.bond_indices
                )
            )
            .kintree
        )
    forward_scan_paths = KinTreeScanOrdering.calculate_from_kintree(
        kintree
    ).forward_scan_paths

    nodes = forward_scan_paths.nodes.numpy()
    scans = forward_scan_paths.scans.numpy()
    gens = forward_scan_paths.gens.numpy()

    n_scans_per_gen = gens[1:, 1] - gens[:-1, 1]

    setattr(restype, "kintree_id", kintree.id.numpy()[1:])
    setattr(restype, "kintree_doftype", kintree.doftype.numpy()[1:])
    setattr(restype, "kintree_parent", kintree.parent.numpy()[1:] - 1)
    setattr(restype, "kintree_frame_x", kintree.frame_x.numpy()[1:] - 1)
    setattr(restype, "kintree_frame_y", kintree.frame_y.numpy()[1:] - 1)
    setattr(restype, "kintree_frame_z", kintree.frame_z.numpy()[1:] - 1)
    setattr(restype, "kintree_nodes", nodes)
    setattr(restype, "kintree_scans", scans)
    setattr(restype, "kintree_gens", gens)
    setattr(restype, "kintree_n_scans_per_gen", n_scans_per_gen)


@validate_args
def coalesce_single_residue_kintrees(pbt: PackedBlockTypes):
    if hasattr(pbt, "kintree_nodes"):
        assert hasattr(pbt, "kintree_n_nodes")
        assert hasattr(pbt, "kintree_nodes")
        assert hasattr(pbt, "kintree_scans")
        assert hasattr(pbt, "kintree_gens")
        assert hasattr(pbt, "kintree_n_scans_per_gen")
        assert hasattr(pbt, "kintree_id")
        assert hasattr(pbt, "kintree_doftype")
        assert hasattr(pbt, "kintree_parent")
        assert hasattr(pbt, "kintree_frame_x")
        assert hasattr(pbt, "kintree_frame_y")
        assert hasattr(pbt, "kintree_frame_z")
        return

    max_n_nodes = max(len(rt.kintree_nodes) for rt in pbt.active_residues)
    max_n_scans = max(rt.kintree_scans.shape[0] for rt in pbt.active_residues)
    max_n_gens = max(rt.kintree_gens.shape[0] for rt in pbt.active_residues)
    max_n_atoms = max(rt.kintree_id.shape[0] for rt in pbt.active_residues)

    rt_n_nodes = numpy.zeros((pbt.n_types,), dtype=numpy.int32)
    rt_nodes = numpy.full((pbt.n_types, max_n_nodes), -1, dtype=numpy.int32)
    rt_scans = numpy.full((pbt.n_types, max_n_scans), -1, dtype=numpy.int32)
    rt_gens = numpy.full((pbt.n_types, max_n_gens, 2), 0, dtype=numpy.int32)
    rt_n_scans_per_gen = numpy.full(
        (pbt.n_types, max_n_gens - 1), -1, dtype=numpy.int32
    )

    rt_id = numpy.full((pbt.n_types, max_n_atoms), -1, dtype=numpy.int32)
    rt_doftype = numpy.full((pbt.n_types, max_n_atoms), -1, dtype=numpy.int32)
    rt_parent = numpy.full((pbt.n_types, max_n_atoms), -1, dtype=numpy.int32)
    rt_frame_x = numpy.full((pbt.n_types, max_n_atoms), -1, dtype=numpy.int32)
    rt_frame_y = numpy.full((pbt.n_types, max_n_atoms), -1, dtype=numpy.int32)
    rt_frame_z = numpy.full((pbt.n_types, max_n_atoms), -1, dtype=numpy.int32)

    for i, rt in enumerate(pbt.active_residues):
        rt_n_nodes[i] = len(rt.kintree_nodes)
        rt_nodes[i, : len(rt.kintree_nodes)] = rt.kintree_nodes
        rt_scans[i, : rt.kintree_scans.shape[0]] = rt.kintree_scans
        rt_gens[i, : rt.kintree_gens.shape[0], :] = rt.kintree_gens
        # fill forward
        rt_gens[i, rt.kintree_gens.shape[0] :, :] = rt.kintree_gens[-1, :]
        rt_n_scans_per_gen[
            i, : rt.kintree_n_scans_per_gen.shape[0]
        ] = rt.kintree_n_scans_per_gen
        rt_id[i, : rt.kintree_id.shape[0]] = rt.kintree_id
        rt_doftype[i, : rt.kintree_id.shape[0]] = rt.kintree_doftype
        rt_parent[i, : rt.kintree_id.shape[0]] = rt.kintree_parent
        rt_frame_x[i, : rt.kintree_id.shape[0]] = rt.kintree_frame_x
        rt_frame_y[i, : rt.kintree_id.shape[0]] = rt.kintree_frame_y
        rt_frame_z[i, : rt.kintree_id.shape[0]] = rt.kintree_frame_z

    setattr(pbt, "kintree_id", rt_id)
    setattr(pbt, "kintree_doftype", rt_doftype)
    setattr(pbt, "kintree_parent", rt_parent)
    setattr(pbt, "kintree_frame_x", rt_frame_x)
    setattr(pbt, "kintree_frame_y", rt_frame_y)
    setattr(pbt, "kintree_frame_z", rt_frame_z)
    setattr(pbt, "kintree_n_nodes", rt_n_nodes)
    setattr(pbt, "kintree_nodes", rt_nodes)
    setattr(pbt, "kintree_scans", rt_scans)
    setattr(pbt, "kintree_gens", rt_gens)
    setattr(pbt, "kintree_n_scans_per_gen", rt_n_scans_per_gen)
