import numpy
import numba
import torch
import attr
import cattr

from tmol.pack.rotamer.build_rotamers import (
    annotate_restype,
    annotate_packed_block_types,
    build_rotamers,
    construct_scans_for_rotamers,
    exclusive_cumsum,
)
from tmol.system.ideal_coords import build_ideal_coords, normalize
from tmol.system.restypes import RefinedResidueType, ResidueTypeSet
from tmol.system.pose import PackedBlockTypes, Pose, Poses
from tmol.score.dunbrack.params import DunbrackParamResolver
from tmol.pack.packer_task import PackerTask, PackerPalette
from tmol.pack.rotamer.chi_sampler import ChiSampler
from tmol.pack.rotamer.dunbrack_chi_sampler import DunbrackChiSampler

from tmol.numeric.dihedrals import coord_dihedrals


def test_annotate_restypes(default_database):
    rts = ResidueTypeSet.from_database(default_database.chemical)

    for rt in rts.residue_types:
        annotate_restype(rt)
        assert hasattr(rt, "kintree_id")
        assert hasattr(rt, "kintree_doftype")
        assert hasattr(rt, "kintree_parent")
        assert hasattr(rt, "kintree_frame_x")
        assert hasattr(rt, "kintree_frame_y")
        assert hasattr(rt, "kintree_frame_z")
        assert hasattr(rt, "kintree_nodes")
        assert hasattr(rt, "kintree_scans")
        assert hasattr(rt, "kintree_gens")
        assert hasattr(rt, "kintree_n_scans_per_gen")

        assert type(rt.kintree_id) == numpy.ndarray
        assert type(rt.kintree_doftype) == numpy.ndarray
        assert type(rt.kintree_parent) == numpy.ndarray
        assert type(rt.kintree_frame_x) == numpy.ndarray
        assert type(rt.kintree_frame_y) == numpy.ndarray
        assert type(rt.kintree_frame_z) == numpy.ndarray

        assert rt.kintree_id.shape == (rt.n_atoms,)
        assert rt.kintree_doftype.shape == (rt.n_atoms,)
        assert rt.kintree_parent.shape == (rt.n_atoms,)
        assert rt.kintree_frame_x.shape == (rt.n_atoms,)
        assert rt.kintree_frame_y.shape == (rt.n_atoms,)
        assert rt.kintree_frame_z.shape == (rt.n_atoms,)


def test_build_rotamers_smoke(ubq_res, default_database):
    torch_device = torch.device("cpu")

    rts = ResidueTypeSet.from_database(default_database.chemical)

    # replace them with residues constructed from the residue types
    # that live in our locally constructed set of refined residue types
    ubq_res = [
        attr.evolve(
            res,
            residue_type=next(
                rt for rt in rts.residue_types if rt.name == res.residue_type.name
            ),
        )
        for res in ubq_res
    ]

    p1 = Pose.from_residues_one_chain(ubq_res[5:11], torch_device)
    p2 = Pose.from_residues_one_chain(ubq_res[:7], torch_device)
    poses = Poses.from_poses([p1, p2], torch_device)
    palette = PackerPalette(rts)
    task = PackerTask(poses, palette)
    task.restrict_to_repacking()

    param_resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    sampler = DunbrackChiSampler.from_database(param_resolver)
    task.add_chi_sampler(sampler)

    build_rotamers(poses, task)


def test_construct_scans_for_rotamers(default_database):
    torch_device = torch.device("cpu")

    rts = ResidueTypeSet.from_database(default_database.chemical)
    leu_rt_list = [rts.restype_map["LEU"][0]]
    pbt = PackedBlockTypes.from_restype_list(leu_rt_list, device=torch_device)

    annotate_restype(leu_rt_list[0])
    annotate_packed_block_types(pbt)

    rt_block_inds = numpy.zeros(3, dtype=numpy.int32)
    rt_for_rot = torch.zeros(3, dtype=torch.int64)

    block_ind_for_rot = rt_block_inds[rt_for_rot]
    block_ind_for_rot_torch = torch.tensor(
        block_ind_for_rot, dtype=torch.int64, device=torch_device
    )
    n_atoms_for_rot = pbt.n_atoms[block_ind_for_rot_torch]
    n_atoms_offset_for_rot = torch.cumsum(n_atoms_for_rot, dim=0)
    n_atoms_offset_for_rot = n_atoms_offset_for_rot.cpu().numpy()
    n_atoms_total = n_atoms_offset_for_rot[-1]
    n_atoms_offset_for_rot = exclusive_cumsum(n_atoms_offset_for_rot)

    nodes, scans, gens = construct_scans_for_rotamers(
        pbt, block_ind_for_rot, n_atoms_for_rot, n_atoms_offset_for_rot
    )

    n_atoms = len(leu_rt_list[0].atoms)
    kt_nodes = pbt.kintree_nodes[0]
    kt_scans = pbt.kintree_scans[0]
    kt_gens = pbt.kintree_gens[0]
    nodes_gold = numpy.concatenate(
        [
            kt_nodes[0 : kt_gens[1, 0]],
            kt_nodes[0 : kt_gens[1, 0]] + n_atoms,
            kt_nodes[0 : kt_gens[1, 0]] + 2 * n_atoms,
            kt_nodes[kt_gens[1, 0] : kt_gens[2, 0]],
            kt_nodes[kt_gens[1, 0] : kt_gens[2, 0]] + n_atoms,
            kt_nodes[kt_gens[1, 0] : kt_gens[2, 0]] + 2 * n_atoms,
            kt_nodes[kt_gens[2, 0] : kt_gens[3, 0]],
            kt_nodes[kt_gens[2, 0] : kt_gens[3, 0]] + n_atoms,
            kt_nodes[kt_gens[2, 0] : kt_gens[3, 0]] + 2 * n_atoms,
        ]
    )
    numpy.testing.assert_equal(nodes_gold, nodes)

    scans_gold = numpy.concatenate(
        [
            kt_scans[0 : kt_gens[1, 1]],
            kt_scans[0 : kt_gens[1, 1]] + kt_gens[1, 0],
            kt_scans[0 : kt_gens[1, 1]] + 2 * kt_gens[1, 0],
            kt_scans[kt_gens[1, 1] : kt_gens[2, 1]],
            kt_scans[kt_gens[1, 1] : kt_gens[2, 1]]
            + 1 * (kt_gens[2, 0] - kt_gens[1, 0]),
            kt_scans[kt_gens[1, 1] : kt_gens[2, 1]]
            + 2 * (kt_gens[2, 0] - kt_gens[1, 0]),
            kt_scans[kt_gens[2, 1] : kt_gens[3, 1]],
            kt_scans[kt_gens[2, 1] : kt_gens[3, 1]]
            + 1 * (kt_gens[3, 0] - kt_gens[2, 0]),
            kt_scans[kt_gens[2, 1] : kt_gens[3, 1]]
            + 2 * (kt_gens[3, 0] - kt_gens[2, 0]),
        ]
    )
    numpy.testing.assert_equal(scans_gold, scans)

    gens_gold = kt_gens * 3
    numpy.testing.assert_equal(gens_gold, gens)


def test_construct_scans_for_rotamers2(default_database):
    torch_device = torch.device("cpu")

    rts = ResidueTypeSet.from_database(default_database.chemical)
    leu_met_rt_list = [rts.restype_map["LEU"][0]] + [rts.restype_map["MET"][0]]
    pbt = PackedBlockTypes.from_restype_list(leu_met_rt_list, device=torch_device)

    annotate_restype(leu_met_rt_list[0])
    annotate_restype(leu_met_rt_list[1])
    annotate_packed_block_types(pbt)

    rt_block_inds = numpy.concatenate(
        [numpy.zeros(1, dtype=numpy.int32), numpy.ones(2, dtype=numpy.int32)]
    )
    rt_for_rot = torch.cat(
        [torch.zeros(1, dtype=torch.int64), torch.ones(2, dtype=torch.int64)]
    )

    block_ind_for_rot = rt_block_inds[rt_for_rot]
    block_ind_for_rot_torch = torch.tensor(
        block_ind_for_rot, dtype=torch.int64, device=torch_device
    )
    n_atoms_for_rot = pbt.n_atoms[block_ind_for_rot_torch]
    n_atoms_offset_for_rot = torch.cumsum(n_atoms_for_rot, dim=0)
    n_atoms_offset_for_rot = n_atoms_offset_for_rot.cpu().numpy()
    n_atoms_total = n_atoms_offset_for_rot[-1]
    n_atoms_offset_for_rot = exclusive_cumsum(n_atoms_offset_for_rot)

    nodes, scans, gens = construct_scans_for_rotamers(
        pbt, block_ind_for_rot, n_atoms_for_rot, n_atoms_offset_for_rot
    )

    leu_n_atoms = len(leu_met_rt_list[0].atoms)
    met_n_atoms = len(leu_met_rt_list[1].atoms)
    kt_nodes = pbt.kintree_nodes
    kt_scans = pbt.kintree_scans
    kt_gens = pbt.kintree_gens
    leu = 0
    met = 1
    nodes_gold = numpy.concatenate(
        [
            kt_nodes[leu, 0 : kt_gens[leu, 1, 0]],
            kt_nodes[met, 0 : kt_gens[met, 1, 0]] + leu_n_atoms,
            kt_nodes[met, 0 : kt_gens[met, 1, 0]] + leu_n_atoms + met_n_atoms,
            kt_nodes[leu, kt_gens[leu, 1, 0] : kt_gens[leu, 2, 0]],
            kt_nodes[met, kt_gens[met, 1, 0] : kt_gens[met, 2, 0]] + leu_n_atoms,
            kt_nodes[met, kt_gens[met, 1, 0] : kt_gens[met, 2, 0]]
            + leu_n_atoms
            + met_n_atoms,
            kt_nodes[leu, kt_gens[leu, 2, 0] : kt_gens[leu, 3, 0]],
            kt_nodes[met, kt_gens[met, 2, 0] : kt_gens[met, 3, 0]] + leu_n_atoms,
            kt_nodes[met, kt_gens[met, 2, 0] : kt_gens[met, 3, 0]]
            + leu_n_atoms
            + met_n_atoms,
        ]
    )
    numpy.testing.assert_equal(nodes_gold, nodes)

    scans_gold = numpy.concatenate(
        [
            kt_scans[leu, 0 : kt_gens[leu, 1, 1]],
            kt_scans[met, 0 : kt_gens[met, 1, 1]] + kt_gens[leu, 1, 0],
            kt_scans[met, 0 : kt_gens[met, 1, 1]]
            + kt_gens[leu, 1, 0]
            + kt_gens[met, 1, 0],
            kt_scans[leu, kt_gens[leu, 1, 1] : kt_gens[leu, 2, 1]],
            kt_scans[met, kt_gens[met, 1, 1] : kt_gens[met, 2, 1]]
            + kt_gens[leu, 2, 0]
            - kt_gens[leu, 1, 0],
            kt_scans[met, kt_gens[met, 1, 1] : kt_gens[met, 2, 1]]
            + kt_gens[leu, 2, 0]
            - kt_gens[leu, 1, 0]
            + kt_gens[met, 2, 0]
            - kt_gens[met, 1, 0],
            kt_scans[leu, kt_gens[leu, 2, 1] : kt_gens[leu, 3, 1]],
            kt_scans[met, kt_gens[met, 2, 1] : kt_gens[met, 3, 1]]
            + kt_gens[leu, 3, 0]
            - kt_gens[leu, 2, 0],
            kt_scans[met, kt_gens[met, 2, 1] : kt_gens[met, 3, 1]]
            + kt_gens[leu, 3, 0]
            - kt_gens[leu, 2, 0]
            + kt_gens[met, 3, 0]
            - kt_gens[met, 2, 0],
        ]
    )
    numpy.testing.assert_equal(scans_gold, scans)

    gens_gold = kt_gens[leu] + 2 * kt_gens[met]
    numpy.testing.assert_equal(gens_gold, gens)


def test_inv_kin_rotamers(default_database, ubq_res):
    torch_device = torch.device("cpu")

    rts = ResidueTypeSet.from_database(default_database.chemical)
    ubq_res = [
        attr.evolve(
            res,
            residue_type=next(
                rt for rt in rts.residue_types if rt.name == res.residue_type.name
            ),
        )
        for res in ubq_res
    ]
    p = Pose.from_residues_one_chain(ubq_res[:3], torch_device)

    leu_met_rt_list = [rts.restype_map["LEU"][0]] + [rts.restype_map["MET"][0]]
    pbt = PackedBlockTypes.from_restype_list(leu_met_rt_list, device=torch_device)

    annotate_restype(leu_met_rt_list[0])
    annotate_restype(leu_met_rt_list[1])
    annotate_packed_block_types(pbt)

    met_rt = leu_met_rt_list[1]

    def ia(val, arr):
        return torch.tensor(
            numpy.concatenate((numpy.array([val]), arr)),
            dtype=torch.int32,
            device=torch_device,
        )

    def fa(val, arr):
        return torch.tensor(
            numpy.concatenate((numpy.array([val]), arr)),
            dtype=torch.float32,
            device=torch_device,
        )

    kt_id = ia(-1, met_rt.kintree_id)
    kt_doftype = ia(0, met_rt.kintree_doftype)
    kt_parent = ia(0, met_rt.kintree_parent)
    kt_frame_x = ia(0, met_rt.kintree_frame_x)
    kt_frame_y = ia(0, met_rt.kintree_frame_y)
    kt_frame_z = ia(0, met_rt.kintree_frame_z)

    from tmol.kinematics.compiled.compiled import inverse_kin

    dofs = inverse_kin(
        p.coords[0], kt_parent, kt_frame_x, kt_frame_y, kt_frame_z, kt_doftype
    )

    print("dofs")
    print(dofs)

    # what atoms should we copy over?
    # everything north of "first sidechain atom"?
    # let's have a map from rt x bb-type --> atom-indices on that rt for those bb
    # and then when we want to map between two rts, we ask "what is their rt compatibility"?
    # and then use that mapping

    # so
    # all canonical aas except proline are class 1
    # pro is class 2
    # gly is class 3
    #
    # class 1 has n, ca, c, o, h, and ha
    # class 2 has n, ca, c, o, and ha
    # class 3 has n, ca, c, o, and the "left" ha

    # how do we tell what classes of backbones there are?
    # we ask:
    # what atoms are upstream of the first sidechain atom
    # for each atom that's upstream of the first sidechain atom
    # who is chemically bound to it, what is the chirality
    # of that connection, and what is the element type of that
    # connection

    # then we need to hash that
    # (how???)
    # atoms then should be sorted along mainchain?? and then
    # with chirality

    # n -- > (0, 0, 0, 7)
    # h -- > (0, 1, 0, 1)
    # ca --> (1, 0, 0, 6)
    # ha --> (1, 1, 1, 1)
    # c  --> (2, 0, 0, 6)
    # o  --> (2, 1, 0, 8)

    # position 0: position along the backbone or backbone you're bonded to
    # position 1: number of bonds from the backbone
    # position 2: chirality: 0 - achiral, 1 - left, 2 - right
    # position 3: element

    # how do I determine chirality?
    #
    # if bb atom has three chemical bonds, then
    # treat it as achiral.
    # if it has four chemical bonds, then
    # measure chirality of 4th bond by taking
    # the dot product of sc-i and the cross
    # product of (p_i - p_{i-1}) and (p_{i+1}, p_i)
    # if it's positive, then chirality value of 1
    # if it's negative, then chirality value of 2

    # and then the 4th column is the element, so, that needs to be encoded somehow...

    # how do we sort atoms further from the backbone?
    # what about when something like: put into the chirality position
    # a counter so that things further from the backbone get noted
    # with a higher count; how can you guarantee uniqueness, though??
    # maybe it should be like an array with an offset based on the chirality
    # of its ancestors back to the backbone where you put
