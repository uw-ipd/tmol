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
from tmol.pack.rotamer.fixed_aa_chi_sampler import FixedAAChiSampler

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
    chem_db = default_database.chemical

    rts = ResidueTypeSet.from_database(chem_db)
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

    param_resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    dun_sampler = DunbrackChiSampler.from_database(param_resolver)
    fixed_sampler = FixedAAChiSampler()
    samplers = (dun_sampler, fixed_sampler)

    leu_met_rt_list = [rts.restype_map["LEU"][0]] + [rts.restype_map["MET"][0]]
    pbt = PackedBlockTypes.from_restype_list(leu_met_rt_list, device=torch_device)

    annotate_restype(leu_met_rt_list[0], samplers, chem_db)
    annotate_restype(leu_met_rt_list[1], samplers, chem_db)
    annotate_packed_block_types(pbt)

    leu_rt = leu_met_rt_list[0]
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
    kt_parent = ia(0, met_rt.kintree_parent + 1)
    kt_frame_x = ia(0, met_rt.kintree_frame_x + 1)
    kt_frame_y = ia(0, met_rt.kintree_frame_y + 1)
    kt_frame_z = ia(0, met_rt.kintree_frame_z + 1)

    from tmol.kinematics.compiled.compiled import inverse_kin

    # print("kt_parent")
    # print(kt_parent)
    # print("kt_frame_x")
    # print(kt_frame_x)
    # print("kt_frame_y")
    # print(kt_frame_y)
    # print("kt_frame_z")
    # print(kt_frame_z)

    dofs_orig = inverse_kin(
        p.coords[0], kt_parent, kt_frame_x, kt_frame_y, kt_frame_z, kt_doftype
    )

    print("dofs")
    print(dofs_orig)

    dofs_new = torch.zeros((leu_rt.n_atoms + 1, 9), dtype=torch.float32)

    dun_sampler_ind = pbt.mc_sampler_mapping[dun_sampler.sampler_name()]
