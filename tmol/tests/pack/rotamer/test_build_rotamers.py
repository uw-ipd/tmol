import numpy
import numba
import torch
import attr
import cattr

from tmol.pack.rotamer.build_rotamers import (
    annotate_restype,
    annotate_packed_block_types,
    build_rotamers,
    construct_kintree_for_rotamers,
    construct_scans_for_rotamers,
    exclusive_cumsum,
    measure_dofs_from_orig_coords,
    rebuild_poses_if_necessary,
    annotate_everything,
    create_dof_inds_to_copy_from_orig_to_rotamers,
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
from tmol.utility.tensor.common_operations import exclusive_cumsum1d


def test_annotate_restypes(default_database, torch_device):
    # torch_device = torch.device("cpu")
    rts = ResidueTypeSet.from_database(default_database.chemical)

    param_resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    dun_sampler = DunbrackChiSampler.from_database(param_resolver)
    fixed_sampler = FixedAAChiSampler()
    samplers = (dun_sampler, fixed_sampler)

    for rt in rts.residue_types:
        annotate_restype(rt, samplers, default_database.chemical)
        assert hasattr(rt, "rotamer_kintree")

        assert type(rt.rotamer_kintree.id) == numpy.ndarray
        assert type(rt.rotamer_kintree.doftype) == numpy.ndarray
        assert type(rt.rotamer_kintree.parent) == numpy.ndarray
        assert type(rt.rotamer_kintree.frame_x) == numpy.ndarray
        assert type(rt.rotamer_kintree.frame_y) == numpy.ndarray
        assert type(rt.rotamer_kintree.frame_z) == numpy.ndarray

        assert rt.rotamer_kintree.id.shape == (rt.n_atoms,)
        assert rt.rotamer_kintree.doftype.shape == (rt.n_atoms,)
        assert rt.rotamer_kintree.parent.shape == (rt.n_atoms,)
        assert rt.rotamer_kintree.frame_x.shape == (rt.n_atoms,)
        assert rt.rotamer_kintree.frame_y.shape == (rt.n_atoms,)
        assert rt.rotamer_kintree.frame_z.shape == (rt.n_atoms,)


def test_build_rotamers_smoke(ubq_res, default_database, torch_device):
    # torch_device = torch.device("cpu")

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

    p1 = Pose.from_residues_one_chain(ubq_res[:3], torch_device)
    p2 = Pose.from_residues_one_chain(ubq_res[:2], torch_device)
    poses = Poses.from_poses([p1, p2], torch_device)
    palette = PackerPalette(rts)
    task = PackerTask(poses, palette)
    leu_set = set(["LEU"])
    for one_pose_rlts in task.rlts:
        for rlt in one_pose_rlts:
            rlt.restrict_absent_name3s(leu_set)

    param_resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    dun_sampler = DunbrackChiSampler.from_database(param_resolver)
    fixed_sampler = FixedAAChiSampler()
    task.add_chi_sampler(dun_sampler)
    task.add_chi_sampler(fixed_sampler)

    new_coords = build_rotamers(poses, task, default_database.chemical)
    # print(new_coords[:50].cpu().numpy())

    # for writing coordinates into a pdb
    # print("new coords")
    # print(new_coords.shape)
    # rot = 4
    # for i in range(0, new_coords.shape[1]):
    #     print(
    #         "%7.3f %7.3f %7.3f"
    #         % (new_coords[rot, i, 0], new_coords[rot, i, 1], new_coords[rot, i, 2])
    #     )


def test_construct_scans_for_rotamers(default_database, torch_device):
    # torch_device = torch.device("cpu")

    rts = ResidueTypeSet.from_database(default_database.chemical)
    leu_rt_list = [rts.restype_map["LEU"][0]]
    pbt = PackedBlockTypes.from_restype_list(leu_rt_list, device=torch_device)

    param_resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    dun_sampler = DunbrackChiSampler.from_database(param_resolver)
    fixed_sampler = FixedAAChiSampler()
    samplers = (dun_sampler, fixed_sampler)

    annotate_restype(leu_rt_list[0], samplers, default_database.chemical)
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
    kt_nodes = pbt.rotamer_kintree.nodes[0]
    kt_scans = pbt.rotamer_kintree.scans[0]
    kt_gens = pbt.rotamer_kintree.gens[0]
    nodes_gold = numpy.concatenate(
        [
            kt_nodes[0 : kt_gens[1, 0]],
            kt_nodes[0:1],
            kt_nodes[1 : kt_gens[1, 0]] + n_atoms,
            kt_nodes[0:1],
            kt_nodes[1 : kt_gens[1, 0]] + 2 * n_atoms,
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


def test_construct_scans_for_rotamers2(default_database, torch_device):
    # torch_device = torch.device("cpu")

    rts = ResidueTypeSet.from_database(default_database.chemical)
    leu_met_rt_list = [rts.restype_map["LEU"][0]] + [rts.restype_map["MET"][0]]
    pbt = PackedBlockTypes.from_restype_list(leu_met_rt_list, device=torch_device)

    param_resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    dun_sampler = DunbrackChiSampler.from_database(param_resolver)
    fixed_sampler = FixedAAChiSampler()
    samplers = (dun_sampler, fixed_sampler)

    annotate_restype(leu_met_rt_list[0], samplers, default_database.chemical)
    annotate_restype(leu_met_rt_list[1], samplers, default_database.chemical)
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
    kt_nodes = pbt.rotamer_kintree.nodes
    kt_scans = pbt.rotamer_kintree.scans
    kt_gens = pbt.rotamer_kintree.gens
    leu = 0
    met = 1
    nodes_gold = numpy.concatenate(
        [
            kt_nodes[leu, 0 : kt_gens[leu, 1, 0]],
            kt_nodes[met, 0:1],
            kt_nodes[met, 1 : kt_gens[met, 1, 0]] + leu_n_atoms,
            kt_nodes[met, 0:1],
            kt_nodes[met, 1 : kt_gens[met, 1, 0]] + leu_n_atoms + met_n_atoms,
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


def test_inv_kin_rotamers(default_database, ubq_res, torch_device):
    # steps:
    # - annotate residue types and pbt with kintrees + mainchain fingerprints
    # - construct unified kintree for measuring internal coordinates out of
    #   the existing residues
    # - reorder the coordinates of the input structure(s)
    # - invoke inverse_kin
    # - create tensor with ideal dofs for rotamers
    # - copy dofs from existing residues for the mainchain fingerprint atoms
    # - assign chi1 dofs to the appropriate atoms
    # - invoke forward_only_kin_op
    # - reindex coordinates

    # torch_device = torch.device("cpu")
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

    # construct an integer tensor with a value concatenated at position 0
    def it(val, arr):
        return torch.tensor(
            numpy.concatenate((numpy.array([val]), arr)),
            dtype=torch.int32,
            device=torch_device,
        )

    met_kt_id = it(-1, met_rt.rotamer_kintree.id)
    met_kt_doftype = it(0, met_rt.rotamer_kintree.doftype)
    met_kt_parent = it(0, met_rt.rotamer_kintree.parent + 1)
    met_kt_frame_x = it(0, met_rt.rotamer_kintree.frame_x + 1)
    met_kt_frame_y = it(0, met_rt.rotamer_kintree.frame_y + 1)
    met_kt_frame_z = it(0, met_rt.rotamer_kintree.frame_z + 1)

    from tmol.kinematics.compiled.compiled import inverse_kin

    coords = torch.cat(
        (
            torch.zeros((1, 3), dtype=torch.float32, device=torch_device),
            p.coords[0][met_kt_id[1:].to(torch.int64)],
        )
    )

    dofs_orig = inverse_kin(
        coords,
        met_kt_parent,
        met_kt_frame_x,
        met_kt_frame_y,
        met_kt_frame_z,
        met_kt_doftype,
    )

    dofs_new = torch.cat(
        (
            torch.zeros((1, 9), dtype=torch.float32, device=torch_device),
            torch.tensor(
                leu_rt.rotamer_kintree.dofs_ideal,
                dtype=torch.float32,
                device=torch_device,
            ),
        )
    )

    dun_sampler_ind = pbt.mc_fingerprints.sampler_mapping[dun_sampler.sampler_name()]
    met_max_fp = pbt.mc_fingerprints.max_fingerprint[1]
    for i in range(pbt.mc_fingerprints.atom_mapping.shape[3]):
        leu_at_i = pbt.mc_fingerprints.atom_mapping[dun_sampler_ind, met_max_fp, 0, i]
        met_at_i = pbt.mc_fingerprints.atom_mapping[dun_sampler_ind, met_max_fp, 0, i]
        if leu_at_i >= 0 and met_at_i >= 0:
            leu_ktat_i = leu_rt.rotamer_kintree.kintree_idx[leu_at_i]
            met_ktat_i = met_rt.rotamer_kintree.kintree_idx[met_at_i]
            dofs_new[leu_ktat_i + 1, :] = dofs_orig[met_ktat_i + 1, :]

    # print("dofs_new")
    # print(dofs_new[:, :4])

    dofs_new[
        leu_rt.rotamer_kintree.kintree_idx[leu_rt.atom_to_idx["CB"]] + 1, 3
    ] = numpy.radians(180)
    dofs_new[
        leu_rt.rotamer_kintree.kintree_idx[leu_rt.atom_to_idx["CG"]] + 1, 3
    ] = numpy.radians(-60)

    # forward folding; let's build leu on the met's coords
    def _p(t):
        return torch.nn.Parameter(t, requires_grad=False)

    def _t(t):
        return torch.tensor(t, dtype=torch.int32, device=torch_device)

    leu_kintree = _p(
        torch.stack(
            [
                it(-1, leu_rt.rotamer_kintree.id),
                it(0, leu_rt.rotamer_kintree.doftype),
                it(0, leu_rt.rotamer_kintree.parent + 1),
                it(0, leu_rt.rotamer_kintree.frame_x + 1),
                it(0, leu_rt.rotamer_kintree.frame_y + 1),
                it(0, leu_rt.rotamer_kintree.frame_z + 1),
            ],
            dim=1,
        ).to(torch_device)
    )

    new_coords = torch.ops.tmol.forward_only_kin_op(
        dofs_new,
        _p(_t(leu_rt.rotamer_kintree.nodes)),
        _p(_t(leu_rt.rotamer_kintree.scans)),
        _p(torch.tensor(leu_rt.rotamer_kintree.gens, dtype=torch.int32)),  ## CPU!
        leu_kintree,
    )
    assert new_coords.shape == (leu_rt.n_atoms + 1, 3)

    reordered_coords = torch.zeros(
        (leu_rt.n_atoms, 3), dtype=torch.float32, device=torch_device
    )
    reordered_coords[leu_rt.rotamer_kintree.id] = new_coords[1:]

    # for writing coordinates into a pdb
    # print("new coords")
    # for i in range(0, reordered_coords.shape[0]):
    #    print("%6.3f %7.3f %7.3f" % (reordered_coords[i,0], reordered_coords[i,1], reordered_coords[i,2]))

    # make sure that the coordinates of the mainchain atoms that should
    # have been "copied" from the original position are in essentially the same
    # position
    for at in ("N", "H", "CA", "HA", "C", "O"):
        at_met = met_rt.atom_to_idx[at]
        at_leu = leu_rt.atom_to_idx[at]
        assert torch.norm(p.coords[0, at_met, :] - reordered_coords[at_leu, :]) < 1e-5


def test_construct_kintree_for_rotamers(default_database, ubq_res, torch_device):
    # torch_device = torch.device("cpu")
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

    kt1 = construct_kintree_for_rotamers(
        pbt,
        numpy.zeros(1, dtype=numpy.int32),
        leu_rt.n_atoms,
        torch.full((1,), leu_rt.n_atoms, dtype=torch.int32, device=torch_device),
        numpy.ones((1,), dtype=numpy.int32),
        torch_device,
    )

    def cat(val, arr):
        return numpy.concatenate((numpy.array([val], dtype=numpy.int32), arr))

    gold_leu_kintree1_id = cat(-1, leu_rt.rotamer_kintree.id + 1)
    gold_leu_kintree1_doftype = cat(0, leu_rt.rotamer_kintree.doftype)
    gold_leu_kintree1_parent = cat(0, leu_rt.rotamer_kintree.parent + 1)
    gold_leu_kintree1_frame_x = cat(0, leu_rt.rotamer_kintree.frame_x + 1)
    gold_leu_kintree1_frame_y = cat(0, leu_rt.rotamer_kintree.frame_y + 1)
    gold_leu_kintree1_frame_z = cat(0, leu_rt.rotamer_kintree.frame_z + 1)

    numpy.testing.assert_equal(gold_leu_kintree1_id, kt1.id.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kintree1_doftype, kt1.doftype.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kintree1_parent, kt1.parent.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kintree1_frame_x, kt1.frame_x.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kintree1_frame_y, kt1.frame_y.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kintree1_frame_z, kt1.frame_z.cpu().numpy())

    kt2 = construct_kintree_for_rotamers(
        pbt,
        numpy.zeros(2, dtype=numpy.int32),
        2 * leu_rt.n_atoms,
        torch.full((2,), leu_rt.n_atoms, dtype=torch.int32),
        numpy.arange(2, dtype=numpy.int32) * pbt.max_n_atoms,
        torch_device,
    )

    def cat2(val, arr1, arr2):
        return numpy.concatenate((numpy.array([val], dtype=numpy.int32), arr1, arr2))

    gold_leu_kintree2_id = cat2(
        -1, leu_rt.rotamer_kintree.id, leu_rt.rotamer_kintree.id + pbt.max_n_atoms
    )
    gold_leu_kintree2_doftype = cat2(
        0, leu_rt.rotamer_kintree.doftype, leu_rt.rotamer_kintree.doftype
    )
    gold_leu_kintree2_parent = cat2(
        0,
        leu_rt.rotamer_kintree.parent + 1,
        leu_rt.rotamer_kintree.parent + 1 + leu_rt.n_atoms,
    )
    # fix the jump-to-root for the 1st atom in rotamer 2
    gold_leu_kintree2_parent[1 + leu_rt.n_atoms] = 0
    gold_leu_kintree2_frame_x = cat2(
        0,
        leu_rt.rotamer_kintree.frame_x + 1,
        leu_rt.rotamer_kintree.frame_x + 1 + leu_rt.n_atoms,
    )
    gold_leu_kintree2_frame_y = cat2(
        0,
        leu_rt.rotamer_kintree.frame_y + 1,
        leu_rt.rotamer_kintree.frame_y + 1 + leu_rt.n_atoms,
    )
    gold_leu_kintree2_frame_z = cat2(
        0,
        leu_rt.rotamer_kintree.frame_z + 1,
        leu_rt.rotamer_kintree.frame_z + 1 + leu_rt.n_atoms,
    )

    numpy.testing.assert_equal(gold_leu_kintree2_id, kt2.id.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kintree2_doftype, kt2.doftype.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kintree2_parent, kt2.parent.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kintree2_frame_x, kt2.frame_x.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kintree2_frame_y, kt2.frame_y.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kintree2_frame_z, kt2.frame_z.cpu().numpy())


def test_construct_kintree_for_rotamers2(default_database, ubq_res, torch_device):
    # torch_device = torch.device("cpu")
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

    kt1 = construct_kintree_for_rotamers(
        pbt,
        numpy.zeros(1, dtype=numpy.int32),
        leu_rt.n_atoms,
        torch.full((1,), leu_rt.n_atoms, dtype=torch.int32),
        numpy.zeros(1, dtype=numpy.int32),
        torch_device,
    )

    def cat(val, arr):
        return numpy.concatenate((numpy.array([val], dtype=numpy.int32), arr))

    gold_leu_kintree1_id = cat(-1, leu_rt.rotamer_kintree.id)
    gold_leu_kintree1_doftype = cat(0, leu_rt.rotamer_kintree.doftype)
    gold_leu_kintree1_parent = cat(0, leu_rt.rotamer_kintree.parent + 1)
    gold_leu_kintree1_frame_x = cat(0, leu_rt.rotamer_kintree.frame_x + 1)
    gold_leu_kintree1_frame_y = cat(0, leu_rt.rotamer_kintree.frame_y + 1)
    gold_leu_kintree1_frame_z = cat(0, leu_rt.rotamer_kintree.frame_z + 1)

    numpy.testing.assert_equal(gold_leu_kintree1_id, kt1.id.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kintree1_doftype, kt1.doftype.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kintree1_parent, kt1.parent.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kintree1_frame_x, kt1.frame_x.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kintree1_frame_y, kt1.frame_y.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kintree1_frame_z, kt1.frame_z.cpu().numpy())

    kt2 = construct_kintree_for_rotamers(
        pbt,
        numpy.zeros(2, dtype=numpy.int32),
        2 * leu_rt.n_atoms,
        torch.full((2,), leu_rt.n_atoms, dtype=torch.int32),
        numpy.arange(2, dtype=numpy.int32) * pbt.max_n_atoms,
        torch_device,
    )

    def cat2(val, arr1, arr2):
        return numpy.concatenate((numpy.array([val], dtype=numpy.int32), arr1, arr2))

    gold_leu_kintree2_id = cat2(
        -1, leu_rt.rotamer_kintree.id, leu_rt.rotamer_kintree.id + pbt.max_n_atoms
    )
    gold_leu_kintree2_doftype = cat2(
        0, leu_rt.rotamer_kintree.doftype, leu_rt.rotamer_kintree.doftype
    )
    gold_leu_kintree2_parent = cat2(
        0,
        leu_rt.rotamer_kintree.parent + 1,
        leu_rt.rotamer_kintree.parent + 1 + leu_rt.n_atoms,
    )
    # fix the jump-to-root for the 1st atom in rotamer 2
    gold_leu_kintree2_parent[1 + leu_rt.n_atoms] = 0
    gold_leu_kintree2_frame_x = cat2(
        0,
        leu_rt.rotamer_kintree.frame_x + 1,
        leu_rt.rotamer_kintree.frame_x + 1 + leu_rt.n_atoms,
    )
    gold_leu_kintree2_frame_y = cat2(
        0,
        leu_rt.rotamer_kintree.frame_y + 1,
        leu_rt.rotamer_kintree.frame_y + 1 + leu_rt.n_atoms,
    )
    gold_leu_kintree2_frame_z = cat2(
        0,
        leu_rt.rotamer_kintree.frame_z + 1,
        leu_rt.rotamer_kintree.frame_z + 1 + leu_rt.n_atoms,
    )

    numpy.testing.assert_equal(gold_leu_kintree2_id, kt2.id.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kintree2_doftype, kt2.doftype.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kintree2_parent, kt2.parent.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kintree2_frame_x, kt2.frame_x.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kintree2_frame_y, kt2.frame_y.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kintree2_frame_z, kt2.frame_z.cpu().numpy())


def test_measure_original_dofs(ubq_res, default_database, torch_device):
    # torch_device = torch.device("cpu")
    chem_db = default_database.chemical

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

    p1 = Pose.from_residues_one_chain(ubq_res[:2], torch_device)
    poses = Poses.from_poses([p1], torch_device)
    palette = PackerPalette(rts)
    task = PackerTask(poses, palette)
    task.restrict_to_repacking()

    param_resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    dun_sampler = DunbrackChiSampler.from_database(param_resolver)
    fixed_sampler = FixedAAChiSampler()
    task.add_chi_sampler(dun_sampler)
    task.add_chi_sampler(fixed_sampler)
    samplers = (dun_sampler, fixed_sampler)

    pbt = poses.packed_block_types
    for rt in pbt.active_block_types:
        annotate_restype(rt, samplers, chem_db)
    annotate_packed_block_types(pbt)

    block_inds = poses.block_inds.view(-1)
    real_block_inds = block_inds != -1
    nz_real_block_inds = torch.nonzero(real_block_inds).flatten()
    real_block_inds_numpy = nz_real_block_inds.cpu().numpy().astype(numpy.int32)
    block_inds = block_inds[block_inds != -1]
    res_n_atoms = pbt.n_atoms[block_inds.to(torch.int64)]
    n_total_atoms = torch.sum(res_n_atoms).item()

    kintree = construct_kintree_for_rotamers(
        pbt,
        block_inds.cpu().numpy(),
        n_total_atoms,
        res_n_atoms,
        nz_real_block_inds.cpu().numpy().astype(numpy.int32) * pbt.max_n_atoms,
        torch_device,
    )

    dofs = measure_dofs_from_orig_coords(poses.coords.view(-1), kintree)

    # let's refold and make sure the coordinates are the same?
    # forward folding; let's build leu on the met's coords
    def _p(t):
        return torch.nn.Parameter(t, requires_grad=False)

    kintree_stacked = _p(
        torch.stack(
            [
                kintree.id,
                kintree.doftype,
                kintree.parent,
                kintree.frame_x,
                kintree.frame_y,
                kintree.frame_z,
            ],
            dim=1,
        ).to(torch_device)
    )
    n_atoms_offset_for_rot = (
        exclusive_cumsum1d(res_n_atoms).cpu().numpy().astype(numpy.int64)
    )
    nodes, scans, gens = construct_scans_for_rotamers(
        pbt, real_block_inds_numpy, res_n_atoms, n_atoms_offset_for_rot
    )

    new_kin_coords = torch.ops.tmol.forward_only_kin_op(
        dofs,
        _p(torch.tensor(nodes, dtype=torch.int32, device=torch_device)),
        _p(torch.tensor(scans, dtype=torch.int32, device=torch_device)),
        _p(torch.tensor(gens, dtype=torch.int32, device=torch.device("cpu"))),
        kintree_stacked,
    )

    new_coords = torch.zeros_like(poses.coords).view(-1, 3)
    new_coords[kintree.id.to(torch.int64)] = new_kin_coords

    # print(new_kin_coords)

    # for writing coordinates into a pdb
    # print("new coords")
    # for i in range(0, new_coords.shape[0]):
    #     print(
    #         "%7.3f %7.3f %7.3f" % (new_coords[i, 0], new_coords[i, 1], new_coords[i, 2])
    #     )


def test_measure_original_dofs2(ubq_res, default_database, torch_device):
    # torch_device = torch.device("cpu")
    chem_db = default_database.chemical

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
    dun_sampler = DunbrackChiSampler.from_database(param_resolver)
    fixed_sampler = FixedAAChiSampler()
    task.add_chi_sampler(dun_sampler)
    task.add_chi_sampler(fixed_sampler)
    samplers = (dun_sampler, fixed_sampler)

    pbt = poses.packed_block_types
    for rt in pbt.active_block_types:
        annotate_restype(rt, samplers, chem_db)
    annotate_packed_block_types(pbt)

    block_inds = poses.block_inds.view(-1)
    real_block_inds = block_inds != -1
    nz_real_block_inds = torch.nonzero(real_block_inds).flatten()
    block_inds = block_inds[block_inds != -1]
    res_n_atoms = pbt.n_atoms[block_inds.to(torch.int64)]
    n_total_atoms = torch.sum(res_n_atoms).item()

    kintree = construct_kintree_for_rotamers(
        pbt,
        block_inds.cpu().numpy(),
        n_total_atoms,
        res_n_atoms,
        nz_real_block_inds.cpu().numpy().astype(numpy.int32) * pbt.max_n_atoms,
        torch_device,
    )

    dofs = measure_dofs_from_orig_coords(poses.coords.view(-1), kintree)
    # print("dofs")
    # print(dofs[:, :4])

    # let's refold and make sure the coordinates are the same?
    # forward folding; let's build leu on the met's coords
    def _p(t):
        return torch.nn.Parameter(t, requires_grad=False)

    kintree_stacked = _p(
        torch.stack(
            [
                kintree.id,
                kintree.doftype,
                kintree.parent,
                kintree.frame_x,
                kintree.frame_y,
                kintree.frame_z,
            ],
            dim=1,
        ).to(torch_device)
    )
    n_atoms_offset_for_rot = (
        exclusive_cumsum1d(res_n_atoms).cpu().numpy().astype(numpy.int64)
    )

    nodes, scans, gens = construct_scans_for_rotamers(
        pbt, block_inds.cpu().numpy(), res_n_atoms, n_atoms_offset_for_rot
    )

    new_kin_coords = torch.ops.tmol.forward_only_kin_op(
        dofs,
        _p(torch.tensor(nodes, dtype=torch.int32, device=torch_device)),
        _p(torch.tensor(scans, dtype=torch.int32, device=torch_device)),
        _p(torch.tensor(gens, dtype=torch.int32, device=torch.device("cpu"))),
        kintree_stacked,
    )

    new_coords = torch.zeros_like(poses.coords).view(-1, 3)
    new_coords[kintree.id.to(torch.int64)] = new_kin_coords
    new_coords = new_coords.view(poses.coords.shape)

    for i in range(poses.coords.shape[0]):
        for j in range(poses.coords.shape[1]):
            if poses.block_inds[i, j] == -1:
                continue
            j_n_atoms = poses.residues[i][j].residue_type.n_atoms
            numpy.testing.assert_almost_equal(
                poses.coords[i, j, :j_n_atoms].cpu().numpy(),
                new_coords[i, j, :j_n_atoms].cpu().numpy(),
                decimal=5,
            )

    # print(new_kin_coords)

    # for writing coordinates into a pdb
    # print("new coords")
    # for i in range(0, new_coords.shape[0]):
    #     print(
    #         "%7.3f %7.3f %7.3f" % (new_coords[i, 0], new_coords[i, 1], new_coords[i, 2])
    #     )


def test_create_dof_inds_to_copy_from_orig_to_rotamers(
    ubq_res, default_database, torch_device
):
    # torch_device = torch.device("cpu")

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

    p1 = Pose.from_residues_one_chain(ubq_res[:2], torch_device)
    p2 = Pose.from_residues_one_chain(ubq_res[:3], torch_device)
    poses = Poses.from_poses([p1, p2], torch_device)
    palette = PackerPalette(rts)
    task = PackerTask(poses, palette)
    leu_set = set(["LEU"])
    for one_pose_rlts in task.rlts:
        for rlt in one_pose_rlts:
            rlt.restrict_absent_name3s(leu_set)

    param_resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    dun_sampler = DunbrackChiSampler.from_database(param_resolver)
    fixed_sampler = FixedAAChiSampler()
    task.add_chi_sampler(dun_sampler)
    task.add_chi_sampler(fixed_sampler)

    poses, samplers = rebuild_poses_if_necessary(poses, task)
    pbt = poses.packed_block_types
    assert pbt.active_block_types[0].name == "LEU"
    leu_rt = pbt.active_block_types[0]
    annotate_everything(default_database.chemical, samplers, pbt)

    rt_for_rot = torch.tensor(
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=torch.int64, device=torch_device
    )
    block_ind_for_rot = torch.tensor(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int64, device=torch_device
    )

    # [pbt.mc_fingerprints.sampler_mapping[dun_sampler.sampler_name()] ] * 10
    sampler_for_rotamer = torch.zeros(10, dtype=torch.int64, device=torch_device)

    n_dof_atoms_offset_for_rot = (
        torch.arange(10, dtype=torch.int32, device=torch_device) * 19
    )

    dst, srt = create_dof_inds_to_copy_from_orig_to_rotamers(
        poses,
        task,
        samplers,
        rt_for_rot,
        block_ind_for_rot,
        sampler_for_rotamer,
        n_dof_atoms_offset_for_rot,
    )

    # OK, what do we expect?
    # dst should be the atoms for N, H, CA, HA, C, O
    # in kintree order with an offset of 19 for each
    # successive batch.

    fingerprint_atoms = "N", "H", "CA", "HA", "C", "O"

    def fp_kto(rt):
        return [
            rt.rotamer_kintree.kintree_idx[rt.atom_to_idx[at_name]]
            for at_name in fingerprint_atoms
        ]

    dst_gold_template = numpy.array(fp_kto(leu_rt), dtype=numpy.int64)
    dst_gold = numpy.arange(10).repeat(6) * 19 + numpy.tile(dst_gold_template, 10) + 1

    numpy.testing.assert_equal(dst_gold, dst.cpu().numpy())

    src_fpats_kto = numpy.array(
        fp_kto(pbt.active_block_types[poses.block_inds[0, 0]])
        + fp_kto(pbt.active_block_types[poses.block_inds[0, 1]])
        + fp_kto(pbt.active_block_types[poses.block_inds[1, 0]])
        + fp_kto(pbt.active_block_types[poses.block_inds[1, 1]])
        + fp_kto(pbt.active_block_types[poses.block_inds[1, 2]]),
        dtype=numpy.int64,
    )

    def n_ats(i1, i2):
        return pbt.n_atoms[poses.block_inds[i1, i2]]

    src_dof_offsets = numpy.cumsum(
        [0, n_ats(0, 0), n_ats(0, 1), n_ats(1, 0), n_ats(1, 1)]
    ).repeat(6)

    src_gold = src_fpats_kto + src_dof_offsets + 1

    numpy.testing.assert_equal(src_gold, src_gold)


def test_create_dof_inds_to_copy_from_orig_to_rotamers2(
    ubq_res, default_database, torch_device
):
    # torch_device = torch.device("cpu")

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

    p = Pose.from_residues_one_chain(ubq_res[:5], torch_device)
    poses = Poses.from_poses([p] * 3, torch_device)
    palette = PackerPalette(rts)
    task = PackerTask(poses, palette)
    task.restrict_to_repacking()

    param_resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    dun_sampler = DunbrackChiSampler.from_database(param_resolver)
    fixed_sampler = FixedAAChiSampler()
    task.add_chi_sampler(dun_sampler)
    task.add_chi_sampler(fixed_sampler)

    poses, samplers = rebuild_poses_if_necessary(poses, task)
    pbt = poses.packed_block_types
    annotate_everything(default_database.chemical, samplers, pbt)

    rt_for_rot = torch.div(torch.arange(30, dtype=torch.int64, device=torch_device), 2)

    block_ind_for_rot = torch.remainder(
        torch.div(torch.arange(30, dtype=torch.int64, device=torch_device), 2), 5
    )

    sampler_for_rotamer = torch.zeros(30, dtype=torch.int64, device=torch_device)

    n_dof_atoms_offset_for_rot = exclusive_cumsum1d(pbt.n_atoms[block_ind_for_rot])

    dst, src = create_dof_inds_to_copy_from_orig_to_rotamers(
        poses,
        task,
        samplers,
        rt_for_rot,
        block_ind_for_rot,
        sampler_for_rotamer,
        n_dof_atoms_offset_for_rot,
    )

    src = src.cpu().numpy()
    dst = dst.cpu().numpy()

    n_src = src.shape[0]
    n_dst = dst.shape[0]
    assert n_src == n_dst

    n_dofs_per_pose = n_src // 3
    assert 3 * n_dofs_per_pose == n_src

    n_ats_per_pose = torch.sum(pbt.n_atoms[poses.block_inds[0].to(torch.int64)]).item()
    n_rot_ats_per_pose = torch.sum(pbt.n_atoms[block_ind_for_rot[:10]]).item()

    src0 = src[:n_dofs_per_pose]
    src1 = src[n_dofs_per_pose : 2 * n_dofs_per_pose]
    src2 = src[2 * n_dofs_per_pose : 3 * n_dofs_per_pose]

    dst0 = dst[:n_dofs_per_pose]
    dst1 = dst[n_dofs_per_pose : 2 * n_dofs_per_pose]
    dst2 = dst[2 * n_dofs_per_pose : 3 * n_dofs_per_pose]

    numpy.testing.assert_equal(src0 - 1, src1 - 1 - n_ats_per_pose)
    numpy.testing.assert_equal(src0 - 1, src2 - 1 - 2 * n_ats_per_pose)

    numpy.testing.assert_equal(dst0 - 1, dst1 - 1 - n_rot_ats_per_pose)
    numpy.testing.assert_equal(dst0 - 1, dst2 - 1 - 2 * n_rot_ats_per_pose)


def test_build_lots_of_rotamers(ubq_res, default_database, torch_device):
    # torch_device = torch.device("cpu")

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

    # subset_len = 10
    # print("resnames")
    # print([res.residue_type.name for res in ubq_res[:subset_len]])

    n_poses = 10
    p = Pose.from_residues_one_chain(ubq_res, torch_device)
    poses = Poses.from_poses([p] * n_poses, torch_device)
    palette = PackerPalette(rts)
    task = PackerTask(poses, palette)
    # leu_set = set(["LEU"])
    # for one_pose_rlts in task.rlts:
    #     for rlt in one_pose_rlts:
    #         rlt.restrict_absent_name3s(leu_set)
    task.restrict_to_repacking()

    param_resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    dun_sampler = DunbrackChiSampler.from_database(param_resolver)
    fixed_sampler = FixedAAChiSampler()
    task.add_chi_sampler(dun_sampler)
    task.add_chi_sampler(fixed_sampler)

    rt_for_rot, block_ind_for_rot, new_coords = build_rotamers(
        poses, task, default_database.chemical
    )

    if True:
        # if torch_device == torch.device("cpu"):
        # for some reason, the number of rotamers built on the GPU is not consistent
        # between poses, and it's not clear to me why they would not be.

        n_rots = new_coords.shape[0]
        # print("n_rots", n_rots)

        # all the rotamers should be the same on all n_poses copies of ubq
        n_rots_per_pose = n_rots // n_poses + 1
        # assert n_rots_per_pose * n_poses == n_rots

        new_coords = new_coords.cpu().numpy()

        # print(new_coords[:50].cpu().numpy())

        # for writing coordinates into a pdb
        # print("new coords")
        # print(new_coords.shape)
        # rot = new_coords.shape[0] - 3  # arg on 74 of last pose
        for i in range(1, n_poses):
            i_offset = i * n_rots_per_pose
            all_good = True
            for j in range(0, n_rots_per_pose):
                for k in range(0, new_coords.shape[1]):

                    dist = numpy.linalg.norm(
                        new_coords[j, k, :] - new_coords[i_offset + j, k, :]
                    )
                    if dist < 1e-5:
                        continue

                    all_good = False
                    # print("rot discrepancy")
                    # print("rt:", rt_for_rot[j], rt_for_rot[i_offset + j])
                    # print(
                    #     "block_ind:", block_ind_for_rot[j], block_ind_for_rot[i_offset + j]
                    # )
                    print(
                        "%4d %7d %3d %7.3f -- %7.3f %7.3f %7.3f vs %7.3f %7.3f %7.3f"
                        % (
                            i,
                            j,
                            k,
                            numpy.linalg.norm(
                                new_coords[j, k, :] - new_coords[i_offset + j, k, :]
                            ),
                            new_coords[j, k, 0],
                            new_coords[j, k, 1],
                            new_coords[j, k, 2],
                            new_coords[i_offset + j, k, 0],
                            new_coords[i_offset + j, k, 1],
                            new_coords[i_offset + j, k, 2],
                        )
                    )
                    # numpy.testing.assert_almost_equal(
                    #     new_coords[j, k, :], new_coords[i_offset + j, k, :]
                    # )
            assert all_good

        for i in range(1, n_poses):
            numpy.testing.assert_almost_equal(
                new_coords[:n_rots_per_pose],
                new_coords[(n_rots_per_pose * i) : (n_rots_per_pose * (i + 1))],
                decimal=5,
            )
