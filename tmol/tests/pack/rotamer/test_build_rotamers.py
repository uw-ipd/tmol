import numpy
import torch

from tmol.pack.rotamer.build_rotamers import (
    RotamerSet,
    annotate_restype,
    annotate_packed_block_types,
    build_rotamers,
    construct_kinforest_for_conformers,
    construct_scans_for_conformers,
    exc_cumsum_from_inc_cumsum,
    measure_pose_dofs,
    measure_dofs_from_orig_coords,
    rebuild_poses_if_necessary,
    annotate_everything,
    merge_conformer_samples,
    # copy_dofs_from_orig_to_rotamers,
    # assign_dofs_from_samples,
    # create_dof_inds_to_copy_from_orig_to_rotamers,
    calculate_rotamer_coords,
    get_rotamer_origin_data,
)
from tmol.pack.rotamer.chi_sampler import (
    # copy_dofs_from_orig_to_rotamers_for_sampler
    create_dof_inds_to_copy_from_orig_to_rotamers_for_sampler,
    # assign_chi_dofs_from_samples
)


from tmol.io import pose_stack_from_pdb

from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.pack.packer_task import PackerTask, PackerPalette
from tmol.pack.rotamer.fixed_aa_chi_sampler import FixedAAChiSampler

from tmol.kinematics.compiled.compiled_ops import forward_only_op

from tmol.utility.tensor.common_operations import exclusive_cumsum1d, stretch

from tmol.tests.data import no_termini_pose_stack_from_pdb


def test_annotate_restypes(
    default_database, fresh_default_restype_set, torch_device, dun_sampler
):
    fixed_sampler = FixedAAChiSampler()
    samplers = (dun_sampler, fixed_sampler)

    for rt in fresh_default_restype_set.residue_types:
        annotate_restype(rt, samplers, default_database.chemical)
        assert hasattr(rt, "rotamer_kinforest")

        assert isinstance(rt.rotamer_kinforest.id, numpy.ndarray)
        assert isinstance(rt.rotamer_kinforest.doftype, numpy.ndarray)
        assert isinstance(rt.rotamer_kinforest.parent, numpy.ndarray)
        assert isinstance(rt.rotamer_kinforest.frame_x, numpy.ndarray)
        assert isinstance(rt.rotamer_kinforest.frame_y, numpy.ndarray)
        assert isinstance(rt.rotamer_kinforest.frame_z, numpy.ndarray)

        assert rt.rotamer_kinforest.id.shape == (rt.n_atoms,)
        assert rt.rotamer_kinforest.doftype.shape == (rt.n_atoms,)
        assert rt.rotamer_kinforest.parent.shape == (rt.n_atoms,)
        assert rt.rotamer_kinforest.frame_x.shape == (rt.n_atoms,)
        assert rt.rotamer_kinforest.frame_y.shape == (rt.n_atoms,)
        assert rt.rotamer_kinforest.frame_z.shape == (rt.n_atoms,)


def test_build_rotamers_smoke(default_database, ubq_pdb, torch_device, dun_sampler):
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=11)
    p2 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=10)

    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)
    restype_set = poses.packed_block_types.restype_set
    palette = PackerPalette(restype_set)
    task = PackerTask(poses, palette)

    fixed_sampler = FixedAAChiSampler()
    task.add_conformer_sampler(dun_sampler)
    task.add_conformer_sampler(fixed_sampler)

    poses, rotamer_set = build_rotamers(poses, task, default_database.chemical)
    assert rotamer_set is not None


def test_construct_scans_for_rotamers(
    default_database, fresh_default_restype_set, torch_device, dun_sampler
):

    leu_rt_list = [fresh_default_restype_set.restype_map["LEU"][0]]
    pbt = PackedBlockTypes.from_restype_list(
        default_database.chemical,
        fresh_default_restype_set,
        leu_rt_list,
        device=torch_device,
    )

    fixed_sampler = FixedAAChiSampler()
    samplers = (dun_sampler, fixed_sampler)

    annotate_restype(leu_rt_list[0], samplers, default_database.chemical)
    annotate_packed_block_types(pbt)

    rt_block_type_ind = numpy.zeros(3, dtype=numpy.int32)
    rt_for_rot = torch.zeros(3, dtype=torch.int64)

    block_ind_for_rot = rt_block_type_ind[rt_for_rot]
    block_ind_for_rot_torch = torch.tensor(
        block_ind_for_rot, dtype=torch.int64, device=torch_device
    )
    n_atoms_for_rot = pbt.n_atoms[block_ind_for_rot_torch]
    n_atoms_offset_for_rot = torch.cumsum(n_atoms_for_rot, dim=0)
    n_atoms_offset_for_rot = n_atoms_offset_for_rot.cpu().numpy()
    n_atoms_offset_for_rot = exc_cumsum_from_inc_cumsum(n_atoms_offset_for_rot)

    nodes, scans, gens = construct_scans_for_conformers(
        pbt, block_ind_for_rot, n_atoms_for_rot, n_atoms_offset_for_rot
    )

    n_atoms = len(leu_rt_list[0].atoms)
    kt_nodes = pbt.rotamer_kinforest.nodes[0]
    kt_scans = pbt.rotamer_kinforest.scans[0]
    kt_gens = pbt.rotamer_kinforest.gens[0]
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


def test_construct_scans_for_rotamers2(
    default_database, fresh_default_restype_set, torch_device, dun_sampler
):
    leu_met_rt_list = [fresh_default_restype_set.restype_map["LEU"][0]] + [
        fresh_default_restype_set.restype_map["MET"][0]
    ]
    pbt = PackedBlockTypes.from_restype_list(
        default_database.chemical,
        fresh_default_restype_set,
        leu_met_rt_list,
        device=torch_device,
    )

    fixed_sampler = FixedAAChiSampler()
    samplers = (dun_sampler, fixed_sampler)

    annotate_restype(leu_met_rt_list[0], samplers, default_database.chemical)
    annotate_restype(leu_met_rt_list[1], samplers, default_database.chemical)
    annotate_packed_block_types(pbt)

    rt_block_type_ind = numpy.concatenate(
        [numpy.zeros(1, dtype=numpy.int32), numpy.ones(2, dtype=numpy.int32)]
    )
    rt_for_rot = torch.cat(
        [torch.zeros(1, dtype=torch.int64), torch.ones(2, dtype=torch.int64)]
    )

    block_ind_for_rot = rt_block_type_ind[rt_for_rot]
    block_ind_for_rot_torch = torch.tensor(
        block_ind_for_rot, dtype=torch.int64, device=torch_device
    )
    n_atoms_for_rot = pbt.n_atoms[block_ind_for_rot_torch]
    n_atoms_offset_for_rot = torch.cumsum(n_atoms_for_rot, dim=0)
    n_atoms_offset_for_rot = n_atoms_offset_for_rot.cpu().numpy()
    n_atoms_offset_for_rot = exc_cumsum_from_inc_cumsum(n_atoms_offset_for_rot)

    nodes, scans, gens = construct_scans_for_conformers(
        pbt, block_ind_for_rot, n_atoms_for_rot, n_atoms_offset_for_rot
    )

    leu_n_atoms = len(leu_met_rt_list[0].atoms)
    met_n_atoms = len(leu_met_rt_list[1].atoms)
    kt_nodes = pbt.rotamer_kinforest.nodes
    kt_scans = pbt.rotamer_kinforest.scans
    kt_gens = pbt.rotamer_kinforest.gens
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


def test_inv_kin_rotamers(default_database, ubq_pdb, torch_device, dun_sampler):
    # steps:
    # - annotate residue types and pbt with kinforests + mainchain fingerprints
    # - construct unified kinforest for measuring internal coordinates out of
    #   the existing residues
    # - reorder the coordinates of the input structure(s)
    # - invoke inverse_kin
    # - create tensor with ideal dofs for rotamers
    # - copy dofs from existing residues for the mainchain fingerprint atoms
    # - assign chi1 dofs to the appropriate atoms
    # - invoke forward_only_kin_op
    # - reindex coordinates

    chem_db = default_database.chemical

    p = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=4)
    restype_set = p.packed_block_types.restype_set

    fixed_sampler = FixedAAChiSampler()
    samplers = (dun_sampler, fixed_sampler)

    leu_met_rt_list = [restype_set.restype_map["LEU"][0]] + [
        restype_set.restype_map["MET"][0]
    ]
    pbt = PackedBlockTypes.from_restype_list(
        default_database.chemical,
        restype_set,
        leu_met_rt_list,
        device=torch_device,
    )

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

    met_kt_id = it(-1, met_rt.rotamer_kinforest.id)
    met_kt_doftype = it(0, met_rt.rotamer_kinforest.doftype)
    met_kt_parent = it(0, met_rt.rotamer_kinforest.parent + 1)
    met_kt_frame_x = it(0, met_rt.rotamer_kinforest.frame_x + 1)
    met_kt_frame_y = it(0, met_rt.rotamer_kinforest.frame_y + 1)
    met_kt_frame_z = it(0, met_rt.rotamer_kinforest.frame_z + 1)

    from tmol.kinematics.compiled.compiled_inverse_kin import inverse_kin

    coords = torch.cat(
        (
            torch.zeros((1, 3), dtype=torch.float32, device=torch_device),
            p.coords[0, met_kt_id[1:].to(torch.int64)],
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
                leu_rt.rotamer_kinforest.dofs_ideal,
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
            leu_ktat_i = leu_rt.rotamer_kinforest.kinforest_idx[leu_at_i]
            met_ktat_i = met_rt.rotamer_kinforest.kinforest_idx[met_at_i]
            dofs_new[leu_ktat_i + 1, :] = dofs_orig[met_ktat_i + 1, :]

    dofs_new[
        leu_rt.rotamer_kinforest.kinforest_idx[leu_rt.atom_to_idx["CB"]] + 1, 3
    ] = numpy.radians(180)
    dofs_new[
        leu_rt.rotamer_kinforest.kinforest_idx[leu_rt.atom_to_idx["CG"]] + 1, 3
    ] = numpy.radians(-60)

    # forward folding; let's build leu on the met's coords
    def _p(t):
        return torch.nn.Parameter(t, requires_grad=False)

    def _t(t):
        return torch.tensor(t, dtype=torch.int32, device=torch_device)

    leu_kinforest = _p(
        torch.stack(
            [
                it(-1, leu_rt.rotamer_kinforest.id),
                it(0, leu_rt.rotamer_kinforest.doftype),
                it(0, leu_rt.rotamer_kinforest.parent + 1),
                it(0, leu_rt.rotamer_kinforest.frame_x + 1),
                it(0, leu_rt.rotamer_kinforest.frame_y + 1),
                it(0, leu_rt.rotamer_kinforest.frame_z + 1),
            ],
            dim=1,
        ).to(torch_device)
    )

    new_coords = forward_only_op(
        dofs_new,
        _p(_t(leu_rt.rotamer_kinforest.nodes)),
        _p(_t(leu_rt.rotamer_kinforest.scans)),
        _p(torch.tensor(leu_rt.rotamer_kinforest.gens, dtype=torch.int32)),  # CPU!
        leu_kinforest,
    )
    assert new_coords.shape == (leu_rt.n_atoms + 1, 3)

    reordered_coords = torch.zeros(
        (leu_rt.n_atoms, 3), dtype=torch.float32, device=torch_device
    )
    reordered_coords[leu_rt.rotamer_kinforest.id] = new_coords[1:]

    # make sure that the coordinates of the mainchain atoms that should
    # have been "copied" from the original position are in essentially the same
    # position
    for at in ("N", "H", "CA", "HA", "C", "O"):
        at_met = met_rt.atom_to_idx[at]
        at_leu = leu_rt.atom_to_idx[at]
        assert torch.norm(p.coords[0, at_met, :] - reordered_coords[at_leu, :]) < 1e-5


def test_construct_kinforest_for_rotamers(
    default_database, fresh_default_restype_set, torch_device, dun_sampler
):
    # torch_device = torch.device("cpu")
    chem_db = default_database.chemical

    fixed_sampler = FixedAAChiSampler()
    samplers = (dun_sampler, fixed_sampler)

    leu_met_rt_list = [fresh_default_restype_set.restype_map["LEU"][0]] + [
        fresh_default_restype_set.restype_map["MET"][0]
    ]
    pbt = PackedBlockTypes.from_restype_list(
        default_database.chemical,
        fresh_default_restype_set,
        leu_met_rt_list,
        device=torch_device,
    )

    annotate_restype(leu_met_rt_list[0], samplers, chem_db)
    annotate_restype(leu_met_rt_list[1], samplers, chem_db)
    annotate_packed_block_types(pbt)

    leu_rt = leu_met_rt_list[0]

    kt1 = construct_kinforest_for_conformers(
        pbt,
        numpy.zeros(1, dtype=numpy.int32),
        leu_rt.n_atoms,
        torch.full((1,), leu_rt.n_atoms, dtype=torch.int32, device=torch_device),
        numpy.ones((1,), dtype=numpy.int64),
        torch_device,
    )

    def cat(val, arr):
        return numpy.concatenate((numpy.array([val], dtype=numpy.int32), arr))

    gold_leu_kinforest1_id = cat(-1, leu_rt.rotamer_kinforest.id + 1)
    gold_leu_kinforest1_doftype = cat(0, leu_rt.rotamer_kinforest.doftype)
    gold_leu_kinforest1_parent = cat(0, leu_rt.rotamer_kinforest.parent + 1)
    gold_leu_kinforest1_frame_x = cat(0, leu_rt.rotamer_kinforest.frame_x + 1)
    gold_leu_kinforest1_frame_y = cat(0, leu_rt.rotamer_kinforest.frame_y + 1)
    gold_leu_kinforest1_frame_z = cat(0, leu_rt.rotamer_kinforest.frame_z + 1)

    numpy.testing.assert_equal(gold_leu_kinforest1_id, kt1.id.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kinforest1_doftype, kt1.doftype.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kinforest1_parent, kt1.parent.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kinforest1_frame_x, kt1.frame_x.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kinforest1_frame_y, kt1.frame_y.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kinforest1_frame_z, kt1.frame_z.cpu().numpy())

    kt2 = construct_kinforest_for_conformers(
        pbt,
        numpy.zeros(2, dtype=numpy.int32),
        2 * leu_rt.n_atoms,
        torch.full((2,), leu_rt.n_atoms, dtype=torch.int32),
        numpy.arange(2, dtype=numpy.int64)
        * leu_rt.n_atoms,  # rot-coords layout is compact
        torch_device,
    )

    def cat2(val, arr1, arr2):
        return numpy.concatenate((numpy.array([val], dtype=numpy.int32), arr1, arr2))

    gold_leu_kinforest2_id = cat2(
        -1, leu_rt.rotamer_kinforest.id, leu_rt.rotamer_kinforest.id + pbt.max_n_atoms
    )
    gold_leu_kinforest2_doftype = cat2(
        0, leu_rt.rotamer_kinforest.doftype, leu_rt.rotamer_kinforest.doftype
    )
    gold_leu_kinforest2_parent = cat2(
        0,
        leu_rt.rotamer_kinforest.parent + 1,
        leu_rt.rotamer_kinforest.parent + 1 + leu_rt.n_atoms,
    )
    # fix the jump-to-root for the 1st atom in rotamer 2
    gold_leu_kinforest2_parent[1 + leu_rt.n_atoms] = 0
    gold_leu_kinforest2_frame_x = cat2(
        0,
        leu_rt.rotamer_kinforest.frame_x + 1,
        leu_rt.rotamer_kinforest.frame_x + 1 + leu_rt.n_atoms,
    )
    gold_leu_kinforest2_frame_y = cat2(
        0,
        leu_rt.rotamer_kinforest.frame_y + 1,
        leu_rt.rotamer_kinforest.frame_y + 1 + leu_rt.n_atoms,
    )
    gold_leu_kinforest2_frame_z = cat2(
        0,
        leu_rt.rotamer_kinforest.frame_z + 1,
        leu_rt.rotamer_kinforest.frame_z + 1 + leu_rt.n_atoms,
    )

    numpy.testing.assert_equal(gold_leu_kinforest2_id, kt2.id.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kinforest2_doftype, kt2.doftype.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kinforest2_parent, kt2.parent.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kinforest2_frame_x, kt2.frame_x.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kinforest2_frame_y, kt2.frame_y.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kinforest2_frame_z, kt2.frame_z.cpu().numpy())


def test_construct_kinforest_for_rotamers2(
    default_database, fresh_default_restype_set, torch_device, dun_sampler
):
    # torch_device = torch.device("cpu")
    chem_db = default_database.chemical

    fixed_sampler = FixedAAChiSampler()
    samplers = (dun_sampler, fixed_sampler)

    leu_met_rt_list = [fresh_default_restype_set.restype_map["LEU"][0]] + [
        fresh_default_restype_set.restype_map["MET"][0]
    ]
    pbt = PackedBlockTypes.from_restype_list(
        default_database.chemical,
        fresh_default_restype_set,
        leu_met_rt_list,
        device=torch_device,
    )

    annotate_restype(leu_met_rt_list[0], samplers, chem_db)
    annotate_restype(leu_met_rt_list[1], samplers, chem_db)
    annotate_packed_block_types(pbt)

    leu_rt = leu_met_rt_list[0]

    kt1 = construct_kinforest_for_conformers(
        pbt,
        numpy.zeros(1, dtype=numpy.int32),
        leu_rt.n_atoms,
        torch.full((1,), leu_rt.n_atoms, dtype=torch.int32),
        numpy.zeros(1, dtype=numpy.int64),
        torch_device,
    )

    def cat(val, arr):
        return numpy.concatenate((numpy.array([val], dtype=numpy.int32), arr))

    gold_leu_kinforest1_id = cat(-1, leu_rt.rotamer_kinforest.id)
    gold_leu_kinforest1_doftype = cat(0, leu_rt.rotamer_kinforest.doftype)
    gold_leu_kinforest1_parent = cat(0, leu_rt.rotamer_kinforest.parent + 1)
    gold_leu_kinforest1_frame_x = cat(0, leu_rt.rotamer_kinforest.frame_x + 1)
    gold_leu_kinforest1_frame_y = cat(0, leu_rt.rotamer_kinforest.frame_y + 1)
    gold_leu_kinforest1_frame_z = cat(0, leu_rt.rotamer_kinforest.frame_z + 1)

    numpy.testing.assert_equal(gold_leu_kinforest1_id, kt1.id.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kinforest1_doftype, kt1.doftype.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kinforest1_parent, kt1.parent.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kinforest1_frame_x, kt1.frame_x.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kinforest1_frame_y, kt1.frame_y.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kinforest1_frame_z, kt1.frame_z.cpu().numpy())

    kt2 = construct_kinforest_for_conformers(
        pbt,
        numpy.zeros(2, dtype=numpy.int32),
        2 * leu_rt.n_atoms,
        torch.full((2,), leu_rt.n_atoms, dtype=torch.int32),
        numpy.arange(2, dtype=numpy.int64) * pbt.max_n_atoms,
        torch_device,
    )

    def cat2(val, arr1, arr2):
        return numpy.concatenate((numpy.array([val], dtype=numpy.int32), arr1, arr2))

    gold_leu_kinforest2_id = cat2(
        -1, leu_rt.rotamer_kinforest.id, leu_rt.rotamer_kinforest.id + pbt.max_n_atoms
    )
    gold_leu_kinforest2_doftype = cat2(
        0, leu_rt.rotamer_kinforest.doftype, leu_rt.rotamer_kinforest.doftype
    )
    gold_leu_kinforest2_parent = cat2(
        0,
        leu_rt.rotamer_kinforest.parent + 1,
        leu_rt.rotamer_kinforest.parent + 1 + leu_rt.n_atoms,
    )
    # fix the jump-to-root for the 1st atom in rotamer 2
    gold_leu_kinforest2_parent[1 + leu_rt.n_atoms] = 0
    gold_leu_kinforest2_frame_x = cat2(
        0,
        leu_rt.rotamer_kinforest.frame_x + 1,
        leu_rt.rotamer_kinforest.frame_x + 1 + leu_rt.n_atoms,
    )
    gold_leu_kinforest2_frame_y = cat2(
        0,
        leu_rt.rotamer_kinforest.frame_y + 1,
        leu_rt.rotamer_kinforest.frame_y + 1 + leu_rt.n_atoms,
    )
    gold_leu_kinforest2_frame_z = cat2(
        0,
        leu_rt.rotamer_kinforest.frame_z + 1,
        leu_rt.rotamer_kinforest.frame_z + 1 + leu_rt.n_atoms,
    )

    numpy.testing.assert_equal(gold_leu_kinforest2_id, kt2.id.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kinforest2_doftype, kt2.doftype.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kinforest2_parent, kt2.parent.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kinforest2_frame_x, kt2.frame_x.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kinforest2_frame_y, kt2.frame_y.cpu().numpy())
    numpy.testing.assert_equal(gold_leu_kinforest2_frame_z, kt2.frame_z.cpu().numpy())


def test_measure_original_dofs(default_database, ubq_pdb, torch_device, dun_sampler):
    chem_db = default_database.chemical

    poses = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=25)

    restype_set = poses.packed_block_types.restype_set
    palette = PackerPalette(restype_set)
    task = PackerTask(poses, palette)
    task.restrict_to_repacking()

    fixed_sampler = FixedAAChiSampler()
    task.add_conformer_sampler(dun_sampler)
    task.add_conformer_sampler(fixed_sampler)
    samplers = (dun_sampler, fixed_sampler)

    pbt = poses.packed_block_types
    for rt in pbt.active_block_types:
        annotate_restype(rt, samplers, chem_db)
    annotate_packed_block_types(pbt)

    block_type_ind = poses.block_type_ind.view(-1)
    real_block_type_ind = block_type_ind != -1
    nz_real_block_type_ind = torch.nonzero(real_block_type_ind).flatten()
    real_block_type_ind_numpy = nz_real_block_type_ind.cpu().numpy().astype(numpy.int32)
    block_type_ind = block_type_ind[block_type_ind != -1]
    block_type_ind_numpy = block_type_ind.cpu().numpy()
    res_n_atoms = pbt.n_atoms[block_type_ind.to(torch.int64)]
    n_total_atoms = torch.sum(res_n_atoms).item()

    kinforest = construct_kinforest_for_conformers(
        pbt,
        block_type_ind.cpu().numpy(),
        n_total_atoms,
        res_n_atoms,
        poses.block_coord_offset64.flatten().cpu().numpy(),
        torch_device,
    )

    dofs = measure_dofs_from_orig_coords(poses.coords.view(-1, 3), kinforest)

    # let's refold and make sure the coordinates are the same?
    # forward folding; let's build leu on the met's coords
    def _p(t):
        return torch.nn.Parameter(t, requires_grad=False)

    kinforest_stacked = _p(
        torch.stack(
            [
                kinforest.id,
                kinforest.doftype,
                kinforest.parent,
                kinforest.frame_x,
                kinforest.frame_y,
                kinforest.frame_z,
            ],
            dim=1,
        ).to(torch_device)
    )
    n_atoms_offset_for_rot = (
        exclusive_cumsum1d(res_n_atoms).cpu().numpy().astype(numpy.int64)
    )
    nodes, scans, gens = construct_scans_for_conformers(
        pbt, block_type_ind_numpy, res_n_atoms, n_atoms_offset_for_rot
    )

    new_kin_coords = forward_only_op(
        dofs,
        _p(torch.tensor(nodes, dtype=torch.int32, device=torch_device)),
        _p(torch.tensor(scans, dtype=torch.int32, device=torch_device)),
        _p(torch.tensor(gens, dtype=torch.int32, device=torch.device("cpu"))),
        kinforest_stacked,
    )

    new_coords = torch.zeros_like(poses.coords).view(-1, 3)
    new_coords[kinforest.id.to(torch.int64)] = new_kin_coords

    # fd no assert?


def test_measure_original_dofs2(default_database, ubq_pdb, torch_device, dun_sampler):
    chem_db = default_database.chemical

    p1 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=11)
    p2 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=76)

    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)
    restype_set = poses.packed_block_types.restype_set
    palette = PackerPalette(restype_set)
    task = PackerTask(poses, palette)
    task.restrict_to_repacking()

    fixed_sampler = FixedAAChiSampler()
    task.add_conformer_sampler(dun_sampler)
    task.add_conformer_sampler(fixed_sampler)
    samplers = (dun_sampler, fixed_sampler)

    pbt = poses.packed_block_types
    for rt in pbt.active_block_types:
        annotate_restype(rt, samplers, chem_db)
    annotate_packed_block_types(pbt)

    block_type_ind = poses.block_type_ind.view(-1)
    real_block_type_ind = block_type_ind != -1
    # nz_real_block_type_ind = torch.nonzero(real_block_type_ind).flatten()
    block_type_ind = block_type_ind[block_type_ind != -1]
    res_n_atoms = pbt.n_atoms[block_type_ind.to(torch.int64)]
    n_total_atoms = torch.sum(res_n_atoms).item()

    n_poses = poses.coords.shape[0]
    max_n_atoms_per_pose = poses.coords.shape[1]
    max_n_blocks_per_pose = poses.block_coord_offset.shape[1]
    per_pose_offset = max_n_atoms_per_pose * stretch(
        torch.arange(n_poses, dtype=torch.int64, device=poses.device),
        max_n_blocks_per_pose,
    )
    orig_atom_offset_for_poses_blocks = (
        (
            poses.block_coord_offset64.flatten()[real_block_type_ind]
            + per_pose_offset[real_block_type_ind]
        )
        .cpu()
        .numpy()
    )

    kinforest = construct_kinforest_for_conformers(
        pbt,
        block_type_ind.cpu().numpy(),
        n_total_atoms,
        res_n_atoms,
        orig_atom_offset_for_poses_blocks,
        torch_device,
    )

    dofs = measure_dofs_from_orig_coords(poses.coords.view(-1), kinforest)

    # let's refold and make sure the coordinates are the same?
    # forward folding; let's build leu on the met's coords
    def _p(t):
        return torch.nn.Parameter(t, requires_grad=False)

    kinforest_stacked = _p(
        torch.stack(
            [
                kinforest.id,
                kinforest.doftype,
                kinforest.parent,
                kinforest.frame_x,
                kinforest.frame_y,
                kinforest.frame_z,
            ],
            dim=1,
        ).to(torch_device)
    )
    n_atoms_offset_for_rot = (
        exclusive_cumsum1d(res_n_atoms).cpu().numpy().astype(numpy.int64)
    )

    nodes, scans, gens = construct_scans_for_conformers(
        pbt, block_type_ind.cpu().numpy(), res_n_atoms, n_atoms_offset_for_rot
    )

    new_kin_coords = forward_only_op(
        dofs,
        _p(torch.tensor(nodes, dtype=torch.int32, device=torch_device)),
        _p(torch.tensor(scans, dtype=torch.int32, device=torch_device)),
        _p(torch.tensor(gens, dtype=torch.int32, device=torch.device("cpu"))),
        kinforest_stacked,
    )

    new_coords = torch.zeros_like(poses.coords).view(-1, 3)
    new_coords[kinforest.id.to(torch.int64)] = new_kin_coords
    new_coords = new_coords.view(poses.coords.shape)

    for i in range(poses.coords.shape[0]):
        for j in range(poses.block_coord_offset.shape[1]):
            if poses.block_type_ind[i, j] == -1:
                continue
            j_n_atoms = poses.block_type(i, j).n_atoms
            numpy.testing.assert_almost_equal(
                poses.coords[i, j, :j_n_atoms].cpu().numpy(),
                new_coords[i, j, :j_n_atoms].cpu().numpy(),
                decimal=5,
            )

    # for writing coordinates into a pdb
    # print("new coords")
    # for i in range(0, new_coords.shape[0]):
    #     print(
    #         "%7.3f %7.3f %7.3f" %
    #         (
    #             new_coords[i, 0],
    #             new_coords[i, 1],
    #             new_coords[i, 2]
    #         )
    #     )


def test_create_dof_inds_to_copy_from_orig_to_rotamers(
    default_database, ubq_pdb, torch_device, dun_sampler
):
    """This test makes sure that the DOFs from the starting
    backbone are copied over. Part of the validity of its
    logic depends on all of the residues being mid-protein
    (i.e. not termini); this is a requirement of the test,
    not of the code itself, which is happy to build
    rotamers for termini.
    """
    p1 = no_termini_pose_stack_from_pdb(
        ubq_pdb, torch_device, residue_start=1, residue_end=3
    )
    p2 = no_termini_pose_stack_from_pdb(
        ubq_pdb, torch_device, residue_start=1, residue_end=4
    )
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)
    restype_set = poses.packed_block_types.restype_set

    pbt = poses.packed_block_types
    pbt_namelist = [x.name for x in pbt.active_block_types]
    # assert "LEU" not in pbt_namelist

    palette = PackerPalette(restype_set)
    task = PackerTask(poses, palette)
    leu_set = set(["LEU"])
    for one_pose_blts in task.blts:
        for blt in one_pose_blts:
            blt.restrict_absent_name3s(leu_set)

    fixed_sampler = FixedAAChiSampler()
    task.add_conformer_sampler(dun_sampler)
    task.add_conformer_sampler(fixed_sampler)

    poses, samplers = rebuild_poses_if_necessary(poses, task)
    pbt = poses.packed_block_types

    pbt_namelist = [x.name for x in pbt.active_block_types]
    assert "LEU" in pbt_namelist  # may have a variant attached?

    for i, rt in enumerate(pbt.active_block_types):
        if rt.name == "LEU":
            leu_rt = rt
            leu_ind = i
            break
    annotate_everything(default_database.chemical, samplers, pbt)

    gbt_for_rot = torch.tensor(
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=torch.int64, device=torch_device
    )
    block_type_ind_for_rot = torch.full(
        (10,), leu_ind, dtype=torch.int64, device=torch_device
    )

    # [pbt.mc_fingerprints.sampler_mapping[dun_sampler.sampler_name()] ] * 10
    # sampler_for_rotamer = torch.zeros(10, dtype=torch.int64, device=torch_device)
    conf_inds_for_dun_sampler = torch.arange(10, dtype=torch.int64, device=torch_device)
    sampler_n_rots_for_gbt = torch.full((5,), 2, dtype=torch.int32, device=torch_device)
    sampler_gbt_for_rotamer = gbt_for_rot.to(torch.int32)

    n_dof_atoms_offset_for_rot = (
        torch.arange(10, dtype=torch.int64, device=torch_device) * 19
    )

    dst, src = create_dof_inds_to_copy_from_orig_to_rotamers_for_sampler(
        poses,
        task,
        dun_sampler.sampler_name(),
        gbt_for_rot,
        block_type_ind_for_rot,
        conf_inds_for_dun_sampler,
        sampler_n_rots_for_gbt,
        sampler_gbt_for_rotamer,
        n_dof_atoms_offset_for_rot,
    )

    # OK, what do we expect?
    # dst should be the atoms for N, H, CA, HA, C, O
    # in kinforest order with an offset of 19 for each
    # successive batch.

    # fd hardcoded "H" fails w/ terminal variants
    fingerprint_atoms = "N", "H", "CA", "HA", "C", "O"

    def fp_kto(rt):
        return [
            rt.rotamer_kinforest.kinforest_idx[rt.atom_to_idx[at_name]]
            for at_name in fingerprint_atoms
        ]

    dst_gold_template = numpy.array(fp_kto(leu_rt), dtype=numpy.int64)
    dst_gold = numpy.arange(10).repeat(6) * 19 + numpy.tile(dst_gold_template, 10) + 1

    numpy.testing.assert_equal(dst_gold, dst.cpu().numpy())

    src_fpats_kto = numpy.array(
        fp_kto(pbt.active_block_types[poses.block_type_ind[0, 0]])
        + fp_kto(pbt.active_block_types[poses.block_type_ind[0, 1]])
        + fp_kto(pbt.active_block_types[poses.block_type_ind[1, 0]])
        + fp_kto(pbt.active_block_types[poses.block_type_ind[1, 1]])
        + fp_kto(pbt.active_block_types[poses.block_type_ind[1, 2]]),
        dtype=numpy.int64,
    )

    def n_ats(i1, i2):
        return pbt.n_atoms[poses.block_type_ind[i1, i2]]

    src_dof_offsets = numpy.cumsum(
        [0, n_ats(0, 0).cpu(), n_ats(0, 1).cpu(), n_ats(1, 0).cpu(), n_ats(1, 1).cpu()]
    ).repeat(6)

    src_gold = src_fpats_kto + src_dof_offsets + 1

    numpy.testing.assert_equal(src_gold, src_gold)


def test_create_dof_inds_to_copy_from_orig_to_rotamers2(
    default_database, ubq_pdb, torch_device, dun_sampler
):
    p = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=6)
    poses = PoseStackBuilder.from_poses([p] * 3, torch_device)
    restype_set = poses.packed_block_types.restype_set
    palette = PackerPalette(restype_set)
    task = PackerTask(poses, palette)
    task.restrict_to_repacking()

    gbt_for_rot_list = []
    count_gbt = 0

    for i, one_pose_blts in enumerate(task.blts):
        for j, blt in enumerate(one_pose_blts):
            for k, bt in enumerate(blt.considered_block_types):
                if blt.block_type_allowed[k]:
                    # print("allowed block type", i, j, k, bt.name)
                    gbt_for_rot_list.append(count_gbt)
                    gbt_for_rot_list.append(count_gbt)
                count_gbt += 1

    fixed_sampler = FixedAAChiSampler()
    task.add_conformer_sampler(dun_sampler)
    task.add_conformer_sampler(fixed_sampler)

    poses, samplers = rebuild_poses_if_necessary(poses, task)
    pbt = poses.packed_block_types
    annotate_everything(default_database.chemical, samplers, pbt)

    # gbt_for_rot = torch.floor_divide(
    #     torch.arange(36, dtype=torch.int64, device=torch_device), 2
    # )
    gbt_for_rot = torch.tensor(gbt_for_rot_list, dtype=torch.int64, device=torch_device)

    block_type_ind_for_rot = torch.remainder(
        torch.floor_divide(torch.arange(36, dtype=torch.int64, device=torch_device), 2),
        6,
    )

    # sampler_for_rotamer = torch.zeros(36, dtype=torch.int64, device=torch_device)
    conf_inds_for_dun_sampler = torch.arange(36, dtype=torch.int64, device=torch_device)
    sampler_n_rots_for_gbt = torch.full(
        (18,), 2, dtype=torch.int32, device=torch_device
    )
    sampler_gbt_for_rotamer = gbt_for_rot.to(torch.int32)

    n_dof_atoms_offset_for_rot = exclusive_cumsum1d(pbt.n_atoms[block_type_ind_for_rot])

    dst, src = create_dof_inds_to_copy_from_orig_to_rotamers_for_sampler(
        poses,
        task,
        dun_sampler.sampler_name(),
        gbt_for_rot,
        block_type_ind_for_rot,
        conf_inds_for_dun_sampler,
        sampler_n_rots_for_gbt,
        sampler_gbt_for_rotamer,
        n_dof_atoms_offset_for_rot,
    )

    src = src.cpu().numpy()
    dst = dst.cpu().numpy()

    n_src = src.shape[0]
    n_dst = dst.shape[0]
    assert n_src == n_dst

    n_dofs_per_pose = n_src // 3
    assert 3 * n_dofs_per_pose == n_src

    n_ats_per_pose = torch.sum(
        pbt.n_atoms[poses.block_type_ind[0].to(torch.int64)]
    ).item()
    n_rot_ats_per_pose = torch.sum(pbt.n_atoms[block_type_ind_for_rot[:12]]).item()

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


def test_build_lots_of_rotamers(default_database, ubq_pdb, torch_device, dun_sampler):
    n_poses = 6

    p = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=76)
    poses = PoseStackBuilder.from_poses([p] * n_poses, torch_device)
    restype_set = poses.packed_block_types.restype_set
    palette = PackerPalette(restype_set)
    task = PackerTask(poses, palette)
    task.restrict_to_repacking()

    fixed_sampler = FixedAAChiSampler()
    task.add_conformer_sampler(dun_sampler)
    task.add_conformer_sampler(fixed_sampler)

    poses, rotamer_set = build_rotamers(poses, task, default_database.chemical)
    numpy.testing.assert_array_less(
        rotamer_set.block_type_ind_for_rot.cpu().numpy(),
        len(poses.packed_block_types.active_block_types),
    )

    # print("rotamer_set.coords.shape")
    # print(rotamer_set.coords.shape)
    n_atoms = rotamer_set.coords.shape[0]

    # all the rotamers should be the same on all n_poses copies of ubq
    n_atoms_per_pose = n_atoms // n_poses
    assert n_atoms_per_pose * n_poses == n_atoms

    new_coords = rotamer_set.coords.cpu().numpy()
    # print("new_coords[:10]")
    # print(new_coords[:10])
    # print("new_coords[n_atoms_per_pose:(n_atoms_per_pose + 10)]")
    # print(new_coords[n_atoms_per_pose : (n_atoms_per_pose + 10)])

    for i in range(1, n_poses):
        numpy.testing.assert_almost_equal(
            new_coords[:n_atoms_per_pose],
            new_coords[(n_atoms_per_pose * i) : (n_atoms_per_pose * (i + 1))],
            decimal=5,
        )


def test_create_dofs_for_many_rotamers(
    default_database, ubq_pdb, torch_device, dun_sampler
):
    n_poses = 6

    p = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=76)
    restype_set = p.packed_block_types.restype_set
    poses = PoseStackBuilder.from_poses([p] * n_poses, torch_device)
    palette = PackerPalette(restype_set)
    task = PackerTask(poses, palette)
    task.restrict_to_repacking()

    fixed_sampler = FixedAAChiSampler()
    task.add_conformer_sampler(dun_sampler)
    task.add_conformer_sampler(fixed_sampler)
    chem_db = default_database.chemical

    ###########################################
    # Now the contents of build rotamers right
    # up to the coordinate calc
    ###########################################
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

    ###########################################
    # ok, now let's make sure that rot_dofs_kto
    # is a perfect copy from beginning to end
    ###########################################

    rot_dofs_kto = conf_dofs_kto.cpu().numpy()
    n_rot_atoms = rot_dofs_kto.shape[0] - 1

    # all the rotamers should be the same on all n_poses copies of ubq
    # there's going to be one extra atom for the root
    n_rot_atoms_per_pose = n_rot_atoms // n_poses
    assert n_rot_atoms_per_pose * n_poses == n_rot_atoms

    for i in range(1, n_poses):
        numpy.testing.assert_almost_equal(
            rot_dofs_kto[1 : n_rot_atoms_per_pose + 1],
            rot_dofs_kto[
                (n_rot_atoms_per_pose * i + 1) : (n_rot_atoms_per_pose * (i + 1) + 1)
            ],
            decimal=5,
        )


def test_new_rotamer_building_logic1(
    default_database, ubq_pdb, torch_device, dun_sampler
):
    n_poses = 6
    p = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=76)
    restype_set = p.packed_block_types.restype_set
    poses = PoseStackBuilder.from_poses([p] * n_poses, torch_device)

    palette = PackerPalette(restype_set)
    task = PackerTask(poses, palette)
    task.restrict_to_repacking()

    fixed_sampler = FixedAAChiSampler()
    task.add_conformer_sampler(dun_sampler)
    task.add_conformer_sampler(fixed_sampler)
    chem_db = default_database.chemical
    ###########################################
    # Now the contents of build rotamers right
    ###########################################

    # Step 1
    # print("A n poses:", poses.n_poses, poses.block_type_ind.shape)
    poses, samplers = rebuild_poses_if_necessary(poses, task)
    # print("B n poses:", poses.n_poses, poses.block_type_ind.shape)
    pbt = poses.packed_block_types

    # Step 2
    annotate_everything(chem_db, samplers, pbt)

    # Step 3
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
    # for i, samples in enumerate(conformer_samples):
    #     print("i", i, "samples", samples[0].shape, samples[1].shape)

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

    gbt_for_conformer_torch = _t(gbt_for_conformer, torch.int64)

    # apl: fingers crossed this is no longer true!
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
        # print("fill dofs for samples", sampler.sampler_name())
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

    # return (
    #     poses,
    #     RotamerSet(
    #         n_rots_for_pose=n_rots_for_pose,
    #         rot_offset_for_pose=rot_offset_for_pose,
    #         n_rots_for_block=n_rots_for_block,
    #         rot_offset_for_block=rot_offset_for_block,
    #         pose_for_rot=pose_for_rot,
    #         block_type_ind_for_rot=block_type_ind_for_conformer_torch,
    #         block_ind_for_rot=block_ind_for_rot,
    #         coords=rotamer_coords,
    #     ),
    # )


def test_new_rotamer_building_logic2(
    default_database, ubq_pdb, torch_device, dun_sampler
):
    n_poses = 6
    p = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=76)
    restype_set = p.packed_block_types.restype_set
    poses = PoseStackBuilder.from_poses([p] * n_poses, torch_device)

    palette = PackerPalette(restype_set)
    task = PackerTask(poses, palette)
    task.restrict_to_repacking()

    fixed_sampler = FixedAAChiSampler()
    task.add_conformer_sampler(dun_sampler)
    task.add_conformer_sampler(fixed_sampler)
    chem_db = default_database.chemical
    ###########################################

    poses, rotamer_set = build_rotamers(poses, task, chem_db)
