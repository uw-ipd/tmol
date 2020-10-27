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
from tmol.system.restypes import RefinedResidueType, ResidueTypeSet
from tmol.system.pose import PackedBlockTypes, Pose, Poses
from tmol.score.dunbrack.params import DunbrackParamResolver
from tmol.pack.packer_task import PackerTask, PackerPalette
from tmol.pack.rotamer.chi_sampler import ChiSampler
from tmol.pack.rotamer.dunbrack_chi_sampler import DunbrackChiSampler


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


@numba.jit(nopython=True)
def bfs_sidechain_atoms_jit(parents, sc_roots):
    n_atoms = parents.shape[0]
    n_children = numpy.zeros(n_atoms, dtype=numpy.int32)
    for i in range(n_atoms):
        n_children[parents[i]] += 1
    children_start = numpy.concatenate(
        (numpy.zeros(1, dtype=numpy.int32), numpy.cumsum(n_children)[:-1])
    )

    child_count = numpy.zeros(n_atoms, dtype=numpy.int32)
    children = numpy.full(n_atoms, -1, dtype=numpy.int32)
    for i in range(n_atoms):
        parent = parents[i]
        children[children_start[parent] + child_count[parent]] = i
        child_count[parent] += 1

    bfs_count_end = 0
    bfs_curr = 0
    bfs_list = numpy.full(n_atoms, -1, dtype=numpy.int32)
    visited = numpy.zeros(n_atoms, dtype=numpy.int32)
    for root in sc_roots:
        # put the root children in the bfs list
        visited[root] = 1
        for child_ind in range(
            children_start[root], children_start[root] + child_count[root]
        ):
            bfs_list[bfs_count_end] = children[child_ind]
            bfs_count_end += 1
        while bfs_curr != bfs_count_end:
            node = bfs_list[bfs_curr]
            bfs_curr += 1
            if visited[node]:
                # can happen when the root of the kintree is given
                # as a sidechain root
                continue
            # add node's children to the bfs_list
            for child_ind in range(
                children_start[node], children_start[node] + child_count[node]
            ):
                bfs_list[bfs_count_end] = children[child_ind]
                bfs_count_end += 1
            visited[node] = 1
    return visited


def bfs_sidechain_atoms(restype, sc_roots):
    # first descend through the sidechain
    id = restype.kintree_id
    parents = restype.kintree_parent
    parents[parents < 0] = 0
    parents[id] = id[parents]
    return bfs_sidechain_atoms_jit(parents, numpy.array(sc_roots, dtype=numpy.int32))


@numba.jit(nopython=True)
def eye4():
    """Create the identity homogeneous transform
    Only necessary because numpy.eye(4, dtype=numpy.float32)
    is strangely unsupported in numpy"""

    m = numpy.zeros((4, 4), dtype=numpy.float32)
    m[0, 0] = 1
    m[1, 1] = 1
    m[2, 2] = 1
    m[3, 3] = 1
    return m


@numba.jit
def normalize(v):
    return v / numpy.linalg.norm(v)


@numba.jit(nopython=True)
def frame_from_coords(p1, p2, p3):
    ht = eye4()
    # ht = numpy.eye(4,4, dtype=numpy.float32)
    z = normalize(p3 - p2)
    v21 = normalize(p1 - p2)
    y = normalize(v21 - numpy.dot(z, v21) * z)
    x = normalize(numpy.cross(y, z))

    ht[0:3, 0] = x
    ht[0:3, 1] = y
    ht[0:3, 2] = z
    ht[0:3, 3] = p3
    return ht


# @numba.jit
@numba.jit(nopython=True)
def rot_x(rot):
    ht = eye4()
    crot = numpy.cos(rot)
    srot = numpy.sin(rot)
    # print("rot x", rot, crot, srot)
    ht[1, 1] = crot
    ht[2, 1] = srot
    ht[1, 2] = -srot
    ht[2, 2] = crot
    return ht


# @numba.jit
@numba.jit(nopython=True)
def rot_z(rot):
    ht = eye4()
    crot = numpy.cos(rot)
    srot = numpy.sin(rot)
    # print("rot z", rot, crot, srot)
    ht[0, 0] = crot
    ht[1, 0] = srot
    ht[0, 1] = -srot
    ht[1, 1] = crot
    return ht


# @numba.jit
@numba.jit(nopython=True)
def trans_z(trans):
    ht = eye4()
    ht[2, 3] = trans
    return ht


# @numba.jit
@numba.jit(nopython=True)
def build_coords_from_icoors(icoors_ancestors, icoors_geom):
    # start with atom 1 at the origin
    # place atom 2 along the x axis
    # place atom 3 in the x-y plane
    # place all other atoms

    n_atoms = icoors_ancestors.shape[0]
    coords = numpy.zeros((n_atoms, 3), dtype=numpy.float32)
    coords[1, 0] = icoors_geom[1, 2]
    # coord 2
    # in the x-y plane
    ht_1 = eye4()
    ht_1[:3, 3] = coords[1, :]
    rot_2 = rot_z(-icoors_geom[2, 1])
    trans_2 = eye4()
    trans_2[0, 3] = icoors_geom[2, 2]
    ht_2 = ht_1 @ rot_2 @ trans_2
    coords[2, :] = ht_2[:3, 3]

    for i in range(3, icoors_ancestors.shape[0]):
        # print("ancestors", i)
        ht_i = frame_from_coords(
            coords[icoors_ancestors[i, 2], :],
            coords[icoors_ancestors[i, 1], :],
            coords[icoors_ancestors[i, 0], :],
        )

        ht_rot_z = rot_z(icoors_geom[i, 0])
        ht_rot_x = rot_x(-icoors_geom[i, 1])
        ht_trans_z = trans_z(icoors_geom[i, 2])

        # temp = numpy.matmul(ht_trans_z, ht_rot_x)
        # temp = numpy.matmul(temp, ht_rot_z)
        # ht_i = numpy.matmul(temp, ht_i)

        ht_i = ht_i @ ht_rot_z @ ht_rot_x @ ht_trans_z
        coords[i, :3] = ht_i[:3, 3]
    return coords


def build_ideal_coords(restype):

    # lets build a kintree using not the prioritized bonds,
    # but the icoors; let's not even use the scan algorithm.
    # let's just build the coordinates directly from the
    # tree provided in the icoors.

    return build_coords_from_icoors(restype.icoors_ancestors, restype.icoors_geom)


def test_build_ideal_coords(default_database):
    torch_device = torch.device("cpu")
    rts = ResidueTypeSet.from_database(default_database.chemical)
    leu_rt = rts.restype_map["LEU"][0]

    coords = build_ideal_coords(leu_rt)
    print("leu ideal coords")
    for i in range(coords.shape[0]):
        print(leu_rt.icoors[i].name, coords[i, :])

    print("angle at up")
    n_ind = leu_rt.icoors_index["CA"]
    ca_ind = leu_rt.icoors_index["C"]
    cb_ind = leu_rt.icoors_index["up"]
    print(
        numpy.degrees(
            numpy.arccos(
                numpy.dot(
                    normalize(coords[n_ind, :] - coords[ca_ind, :]),
                    normalize(coords[cb_ind, :] - coords[ca_ind, :]),
                )
            )
        )
    )
    print("dist to up", numpy.linalg.norm(coords[ca_ind, :] - coords[cb_ind]))


def test_identify_sidechain_atoms_from_roots(default_database):
    torch_device = torch.device("cpu")
    rts = ResidueTypeSet.from_database(default_database.chemical)
    leu_rt = rts.restype_map["LEU"][0]

    annotate_restype(leu_rt)

    sc_ats = bfs_sidechain_atoms(leu_rt, [leu_rt.atom_to_idx["CB"]])
    atom_names = numpy.array([at.name for at in leu_rt.atoms], dtype=str)
    sc_ats = set(atom_names[sc_ats != 0])
    gold_sc_ats = [
        "CB",
        "1HB",
        "2HB",
        "CG",
        "HG",
        "CD1",
        "1HD1",
        "2HD1",
        "3HD1",
        "CD2",
        "1HD2",
        "2HD2",
        "3HD2",
    ]
    assert len(gold_sc_ats) == len(sc_ats)
    for gold_at in gold_sc_ats:
        assert gold_at in sc_ats


def create_non_sidechain_fingerprint(rt, parents, sc_atoms):
    non_sc_atoms = numpy.nonzero(sc_atoms == 0)
    mc_at_names = rt.properties.polymer.mainchain_atoms
    mc_atoms = numpy.array(
        [rt.atom_to_idx[at] for at in mc_at_names], dtype=numpy.int32
    )
    mc_ind = numpy.full(rt.n_atoms, -1, dtype=numpy.int32)
    mc_ind[mc_atoms] = numpy.arange(mc_atoms.shape[0], dtype=numpy.int32)

    # count the number of bonds for each atom
    n_bonds = numpy.zeros(rt.n_atoms, dtype=numpy.int32)
    for i in range(rt.bond_indices.shape[0]):
        n_bonds[rt.bond_indices[i, 0]] += 1
    for conn in rt.connection_to_idx:
        n_bonds[rt.connection_to_idx[conn]] += 1

    mc_ancestors = numpy.full(rt.n_atoms, -1, dtype=numpy.int32)
    chiralities = numpy.full(rt.n_atoms, -1, dtype=numpy.int32)
    for at in non_sc_atoms:
        # find the index of the mc atom this branches from using the kintree
        mc_anc = mc_ind[at]
        bonds_from_mc = 0
        atom = at
        for i in range(rt.n_atoms):
            if mc_anc == -1:
                break
            par = parents[atom]
            mc_anc = mc_ind[par]
            atom = par
            bonds_from_mc += 1
        # now lets figure out the chirality of this atom??
        if bonds_from_mc == 0:
            chirality = 0
        elif bonds_from_mc == 1:
            # ok, let's figure out the number of bonds
            # that the mc atom has
            mc_n_bonds = n_bonds[mc_anc]
            if mc_n_bonds == 4:
                # where is the
                pass
            else:
                chirality = 0


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
