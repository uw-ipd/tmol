import torch
import numpy
from tmol.pose.pose_stack import PoseStack
from tmol.pose.pose_kinematics import get_bonds_for_named_torsions


def test_get_bonds_for_named_torsions(ubq_res):
    torch_device = torch.device("cpu")
    p1 = PoseStack.one_structure_from_polymeric_residues(ubq_res[:40], torch_device)
    p2 = PoseStack.one_structure_from_polymeric_residues(ubq_res[:60], torch_device)
    pose_stack = PoseStack.from_poses((p1, p2), torch_device)

    middle_bond_ats = get_bonds_for_named_torsions(pose_stack)

    def resolve_atom(pose_ind, res_ind, res, uaid):
        if uaid.atom is None:
            conn_ind = res.residue_type.connection_to_cidx[uaid.connection]
            other_res_ind = pose_stack.inter_residue_connections[
                pose_ind, res_ind, conn_ind, 0
            ]
            if other_res_ind == -1:
                return pose_ind, -1, -1
            other_res_conn = pose_stack.inter_residue_connections[
                pose_ind, res_ind, conn_ind, 1
            ]
            other_res_block_type = pose_stack.block_type_ind[0, res_ind]

            other_res_atom = pose_stack.packed_block_types.atom_downstream_of_conn[
                other_res_block_type, other_res_conn, uaid.bond_sep_from_conn
            ]
            return pose_ind, other_res_ind, other_res_atom
        else:
            return pose_ind, res_ind, res.residue_type.atom_to_idx[uaid.atom]

    def atom_ind_to_global_index(pose_ind, res_ind, at_ind):
        if res_ind == -1 or at_ind == -1:
            return -1
        return (
            pose_ind * pose_stack.max_n_pose_atoms
            + pose_stack.block_coord_offset[pose_ind, res_ind]
            + at_ind
        )

    tor_at_inds = []
    for pose_ind in range(2):
        for i, res in enumerate(pose_stack.residues[pose_ind]):
            for j, tor in enumerate(res.residue_type.torsions):
                at1 = atom_ind_to_global_index(*resolve_atom(pose_ind, i, res, tor.b))
                at2 = atom_ind_to_global_index(*resolve_atom(pose_ind, i, res, tor.c))
                if at1 != -1 and at2 != -1:
                    tor_at_inds.append((at1, at2))
    middle_bond_ats_gold = numpy.array(tor_at_inds, dtype=numpy.int64)
    numpy.testing.assert_equal(middle_bond_ats_gold, middle_bond_ats.cpu().numpy())


# def test_build_kintree_for_pose(ubq_res, torch_device):
#    pass
