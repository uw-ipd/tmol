import torch
import numpy


class ConstraintWholePoseScoringModule(torch.nn.Module):
    def __init__(
        self,
        pose_stack_block_coord_offset,
        pose_stack_block_types,
        pose_stack_inter_block_connections,
        bt_atom_downstream_of_conn,
    ):
        super(ConstraintWholePoseScoringModule, self).__init__()

        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        def _t(ts):
            return tuple(map(lambda t: t.to(torch.float), ts))

        self.pose_stack_block_coord_offset = _p(pose_stack_block_coord_offset)
        self.pose_stack_block_types = _p(pose_stack_block_types)
        self.pose_stack_inter_block_connections = _p(pose_stack_inter_block_connections)
        self.bt_atom_downstream_of_conn = _p(bt_atom_downstream_of_conn)

    def forward(self, coords):
        print("HELLO")

        n_cnstrs = 3
        cnstrs = torch.full(
            (n_cnstrs, 2, 2), 0, dtype=torch.int32, device=coords.device
        )
        for i in range(n_cnstrs):
            cnstrs[i, 0, 0] = i
            cnstrs[i, 0, 1] = i
            cnstrs[i, 1, 0] = i + 1
            cnstrs[i, 1, 1] = i + 1

        atom_residue_indices = cnstrs[:, :, 0].flatten()
        atom_atom_indices = cnstrs[:, :, 1].flatten()

        atom_global_atom_indices = (
            self.pose_stack_block_coord_offset.index_select(1, atom_residue_indices)[0]
            + atom_atom_indices
        )
        print(atom_residue_indices)
        print(atom_atom_indices)
        print(atom_global_atom_indices)

        atom1_residue_indices = cnstrs[:, 0, 0]
        atom1_atom_indices = cnstrs[:, 0, 1]

        atom2_residue_indices = cnstrs[:, 1, 0]
        atom2_atom_indices = cnstrs[:, 1, 1]

        atom1_global_atom_indices = (
            self.pose_stack_block_coord_offset.index_select(1, atom1_residue_indices)[0]
            + atom1_atom_indices
        )
        atom2_global_atom_indices = (
            self.pose_stack_block_coord_offset.index_select(1, atom2_residue_indices)[0]
            + atom2_atom_indices
        )

        print(atom1_global_atom_indices)
        print(atom2_global_atom_indices)

        ideal = 4
        harm = lambda dist: (dist - ideal) ** 2

        # print(coords[0, global_atom_indices])
        print("ATOM1")
        print(coords[0].index_select(0, atom1_global_atom_indices))
        print("ATOM2")
        print(coords[0].index_select(0, atom2_global_atom_indices))
        c = coords[0]
        diff = c.index_select(0, atom1_global_atom_indices) - c.index_select(
            0, atom2_global_atom_indices
        )
        dist = diff.pow(2).sum(1).sqrt()

        print("DIST")
        print(dist)

        ret = torch.unsqueeze(torch.unsqueeze(torch.sum(harm(dist), 0), 0), 0)
        print(ret)
        return ret
        pass
