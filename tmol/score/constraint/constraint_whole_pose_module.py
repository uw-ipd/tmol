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

    def forward(self, coords, output_block_pair_energies=False):
        n_cnstrs = 3
        cnstrs = torch.full(
            (n_cnstrs, 2, 2), 0, dtype=torch.int32, device=coords.device
        )

        # constraint_index x (type, pose, atm1_block, atm1_atm, ...)
        """for i in range(n_cnstrs):
            cnstrs[i, 0, 0] = 0
            cnstrs[i, 0, 1] = 2*i
            cnstrs[i, 1, 0] = 0
            cnstrs[i, 1, 1] = 2*i + 1"""

        cnstrs[0, 0, 0] = 0
        cnstrs[0, 0, 1] = 0
        cnstrs[0, 1, 0] = 1
        cnstrs[0, 1, 1] = 1

        cnstrs[1, 0, 0] = 1
        cnstrs[1, 0, 1] = 0
        cnstrs[1, 1, 0] = 2
        cnstrs[1, 1, 1] = 1

        cnstrs[2, 0, 0] = 0
        cnstrs[2, 0, 1] = 0
        cnstrs[2, 1, 0] = 1
        cnstrs[2, 1, 1] = 1

        atom_residue_indices = cnstrs[:, :, 0].flatten()
        atom_atom_indices = cnstrs[:, :, 1].flatten()

        atom_global_atom_indices = (
            self.pose_stack_block_coord_offset.index_select(1, atom_residue_indices)[0]
            + atom_atom_indices
        )

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

        ideal = 4
        harm = lambda atom1, atom2: ((atom1 - atom2).pow(2).sum(2).sqrt() - ideal) ** 2
        harm2 = lambda atom1, atom2: ((atom1 - atom2).pow(2).sum(2).sqrt() - ideal) ** 4
        consta = lambda atom1, atom2: torch.full_like(
            atom1.sum(2), 0.1
        )  # .sum(2)*0 + 10
        # consta = lambda atom1, atom2: ((atom1-atom2).pow(2).sum(2).sqrt() - ideal) ** 2
        lambdas = [harm, consta]
        constr_fns = torch.tensor([0, 0, 1], dtype=torch.int32, device=coords.device)

        def score_cnstrs(types, atom1, atom2):
            cnstr_scores = torch.full(
                (len(types),), 0, dtype=torch.float32, device=coords.device
            )

            for ind, lam in enumerate(lambdas):
                # get the constraints that match this type
                c_inds = torch.nonzero(types == ind)

                atom1_sq = atom1[c_inds].squeeze(1)
                atom2_sq = atom2[c_inds].squeeze(1)
                result = lam(atom1[c_inds], atom2[c_inds])
                # print("lambda: ", ind, result)
                cnstr_scores[c_inds] += result

            # print(cnstr_scores)
            return cnstr_scores

        # print(coords[0, global_atom_indices])
        # print(coords[0].index_select(0, atom1_global_atom_indices))
        # print(coords[0].index_select(0, atom2_global_atom_indices))
        c = coords[0]
        atom1_coords = c.index_select(0, atom1_global_atom_indices)
        atom2_coords = c.index_select(0, atom2_global_atom_indices)

        nblocks = self.pose_stack_block_coord_offset.size(1)

        def distribute_scores(scores, atom1_res_inds, atom2_res_inds):
            block_scores = torch.full(
                (
                    nblocks,
                    nblocks,
                ),
                0,
                dtype=torch.float32,
                device=coords.device,
            )

            # print(scores)
            # print(scores != 500000)
            # print(scores > -1000)
            # block_scores[atom1_res_inds, atom2_res_inds].scatter_add(0, scores != 5000000, scores)
            flattened = block_scores.view(-1)
            indices1 = atom1_res_inds * nblocks + atom2_res_inds
            indices2 = atom2_res_inds * nblocks + atom1_res_inds
            # print(indices)
            flattened.index_add_(0, indices1, scores / 2)
            flattened.index_add_(0, indices2, scores / 2)
            # block_scores.scatter_add(0, (atom2_res_inds, atom1_res_inds), scores/2)
            # block_scores[atom1_res_inds, atom2_res_inds] = block_scores[atom1_res_inds, atom2_res_inds] + scores/2
            # block_scores[atom2_res_inds, atom1_res_inds] = block_scores[atom2_res_inds, atom1_res_inds] + scores/2
            # block_scores[atom2_res_inds, atom1_res_inds] += scores/2
            return block_scores

        scores = score_cnstrs(constr_fns, atom1_coords, atom2_coords)
        scores = distribute_scores(scores, atom1_residue_indices, atom2_residue_indices)

        # print(scores)

        # ret = torch.unsqueeze(torch.unsqueeze(torch.sum(outer_func(constr_fns, atom1_coords, atom2_coords), 0), 0), 0)

        if output_block_pair_energies:
            pass
            # score = torch.diag_embed(score)
        else:
            # for each pose, sum up the block scores
            scores = torch.sum(scores, 1)

        # wrap this all in an extra dim (the output expects an outer dim to separate sub-terms)
        scores = torch.unsqueeze(torch.unsqueeze(scores, 0), 0)
        return scores
