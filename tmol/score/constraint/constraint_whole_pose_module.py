import torch


class ConstraintWholePoseScoringModule(torch.nn.Module):
    def __init__(
        self,
        pose_stack_block_coord_offset,
        pose_stack_block_types,
        pose_stack_inter_block_connections,
        bt_atom_downstream_of_conn,
        constraint_set,
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

        self.constraint_function_inds = _p(constraint_set.constraint_function_inds)
        self.constraint_params = _p(constraint_set.constraint_params)
        self.constraint_functions = constraint_set.constraint_functions

        self.constraint_atom_pose_inds = constraint_set.constraint_atoms[:, :, 0]
        self.constraint_atom_residue_inds = constraint_set.constraint_atoms[:, :, 1]
        constraint_atom_atom_inds = constraint_set.constraint_atoms[:, :, 2]

        atom_offsets = self.pose_stack_block_coord_offset[
            self.constraint_atom_pose_inds.view(-1),
            self.constraint_atom_residue_inds.view(-1),
        ].view(self.constraint_atom_residue_inds.size())

        self.atom_global_indices = atom_offsets + constraint_atom_atom_inds

    def forward(self, coords, output_block_pair_energies=False):

        def score_cnstrs(functions, types, atom_coords, params):
            cnstr_scores = torch.full(
                (len(types),), 0, dtype=torch.float32, device=coords.device
            )

            for ind, fn in enumerate(functions):

                # get the constraints that match this constraint type
                c_inds = torch.nonzero(types == ind)

                fn_coords = atom_coords[c_inds].squeeze(1)
                result = fn(fn_coords, params[c_inds].squeeze(1))
                result = result.unsqueeze(1)
                cnstr_scores[c_inds] += result

            return cnstr_scores

        atom_coords = coords[
            self.constraint_atom_pose_inds.view(-1), self.atom_global_indices.view(-1)
        ].view(self.atom_global_indices.size(0), self.atom_global_indices.size(1), 3)

        nblocks = self.pose_stack_block_coord_offset.size(1)
        nposes = self.pose_stack_block_coord_offset.size(0)

        def distribute_scores(scores, atom_pose_inds, atom_res_inds):
            block_scores = torch.full(
                (
                    nposes,
                    nblocks,
                    nblocks,
                ),
                0,
                dtype=torch.float32,
                device=coords.device,
            )

            flattened = block_scores.view(-1)
            indices1 = (
                atom_pose_inds[:, 0] * (nblocks**2)
                + atom_res_inds[:, 0] * nblocks
                + atom_res_inds[:, 3]
            )
            indices2 = (
                atom_pose_inds[:, 0] * (nblocks**2)
                + atom_res_inds[:, 3] * nblocks
                + atom_res_inds[:, 0]
            )
            flattened.index_add_(0, indices1, scores / 2)
            flattened.index_add_(0, indices2, scores / 2)
            return block_scores

        scores = score_cnstrs(
            self.constraint_functions,
            self.constraint_function_inds,
            atom_coords,
            self.constraint_params,
        )
        scores = distribute_scores(
            scores, self.constraint_atom_pose_inds, self.constraint_atom_residue_inds
        )

        if output_block_pair_energies:
            pass
        else:
            # for each pose, sum up the block scores
            scores = torch.sum(scores, (1, 2))

        # wrap this all in an extra dim (the output expects an outer dim to separate sub-terms)
        scores = torch.unsqueeze(scores, 0)
        return scores
