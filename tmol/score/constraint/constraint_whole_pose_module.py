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
        MAX_N_ATOMS = 4

        ideal = 4
        harm = (
            lambda atoms: (
                (atoms[:, :, 0] - atoms[:, :, 1]).pow(2).sum(2).sqrt() - ideal
            )
            ** 2
        )

        def harmfunc(atoms):
            atoms1 = atoms[:, 0]
            atoms2 = atoms[:, 1]
            diff = atoms1 - atoms2
            return (diff.pow(2).sum(1).sqrt() - 4) ** 2

        lambdas = [harmfunc]

        # constr# x type,pose,num_atms,num_params
        cnstrs = torch.full((n_cnstrs, 4), 0, dtype=torch.int32, device=coords.device)

        cnstrs[0] = torch.tensor([0, 0, 2, 0])
        cnstrs[1] = torch.tensor([0, 0, 2, 0])
        cnstrs[2] = torch.tensor([0, 0, 2, 0])

        # cnstr# x 4(atoms) x res_ind,atom_ind
        cnstr_atoms = torch.full(
            (n_cnstrs, MAX_N_ATOMS, 2), 0, dtype=torch.int32, device=coords.device
        )

        cnstr_atoms[0, 0, 0] = 0
        cnstr_atoms[0, 0, 1] = 0
        cnstr_atoms[0, 1, 0] = 1
        cnstr_atoms[0, 1, 1] = 1

        cnstr_atoms[1, 0, 0] = 1
        cnstr_atoms[1, 0, 1] = 0
        cnstr_atoms[1, 1, 0] = 2
        cnstr_atoms[1, 1, 1] = 1

        cnstr_atoms[2, 0, 0] = 0
        cnstr_atoms[2, 0, 1] = 0
        cnstr_atoms[2, 1, 0] = 1
        cnstr_atoms[2, 1, 1] = 1

        atom_pose_indices = (
            cnstrs[:, 1].unsqueeze(-1).expand(-1, MAX_N_ATOMS).reshape(-1)
        )
        atom_residue_indices = cnstr_atoms[:, :, 0]
        atom_atom_indices = cnstr_atoms[:, :, 1]

        atom_offsets = self.pose_stack_block_coord_offset[
            atom_pose_indices, atom_residue_indices.view(-1)
        ].view(atom_residue_indices.size())

        atom_global_indices = atom_offsets + atom_atom_indices

        def score_cnstrs(types, atom_coords):
            cnstr_scores = torch.full(
                (len(types),), 0, dtype=torch.float32, device=coords.device
            )

            for ind, lam in enumerate(lambdas):
                # get the constraints that match this constraint type
                c_inds = torch.nonzero(types == ind)

                print("ATOM_COORDS:", atom_coords[c_inds])

                lam_coords = atom_coords[c_inds].squeeze(1)
                result = lam(lam_coords)
                result = result.unsqueeze(1)
                cnstr_scores[c_inds] += result

            return cnstr_scores

        # atom_coords = coords[cnstrs[:,1], atom_global_indices]
        atom_coords = coords[atom_pose_indices, atom_global_indices.view(-1)].view(
            atom_global_indices.size(0), atom_global_indices.size(1), 3
        )

        nblocks = self.pose_stack_block_coord_offset.size(1)
        nposes = self.pose_stack_block_coord_offset.size(0)

        def distribute_scores(scores, atom_res_inds):
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
                cnstrs[:, 1] * (nblocks**2)
                + atom_res_inds[:, 0] * nblocks
                + atom_res_inds[:, 1]
            )
            indices2 = (
                cnstrs[:, 1] * (nblocks**2)
                + atom_res_inds[:, 1] * nblocks
                + atom_res_inds[:, 0]
            )
            flattened.index_add_(0, indices1, scores / 2)
            flattened.index_add_(0, indices2, scores / 2)
            return block_scores

        scores = score_cnstrs(cnstrs[:, 0], atom_coords)
        scores = distribute_scores(scores, atom_residue_indices)

        if output_block_pair_energies:
            pass
        else:
            # for each pose, sum up the block scores
            scores = torch.sum(scores, (1, 2))

        # wrap this all in an extra dim (the output expects an outer dim to separate sub-terms)
        scores = torch.unsqueeze(scores, 0)
        return scores
