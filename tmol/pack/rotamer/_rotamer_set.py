import attr
import numpy
import torch
from tmol.types import ValidateAttrs
from tmol.types import Tensor


@attr.s(auto_attribs=True, slots=True, frozen=True)
class RotamerSet(ValidateAttrs):
    n_rots_for_pose: Tensor[torch.int64][:]
    rot_offset_for_pose: Tensor[torch.int64][:]
    n_rots_for_block: Tensor[torch.int64][:, :]
    rot_offset_for_block: Tensor[torch.int64][:, :]
    pose_for_rot: Tensor[torch.int64][:]
    block_type_ind_for_rot: Tensor[torch.int64][:]
    block_ind_for_rot: Tensor[torch.int32][:]
    coord_offset_for_rot: Tensor[torch.int32][:]
    coords: Tensor[torch.float32][:, 3]

    first_rot_block_type: Tensor[torch.int64][:, :] = attr.ib()

    @first_rot_block_type.default
    def _block_type_for_first_rot_for_block(self):
        block_type_for_first_rot_for_block = torch.full_like(
            self.rot_offset_for_block, -1
        )
        does_block_type_have_rots = self.n_rots_for_block != 0
        block_type_for_first_rot_for_block[does_block_type_have_rots] = (
            self.block_type_ind_for_rot[
                self.rot_offset_for_block[does_block_type_have_rots]
            ]
        )
        return block_type_for_first_rot_for_block

    max_n_rots_per_pose: int = attr.ib()

    @max_n_rots_per_pose.default
    def _max_n_rots_per_pose(self):
        return int(torch.max(self.n_rots_for_pose).cpu().item())

    pose_ind_for_atom: Tensor[torch.int64][:] = attr.ib()

    @pose_ind_for_atom.default
    def _pose_ind_for_atom(self):
        n_atoms = self.coords.shape[0]
        pifa = torch.zeros((n_atoms,), dtype=torch.int64, device=self.coords.device)
        # mark the first atom for the first rotamer in each pose after pose 0
        pifa[self.coord_offset_for_rot[self.rot_offset_for_pose[1:]]] = 1
        pifa = torch.cumsum(pifa, dim=0)
        return pifa

    @property
    def n_rotamers_total(self):
        return self.block_ind_for_rot.shape[0]

    def to_atom_array(self, pbt):
        import biotite.structure as struc
        from tmol.io import get_element_from_atom_name

        n_rots = self.n_rotamers_total
        coords = self.coords.cpu().numpy()

        atoms = []
        model_ids = []

        for i in range(n_rots):
            i_offset = int(self.coord_offset_for_rot[i])
            block_type_ind = int(self.block_type_ind_for_rot[i])
            n_atoms = int(pbt.n_atoms[block_type_ind])
            bt = pbt.active_block_types[block_type_ind]
            res_id = int(self.block_ind_for_rot[i]) + 1

            for j in range(n_atoms):
                atom_name = bt.atoms[j].name
                atoms.append(
                    struc.Atom(
                        coords[i_offset + j],
                        chain_id="A",
                        res_id=res_id,
                        res_name=bt.name3,
                        atom_name=atom_name,
                        element=get_element_from_atom_name(atom_name),
                    )
                )
                model_ids.append(i)

        atom_array = struc.array(atoms)
        atom_array.set_annotation("model", numpy.array(model_ids, dtype=numpy.int32))
        return atom_array
