import attr
import numpy
import torch
from tmol.types.attrs import ValidateAttrs
from tmol.types.torch import Tensor


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

    def write_pdb(self, pbt) -> str:
        from io import StringIO
        from tmol.io.pdb_parsing import atom_record_dtype, to_pdb_lines

        n_atoms_total = self.coords.shape[0]
        atom_records = numpy.empty((n_atoms_total,), dtype=atom_record_dtype)
        n_rots = self.block_type_ind_for_rot.shape[0]

        for i in range(n_rots):
            i_offset = self.coord_offset_for_rot[i]
            block_type_ind = self.block_type_ind_for_rot[i]
            n_atoms_for_block_type = pbt.n_atoms[block_type_ind]
            for j in range(n_atoms_for_block_type):
                atom_records[i_offset + j]["modeli"] = i
                atom_records[i_offset + j]["chaini"] = 0
                atom_records[i_offset + j]["resi"] = self.block_ind_for_rot[i] + 1
                atom_records[i_offset + j]["atomi"] = j + 1
                atom_records[i_offset + j]["model"] = f"M{i:05d}"
                atom_records[i_offset + j]["chain"] = "A"
                atom_records[i_offset + j]["resn"] = pbt.active_block_types[
                    block_type_ind
                ].name3
                atom_records[i_offset + j]["atomn"] = (
                    pbt.active_block_types[block_type_ind].atoms[j].name
                )
                atom_records[i_offset + j]["x"] = self.coords[i_offset + j, 0].item()
                atom_records[i_offset + j]["y"] = self.coords[i_offset + j, 1].item()
                atom_records[i_offset + j]["z"] = self.coords[i_offset + j, 2].item()
                atom_records[i_offset + j]["insert"] = ""
                atom_records[i_offset + j]["occupancy"] = 1.0
                atom_records[i_offset + j]["b"] = 0.0

        buf = StringIO()
        buf.writelines(to_pdb_lines(atom_records))
        return buf.getvalue()
