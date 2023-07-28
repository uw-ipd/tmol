import attr

import torch
import pandas

from typing import Sequence

from tmol.types.torch import Tensor

from tmol.chemical.restypes import RefinedResidueType, Residue
from tmol.utility.tensor.common_operations import join_tensors_and_report_real_entries


def residue_types_from_residues(residues):
    rt_dict = {}
    for res in residues:
        if id(res.residue_type) not in rt_dict:
            rt_dict[id(res.residue_type)] = res.residue_type
    return [rt for addr, rt in rt_dict.items()]


@attr.s(auto_attribs=True)
class PackedBlockTypes:
    active_block_types: Sequence[RefinedResidueType]
    restype_index: pandas.Index

    max_n_atoms: int
    n_atoms: Tensor[torch.int32][:]  # dim: n_types
    atom_is_real: Tensor[torch.uint8][:, :]  # dim: n_types x max_n_atoms

    atom_downstream_of_conn: Tensor[torch.int32][:, :, :]

    atom_paths_from_conn: Tensor[torch.int32][:, :, :, 3]

    max_n_torsions: int
    n_torsions: Tensor[torch.int32][:]  # dim: n_types x max_n_tors
    torsion_is_real: Tensor[torch.uint8][:, :]  # dim: n_types, max_n_tors
    # unresolved atom ids for all named torsions in the block types
    torsion_uaids: Tensor[torch.int32][:, :, 3]

    max_n_bonds: int
    n_bonds: Tensor[torch.int32][:]
    bond_is_real: Tensor[torch.bool][:, :]
    # the symmetric / redundant list of bonds
    # indexed by n_types x max_n_bonds x 2
    # where the last dimension is
    # 0: atom1-ind
    # 1: atom2-ind
    bond_indices: Tensor[torch.int32][:, :, 2]

    max_n_conn: int
    n_conn: Tensor[torch.int32][:]
    conn_is_real: Tensor[torch.bool][:, :]
    conn_atom: Tensor[torch.int32][:, :]

    down_conn_inds: Tensor[torch.int32][:]
    up_conn_inds: Tensor[torch.int32][:]

    device: torch.device

    @property
    def n_types(self):
        return len(self.active_block_types)

    @classmethod
    def from_restype_list(
        cls, active_block_types: Sequence[RefinedResidueType], device: torch.device
    ):
        max_n_atoms = cls.count_max_n_atoms(active_block_types)
        n_atoms = cls.count_n_atoms(active_block_types, device)
        atom_is_real = cls.determine_real_atoms(max_n_atoms, n_atoms, device)
        restype_index = pandas.Index([restype.name for restype in active_block_types])
        atom_downstream_of_conn = cls.join_atom_downstream_of_conn(
            active_block_types, device
        )
        atom_paths_from_conn = cls.join_atom_paths_from_conn(active_block_types, device)
        n_torsions, torsion_is_real, torsion_uaids = cls.join_torsion_uaids(
            active_block_types, device
        )
        n_bonds, bond_is_real, bond_indices = cls.join_bond_indices(
            active_block_types, device
        )
        n_conn, conn_is_real, conn_atom = cls.join_conn_indices(
            active_block_types, device
        )
        down_conn_inds, up_conn_inds = cls.join_polymeric_connections(
            active_block_types, device
        )

        return cls(
            active_block_types=active_block_types,
            restype_index=restype_index,
            max_n_atoms=max_n_atoms,
            n_atoms=n_atoms,
            atom_is_real=atom_is_real,
            atom_downstream_of_conn=atom_downstream_of_conn,
            atom_paths_from_conn=atom_paths_from_conn,
            max_n_torsions=torsion_is_real.shape[1],
            n_torsions=n_torsions,
            torsion_is_real=torsion_is_real,
            torsion_uaids=torsion_uaids,
            max_n_bonds=bond_is_real.shape[1],
            n_bonds=n_bonds,
            bond_is_real=bond_is_real,
            bond_indices=bond_indices,
            max_n_conn=conn_is_real.shape[1],
            n_conn=n_conn,
            conn_is_real=conn_is_real,
            conn_atom=conn_atom,
            down_conn_inds=down_conn_inds,
            up_conn_inds=up_conn_inds,
            device=device,
        )

    @classmethod
    def count_max_n_atoms(cls, active_block_types: Sequence[RefinedResidueType]):
        return max(len(restype.atoms) for restype in active_block_types)

    @classmethod
    def count_n_atoms(
        cls, active_block_types: Sequence[RefinedResidueType], device: torch.device
    ):
        return torch.tensor(
            [len(restype.atoms) for restype in active_block_types],
            dtype=torch.int32,
            device=device,
        )

    @classmethod
    def determine_real_atoms(
        cls, max_n_atoms: int, n_atoms: Tensor[torch.int32][:], device: torch.device
    ):
        n_types = n_atoms.shape[0]
        return (
            torch.remainder(
                torch.arange(n_types * max_n_atoms, dtype=torch.int32, device=device),
                max_n_atoms,
            )
            < n_atoms[
                torch.floor_divide(
                    torch.arange(
                        n_types * max_n_atoms, dtype=torch.int64, device=device
                    ),
                    max_n_atoms,
                )
            ]
        ).reshape(n_atoms.shape[0], max_n_atoms)

    @classmethod
    def join_atom_downstream_of_conn(
        cls, active_block_types: Sequence[Residue], device: torch.device
    ):
        n_restypes = len(active_block_types)
        max_n_conn = max(len(rt.connections) for rt in active_block_types)
        max_n_atoms = max(len(rt.atoms) for rt in active_block_types)
        atom_downstream_of_conn = torch.full(
            (n_restypes, max_n_conn, max_n_atoms), -1, dtype=torch.int32, device=device
        )
        for i, rt in enumerate(active_block_types):
            rt_adoc = rt.atom_downstream_of_conn
            atom_downstream_of_conn[
                i, : rt_adoc.shape[0], : rt_adoc.shape[1]
            ] = torch.tensor(rt_adoc, dtype=torch.int32, device=device)
        return atom_downstream_of_conn

    @classmethod
    def join_atom_paths_from_conn(
        clas, active_block_types: Sequence[Residue], device: torch.device
    ):
        n_restypes = len(active_block_types)
        max_n_conn = max(len(bt.connections) for bt in active_block_types)

        atom_paths_from_conn = torch.full(
            (
                n_restypes,
                max_n_conn,
                13,
                3,
            ),  # TODO: constant should be somewhere globally accessible...
            -1,
            dtype=torch.int32,
            device=device,
        )

        for i, bt in enumerate(active_block_types):
            paths = bt.atom_paths_from_conn
            atom_paths_from_conn[i][0 : len(bt.connections)] = torch.tensor(
                paths, device=device
            )

        return atom_paths_from_conn

    @classmethod
    def join_torsion_uaids(cls, active_block_types, device):
        # unresolved atom ids are three integers:
        #  1st integer: an atom index, -1 if the atom is unresolved
        #  2nd integer: the connection index for this block that the unresolved atom
        #               is on ther other side of, -1 if
        #  3rd integer: the number of chemical bonds into the other block that the
        #               unresolved atom is found.

        ordered_torsions = [
            torch.tensor(bt.ordered_torsions.copy(), device=device)
            for bt in active_block_types
        ]
        return join_tensors_and_report_real_entries(ordered_torsions)

        # n_types = len(active_block_types)
        # max_n_tor = max(len(bt.torsion_to_uaids) for bt in active_block_types)
        # torsion_uaids = torch.full(
        #     (n_types, max_n_tor, 4, 3), -1, dtype=torch.int32, device=device
        # )
        #
        # n_torsions = torch.tensor(
        #     [bt.ordered_torsions.shape[0] for bt in active_block_types],
        #     dtype=torch.int32,
        #     device=device,
        # )
        #
        # for i, bt in enumerate(active_block_types):
        #     torsion_uaids[i, : bt.ordered_torsions.shape[0]] = torch.tensor(
        #         bt.ordered_torsions, dtype=torch.int32, device=device
        #     )
        #
        # n_tors_per_bt_arange_expanded = (
        #     torch.arange(max_n_tor, dtype=torch.int32, device=device)
        #     .repeat(n_types)
        #     .view(n_types, max_n_tor)
        # )
        # torsion_is_real = n_tors_per_bt_arange_expanded < n_torsions.unsqueeze(1)
        #
        # return n_torsions, torsion_is_real, torsion_uaids

    @classmethod
    def join_bond_indices(cls, active_block_types, device):
        bond_indices = [
            torch.tensor(bt.bond_indices.copy(), dtype=torch.int32, device=device)
            for bt in active_block_types
        ]
        return join_tensors_and_report_real_entries(bond_indices)

    @classmethod
    def join_conn_indices(cls, active_block_types, device):
        conn_atoms = [
            torch.tensor(bt.ordered_connection_atoms, dtype=torch.int32, device=device)
            for bt in active_block_types
        ]
        return join_tensors_and_report_real_entries(conn_atoms)

    @classmethod
    def join_polymeric_connections(cls, active_block_types, device):
        down_conn_inds = torch.tensor(
            [bt.down_connection_ind for bt in active_block_types],
            dtype=torch.int32,
            device=device,
        )
        up_conn_inds = torch.tensor(
            [bt.up_connection_ind for bt in active_block_types],
            dtype=torch.int32,
            device=device,
        )
        return down_conn_inds, up_conn_inds

    def inds_for_res(self, residues: Sequence[Residue]):
        return self.restype_index.get_indexer(
            [res.residue_type.name for res in residues]
        )

    def inds_for_restypes(self, res_types: Sequence[RefinedResidueType]):
        return self.restype_index.get_indexer(
            [residue_type.name for residue_type in res_types]
        )

    def cpu(self):
        def cpu_equiv(x):
            return x.cpu() if hasattr(x, "cpu") else x

        new_inst = PackedBlockTypes(
            active_block_types=cpu_equiv(self.active_block_types),
            restype_index=cpu_equiv(self.restype_index),
            max_n_atoms=cpu_equiv(self.max_n_atoms),
            n_atoms=cpu_equiv(self.n_atoms),
            atom_is_real=cpu_equiv(self.atom_is_real),
            atom_downstream_of_conn=cpu_equiv(self.atom_downstream_of_conn),
            max_n_torsions=cpu_equiv(self.max_n_torsions),
            n_torsions=cpu_equiv(self.n_torsions),
            torsion_is_real=cpu_equiv(self.torsion_is_real),
            torsion_uaids=cpu_equiv(self.torsion_uaids),
            max_n_bonds=cpu_equiv(self.max_n_bonds),
            n_bonds=cpu_equiv(self.n_bonds),
            bond_is_real=cpu_equiv(self.bond_is_real),
            bond_indices=cpu_equiv(self.bond_indices),
            max_n_conn=cpu_equiv(self.conn_is_real.shape[1]),
            n_conn=cpu_equiv(self.n_conn),
            conn_is_real=cpu_equiv(self.conn_is_real),
            conn_atom=cpu_equiv(self.conn_atom),
            down_conn_inds=cpu_equiv(self.down_conn_inds),
            up_conn_inds=cpu_equiv(self.up_conn_inds),
            device=cpu_equiv(self.device),
        )
        for self_key in self.__dict__:
            if self_key not in new_inst.__dict__:
                setattr(new_inst, self_key, cpu_equiv(self.__dict___[self_key]))
        return new_inst
