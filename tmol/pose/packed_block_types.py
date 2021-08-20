import attr
import toolz

import itertools

import numpy
import torch
import pandas
import sparse
import scipy.sparse.csgraph as csgraph

from typing import Sequence, Tuple

from tmol.types.array import NDArray
from tmol.types.torch import Tensor

from tmol.chemical.restypes import RefinedResidueType, Residue
from tmol.system.datatypes import connection_metadata_dtype


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
    atom_is_real: Tensor[torch.uint8][:, :]  # dim: n_types

    atom_downstream_of_conn: Tensor[torch.int32][:, :, :]

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

        return cls(
            active_block_types=active_block_types,
            restype_index=restype_index,
            max_n_atoms=max_n_atoms,
            n_atoms=n_atoms,
            atom_is_real=atom_is_real,
            atom_downstream_of_conn=atom_downstream_of_conn,
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

    def inds_for_res(self, residues: Sequence[Residue]):
        return self.restype_index.get_indexer(
            [res.residue_type.name for res in residues]
        )

    def inds_for_restypes(self, res_types: Sequence[RefinedResidueType]):
        return self.restype_index.get_indexer(
            [residue_type.name for residue_type in res_types]
        )
