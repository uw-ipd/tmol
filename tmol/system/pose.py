import attr
import cattr

import itertools

import numpy
import torch
import pandas
import sparse
import scipy.sparse.csgraph as csgraph

from typing import Sequence, Tuple

from tmol.types.array import NDArray
from tmol.types.torch import Tensor

from .restypes import ResidueType, Residue
from .datatypes import (
    atom_metadata_dtype,
    torsion_metadata_dtype,
    connection_metadata_dtype,
    partial_atom_id_dtype,
)

from tmol.database.chemical import ChemicalDatabase


def residue_types_from_residues(residues):
    rt_dict = {}
    for res in residues:
        if id(res.residue_type) not in rt_dict:
            rt_dict[id(res.residue_type)] = res.residue_type
    return [rt for addr, rt in rt_dict.items()]


@attr.s(auto_attribs=True)
class PackedBlockTypes:
    active_residues: Sequence[ResidueType]
    restype_index: pandas.Index

    max_n_atoms: int
    n_atoms: Tensor(int)[:]  # dim: ntypes

    device: torch.device

    @property
    def n_types(self):
        return len(self.active_residues)

    @classmethod
    def from_restype_list(
        cls,
        active_residues: Sequence[ResidueType],
        chem_db: ChemicalDatabase,
        device=torch.device,
    ):
        max_n_atoms = cls.count_max_n_atoms(active_residues)
        n_atoms = cls.count_n_atoms(active_residues, device)
        restype_index = pandas.Index([restype.name for restype in active_residues])

        return cls(
            active_residues=active_residues,
            restype_index=restype_index,
            max_n_atoms=max_n_atoms,
            n_atoms=n_atoms,
            device=device,
        )

    @classmethod
    def count_max_n_atoms(cls, active_residues: Sequence[ResidueType]):
        return max(len(restype.atoms) for restype in active_residues)

    @classmethod
    def count_n_atoms(
        cls, active_residues: Sequence[ResidueType], device: torch.device
    ):
        return torch.tensor(
            [len(restype.atoms) for restype in active_residues],
            dtype=torch.int32,
            device=device,
        )

    def inds_for_res(self, residues: Sequence[Residue]):
        return self.restype_index.get_indexer(
            [res.residue_type.name for res in residues]
        )


@attr.s(auto_attribs=True)
class Pose:
    packed_block_types: PackedBlockTypes

    residues: Sequence[Residue]

    # n-blocks x max-n-atoms
    coords: NDArray(float)[:, :, 3]

    # n-blocks x n-blocks x max-n-interblock-conn x max-n-interblock-conn
    # Why so much data? Imagine two Bpy-Ala NCAAs binding to a Zn2+ ion.
    # The path distance of N1 atom on Bpy-Ala X and the N2 atom on Bpy-Ala Y
    # is 2: they each are bonded directly to the Zn2+. If we were to suppose
    # either that residue X and Y were only bonded by a single inter-residue
    # connection, or that X's N1 only bonded to Y at a single inter-residue
    # connection, then we would miss the complexity of their arrangement.
    # If we mark N1 of X as bonded to N2 of Y and then asked "what's the
    # bond-separation distance of N1 on X to N1 on Y, we would walk from
    # X-N1 to the Zn2+ to Y-N2 to Y-C6 to Y-C3 and then to Y-N1, tallying
    # a bond separation of 5 instead of 2, and incorrectly count the
    # X-N1--Y-N1 interaction at full strength (and perhaps a collision).
    inter_block_bondsep: NDArray(int)[:, :, :, :]

    # For each block, what is the index of the block type in the PackedBlockTypes
    # structure?
    block_inds: NDArray(int)[:]

    @classmethod
    def from_residues_one_chain(
        cls, res: Sequence[Residue], chem_db: ChemicalDatabase, device: torch.device
    ):
        rt_list = residue_types_from_residues(res)
        packed_block_types = PackedBlockTypes.from_restype_list(
            rt_list, chem_db, device
        )
        inter_block_bondsep = cls.resolve_single_chain_inter_block_bondsep(res)
        coords = cls.pack_coords(packed_block_types, res)
        attached_res = [
            r.attach_to(coords[rind, 0 : len(r.residue_type.atoms), :])
            for rind, r in enumerate(res)
        ]
        block_inds = packed_block_types.inds_for_res(res)

        return cls(
            packed_block_types=packed_block_types,
            residues=attached_res,
            coords=coords,
            inter_block_bondsep=inter_block_bondsep,
            block_inds=block_inds,
        )

    @classmethod
    def resolve_single_chain_inter_block_bondsep(cls, res: Sequence[Residue]):
        # Just a linear set of connections up<->down
        # Logic taken mostly from PackedResidueSystem's from_residues

        max_n_atoms = max(r.coords.shape[0] for r in res)

        ### Index residue connectivity
        # Generate a table of residue connections, with "from" and "to" entries
        # for *both* directions across the connection.

        residue_connections = []
        for i, j in zip(range(len(res) - 1), range(1, len(res))):
            valid_connection = (
                "up" in res[i].residue_type.connection_to_idx
                and "down" in res[j].residue_type.connection_to_idx
            )

            if valid_connection:
                residue_connections.extend(
                    [(i, "up", i + 1, "down"), (i + 1, "down", i, "up")]
                )
            else:
                # TODO add logging
                pass

        for f_i, f_n, t_i, t_n in residue_connections:
            assert (
                f_n in res[f_i].residue_type.connection_to_idx
            ), f"residue missing named connection: {f_n!r} res:\n{res[f_i]}"
            assert (
                t_n in res[t_i].residue_type.connection_to_idx
            ), f"residue missing named connection: {t_n!r} res:\n{res[t_i]}"

        connection_index = pandas.DataFrame.from_records(
            residue_connections,
            columns=pandas.MultiIndex.from_tuples(
                [("from", "resi"), ("from", "cname"), ("to", "resi"), ("to", "cname")]
            ),
        )

        # Unpack the connection metadata table
        connection_metadata = numpy.empty(
            len(connection_index), dtype=connection_metadata_dtype
        )

        connection_metadata["from_residue_index"] = connection_index["from"]["resi"]
        connection_metadata["from_connection_name"] = connection_index["from"]["cname"]

        connection_metadata["to_residue_index"] = connection_index["to"]["resi"]
        connection_metadata["to_connection_name"] = connection_index["to"]["cname"]

        # Generate an index of all the connection atoms in the system,
        # resolving the internal and global index of the connection atoms
        connection_atoms = pandas.DataFrame.from_records(
            [
                (ri, cidx, cname, c_aidx, c_aidx + max_n_atoms * ri)
                for ri, r in enumerate(res)
                for cidx, (cname, c_aidx) in enumerate(
                    r.residue_type.connection_to_idx.items()
                )
            ],
            columns=["resi", "cidx", "cname", "internal_aidx", "aidx"],
        )

        # Merge against the connection table to generate a connection entry
        # with the residue index, the connection name, the local atom index,
        # and the global atom index for the connection by merging on the
        # "cname", "resi" columns.
        #
        # columns:
        # cname resi internal_aidx  aidx
        from_connections = pandas.merge(connection_index["from"], connection_atoms)
        to_connections = pandas.merge(connection_index["to"], connection_atoms)

        for c in from_connections.columns:
            connection_index["from", c] = from_connections[c]
        for c in to_connections.columns:
            connection_index["to", c] = to_connections[c]

        inter_res_bonds = numpy.vstack(
            [from_connections["aidx"].values, to_connections["aidx"].values]
        ).T

        return cls.resolve_inter_block_bondsep(res, inter_res_bonds, connection_atoms)

    @classmethod
    def resolve_inter_block_bondsep(
        cls,
        res: Sequence[Residue],
        inter_res_bonds: NDArray(int)[:, 2],
        connection_atoms: pandas.DataFrame,
    ):
        n_res = len(res)
        max_n_atoms = max(r.coords.shape[0] for r in res)
        max_n_conn = max(len(r.residue_type.connections) for r in res)

        ### Generate the bond graph
        #
        intra_res_bonds = numpy.concatenate(
            [
                r.residue_type.bond_indices + rind * max_n_atoms
                for rind, r in zip(numpy.arange(len(res), dtype=int), res)
            ]
        )

        bonds = numpy.concatenate([intra_res_bonds, inter_res_bonds])
        bond_graph = sparse.COO(
            bonds.T,
            data=numpy.full(len(bonds), True),
            shape=(max_n_atoms * n_res, max_n_atoms * n_res),
            cache=True,
        )

        min_bond_dist = csgraph.dijkstra(
            bond_graph.tocsr(), directed=False, unweighted=True, limit=6
        )

        # ok, now we want to ask: for each pair of connections, what is
        # the minimum number of chemical bonds that connects them.

        inter_block_bondsep = numpy.full(
            (n_res, n_res, max_n_conn, max_n_conn), 6, dtype=numpy.int32
        )
        min_bond_dist[min_bond_dist == numpy.inf] = 6

        ainds_conn1 = connection_atoms["aidx"].values
        ainds_conn2 = connection_atoms["aidx"].values
        rinds_conn1 = connection_atoms["resi"].values
        rinds_conn2 = connection_atoms["resi"].values
        cinds_conn1 = connection_atoms["cidx"].values
        cinds_conn2 = connection_atoms["cidx"].values

        ainds_all_pairs = numpy.array(
            list(itertools.product(ainds_conn1, ainds_conn2)), dtype=int
        )
        rinds_all_pairs = numpy.array(
            list(itertools.product(rinds_conn1, rinds_conn2)), dtype=int
        )
        cinds_all_pairs = numpy.array(
            list(itertools.product(cinds_conn1, cinds_conn2)), dtype=int
        )

        inter_block_bondsep[
            rinds_all_pairs[:, 0],
            rinds_all_pairs[:, 1],
            cinds_all_pairs[:, 0],
            cinds_all_pairs[:, 1],
        ] = min_bond_dist[ainds_all_pairs[:, 0], ainds_all_pairs[:, 1]]

        return inter_block_bondsep

    @classmethod
    def pack_coords(cls, packed_block_types: PackedBlockTypes, res: Sequence[Residue]):
        coords = numpy.zeros(
            (len(res), packed_block_types.max_n_atoms, 3), dtype=numpy.float64
        )
        for i, r in enumerate(res):
            coords[i, : len(r.residue_type.atoms)] = r.coords
        return coords


@attr.s(auto_attribs=True)
class Poses:

    packed_block_types: PackedBlockTypes
    residues: Sequence[Sequence[Residue]]

    coords: NDArray(float)[:, :, :, 3]
    inter_block_bondsep: NDArray(int)[:, :, :, :, :]
    block_inds: NDArray(float)[:, :]

    @classmethod
    def from_poses(
        cls, poses: Sequence[Pose], chem_db: ChemicalDatabase, device: torch.device
    ):
        all_res = [res for pose in poses for res in pose.residues]
        restypes = residue_types_from_residues(all_res)
        packed_block_types = PackedBlockTypes.from_restype_list(
            restypes, chem_db, device
        )

        max_n_blocks = max(len(pose.residues) for pose in poses)
        coords = cls.pack_coords(packed_block_types, poses, max_n_blocks)

        residues = [
            [
                r.attach_to(coords[i, j, : len(r.residue_type.atoms)])
                for j, r in enumerate(pose.residues)
            ]
            for i, pose in enumerate(poses)
        ]

        inter_block_bondsep = cls.interblock_bondsep_from_poses(
            packed_block_types, poses, max_n_blocks
        )
        block_inds = cls.resolve_block_inds(packed_block_types, poses, max_n_blocks)

        return cls(
            packed_block_types=packed_block_types,
            residues=residues,
            coords=coords,
            inter_block_bondsep=inter_block_bondsep,
            block_inds=block_inds,
        )

    @classmethod
    def pack_coords(
        cls,
        packed_block_types: PackedBlockTypes,
        poses: Sequence[Pose],
        max_n_blocks: int,
    ):
        coords = numpy.zeros(
            (len(poses), max_n_blocks, packed_block_types.max_n_atoms, 3),
            dtype=numpy.float64,
        )
        for i, p in enumerate(poses):
            for j, r in enumerate(p.residues):
                coords[i, j, : len(r.residue_type.atoms)] = r.coords
        return coords

    @classmethod
    def interblock_bondsep_from_poses(
        cls,
        packed_block_types: PackedBlockTypes,
        poses: Sequence[Pose],
        max_n_blocks: int,
    ):
        n_poses = len(poses)
        max_n_conn = max(
            len(rt.connections) for rt in packed_block_types.active_residues
        )
        inter_block_bondsep = numpy.full(
            (n_poses, max_n_blocks, max_n_blocks, max_n_conn, max_n_conn),
            6,
            dtype=numpy.int32,
        )
        for i, pose in enumerate(poses):
            i_nblocks = len(pose.residues)
            i_nconn = pose.inter_block_bondsep.shape[2]
            inter_block_bondsep[
                i, :i_nblocks, :i_nblocks, :i_nblocks, :i_nblocks
            ] = pose.inter_block_bondsep
        return inter_block_bondsep

    @classmethod
    def resolve_block_inds(
        cls,
        packed_block_types: PackedBlockTypes,
        poses: Sequence[Pose],
        max_n_blocks: int,
    ):
        block_inds = numpy.full((len(poses), max_n_blocks), -1, dtype=numpy.int32)
        for i, pose in enumerate(poses):
            block_inds[i, : len(pose.residues)] = packed_block_types.inds_for_res(
                pose.residues
            )
        return block_inds
