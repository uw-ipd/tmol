import attr
import cattr

import itertools

import numpy
import pandas
import sparse
import scipy.sparse.csgraph as csgraph

from typing import Sequence, Tuple

from tmol.types.array import NDArray

from .restypes import ResidueType, Residue
from .datatypes import (
    atom_metadata_dtype,
    torsion_metadata_dtype,
    connection_metadata_dtype,
    partial_atom_id_dtype,
)

from tmol.database.chemical import ChemicalDatabase


@attr.s(auto_attribs=True)
class PackedBlockTypes:
    active_residues: Sequence[ResidueType]

    max_n_atoms: int
    n_atoms: NDArray(int)[:]  # ntypes

    @property
    def n_types(self):
        return len(self.active_residues)

    @classmethod
    def from_restype_list(
        cls, active_residues: Sequence[ResidueType], chem_db: ChemicalDatabase
    ):
        max_n_atoms = cls.count_max_n_atoms(active_residues)
        n_atoms = cls.count_n_atoms(active_residues)

        return cls(
            active_residues=active_residues, max_n_atoms=max_n_atoms, n_atoms=n_atoms
        )

    @classmethod
    def count_max_n_atoms(cls, active_residues: Sequence[ResidueType]):
        return max(len(restype.atoms) for restype in active_residues)

    @classmethod
    def count_n_atoms(cls, active_residues: Sequence[ResidueType]):
        return numpy.array(
            [len(restype.atoms) for restype in active_residues], dtype=numpy.int32
        )


@attr.s(auto_attribs=True)
class Pose:
    packed_block_types: PackedBlockTypes
    coords: NDArray(float)[:, :, 3]
    inter_block_separation: NDArray(int)[:, :, :, :]

    @classmethod
    def from_residues_one_chain(cls, res: Sequence[Residue]):
        inter_block_separation = cls.resolve_single_chain_inter_block_separation(res)

    @classmethod
    def resolve_single_chain_inter_block_separation(cls, res: Sequence[Residue]):
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

        return cls.resolve_inter_block_separation(
            res, inter_res_bonds, connection_atoms
        )

    @classmethod
    def resolve_inter_block_separation(
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
                r.residue_type.bond_indicies + rind * max_n_atoms
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

        inter_block_separation = numpy.full(
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

        inter_block_separation[
            rinds_all_pairs[:, 0],
            rinds_all_pairs[:, 1],
            cinds_all_pairs[:, 0],
            cinds_all_pairs[:, 1],
        ] = min_bond_dist[ainds_all_pairs[:, 0], ainds_all_pairs[:, 1]]

        return inter_block_separation


@attr.s(auto_attribs=True)
class Poses:

    packed_block_types: PackedBlockTypes

    coords: NDArray(float)[:, :, :, 3]
    block_types: NDArray(float)[:, :]
    inter_block_separation: NDArray(int)[:, :, :, :, :]

    @classmethod
    def from_residues_one_chain(cls, res: Sequence[Residue]):
        """Intialize a single pose"""
