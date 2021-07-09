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

from .restypes import RefinedResidueType, Residue
from .datatypes import connection_metadata_dtype


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
                torch.div(
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


@attr.s(auto_attribs=True)
class Pose:
    packed_block_types: PackedBlockTypes

    residues: Sequence[Residue]

    # n-blocks x max-n-atoms
    residue_coords: NDArray[numpy.float64][:, :, 3]
    coords: Tensor[torch.float32][:, :, 3]

    # which blocks are connected to which other blocks by which connection
    # n_blocks x max_n_connections x [other_res, other_res_conn]
    inter_residue_connections: Tensor[torch.int32][:, :, 2]

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
    inter_block_bondsep: Tensor[torch.int32][:, :, :, :]

    # For each block, what is the index of the block type in the PackedBlockTypes
    # structure?
    block_type_ind: Tensor[torch.int32][:]

    device: torch.device

    @classmethod
    def from_residues_one_chain(cls, res: Sequence[Residue], device: torch.device):
        rt_list = residue_types_from_residues(res)
        packed_block_types = PackedBlockTypes.from_restype_list(rt_list, device)
        residue_connections = cls.resolve_single_chain_connections(res)
        inter_residue_connections = cls.create_inter_residue_connections(
            res, residue_connections, device
        )
        inter_block_bondsep = cls.determine_inter_block_bondsep(
            res, residue_connections, device
        )

        residue_coords = cls.pack_coords(packed_block_types, res)
        coords = torch.tensor(residue_coords, dtype=torch.float32, device=device)
        attached_res = [
            r.attach_to(residue_coords[rind, 0 : len(r.residue_type.atoms), :])
            for rind, r in enumerate(res)
        ]
        block_type_ind = torch.tensor(
            packed_block_types.inds_for_res(res), dtype=torch.int32, device=device
        )

        return cls(
            packed_block_types=packed_block_types,
            residues=attached_res,
            coords=coords,
            residue_coords=residue_coords,
            inter_residue_connections=inter_residue_connections,
            inter_block_bondsep=inter_block_bondsep,
            block_type_ind=block_type_ind,
            device=device,
        )

    @classmethod
    def resolve_single_chain_connections(cls, res: Sequence[Residue]):
        # return a list of (int,str,int,str) quadrouples that say residue
        # i is connected to residue i+1 from it's "up" connection to
        # residue i+1's "down" connection and vice versa for all i

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

        return residue_connections

    # @validate_args
    @classmethod
    def create_inter_residue_connections(
        cls,
        res: Sequence[Residue],
        residue_connections: Sequence[Tuple[int, str, int, str]],
        device: torch.device,
    ) -> Tensor[torch.int32][:, :, 2]:
        """Return a torch tensor of integer-indices describing
        which residues are bound to which other residues through
        which connections using the indices of those
        connections and not their names
        """

        max_n_conn = max(len(r.residue_type.connections) for r in res)
        connection_inds = pandas.DataFrame.from_records(
            [
                (r_ind, conn.name, c_ind)
                for r_ind, r in enumerate(res)
                for c_ind, conn in enumerate(r.residue_type.connections)
            ],
            columns=["resi", "cname", "cind"],
        )

        inter_conn_inds = toolz.reduce(
            toolz.curry(pandas.merge)(how="left", copy=False),
            (
                pandas.DataFrame.from_records(
                    residue_connections,
                    columns=["from_res_ind", "from_conn", "to_res_ind", "to_conn"],
                ),
                connection_inds.rename(
                    columns={
                        "resi": "from_res_ind",
                        "cname": "from_conn",
                        "cind": "from_conn_ind",
                    }
                ),
                connection_inds.rename(
                    columns={
                        "resi": "to_res_ind",
                        "cname": "to_conn",
                        "cind": "to_conn_ind",
                    }
                ),
            ),
        )[["from_res_ind", "from_conn_ind", "to_res_ind", "to_conn_ind"]].values.astype(
            numpy.int32
        )

        # .values is deprecated, and to_numpy  is preferred, but the
        # version of pandas I'm currently using does not yet have to_numpy
        #
        # .to_numpy(
        #    dtype=numpy.int32
        # )

        inter_residue_connections = numpy.full(
            (len(res), max_n_conn, 2), -1, dtype=numpy.int32
        )
        inter_residue_connections[
            inter_conn_inds[:, 0], inter_conn_inds[:, 1], :
        ] = inter_conn_inds[:, 2:]
        inter_residue_connections = torch.tensor(
            inter_residue_connections, dtype=torch.int32, device=device
        )
        return inter_residue_connections

    @classmethod
    def determine_inter_block_bondsep(
        cls,
        res: Sequence[Residue],
        residue_connections: Sequence[Tuple[int, str, int, str]],
        device: torch.device,
    ) -> Tensor[torch.int32][:, :, :, :]:
        # Just a linear set of connections up<->down
        # Logic taken mostly from PackedResidueSystem's from_residues

        max_n_atoms = max(r.coords.shape[0] for r in res)

        ### Index residue connectivity
        # Generate a table of residue connections, with "from" and "to" entries
        # for *both* directions across the connection.

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

        return cls.resolve_inter_block_bondsep(
            res, inter_res_bonds, connection_atoms, device
        )

    @classmethod
    def resolve_inter_block_bondsep(
        cls,
        res: Sequence[Residue],
        inter_res_bonds: NDArray[int][:, 2],
        connection_atoms: pandas.DataFrame,
        device: torch.device,
    ) -> Tensor[torch.int32][:, :, :, :]:
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

        return torch.tensor(inter_block_bondsep, dtype=torch.int32, device=device)

    @classmethod
    def pack_coords(cls, packed_block_types: PackedBlockTypes, res: Sequence[Residue]):
        coords = numpy.zeros(
            (len(res), packed_block_types.max_n_atoms, 3), dtype=numpy.float64
        )
        for i, r in enumerate(res):
            coords[i, : len(r.residue_type.atoms)] = r.coords
        return coords

    @property
    def n_residues(self):
        return len(self.residues)

    @property
    def max_n_atoms(self):
        return self.packed_block_types.max_n_atoms


@attr.s(auto_attribs=True)
class Poses:

    packed_block_types: PackedBlockTypes
    residues: Sequence[Sequence[Residue]]

    residue_coords: NDArray[numpy.float32][:, :, :, 3]
    coords: Tensor[torch.float32][:, :, :, 3]

    inter_residue_connections: Tensor[torch.int32][:, :, :, 2]
    inter_block_bondsep: Tensor[torch.int32][:, :, :, :, :]
    block_type_ind: Tensor[torch.int32][:, :]

    device: torch.device

    @classmethod
    def from_poses(cls, poses: Sequence[Pose], device: torch.device):
        all_res = [res for pose in poses for res in pose.residues]
        restypes = residue_types_from_residues(all_res)
        packed_block_types = PackedBlockTypes.from_restype_list(restypes, device)

        max_n_blocks = max(len(pose.residues) for pose in poses)
        coords = cls.pack_coords(packed_block_types, poses, max_n_blocks, device)
        residue_coords = coords.cpu().numpy().astype(numpy.float64)

        residues = [
            [
                r.attach_to(residue_coords[i, j, : len(r.residue_type.atoms)])
                for j, r in enumerate(pose.residues)
            ]
            for i, pose in enumerate(poses)
        ]

        inter_residue_connections = cls.inter_residue_connections_from_poses(
            packed_block_types, poses, max_n_blocks, device
        )
        inter_block_bondsep = cls.interblock_bondsep_from_poses(
            packed_block_types, poses, max_n_blocks, device
        )
        block_type_ind = cls.resolve_block_type_ind(
            packed_block_types, poses, max_n_blocks, device
        )

        return cls(
            packed_block_types=packed_block_types,
            residues=residues,
            coords=coords,
            residue_coords=residue_coords,
            inter_residue_connections=inter_residue_connections,
            inter_block_bondsep=inter_block_bondsep,
            block_type_ind=block_type_ind,
            device=device,
        )

    @classmethod
    def pack_coords(
        cls,
        packed_block_types: PackedBlockTypes,
        poses: Sequence[Pose],
        max_n_blocks: int,
        device: torch.device,
    ) -> Tensor[torch.float32][:, :, :, 3]:
        coords = torch.zeros(
            (len(poses), max_n_blocks, packed_block_types.max_n_atoms, 3),
            dtype=torch.float32,
            device=device,
        )
        for i, p in enumerate(poses):
            coords[i, : p.n_residues, : p.max_n_atoms] = p.coords
        return coords

    @classmethod
    def inter_residue_connections_from_poses(
        cls,
        packed_block_types: PackedBlockTypes,
        poses: Sequence[Pose],
        max_n_blocks: int,
        device: torch.device,
    ) -> Tensor[torch.int32][:, :, :, 2]:
        n_poses = len(poses)
        max_n_conn = max(
            len(rt.connections) for rt in packed_block_types.active_block_types
        )
        inter_residue_connections = torch.full(
            (n_poses, max_n_blocks, max_n_conn, 2), -1, dtype=torch.int32, device=device
        )
        for i, pose in enumerate(poses):
            inter_residue_connections[
                i,
                : pose.inter_residue_connections.shape[0],
                : pose.inter_residue_connections.shape[1],
            ] = pose.inter_residue_connections
        return inter_residue_connections

    @classmethod
    def interblock_bondsep_from_poses(
        cls,
        packed_block_types: PackedBlockTypes,
        poses: Sequence[Pose],
        max_n_blocks: int,
        device: torch.device,
    ) -> Tensor[torch.int32][:, :, :, :, :]:
        n_poses = len(poses)
        max_n_conn = max(
            len(rt.connections) for rt in packed_block_types.active_block_types
        )
        inter_block_bondsep = torch.full(
            (n_poses, max_n_blocks, max_n_blocks, max_n_conn, max_n_conn),
            6,
            dtype=torch.int32,
            device=device,
        )
        for i, pose in enumerate(poses):
            i_nblocks = len(pose.residues)
            i_nconn = pose.inter_block_bondsep.shape[2]
            inter_block_bondsep[
                i, :i_nblocks, :i_nblocks, :i_nconn, :i_nconn
            ] = pose.inter_block_bondsep
        return inter_block_bondsep

    @classmethod
    def resolve_block_type_ind(
        cls,
        packed_block_types: PackedBlockTypes,
        poses: Sequence[Pose],
        max_n_blocks: int,
        device=torch.device,
    ):
        block_type_ind = torch.full(
            (len(poses), max_n_blocks), -1, dtype=torch.int32, device=device
        )
        for i, pose in enumerate(poses):
            block_type_ind[i, : len(pose.residues)] = torch.tensor(
                packed_block_types.inds_for_res(pose.residues),
                dtype=torch.int32,
                device=device,
            )
        return block_type_ind
