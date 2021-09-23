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

from tmol.chemical.restypes import (
    RefinedResidueType,
    Residue,
    find_simple_polymeric_connections,
)
from tmol.pose.packed_block_types import PackedBlockTypes, residue_types_from_residues
from tmol.system.datatypes import connection_metadata_dtype
from tmol.utility.tensor.common_operations import exclusive_cumsum1d


@attr.s(auto_attribs=True)
class PoseStack:

    packed_block_types: PackedBlockTypes
    residues: Sequence[Sequence[Residue]]

    residue_coords: NDArray[numpy.float32][:, :, :, 3]

    # coordinates are held as [n-poses x max-n-atoms x 3]
    # where the offset for each residue are held in the
    # block_coord_offset tensor [n-poses x max-n-blocks]
    coords: Tensor[torch.float32][:, :, 3]
    block_coord_offset: Tensor[torch.int32][:, :]

    inter_residue_connections: Tensor[torch.int32][:, :, :, 2]
    inter_block_bondsep: Tensor[torch.int32][:, :, :, :, :]
    block_type_ind: Tensor[torch.int32][:, :]

    device: torch.device

    def __len__(self):
        """return the number of PoseStack held in this stack"""
        return len(self.residues)

    @classmethod
    def one_structure_from_polymeric_residues(
        cls, res: Sequence[Residue], device: torch.device
    ):
        residue_connections = find_simple_polymeric_connections(res)
        return cls.one_structure_from_residues_and_connections(
            res, residue_connections, device
        )

    @classmethod
    def one_structure_from_residues_and_connections(
        cls,
        res: Sequence[Residue],
        residue_connections: Sequence[Tuple[int, str, int, str]],
        device: torch.device,
    ):
        rt_list = residue_types_from_residues(res)
        packed_block_types = PackedBlockTypes.from_restype_list(rt_list, device)

        inter_residue_connections = cls.create_inter_residue_connections(
            res, residue_connections, device
        )
        inter_block_bondsep = cls.determine_single_structure_inter_block_bondsep(
            res, residue_connections, device
        )

        block_type_ind = torch.tensor(
            packed_block_types.inds_for_res(res), dtype=torch.int32, device=device
        ).unsqueeze(0)

        residue_coords, block_coord_offset, attached_res = cls.pack_residue_coords(
            packed_block_types, block_type_ind, [res]
        )
        coords = torch.tensor(residue_coords, dtype=torch.float32, device=device)
        attached_res = cls.attach_residues(block_coord_offset, [res], residue_coords)

        return cls(
            packed_block_types=packed_block_types,
            residues=attached_res,
            residue_coords=residue_coords,
            coords=coords,
            block_coord_offset=block_coord_offset,
            inter_residue_connections=torch.unsqueeze(inter_residue_connections, 0),
            inter_block_bondsep=torch.unsqueeze(inter_block_bondsep, 0),
            block_type_ind=block_type_ind,
            device=device,
        )

    @classmethod
    def from_poses(cls, pose_stacks: Sequence["PoseStack"], device: torch.device):
        all_res = [
            res
            for pose_stack in pose_stacks
            for reslist in pose_stack.residues
            for res in reslist
        ]
        restypes = residue_types_from_residues(all_res)
        packed_block_types = PackedBlockTypes.from_restype_list(restypes, device)

        max_n_blocks = max(
            len(reslist)
            for pose_stack in pose_stacks
            for reslist in pose_stack.residues
        )
        coords, block_coord_offset = cls.pack_pose_stack_coords(
            packed_block_types, pose_stacks, max_n_blocks, device
        )
        residue_coords = coords.cpu().numpy().astype(numpy.float64)

        ps_offset = exclusive_cumsum1d(
            torch.tensor([len(ps) for ps in pose_stacks], dtype=torch.int64)
        )
        block_coord_offset_cpu = block_coord_offset.cpu()
        residues = cls.attach_residues(
            block_coord_offset_cpu,
            [
                one_pose_residues
                for pose_stack in pose_stacks
                for one_pose_residues in pose_stack.residues
            ],
            residue_coords,
        )

        inter_residue_connections = cls.inter_residue_connections_from_pose_stacks(
            packed_block_types, pose_stacks, ps_offset, max_n_blocks, device
        )
        inter_block_bondsep = cls.interblock_bondsep_from_pose_stacks(
            packed_block_types, pose_stacks, ps_offset, max_n_blocks, device
        )
        block_type_ind = cls.resolve_block_type_ind(
            packed_block_types, pose_stacks, ps_offset, max_n_blocks, device
        )

        return cls(
            packed_block_types=packed_block_types,
            residues=residues,
            residue_coords=residue_coords,
            coords=coords,
            block_coord_offset=block_coord_offset,
            inter_residue_connections=inter_residue_connections,
            inter_block_bondsep=inter_block_bondsep,
            block_type_ind=block_type_ind,
            device=device,
        )

    def rebuild_with_new_packed_block_types(self, packed_block_types) -> "PoseStack":
        """Create a new PoseStack object replacing the existing PackedBlockTypes
        object with a new one, and then rebuilding the other data members that
        depend on it.
        """
        # The input packed_block_types must contain the block types of
        # the PoseStack's existing set of in-use residue types (but not necessarily
        # all of the block types that its PackedBlockTypes object holds)
        for pose_res_list in self.residues:
            for res in pose_res_list:
                assert res.residue_type in packed_block_types.active_block_types

        coords = self.coords.clone()

        block_type_ind = torch.full_like(self.block_type_ind, -1)
        # this could be more efficient if we mapped orig_block_type to new_block_type
        for i, res in enumerate(self.residues):
            block_type_ind[i, : len(res)] = torch.tensor(
                packed_block_types.inds_for_res(res),
                dtype=torch.int32,
                device=self.device,
            )

        residue_coords = coords.to(dtype=torch.float64).cpu().numpy()

        residues = self.attach_residues(
            self.block_coord_offset.cpu(), self.residues, residue_coords
        )

        return PoseStack(
            packed_block_types=packed_block_types,
            residues=residues,
            residue_coords=residue_coords,
            coords=coords,
            block_coord_offset=self.block_coord_offset,
            inter_residue_connections=self.inter_residue_connections,
            inter_block_bondsep=self.inter_block_bondsep,
            block_type_ind=block_type_ind,
            device=self.device,
        )

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
    def determine_single_structure_inter_block_bondsep(
        cls,
        res: Sequence[Residue],
        residue_connections: Sequence[Tuple[int, str, int, str]],
        device: torch.device,
    ) -> Tensor[torch.int32][:, :, :, :]:
        """
        With a list of blocks (residues) from a single structure,
        construct the set of chemical-bond-path-distances between all pairs of
        inter-block connection points in an
        nblock x nblock x max-n-conn x max-n-conn tensor.
        This code relies on the named residue connections (strings) of
        the (assumed) polymer residues, "up" and "down," and is therefore not
        as fast as perhaps an integer-based method. That said, it uses pandas
        for the string comparison logic and is efficient on that level.
        """

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

        # # Unpack the connection metadata table
        # connection_metadata = numpy.empty(
        #     len(connection_index), dtype=connection_metadata_dtype
        # )
        #
        # connection_metadata["from_residue_index"] = connection_index["from"]["resi"]
        # connection_metadata["from_connection_name"] = connection_index["from"]["cname"]
        #
        # connection_metadata["to_residue_index"] = connection_index["to"]["resi"]
        # connection_metadata["to_connection_name"] = connection_index["to"]["cname"]

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

        # for c in from_connections.columns:
        #     connection_index["from", c] = from_connections[c]
        # for c in to_connections.columns:
        #     connection_index["to", c] = to_connections[c]

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
        """
        Resolve chemical-bond path distances between all pairs of interblock connections
        in a single structure given a list of the interblock chemical bonds.
        """

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
    def pack_residue_coords(
        cls,
        packed_block_types: PackedBlockTypes,
        block_type_ind: Tensor[torch.int32][:, :],
        res: Sequence[Sequence[Residue]],
    ):
        device = block_type_ind.device
        btind64 = block_type_ind.to(torch.int64)
        n_ats = torch.zeros(block_type_ind.shape, dtype=torch.int32, device=device)

        n_ats[btind64 != -1] = packed_block_types.n_atoms[btind64[btind64 != -1]]
        n_ats_inccumsum = torch.cumsum(n_ats, dim=1, dtype=torch.int32)
        max_n_ats = torch.max(n_ats_inccumsum[:, -1])
        block_coord_offset = torch.cat(
            (
                torch.zeros(
                    (block_type_ind.shape[0], 1), dtype=torch.int32, device=device
                ),
                n_ats_inccumsum[:, :-1],
            ),
            dim=1,
        )
        block_coord_offset_cpu = block_coord_offset.cpu()

        coords = numpy.zeros((len(res), max_n_ats, 3), dtype=numpy.float64)
        attached_res = []
        for i, rlist in enumerate(res):
            attached_res.append([])
            for j, r in enumerate(rlist):
                j_offset = block_coord_offset_cpu[i, j]
                coords[i, j_offset : (j_offset + len(r.residue_type.atoms))] = r.coords
                attached_res[i].append(
                    r.attach_to(
                        coords[i, j_offset : (j_offset + len(r.residue_type.atoms))]
                    )
                )
        return coords, block_coord_offset, attached_res

    @classmethod
    def pack_pose_stack_coords(
        cls,
        packed_block_types: PackedBlockTypes,
        pose_stacks: Sequence["PoseStack"],
        max_n_blocks: int,
        device: torch.device,
    ) -> Tensor[torch.float32][:, :, :, 3]:

        n_poses = sum(len(ps) for ps in pose_stacks)
        max_n_atoms = max(ps.coords.shape[1] for ps in pose_stacks)
        max_n_blocks = max(ps.block_coord_offset.shape[1] for ps in pose_stacks)
        coords = torch.zeros(
            (n_poses, max_n_atoms, 3), dtype=torch.float32, device=device
        )
        block_coord_offset = torch.zeros(
            (n_poses, max_n_blocks), dtype=torch.int32, device=device
        )
        count = 0
        for p in pose_stacks:
            coords[count : (count + len(p)), : p.coords.shape[1]] = p.coords
            block_coord_offset[
                count : (count + len(p)), : p.block_coord_offset.shape[1]
            ] = p.block_coord_offset
            count += len(p)
        return coords, block_coord_offset

    @classmethod
    def attach_residues(
        cls,
        block_coord_offset_cpu: Tensor[torch.int32][:],
        orig_res: Sequence[Sequence[Residue]],
        residue_coords: NDArray[numpy.float64][:, :, 3],
    ) -> Sequence[Sequence[Residue]]:

        return [
            [
                r.attach_to(
                    residue_coords[
                        i,
                        block_coord_offset_cpu[i, j] : (
                            block_coord_offset_cpu[i, j] + len(r.residue_type.atoms)
                        ),
                    ]
                )
                for j, r in enumerate(orig_res_list)
            ]
            for i, orig_res_list in enumerate(orig_res)
        ]

    @classmethod
    def inter_residue_connections_from_pose_stacks(
        cls,
        packed_block_types: PackedBlockTypes,
        pose_stacks: Sequence["PoseStack"],
        ps_offsets: Tensor[torch.int64][:],
        max_n_blocks: int,
        device: torch.device,
    ) -> Tensor[torch.int32][:, :, :, 2]:
        n_poses = sum(len(ps) for ps in pose_stacks)
        max_n_conn = max(
            len(rt.connections) for rt in packed_block_types.active_block_types
        )
        inter_residue_connections = torch.full(
            (n_poses, max_n_blocks, max_n_conn, 2), -1, dtype=torch.int32, device=device
        )
        for i, pose_stack in enumerate(pose_stacks):
            offset = ps_offsets[i]
            inter_residue_connections[
                offset : (offset + len(pose_stack)),
                : pose_stack.inter_residue_connections.shape[1],
                : pose_stack.inter_residue_connections.shape[2],
            ] = pose_stack.inter_residue_connections
        return inter_residue_connections

    @classmethod
    def interblock_bondsep_from_pose_stacks(
        cls,
        packed_block_types: PackedBlockTypes,
        pose_stacks: Sequence["PoseStack"],
        ps_offsets: Tensor[torch.int64][:],
        max_n_blocks: int,
        device: torch.device,
    ) -> Tensor[torch.int32][:, :, :, :, :]:
        n_poses = sum(len(ps) for ps in pose_stacks)
        max_n_conn = max(
            len(rt.connections) for rt in packed_block_types.active_block_types
        )
        inter_block_bondsep = torch.full(
            (n_poses, max_n_blocks, max_n_blocks, max_n_conn, max_n_conn),
            6,
            dtype=torch.int32,
            device=device,
        )
        for i, pose_stack in enumerate(pose_stacks):
            offset = ps_offsets[i]
            i_nblocks = pose_stack.inter_block_bondsep.shape[1]
            i_nconn = pose_stack.inter_block_bondsep.shape[3]
            inter_block_bondsep[
                offset : (offset + len(pose_stack)),
                :i_nblocks,
                :i_nblocks,
                :i_nconn,
                :i_nconn,
            ] = pose_stack.inter_block_bondsep
        return inter_block_bondsep

    @classmethod
    def resolve_block_type_ind(
        cls,
        packed_block_types: PackedBlockTypes,
        pose_stacks: Sequence["PoseStack"],
        ps_offsets: Tensor[torch.int64][:],
        max_n_blocks: int,
        device=torch.device,
    ):
        n_poses = sum(len(ps) for ps in pose_stacks)
        block_type_ind = torch.full(
            (n_poses, max_n_blocks), -1, dtype=torch.int32, device=device
        )
        for i, pose_stack in enumerate(pose_stacks):
            offset = ps_offsets[i]
            n_blocks = pose_stack.block_type_ind.shape[1]
            mapping = torch.cat(
                (
                    torch.tensor(
                        packed_block_types.inds_for_restypes(
                            pose_stack.packed_block_types.active_block_types
                        ),
                        dtype=torch.int32,
                        device=device,
                    ),
                    torch.full((1,), -1, dtype=torch.int32, device=device),
                )
            )
            remapped = mapping[pose_stack.block_type_ind.to(torch.int64)]

            block_type_ind[
                offset : (offset + len(pose_stack)), : remapped.shape[1]
            ] = remapped
        return block_type_ind
