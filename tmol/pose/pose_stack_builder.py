import toolz

import itertools

import numpy
import torch
import pandas
import sparse
import scipy.sparse.csgraph as csgraph

from typing import List, Tuple

from tmol.types.array import NDArray
from tmol.types.torch import Tensor

from tmol.chemical.restypes import (
    # RefinedResidueType,
    Residue,
    find_simple_polymeric_connections,
)
from tmol.pose.packed_block_types import PackedBlockTypes, residue_types_from_residues
from tmol.pose.pose_stack import PoseStack

# from tmol.system.datatypes import connection_metadata_dtype
from tmol.utility.tensor.common_operations import exclusive_cumsum1d, exclusive_cumsum2d
from tmol.types.functional import validate_args


class PoseStackBuilder:
    @classmethod
    @validate_args
    def one_structure_from_polymeric_residues(
        cls, res: List[Residue], device: torch.device
    ) -> PoseStack:
        residue_connections = find_simple_polymeric_connections(res)
        return cls.one_structure_from_residues_and_connections(
            res, residue_connections, device
        )

    @classmethod
    @validate_args
    def one_structure_from_residues_and_connections(
        cls,
        res: List[Residue],
        residue_connections: List[Tuple[int, str, int, str]],
        device: torch.device,
    ) -> PoseStack:
        rt_list = residue_types_from_residues(res)
        packed_block_types = PackedBlockTypes.from_restype_list(rt_list, device)

        inter_residue_connections = cls._create_inter_residue_connections(
            res, residue_connections, device
        )
        inter_block_bondsep = cls._determine_single_structure_inter_block_bondsep(
            res, residue_connections, device
        )

        block_type_ind = torch.tensor(
            packed_block_types.inds_for_res(res), dtype=torch.int32, device=device
        ).unsqueeze(0)

        residue_coords, block_coord_offset, attached_res = cls._pack_residue_coords(
            packed_block_types, block_type_ind, [res]
        )
        coords = torch.tensor(residue_coords, dtype=torch.float32, device=device)
        attached_res = cls._attach_residues(block_coord_offset, [res], residue_coords)

        def i64(t):
            return t.to(torch.int64)

        return PoseStack(
            packed_block_types=packed_block_types,
            residues=attached_res,
            residue_coords=residue_coords,
            coords=coords,
            block_coord_offset=block_coord_offset,
            block_coord_offset64=i64(block_coord_offset),
            inter_residue_connections=inter_residue_connections,
            inter_residue_connections64=i64(inter_residue_connections),
            inter_block_bondsep=inter_block_bondsep,
            inter_block_bondsep64=i64(inter_block_bondsep),
            block_type_ind=block_type_ind,
            block_type_ind64=i64(block_type_ind),
            device=device,
        )

    @classmethod
    @validate_args
    def from_poses(
        cls, pose_stacks: List[PoseStack], device: torch.device
    ) -> PoseStack:
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
        coords, block_coord_offset = cls._pack_pose_stack_coords(
            packed_block_types, pose_stacks, max_n_blocks, device
        )
        residue_coords = coords.cpu().numpy().astype(numpy.float64)

        ps_offset = exclusive_cumsum1d(
            torch.tensor([len(ps) for ps in pose_stacks], dtype=torch.int64)
        )
        block_coord_offset_cpu = block_coord_offset.cpu()
        residues = cls._attach_residues(
            block_coord_offset_cpu,
            [
                one_pose_residues
                for pose_stack in pose_stacks
                for one_pose_residues in pose_stack.residues
            ],
            residue_coords,
        )

        inter_residue_connections = cls._inter_residue_connections_from_pose_stacks(
            packed_block_types, pose_stacks, ps_offset, max_n_blocks, device
        )
        inter_block_bondsep = cls._interblock_bondsep_from_pose_stacks(
            packed_block_types, pose_stacks, ps_offset, max_n_blocks, device
        )
        block_type_ind = cls._resolve_block_type_ind(
            packed_block_types, pose_stacks, ps_offset, max_n_blocks, device
        )

        def i64(t):
            return t.to(torch.int64)

        return PoseStack(
            packed_block_types=packed_block_types,
            residues=residues,
            residue_coords=residue_coords,
            coords=coords,
            block_coord_offset=block_coord_offset,
            block_coord_offset64=i64(block_coord_offset),
            inter_residue_connections=inter_residue_connections,
            inter_residue_connections64=i64(inter_residue_connections),
            inter_block_bondsep=inter_block_bondsep,
            inter_block_bondsep64=i64(inter_block_bondsep),
            block_type_ind=block_type_ind,
            block_type_ind64=i64(block_type_ind),
            device=device,
        )

    @classmethod
    # @validate_args
    def pose_stack_from_monomer_polymer_sequences(
        cls,
        packed_block_types: PackedBlockTypes,
        sequences: List[List[str]],
    ):
        cls._annotate_pbt_w_canonical_aa1lc_lookup(packed_block_types)
        cls._annotate_pbt_w_polymeric_down_up_bondsep_dist(packed_block_types)

        pbt = packed_block_types
        device = pbt.device
        n_poses = len(sequences)

        n_res = numpy.array([len(x) for x in sequences], dtype=numpy.int32)
        max_n_res = numpy.amax(n_res).item()

        (
            real_res,
            n_res,
            block_type_ind,
            block_type_ind64,
        ) = cls._block_type_indices_from_sequences(
            pbt, n_poses, n_res, max_n_res, sequences
        )

        # inter residue connections:
        # 1) we will just say that there's an up chemical bond at residue i to
        # every down connection at residue i+1 and vice versa for each real
        # residue on each pose, except the first and last residues. This will
        # give us the inter_residue_connections tensor
        #
        # 2) if we know the down-to-up chemical bond separation for each block type
        # then we can use scan to compute the number of chemical bonds separating
        # every pair of residues, this will give us the inter_block_bondsep tensor

        inter_residue_connections64 = (
            cls._inter_residue_connections_for_polymeric_monomers(
                pbt, n_poses, max_n_res, real_res, n_res, block_type_ind64
            )
        )

        inter_block_bondsep64 = cls._find_inter_block_separation_for_polymeric_monomers(
            pbt, n_poses, max_n_res, real_res, block_type_ind64
        )

    @classmethod
    @validate_args
    def rebuild_with_new_packed_block_types(
        cls, ps: PoseStack, packed_block_types: PackedBlockTypes
    ):  # -> "PoseStack"
        """Create a new PoseStack object replacing the existing PackedBlockTypes
        object with a new one, and then rebuilding the other data members that
        depend on it.
        """
        # The input packed_block_types must contain the block types of
        # the PoseStack's existing set of in-use residue types (but not necessarily
        # all of the block types that its PackedBlockTypes object holds)
        for pose_res_list in ps.residues:
            for res in pose_res_list:
                assert res.residue_type in packed_block_types.active_block_types

        coords = ps.coords.clone()

        block_type_ind = torch.full_like(ps.block_type_ind, -1)
        # this could be more efficient if we mapped orig_block_type to new_block_type
        for i, res in enumerate(ps.residues):
            block_type_ind[i, : len(res)] = torch.tensor(
                packed_block_types.inds_for_res(res),
                dtype=torch.int32,
                device=ps.device,
            )

        residue_coords = coords.to(dtype=torch.float64).cpu().numpy()

        residues = cls._attach_residues(
            ps.block_coord_offset.cpu(), ps.residues, residue_coords
        )

        def i64(t):
            return t.to(torch.int64)

        return PoseStack(
            packed_block_types=packed_block_types,
            residues=residues,
            residue_coords=residue_coords,
            coords=coords,
            block_coord_offset=ps.block_coord_offset,
            block_coord_offset64=ps.block_coord_offset64,
            inter_residue_connections=ps.inter_residue_connections,
            inter_residue_connections64=ps.inter_residue_connections64,
            inter_block_bondsep=ps.inter_block_bondsep,
            inter_block_bondsep64=ps.inter_block_bondsep64,
            block_type_ind=block_type_ind,
            block_type_ind64=i64(block_type_ind),
            device=ps.device,
        )

    ################# HELPER FUNCTIONS FOR CONSTRUCTION ###############

    @classmethod
    @validate_args
    def _create_inter_residue_connections(
        cls,
        res: List[Residue],
        residue_connections: List[Tuple[int, str, int, str]],
        device: torch.device,
    ) -> Tensor[torch.int32][:, :, :, 2]:
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
            (1, len(res), max_n_conn, 2), -1, dtype=numpy.int32
        )
        inter_residue_connections[
            0, inter_conn_inds[:, 0], inter_conn_inds[:, 1], :
        ] = inter_conn_inds[:, 2:]
        inter_residue_connections = torch.tensor(
            inter_residue_connections, dtype=torch.int32, device=device
        )
        return inter_residue_connections

    @classmethod
    @validate_args
    def _determine_single_structure_inter_block_bondsep(
        cls,
        res: List[Residue],
        residue_connections: List[Tuple[int, str, int, str]],
        device: torch.device,
    ) -> Tensor[torch.int32][:, :, :, :, :]:
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

        return cls._resolve_inter_block_bondsep(
            res, inter_res_bonds, connection_atoms, device
        )

    @classmethod
    @validate_args
    def _resolve_inter_block_bondsep(
        cls,
        res: List[Residue],
        inter_res_bonds: NDArray[int][:, 2],
        connection_atoms: pandas.DataFrame,
        device: torch.device,
    ) -> Tensor[torch.int32][:, :, :, :, :]:
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
            (1, n_res, n_res, max_n_conn, max_n_conn), 6, dtype=numpy.int32
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
            0,
            rinds_all_pairs[:, 0],
            rinds_all_pairs[:, 1],
            cinds_all_pairs[:, 0],
            cinds_all_pairs[:, 1],
        ] = min_bond_dist[ainds_all_pairs[:, 0], ainds_all_pairs[:, 1]]

        return torch.tensor(inter_block_bondsep, dtype=torch.int32, device=device)

    @classmethod
    @validate_args
    def _pack_residue_coords(
        cls,
        packed_block_types: PackedBlockTypes,
        block_type_ind: Tensor[torch.int32][:, :],
        res: List[List[Residue]],
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
    @validate_args
    def _pack_pose_stack_coords(
        cls,
        packed_block_types: PackedBlockTypes,
        pose_stacks,  # : List["PoseStack"],
        max_n_blocks: int,
        device: torch.device,
    ) -> Tuple[Tensor[torch.float32][:, :, 3], Tensor[torch.int32][:, :]]:

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
    @validate_args
    def _attach_residues(
        cls,
        block_coord_offset_cpu: Tensor[torch.int32][:, :],
        orig_res: List[List[Residue]],
        residue_coords: NDArray[numpy.float64][:, :, 3],
    ) -> List[List[Residue]]:

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
    @validate_args
    def _inter_residue_connections_from_pose_stacks(
        cls,
        packed_block_types: PackedBlockTypes,
        pose_stacks,  # : List["PoseStack"],
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
    @validate_args
    def _interblock_bondsep_from_pose_stacks(
        cls,
        packed_block_types: PackedBlockTypes,
        pose_stacks,  # : List["PoseStack"],
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
    @validate_args
    def _resolve_block_type_ind(
        cls,
        packed_block_types: PackedBlockTypes,
        pose_stacks,  #: List["PoseStack"],
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
            # n_blocks = pose_stack.block_type_ind.shape[1]
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

    @classmethod
    @validate_args
    def _annotate_pbt_w_canonical_aa1lc_lookup(cls, pbt: PackedBlockTypes):
        """Annotate the PBT with a pandas dictionary mapping the (unique!) names
        of each of the block types to their index in the active_block_types list,
        including special entries for the l-canonical amino acids based on their
        1-letter codes. To use you would say:

            df_inds = pbt.bt_mapping_w_lcaa_1lc_ind.get_indexer(list_of_names)
            bt_inds = pbt.bt_mapping_w_lcaa_1lc.iloc[df_inds]["bt_ind"].values
        """

        if hasattr(pbt, "bt_mapping_w_lcaa_1lc"):
            assert hasattr(pbt, "bt_mapping_w_lcaa_1lc_ind")
            return

        aa_codes = {
            "ALA": "A",
            "CYS": "C",
            "ASP": "D",
            "GLU": "E",
            "PHE": "F",
            "GLY": "G",
            "HIS": "H",
            "ILE": "I",
            "LYS": "K",
            "LEU": "L",
            "MET": "M",
            "ASN": "N",
            "PRO": "P",
            "GLN": "Q",
            "ARG": "R",
            "SER": "S",
            "THR": "T",
            "VAL": "V",
            "TRP": "W",
            "TYR": "Y",
        }

        sorted_1lc = sorted([v for _, v in aa_codes.items()])
        one_let_co_index = {
            aa1: ind
            for aa1, ind in zip(sorted_1lc, numpy.arange(20, dtype=numpy.int32))
        }

        lcaa_ind = numpy.full((20,), -1, dtype=numpy.int32)
        for i, res in enumerate(pbt.active_block_types):
            if res.name3 in aa_codes:
                if (
                    "NTerm" not in res.properties.connectivity
                    and "CTerm" not in res.properties.connectivity
                ):
                    if res.name3 == "HIS":
                        if res.name == "HIS_D":
                            continue
                    ind = one_let_co_index[aa_codes[res.name3]]
                    assert lcaa_ind[ind] == -1
                    lcaa_ind[ind] = i

        names = list(
            itertools.chain([bt.name for bt in pbt.active_block_types], sorted_1lc)
        )
        df = pandas.DataFrame(
            dict(
                names=names,
                bt_ind=itertools.chain(
                    numpy.arange(len(pbt.active_block_types), dtype=numpy.int32),
                    lcaa_ind,
                ),
            )
        )
        ind = pandas.Index(names)
        setattr(pbt, "bt_mapping_w_lcaa_1lc", df)
        setattr(pbt, "bt_mapping_w_lcaa_1lc_ind", ind)

    @classmethod
    @validate_args
    def _annotate_pbt_w_polymeric_down_up_bondsep_dist(cls, pbt: PackedBlockTypes):

        if hasattr(pbt, "polymeric_down_to_up_nbonds"):
            return

        nbt = pbt.n_types
        polymeric_down_to_up_nbonds = torch.tensor(
            [
                bt.path_distance[
                    bt.ordered_connection_atoms[bt.down_connection_ind],
                    bt.ordered_connection_atoms[bt.up_connection_ind],
                ]
                if bt.down_connection_ind != -1 and bt.up_connection_ind != -1
                else 0
                for bt in pbt.active_block_types
            ],
            dtype=torch.int32,
            device=pbt.device,
        )

        setattr(pbt, "polymeric_down_to_up_nbonds", polymeric_down_to_up_nbonds)

    @classmethod
    @validate_args
    def _block_type_indices_from_sequences(
        cls,
        pbt: PackedBlockTypes,
        n_poses: int,
        n_res: NDArray[numpy.int32][:],
        max_n_res: int,
        sequences: List[List[str]],
    ) -> Tuple[
        Tensor[torch.bool][:, :],
        Tensor[torch.int64][:],
        Tensor[torch.int32][:, :],
        Tensor[torch.int64][:, :],
    ]:
        device = pbt.device
        real_res = (
            numpy.tile(numpy.arange(max_n_res, dtype=numpy.int32), n_poses).reshape(
                (n_poses, max_n_res)
            )
            < n_res[:, None]
        )
        condensed_seqs = list(itertools.chain.from_iterable(sequences))
        condensed_bt_df_inds = pbt.bt_mapping_w_lcaa_1lc_ind.get_indexer(condensed_seqs)
        # print("condensed_bt_df_inds", condensed_bt_df_inds)

        # error checking: all names need to map to a residue type if we are to proceed
        condensed_non_df_inds = condensed_bt_df_inds == -1
        if numpy.any(condensed_non_df_inds):
            condensed_seqs = numpy.array(condensed_seqs)
            undefined_names = condensed_seqs[condensed_non_df_inds]
            nz_real_res_pose_ind, nz_real_res_res_ind = numpy.nonzero(real_res)
            undefined_pose_ind = nz_real_res_pose_ind[condensed_non_df_inds]
            undefined_res_ind = nz_real_res_res_ind[condensed_non_df_inds]
            triples = ", ".join(
                [
                    "({} at pose {} residue {})".format(n, p, r)
                    for n, p, r in zip(
                        undefined_names, undefined_pose_ind, undefined_res_ind
                    )
                ]
            )
            error = "Fatal error: could not resolve residue type by name for the following residues: {}\n".format(
                triples
            )
            raise ValueError(error)

        condensed_bt_inds = pbt.bt_mapping_w_lcaa_1lc["bt_ind"][
            condensed_bt_df_inds
        ].values
        # print("condensed_bt_inds", condensed_bt_inds)
        bt_inds = numpy.full((n_poses, max_n_res), -1, dtype=numpy.int32)
        bt_inds[real_res] = condensed_bt_inds

        block_type_ind = torch.tensor(bt_inds, dtype=torch.int32, device=device)
        block_type_ind64 = block_type_ind.to(dtype=torch.int64)
        real_res = torch.tensor(real_res, dtype=torch.bool, device=device)
        n_res = torch.tensor(n_res, dtype=torch.int64, device=device)

        return real_res, n_res, block_type_ind, block_type_ind64

    @classmethod
    @validate_args
    def _inter_residue_connections_for_polymeric_monomers(
        cls,
        pbt: PackedBlockTypes,
        n_poses: int,
        max_n_res: int,
        real_res: Tensor[torch.bool][:, :],
        n_res: Tensor[torch.int64][:],
        block_type_ind64: Tensor[torch.int64][:, :],
    ) -> Tensor[torch.int64][:, :, :, 2]:
        assert real_res.shape[0] == n_poses
        assert real_res.shape[1] == max_n_res
        assert n_res.shape[0] == n_poses
        assert block_type_ind64.shape[0] == n_poses
        assert block_type_ind64.shape[1] == max_n_res

        device = pbt.device

        # 1) inter_residue_connections:
        max_n_conn = pbt.max_n_conn
        inter_residue_connections64 = torch.full(
            (n_poses, max_n_res, max_n_conn, 2), -1, dtype=torch.int64, device=device
        )

        # let's find the up connection indices of the n-terminal sides of each connection
        # and the down connection indices of the c-terminal sides of each connection
        res_is_real_and_not_n_term = real_res.clone()
        res_is_real_and_not_n_term[:, 0] = False

        res_is_real_and_not_c_term = real_res.clone()
        npose_arange = torch.arange(n_poses, dtype=torch.int64, device=device)
        res_is_real_and_not_c_term[npose_arange, n_res - 1] = False

        connected_up_conn_inds = pbt.up_conn_inds[
            block_type_ind64[res_is_real_and_not_n_term]
        ].to(torch.int64)
        connected_down_conn_inds = pbt.down_conn_inds[
            block_type_ind64[res_is_real_and_not_c_term]
        ].to(torch.int64)

        (
            nz_res_is_real_and_not_n_term_pose_ind,
            nz_res_is_real_and_not_n_term_res_ind,
        ) = torch.nonzero(res_is_real_and_not_n_term, as_tuple=True)
        (
            nz_res_is_real_and_not_c_term_pose_ind,
            nz_res_is_real_and_not_c_term_res_ind,
        ) = torch.nonzero(res_is_real_and_not_c_term, as_tuple=True)

        inter_residue_connections64[
            nz_res_is_real_and_not_c_term_pose_ind,
            nz_res_is_real_and_not_c_term_res_ind,
            connected_up_conn_inds,
            0,
        ] = nz_res_is_real_and_not_n_term_res_ind
        inter_residue_connections64[
            nz_res_is_real_and_not_c_term_pose_ind,
            nz_res_is_real_and_not_c_term_res_ind,
            connected_up_conn_inds,
            1,
        ] = connected_down_conn_inds

        inter_residue_connections64[
            nz_res_is_real_and_not_n_term_pose_ind,
            nz_res_is_real_and_not_n_term_res_ind,
            connected_down_conn_inds,
            0,
        ] = nz_res_is_real_and_not_c_term_res_ind
        inter_residue_connections64[
            nz_res_is_real_and_not_n_term_pose_ind,
            nz_res_is_real_and_not_n_term_res_ind,
            connected_down_conn_inds,
            1,
        ] = connected_up_conn_inds

        return inter_residue_connections64

    @classmethod
    @validate_args
    def _find_inter_block_separation_for_polymeric_monomers(
        cls,
        pbt: PackedBlockTypes,
        n_chains: int,
        max_n_res: int,
        real_res: Tensor[torch.bool][:, :],
        block_type_ind64: Tensor[torch.int64][:, :],
    ) -> Tensor[torch.int64][:, :, :, :, :]:
        return cls._find_inter_block_separation_for_polymeric_monomers_heavy(
            pbt.device,
            pbt.polymeric_down_to_up_nbonds,
            pbt.up_conn_inds,
            pbt.down_conn_inds,
            n_chains,
            max_n_res,
            pbt.max_n_conn,
            real_res,
            block_type_ind64,
        )

    @classmethod
    @validate_args
    def _find_inter_block_separation_for_polymeric_monomers_heavy(
        cls,
        device: torch.device,
        bt_polymeric_down_to_up_nbonds: Tensor[torch.int32][:],
        bt_up_conn_inds: Tensor[torch.int32][:],
        bt_down_conn_inds: Tensor[torch.int32][:],
        n_chains: int,
        max_n_res: int,
        max_n_conn: int,
        real_res: Tensor[torch.bool][:, :],
        block_type_ind64: Tensor[torch.int64][:, :],
    ) -> Tensor[torch.int64][:, :, :, :, :]:
        assert real_res.shape[0] == n_chains
        assert real_res.shape[1] == max_n_res
        assert block_type_ind64.shape[0] == n_chains
        assert block_type_ind64.shape[1] == max_n_res

        # Let's take an example of a single four residue alpha amino acid chain: "MT[Beta-Ala]Q"
        # (This example has been chosen to  ensure that our logic is sound; if all the backbone
        # types had the same number of atoms, then we might compute the right answer for the
        # wrong reason if we worked only with alpha amino acids, but then fail once we attempt
        # to build this tensor for a mix of alpha- and beta amino acids.)
        #
        # What we want is the connection-point distance tensor of:
        #
        # [ [ [[0,2],[2,0]],   [[3,5],[1,3]], [[6,9],[4,7]], [[10,12],[8,10]] ]
        #   [ [[3,1],[5,3]],   [[0,2],[2,0]], [[3,6],[1,4]], [[7,9],[5,7]]    ]
        #   [ [[6,4],[9,7]],   [[3,1],[6,4]], [[0,3],[3,0]], [[4,6],[1,3]]    ]
        #   [ [[10,8],[12,10], [[7,5],[9,7]], [[4,1],[6,3]], [[0, 2],[2,0]]   ]]
        #
        # Where, e.g., the [0, 2, :, :] matrix [[6,9],[4,7]] says:
        # - The N atom on residue 0 is 6 chemical bonds separated from the N atom on residue 2
        # - The N atom on residue 0 is 9 chemical bonds separated from the C atom on residue 2
        # - The C atom on residue 0 is 4 chemical bonds separated from the N atom on residue 2
        # - The C atom on residue 0 is 7 chemical bonds separated from the C atom on residue 2
        #
        # What about if we have [Met:NTerm]T[Beta-Ala][GLN:CTerm] ?
        # Then we would want this tensor instead
        # (note that "up" on res 0 and "down" on res 3 are both ind 0)
        # [ [ [[0,-1],[-1,-1]],  [[1,3],[-1,-1]], [[4,7],[-1,-1]], [[8,-1],[-1,-1]]  ]
        #   [ [[1,-1],[3,-1]],   [[0,2],[2,0]],   [[3,6],[1,4]],   [[7,-1],[5,-1]]   ]
        #   [ [[4,-1],[7,-1]],   [[3,1],[6,4]],   [[0,3],[3,0]],   [[4,-1],[1,-1]]   ]
        #   [ [[8,-1],[-1,-1],   [[7,5],[-1,-1]], [[4,1],[-1,-1]], [[0, -1],[-1,-1]] ]]
        #
        # So how do we build this tensor? Let's think about it in terms of a single chain.
        # We will add an extra dimension later to talk about multiple chains.
        # Let's start by reading out the length of the down-to-up path separation for each
        # block type. In the former case, A = [2, 2, 3, 2] and in the latter case,
        # A = [0, 2, 3, 0] with zero entries for the residues missing one of the connections.
        # What the two cases will have in common is the subset of residues that contain both
        # up and down connections. Let's focus on the former case and return later to the latter.
        # If we were to add 1 to A, we would have the chemical bond path distance between
        # down-to-down or up-to-up pairs.
        # An exclusive cumulative sum of A+1 would give B = [0, 3, 6, 10]:
        # the down-to-down path distance from residue 0 to residue i.
        # We can do a broadcast subtraction C = B[None,:] - B[:, None]
        # giving
        #
        # [[0,    3,  6, 10],
        #  [-3,   0,  3,  7],
        #  [-6,  -3,  0,  4],
        #  [-10, -7, -4,  0]]
        #
        # so that the absolute value of each entry represents the down-to-down path distance
        # Then we can compute the output tensor D:
        # D[i, j, down, down] = abs(C[i,j])
        # when i < j, D[i,j,down,up] = C[i,j] + A[j] = abs(C[i,j] + A[j])
        #             D[i,j,up,down] = C[i,j] - A[i]) = abs(C[i,j] - A[i]))
        #             D[i,j, up, up] = C[i,j] - A[i] + A[j]
        # when j < i, D[i,j,down,up] = -1 * C[i,j] - A[j] = -1 * (C[i,j] + A[j]) = abs(C[i,j] + A[j])
        #             D[i,j,up,down] = -1 * C[i,j] + A[i] = -1 * (C[i,j] - A[i]) = abs(C[i,j] - A[i])
        #             D[i,j, up, up] = -1 * C[i,j] + A[i] - A[j])
        #
        # therefore D[i, j, down, up] = abs(C[i,j] + A[j])
        #       and D[i, j, up, down] = abs(C[i,j] - A[i])
        #           D[i, j,  up,  up] = abs(C[i,j] - A[i] + A[j])

        # A: down_up_separation
        down_up_separation = torch.full(
            (n_chains, max_n_res), 0, dtype=torch.int64, device=device
        )
        down_up_separation[real_res] = bt_polymeric_down_to_up_nbonds[
            block_type_ind64[real_res]
        ].to(torch.int64)

        # B: down_to_down_chain_distance
        down_to_down_chain_distance = exclusive_cumsum2d(down_up_separation + 1)

        # C: pair_distances
        pair_distances = (
            down_to_down_chain_distance[:, None, :]
            - down_to_down_chain_distance[:, :, None]
        )

        # D: inter_block_bondsep
        inter_block_bondsep64 = torch.full(
            (n_chains, max_n_res, max_n_res, max_n_conn, max_n_conn),
            100,
            dtype=torch.int64,
            device=device,
        )

        down_up_distance = torch.abs(pair_distances + down_up_separation[:, None, :])
        up_down_distance = torch.abs(pair_distances - down_up_separation[:, :, None])
        up_up_distance = torch.abs(
            pair_distances
            - down_up_separation[:, :, None]
            + down_up_separation[:, None, :]
        )

        both_res_real = torch.logical_and(real_res[:, :, None], real_res[:, None, :])
        nz_brr = torch.nonzero(both_res_real, as_tuple=False)

        nz_brr_bt_1 = block_type_ind64[nz_brr[:, 0], nz_brr[:, 1]]
        nz_brr_bt_2 = block_type_ind64[nz_brr[:, 0], nz_brr[:, 2]]

        # now we need the indices for the down and up connections that correspond to the nz_brr indices
        nz_brr_upconn_1 = bt_up_conn_inds[nz_brr_bt_1].to(torch.int64)
        nz_brr_upconn_2 = bt_up_conn_inds[nz_brr_bt_2].to(torch.int64)
        nz_brr_downconn_1 = bt_down_conn_inds[nz_brr_bt_1].to(torch.int64)
        nz_brr_downconn_2 = bt_down_conn_inds[nz_brr_bt_2].to(torch.int64)

        # finally we can enter the information for these connections into their positions in the
        # output tensor
        inter_block_bondsep64[
            nz_brr[:, 0], nz_brr[:, 1], nz_brr[:, 2], nz_brr_upconn_1, nz_brr_downconn_2
        ] = up_down_distance[both_res_real]
        inter_block_bondsep64[
            nz_brr[:, 0], nz_brr[:, 1], nz_brr[:, 2], nz_brr_downconn_1, nz_brr_upconn_2
        ] = down_up_distance[both_res_real]
        inter_block_bondsep64[
            nz_brr[:, 0],
            nz_brr[:, 1],
            nz_brr[:, 2],
            nz_brr_downconn_1,
            nz_brr_downconn_2,
        ] = torch.abs(pair_distances[both_res_real])
        inter_block_bondsep64[
            nz_brr[:, 0], nz_brr[:, 1], nz_brr[:, 2], nz_brr_upconn_1, nz_brr_upconn_2
        ] = up_up_distance[both_res_real]

        return inter_block_bondsep64
