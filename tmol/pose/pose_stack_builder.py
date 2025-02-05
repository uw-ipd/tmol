import toolz
import copy

import itertools

import numpy
import torch
import pandas
import scipy.sparse.csgraph as csgraph
import scipy

from typing import List, Tuple, Optional

from tmol.types.array import NDArray
from tmol.types.torch import Tensor

from tmol.chemical.constants import MAX_SIG_BOND_SEPARATION
from tmol.chemical.patched_chemdb import PatchedChemicalDatabase
from tmol.chemical.restypes import (
    RefinedResidueType,
    # Residue,
    # find_simple_polymeric_connections,
    # find_disulfide_connections,
    three2one,
)

from tmol.pose.packed_block_types import PackedBlockTypes, residue_types_from_residues
from tmol.pose.pose_stack import PoseStack


# from tmol.system.datatypes import connection_metadata_dtype
from tmol.utility.tensor.common_operations import (
    exclusive_cumsum1d,
    exclusive_cumsum2d,
    exclusive_cumsum2d_and_totals,
    stretch,
    stretch2,
)
from tmol.types.functional import validate_args


class PoseStackBuilder:

    @classmethod
    @validate_args
    def from_poses(
        cls, pose_stacks: List[PoseStack], device: torch.device
    ) -> PoseStack:
        pbt0 = pose_stacks[0].packed_block_types
        for ps in pose_stacks:
            # all PoseStacks must be built from the same chemical database
            # even if some of the residue types were perhaps created
            # programmatically instead of being read from an input file
            assert pbt0.chem_db is ps.packed_block_types.chem_db
        reuse_pbt = all(
            pose_stack.packed_block_types is pbt0 for pose_stack in pose_stacks
        )
        if reuse_pbt:
            packed_block_types = pbt0
        else:
            all_bt = [
                bt
                for pose_stack in pose_stacks
                for bt in pose_stack.packed_block_types.active_block_types
            ]
            bt_set = {}
            for bt in all_bt:
                if bt.name not in bt_set:
                    bt_set[bt.name] = bt
            uniq_bt = [v for _, v in bt_set.items()]
            packed_block_types = PackedBlockTypes.from_restype_list(
                pbt0.chem_db, pbt0.restype_set, uniq_bt, device
            )

        max_n_blocks = max(pose_stack.max_n_blocks for pose_stack in pose_stacks)
        coords, block_coord_offset = cls._pack_pose_stack_coords(
            packed_block_types, pose_stacks, max_n_blocks, device
        )

        ps_offset = exclusive_cumsum1d(
            torch.tensor([len(ps) for ps in pose_stacks], dtype=torch.int64)
        )
        # block_coord_offset_cpu = block_coord_offset.cpu()

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
    def pose_stack_from_monomer_polymer_sequences(
        cls,
        packed_block_types: PackedBlockTypes,
        sequences,  # List[List[str]], -- too slow to type check
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
        assert real_res.device == device
        assert n_res.device == device
        assert block_type_ind.device == device
        assert block_type_ind64.device == device

        # inter residue connections:
        # 1) we will just say that there's an up chemical bond at residue i to
        # every down connection at residue i+1 and vice versa for each real
        # residue on each pose, except the first and last residues. This will
        # give us the inter_residue_connections tensor
        #
        # 2) if we know the down-to-up chemical bond separation for each block type
        # then we can use scan to compute the number of chemical bonds separating
        # every pair of residues, this will give us the inter_block_bondsep tensor

        # with flake8 and black working against each other, if you want to give
        # a variable a descriptive name, you often have to assign it to a temporary
        # in the line where you assign it from a function with a descriptive name
        # and then assign it to the variable you really want in a second step.
        # what a waste!
        irc64 = cls._inter_residue_connections_for_polymeric_monomers(
            pbt, n_poses, max_n_res, real_res, n_res, block_type_ind64, None
        )
        inter_residue_connections64 = irc64

        inter_block_bondsep64 = cls._find_inter_block_separation_for_polymeric_monomers(
            pbt, n_poses, max_n_res, real_res, block_type_ind64
        )

        n_atoms = torch.zeros((n_poses, max_n_res), dtype=torch.int32, device=device)
        n_atoms[real_res] = pbt.n_atoms[block_type_ind64[real_res]]
        block_coord_offset = exclusive_cumsum2d(n_atoms)

        max_n_atoms = torch.max(torch.sum(n_atoms, dim=1)).item()

        return PoseStack(
            packed_block_types=packed_block_types,
            coords=torch.zeros(
                (n_poses, max_n_atoms, 3), dtype=torch.float32, device=device
            ),
            block_coord_offset=block_coord_offset,
            block_coord_offset64=block_coord_offset.to(torch.int64),
            inter_residue_connections=inter_residue_connections64.to(torch.int32),
            inter_residue_connections64=inter_residue_connections64,
            inter_block_bondsep=inter_block_bondsep64.to(torch.int32),
            inter_block_bondsep64=inter_block_bondsep64,
            block_type_ind=block_type_ind64.to(torch.int32),
            block_type_ind64=block_type_ind64,
            device=device,
        )

    @classmethod
    @validate_args
    def pose_stack_from_monomer_sequences_w_extrapolymeric_conns(
        cls,
        packed_block_types: PackedBlockTypes,
        sequences,  # List[List[str]], -- too slow to type check
    ):
        """Construct a PoseStack given a list of sequences where the disulfide
        connectivity is known. E.g. If there is a disulfide pair between residues
        5 and 20 and another disulfide pair between residues 9 and 15, then
        the sequence would be given as:

        AAAA[CYD--dslf-first]AAA[CYD--dslf-second]AAA ...
        AA[CYD--dslf-second]AAAA[CYD--dslf-first]AAA

        where the string following the double dash, designates 1) the name of
        the inter-residue connection (for CYD, this is "dslf") and then 2) after
        the single dash, a unique identifier so that which pair of residues are
        forming that connection. In this case the two disulfides have the labels
        "first" and "second," but any unique label would suffice.

        """
        cls._annotate_pbt_w_canonical_aa1lc_lookup(packed_block_types)
        cls._annotate_pbt_w_polymeric_down_up_bondsep_dist(packed_block_types)

        pbt = packed_block_types
        device = pbt.device
        n_poses = len(sequences)

        n_res = numpy.array([len(x) for x in sequences], dtype=numpy.int32)
        max_n_res = numpy.amax(n_res).item()

        trimmed_sequences, expoly_connections = cls._find_connections_in_sequences(
            pbt, sequences
        )

        (
            real_res,
            n_res,
            block_type_ind,
            block_type_ind64,
        ) = cls._block_type_indices_from_sequences(
            pbt, n_poses, n_res, max_n_res, trimmed_sequences
        )
        assert real_res.device == device
        assert n_res.device == device
        assert block_type_ind.device == device
        assert block_type_ind64.device == device

        # inter residue connections:
        #
        # 1) First, make sure that the connections provided in the input sequence
        # actually are present on those residue types.
        #
        # 2) a. We will then say that there's an "up" chemical bond at residue i to
        # every "down" connection at residue i+1 and vice versa for each real
        # residue on each pose, except the first and last residues. This will
        # give us the inter_residue_connections tensor. b. Then we will add
        # to this set of inter-residue connections the ones given to us
        # in the connection-annotated sequence.
        #
        # 3) Then, we will construct a graph representing the edge weights
        # between all pairs of connection points. a) Intra-residue connection
        # distances are read out of the PBT object (after an initial annotation)
        # b) the inter-residue connections will then be added from the inter-residue
        # connections noted in the inter_residue_connections64 tensor.
        #
        # 4) Finally, we invoke all-pairs-shortest-path and then read out the
        # intra-block bond separations

        # 1
        resolved_expoly_connections = cls._find_connection_pairs_for_residue_subset(
            pbt, sequences, block_type_ind64, expoly_connections
        )

        # 2a
        irc64 = cls._inter_residue_connections_for_polymeric_monomers(
            pbt, n_poses, max_n_res, real_res, n_res, block_type_ind64, None
        )
        inter_residue_connections64 = irc64

        # 2b add in non-polymeric connections (such as disulfides)
        cls._incorporate_extra_connections_into_inter_res_conn_set(
            resolved_expoly_connections, inter_residue_connections64
        )

        # 3a
        (
            pconn_matrix,
            pconn_offsets,
            block_n_conn,
            pose_n_pconn,
        ) = cls._take_real_conn_conn_intrablock_pairs(pbt, block_type_ind64, real_res)

        # 3b
        cls._incorporate_inter_residue_connections_into_connectivity_graph(
            inter_residue_connections64, pconn_offsets, pconn_matrix
        )

        # 4
        ibb64 = cls._calculate_interblock_bondsep_from_connectivity_graph(
            pbt, block_n_conn, pose_n_pconn, pconn_matrix
        )
        inter_block_bondsep64 = ibb64

        n_atoms = torch.zeros((n_poses, max_n_res), dtype=torch.int32, device=device)
        n_atoms[real_res] = pbt.n_atoms[block_type_ind64[real_res]]
        block_coord_offset = exclusive_cumsum2d(n_atoms)

        max_n_atoms = torch.max(torch.sum(n_atoms, dim=1)).item()

        return PoseStack(
            packed_block_types=packed_block_types,
            coords=torch.zeros(
                (n_poses, max_n_atoms, 3), dtype=torch.float32, device=device
            ),
            block_coord_offset=block_coord_offset,
            block_coord_offset64=block_coord_offset.to(torch.int64),
            inter_residue_connections=inter_residue_connections64.to(torch.int32),
            inter_residue_connections64=inter_residue_connections64,
            inter_block_bondsep=inter_block_bondsep64.to(torch.int32),
            inter_block_bondsep64=inter_block_bondsep64,
            block_type_ind=block_type_ind64.to(torch.int32),
            block_type_ind64=block_type_ind64,
            device=device,
        )

    @classmethod
    @validate_args
    def pose_stack_from_sequences(
        cls,
        packed_block_types: PackedBlockTypes,
        sequences,  # List[List[str]]
        chain_lengths,  # List[List[int]]
        # option 1:
        # chain_lengths: List[List[int]]
        # option 2:
        # sequences_w_chain_annotation
        # [ALA:Nterm]AAAA[ALA:Cterm][ALA:Nterm]AAAAAAAAA[ALA:Cterm]
    ):
        """Construct a PoseStack given a list of sequences where the disulfide
        connectivity is known. E.g. If there is a disulfide pair between
        residues 5 and 20 and another disulfide pair between residues 9 and 15,
        then the sequence would be given as:

        AAAA[CYD--dslf-first]AAA[CYD--dslf-second]AAA ...
        AA[CYD--dslf-second]AAAA[CYD--dslf-first]AAA

        where the string following the double dash, designates 1) the name of the
        inter-residue connection (for CYD, this is "dslf") and then 2) after the
        single dash, a unique identifier so that which pair of residues are forming
        that connection. In this case the two disulfides have the labels "first"
        and "second," but any unique label would suffice.

        """
        cls._annotate_pbt_w_canonical_aa1lc_lookup(packed_block_types)
        cls._annotate_pbt_w_polymeric_down_up_bondsep_dist(packed_block_types)

        pbt = packed_block_types
        device = pbt.device
        n_poses = len(sequences)

        n_res = numpy.array([len(x) for x in sequences], dtype=numpy.int32)
        max_n_res = numpy.amax(n_res).item()

        trimmed_sequences, expoly_connections = cls._find_connections_in_sequences(
            pbt, sequences
        )

        (
            real_res,
            n_res,
            block_type_ind,
            block_type_ind64,
        ) = cls._block_type_indices_from_sequences(
            pbt, n_poses, n_res, max_n_res, trimmed_sequences
        )
        assert real_res.device == device
        assert n_res.device == device
        assert block_type_ind.device == device
        assert block_type_ind64.device == device

        # inter residue connections:
        #
        # 1) First, make sure that the connections provided in the input sequence
        # actually are present on those residue types.
        #
        # 2) a. We will then say that there's an "up" chemical bond at residue i to
        # every "down" connection at residue i+1 and vice versa for each real
        # residue on each pose, except the first and last residues. This will
        # give us the inter_residue_connections tensor. b. Then we will add
        # to this set of inter-residue connections the ones given to us
        # in the connection-annotated sequence. c. Then we will remove the
        # chemical bonds for i-to-i+1 connections that span chains
        #
        # 3) Then, we will construct a graph representing the edge weights
        # between all pairs of connection points. a) Intra-residue connection
        # distances are read out of the PBT object (after an initial annotation)
        # b) the inter-residue connections will then be added from the inter-residue
        # connections noted in the inter_residue_connections64 tensor.
        #
        # 4) Finally, we invoke all-pairs-shortest-path and then read out the
        # intra-block bond separations

        # 1
        resolved_expoly_connections = cls._find_connection_pairs_for_residue_subset(
            pbt, sequences, block_type_ind64, expoly_connections
        )

        # 2a
        irc64 = cls._inter_residue_connections_for_polymeric_monomers(
            pbt, n_poses, max_n_res, real_res, n_res, block_type_ind64, chain_lengths
        )
        inter_residue_connections64 = irc64

        # 2b add in non-polymeric connections (such as disulfides)
        cls._incorporate_extra_connections_into_inter_res_conn_set(
            resolved_expoly_connections, inter_residue_connections64
        )

        # 3a
        (
            pconn_matrix,
            pconn_offsets,
            block_n_conn,
            pose_n_pconn,
        ) = cls._take_real_conn_conn_intrablock_pairs(pbt, block_type_ind64, real_res)

        # 3b
        cls._incorporate_inter_residue_connections_into_connectivity_graph(
            inter_residue_connections64, pconn_offsets, pconn_matrix
        )

        # 4
        ibb64 = cls._calculate_interblock_bondsep_from_connectivity_graph(
            pbt, block_n_conn, pose_n_pconn, pconn_matrix
        )
        inter_block_bondsep64 = ibb64

        n_atoms = torch.zeros((n_poses, max_n_res), dtype=torch.int32, device=device)
        n_atoms[real_res] = pbt.n_atoms[block_type_ind64[real_res]]
        block_coord_offset = exclusive_cumsum2d(n_atoms)

        max_n_atoms = torch.max(torch.sum(n_atoms, dim=1)).item()

        return PoseStack(
            packed_block_types=packed_block_types,
            coords=torch.zeros(
                (n_poses, max_n_atoms, 3), dtype=torch.float32, device=device
            ),
            block_coord_offset=block_coord_offset,
            block_coord_offset64=block_coord_offset.to(torch.int64),
            inter_residue_connections=inter_residue_connections64.to(torch.int32),
            inter_residue_connections64=inter_residue_connections64,
            inter_block_bondsep=inter_block_bondsep64.to(torch.int32),
            inter_block_bondsep64=inter_block_bondsep64,
            block_type_ind=block_type_ind64.to(torch.int32),
            block_type_ind64=block_type_ind64,
            device=device,
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

        for i in range(ps.n_poses):
            for j in range(ps.max_n_blocks):
                if ps.is_real_block(i, j):
                    bt = ps.block_type(i, j)
                    assert numpy.all(packed_block_types.inds_for_restypes([bt]) != -1)

        coords = ps.coords.clone()

        block_type_ind = torch.full_like(
            ps.block_type_ind, -1, device=torch.device("cpu")
        )
        # this could be more efficient if we mapped orig_block_type to new_block_type
        for i in range(ps.n_poses):
            for j in range(ps.max_n_blocks):
                orig_bt_ind = ps.block_type_ind64[i, j]
                if orig_bt_ind >= 0:
                    bt = ps.packed_block_types.active_block_types[orig_bt_ind]
                    block_type_ind[i, j] = packed_block_types.inds_for_restypes(
                        [bt]
                    ).item()
        block_type_ind = block_type_ind.to(ps.device)

        def i64(t):
            return t.to(torch.int64)

        return PoseStack(
            packed_block_types=packed_block_types,
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
    def _find_connection_pairs_for_residue_subset(
        cls,
        pbt: PackedBlockTypes,
        sequences,
        block_types64: Tensor[torch.int64][:, :],
        residue_connections: List[List[Tuple[int, str, int, str]]],
    ) -> List[List[Tuple[int, int, int, int]]]:
        """When there are only a handful of inter-residue connections that
        must be resolved by name, such as disulfides, then handle these
        few connections one-by-one.
        """
        ps_conn_inds = []
        bt_inds = block_types64.cpu()

        def conn_ind_for_bt(bt_ind, c_name):
            bt = pbt.active_block_types[bt_ind]
            return bt.connection_to_cidx[c_name]

        for i, pose_conns in enumerate(residue_connections):
            pose_conn_inds = []
            for r1, c_name1, r2, c_name2 in pose_conns:
                c1, c2 = None, None
                try:
                    bt1_ind = bt_inds[i, r1]
                    bt2_ind = bt_inds[i, r2]
                    c1 = conn_ind_for_bt(bt1_ind, c_name1)
                    c2 = conn_ind_for_bt(bt2_ind, c_name2)
                    pose_conn_inds.append((r1, c1, r2, c2))
                except KeyError:
                    if c1 is None:
                        missing = (c_name1, r1, bt1_ind, c_name2, r2)
                    else:
                        missing = (c_name2, r2, bt2_ind, c_name1, r1)

                    new_err_msg = (
                        "Failed to find connection '"
                        + missing[0]
                        + "' on residue type '"
                        + sequences[i][missing[1]]
                        + "' which is listed as forming a chemical"
                        + " bond to connection '"
                        + missing[3]
                        + "' on residue type '"
                        + sequences[i][missing[4]]
                        + "'\n"
                        + "Valid connection names on '"
                        + sequences[i][missing[1]]
                        + "' are: "
                        + "'"
                        + "', '".join(
                            x
                            for x in pbt.active_block_types[
                                missing[2]
                            ].connection_to_cidx.keys()
                            if x is not None
                        )
                        + "'"
                    )
                    raise ValueError(new_err_msg)

            ps_conn_inds.append(pose_conn_inds)
        return ps_conn_inds

    @classmethod
    @validate_args
    def _read_intra_block_connection_atom_separations(
        cls,
        pbt_conn_at_intrablock_bond_sep: Tensor[torch.int32][:, :, :],
        block_types64: Tensor[torch.int64][:, :],
        real_blocks: Tensor[torch.bool][:, :],
    ) -> Tensor[torch.int32][:, :, :, :]:
        n_poses = block_types64.shape[0]
        max_n_blocks = block_types64.shape[1]
        max_n_conn = pbt_conn_at_intrablock_bond_sep.shape[1]
        assert pbt_conn_at_intrablock_bond_sep.shape[2] == max_n_conn

        intra_block_conn_dists = torch.zeros(
            (n_poses, max_n_blocks, max_n_conn, max_n_conn),
            dtype=torch.int32,
            device=pbt_conn_at_intrablock_bond_sep.device,
        )
        intra_block_conn_dists[real_blocks] = pbt_conn_at_intrablock_bond_sep[
            block_types64[real_blocks]
        ]
        return intra_block_conn_dists

    @classmethod
    # @validate_args
    def _take_real_conn_conn_intrablock_pairs(cls, pbt, block_types64, real_blocks):
        cls._annotate_pbt_w_intraresidue_connection_atom_distances(pbt)
        return cls._take_real_conn_conn_intrablock_pairs_heavy(
            pbt.n_conn, pbt.conn_at_intrablock_bond_sep, block_types64, real_blocks
        )

    @classmethod
    @validate_args
    def _take_real_conn_conn_intrablock_pairs_heavy(
        cls,
        pbt_n_conn: Tensor[torch.int32][:],
        pbt_conn_at_intrablock_bond_sep: Tensor[torch.int32][:, :, :],
        block_types64: Tensor[torch.int64][:, :],
        real_blocks: Tensor[torch.bool][:, :],
    ):
        n_poses = block_types64.shape[0]
        max_n_blocks = block_types64.shape[1]
        # n_blocks_per_pose = torch.sum(real_blocks, axis=1)
        pbt_device = pbt_n_conn.device
        pbt_max_n_conn = pbt_conn_at_intrablock_bond_sep.shape[1]
        assert pbt_conn_at_intrablock_bond_sep.shape[2] == pbt_max_n_conn

        n_conn_for_block = torch.full_like(block_types64, 0, dtype=torch.int32)
        n_conn_for_block[real_blocks] = pbt_n_conn[block_types64[real_blocks]]
        n_conn_for_block_offset, n_conn_totals = exclusive_cumsum2d_and_totals(
            n_conn_for_block
        )
        n_conn_for_block_offset64 = n_conn_for_block_offset.to(torch.int64)
        max_n_pose_conn = torch.max(n_conn_totals)

        pconn_matrix = torch.full(
            (n_poses, max_n_pose_conn, max_n_pose_conn),
            MAX_SIG_BOND_SEPARATION,
            dtype=torch.int32,
            device=pbt_device,
        )

        if max_n_pose_conn == 0:
            # early exit: we are looking at either an empty pose
            # or a bunch of residues with no inter-residue connections;
            # in either case, no work remains.
            return (
                pconn_matrix,
                n_conn_for_block_offset64,
                n_conn_for_block,
                n_conn_totals,
            )

        # Not all blocks have inter-residue connections! We want the subset
        # of real blocks that do.
        block_has_conn = n_conn_for_block > 0

        # let's identify which pairs of connections are part
        # of the same block
        pose_for_block = stretch(
            torch.arange(n_poses, dtype=torch.int64, device=pbt_device), max_n_blocks
        )

        # mark the first connection in each block with a 1
        first_pconn_for_block = torch.zeros(
            (n_poses, max_n_pose_conn), dtype=torch.int32, device=pbt_device
        )
        first_pconn_for_block[
            pose_for_block[block_has_conn.view(-1)],
            n_conn_for_block_offset64[block_has_conn],
        ] = 1
        # then an inclusive cummulative sum will label all of the
        # connections coming from the same block the same; this
        # will be 1 more than the actual block index for the
        # connection, but that's ok for our purposes
        pseudo_block_for_pconn = torch.cumsum(first_pconn_for_block, dim=1)
        are_pconns_from_same_block = (
            pseudo_block_for_pconn[:, None, :] == pseudo_block_for_pconn[:, :, None]
        )

        # Now let's go to the PackedBlockTypes' data describing intra-residue
        # distances for pairs of inter-residue connections on the same block:
        # Read out the intra-block path distances from the PBT annotation
        # which will give us a tensor of [ n-real-blocks x max-n-conn x max-n-conn ]
        # After, we will have to be clever in order to save this data into the
        # fledgling [ n-poses x max-n-pose-conn x max-n-pose-conn ] tensor that
        # will be input into our all-pairs-shortest-path function.
        intra_block_bconn_dists = cls._read_intra_block_connection_atom_separations(
            pbt_conn_at_intrablock_bond_sep, block_types64, real_blocks
        )

        local_ind_for_bconn1 = (
            torch.arange(pbt_max_n_conn, dtype=torch.int64, device=pbt_device)
            .repeat(n_poses * max_n_blocks * pbt_max_n_conn)
            .reshape(n_poses, max_n_blocks, pbt_max_n_conn, pbt_max_n_conn)
        )
        local_ind_for_bconn2 = torch.transpose(local_ind_for_bconn1, 2, 3)

        # n_conn_for_bconn:
        # [n_poses x max_n_blockx x max_n_conn x max_n_conn] tensor stating
        # for entry [i, j, k, l] the number of connections on pose i
        # block j; then we can figure out if any individual pair (k,l)
        # represents a valid intra-block pair or whether k, e.g., exceeds
        # the number of connections for block j.
        n_conn_for_bconn = stretch2(
            n_conn_for_block, pbt_max_n_conn * pbt_max_n_conn
        ).reshape(n_poses, max_n_blocks, pbt_max_n_conn, pbt_max_n_conn)

        valid_local_bconn_pair = torch.logical_and(
            local_ind_for_bconn1 < n_conn_for_bconn,
            local_ind_for_bconn2 < n_conn_for_bconn,
        )

        # pose_ind_for_pconn1
        # [n_poses x max_n_pose_conn x max_n_pose_conn] tensor where
        # entry [i, j, k] is j;
        # pose_ind_for_pconn2, on the otherhand, gives [i, j, k] as k
        # useful to know if both j and k are less than the maximum
        # number of inter-residue connection points for the pose
        pose_ind_for_pconn1 = (
            torch.arange(max_n_pose_conn, dtype=torch.int64, device=pbt_device)
            .repeat(n_poses * max_n_pose_conn)
            .reshape(n_poses, max_n_pose_conn, max_n_pose_conn)
        )
        pose_ind_for_pconn2 = torch.transpose(pose_ind_for_pconn1, 1, 2)

        n_pose_conn_for_pconn = stretch(
            n_conn_totals, max_n_pose_conn * max_n_pose_conn
        ).reshape(n_poses, max_n_pose_conn, max_n_pose_conn)

        valid_pconn_pair = torch.logical_and(
            pose_ind_for_pconn1 < n_pose_conn_for_pconn,
            pose_ind_for_pconn2 < n_pose_conn_for_pconn,
        )

        real_pconns_from_same_block = torch.logical_and(
            are_pconns_from_same_block, valid_pconn_pair
        )

        # here we are at last! fancy indexing to take the subset of
        # real interresidue connection pairs from the
        pconn_matrix[real_pconns_from_same_block] = intra_block_bconn_dists[
            valid_local_bconn_pair
        ]

        return pconn_matrix, n_conn_for_block_offset64, n_conn_for_block, n_conn_totals

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
        device: torch.device,
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

            block_type_ind[offset : (offset + len(pose_stack)), : remapped.shape[1]] = (
                remapped
            )
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

        Note that this will give the base aa type for each 1lc; it will not
        give you the bt indices of the n- and c-termini
        """

        if hasattr(pbt, "bt_mapping_w_lcaa_1lc"):
            assert hasattr(pbt, "bt_mapping_w_lcaa_1lc_ind")
            return

        lcaa_ind = {}
        for i, res in enumerate(pbt.active_block_types):
            one = three2one(res.name)
            if one:
                assert one not in lcaa_ind
                lcaa_ind[one] = i

        names = [*lcaa_ind.keys(), *[bt.name for bt in pbt.active_block_types]]
        indices = [
            *lcaa_ind.values(),
            *range(len(pbt.active_block_types)),
        ]

        df = pandas.DataFrame(dict(names=names, bt_ind=indices))
        ind = pandas.Index(names)
        setattr(pbt, "bt_mapping_w_lcaa_1lc", df)
        setattr(pbt, "bt_mapping_w_lcaa_1lc_ind", ind)

    @classmethod
    @validate_args
    def _annotate_pbt_w_polymeric_down_up_bondsep_dist(cls, pbt: PackedBlockTypes):
        if hasattr(pbt, "polymeric_down_to_up_nbonds"):
            return

        polymeric_down_to_up_nbonds = torch.tensor(
            [
                (
                    bt.path_distance[
                        bt.ordered_connection_atoms[bt.down_connection_ind],
                        bt.ordered_connection_atoms[bt.up_connection_ind],
                    ]
                    if bt.down_connection_ind != -1 and bt.up_connection_ind != -1
                    else 0
                )
                for bt in pbt.active_block_types
            ],
            dtype=torch.int32,
            device=pbt.device,
        )

        setattr(pbt, "polymeric_down_to_up_nbonds", polymeric_down_to_up_nbonds)

    @classmethod
    @validate_args
    def _annotate_bt_w_intraresidue_connection_atom_distances(
        cls, bt: RefinedResidueType
    ):
        """Annotate the block type with a slice of the path-distances data member
        for only the inter-residue connection atoms
        """
        if hasattr(bt, "conn_at_intrablock_bond_sep"):
            return
        n_conns = len(bt.connections)
        ind1 = numpy.repeat(bt.ordered_connection_atoms, n_conns, axis=0).reshape(
            n_conns, n_conns
        )
        ind2 = numpy.transpose(ind1)
        conn_at_intrablock_bond_sep = bt.path_distance[ind1, ind2]
        setattr(bt, "conn_at_intrablock_bond_sep", conn_at_intrablock_bond_sep)

    @classmethod
    @validate_args
    def _annotate_pbt_w_intraresidue_connection_atom_distances(
        cls, pbt: PackedBlockTypes
    ):
        """Note the number of chemical bonds that separate all pairs of
        connection atoms. This information is needed in order to construct the
        starting (weighted) graph describing the chemical bonds in the system
        from which either the limited-Dijkstra or the all-pairs-shortest-paths
        algorithms will generate the chemical separation of the connection
        atoms.
        """
        if hasattr(pbt, "conn_at_intrablock_bond_sep"):
            return
        for bt in pbt.active_block_types:
            cls._annotate_bt_w_intraresidue_connection_atom_distances(bt)

        max_n_conn = pbt.max_n_conn
        conn_at_intrablock_bond_sep = torch.full(
            (pbt.n_types, max_n_conn, max_n_conn),
            -1,
            dtype=torch.int32,
            device=pbt.device,
        )
        for i, bt in enumerate(pbt.active_block_types):
            i_n_conn = len(bt.connections)
            conn_at_intrablock_bond_sep[i, :i_n_conn, :i_n_conn] = torch.tensor(
                bt.conn_at_intrablock_bond_sep, device=pbt.device
            )
        setattr(pbt, "conn_at_intrablock_bond_sep", conn_at_intrablock_bond_sep)

    @classmethod
    @validate_args
    def _find_connections_in_sequences(
        cls,
        pbt: PackedBlockTypes,
        sequences,  # List[List[str]] -- too slow to type check
    ):
        ps_conns = []
        trimmed_seqs = copy.deepcopy(sequences)
        for i in range(len(sequences)):
            labels = {}
            p_conns = []
            completed_labels = {}
            for j, resname in enumerate(sequences[i]):
                if len(resname) < 6:
                    # X--C-I
                    # is the shortest possible string containing
                    # an inter-residue connection
                    continue
                connections = resname.split("--")
                if len(connections) < 2:
                    # no inter-residue connections specified here,
                    # just a long name for the residue type
                    continue
                trimmed_seqs[i][j] = connections[0]
                for conn in connections[1:]:
                    conn_name, conn_label = conn.split("-")
                    if conn_label in labels:
                        partner, partner_conn_name = labels[conn_label]
                        if partner == -1:
                            # error: more than two inter-residue connections have
                            # been given the same connection label
                            prev_conn = completed_labels[conn_label]
                            err_msg = (
                                "Fatal error: found more than two "
                                + "residue-connections with the "
                                + 'same connection label: "'
                                + conn_label
                                + '"'
                                + "\nPreviously encountered between residues "
                                + str(prev_conn[0])
                                + ", conn "
                                + prev_conn[1]
                                + " and "
                                + str(prev_conn[2])
                                + ", conn "
                                + prev_conn[3]
                                + " and "
                                + "now found again for "
                                + str(j)
                                + " as part of the residue "
                                + resname
                                + "\n"
                            )
                            raise ValueError(err_msg)
                        conn = (partner, partner_conn_name, j, conn_name)
                        completed_labels[conn_label] = conn
                        labels[conn_label] = (-1, None)
                        p_conns.append(conn)
                    else:
                        labels[conn_label] = (j, conn_name)
            ps_conns.append(p_conns)
        return trimmed_seqs, ps_conns

    @classmethod
    @validate_args
    def _block_type_indices_from_sequences(
        cls,
        pbt: PackedBlockTypes,
        n_poses: int,
        n_res: NDArray[numpy.int32][:],
        max_n_res: int,
        sequences,  #: List[List[str]], -- too slow to type check
    ) -> Tuple[
        Tensor[torch.bool][:, :],
        Tensor[torch.int64][:],
        Tensor[torch.int32][:, :],
        Tensor[torch.int64][:, :],
    ]:
        device = pbt.device
        # real_res = numpy.full((n_poses, max_n_res), True, dtype=bool)
        real_res = (
            numpy.tile(numpy.arange(max_n_res, dtype=numpy.int32), n_poses).reshape(
                (n_poses, max_n_res)
            )
            < n_res[:, None]
        )

        condensed_seqs = list(itertools.chain.from_iterable(sequences))

        # look up each string in the PBT
        condensed_bt_df_inds = pbt.bt_mapping_w_lcaa_1lc_ind.get_indexer(condensed_seqs)

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
            error = (
                "Fatal error: could not resolve residue type by"
                + " name for the following residues: {}\n".format(triples)
            )
            raise ValueError(error)

        condensed_bt_inds = pbt.bt_mapping_w_lcaa_1lc["bt_ind"][
            condensed_bt_df_inds
        ].values

        bt_inds = numpy.full((n_poses, max_n_res), -1, dtype=numpy.int32)
        bt_inds[real_res] = condensed_bt_inds

        # now convert all numpy arrays into torch tensors: here forward, all
        # calculations are with torch
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
        chain_lengths: Optional[List[List[int]]],
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

        # let's find the up connection indices of the n-terminal sides of
        # each connection and the down connection indices of the c-terminal
        # sides of each connection
        res_is_real_and_not_n_term = real_res.clone()
        res_is_real_and_not_n_term[:, 0] = False

        res_is_real_and_not_c_term = real_res.clone()
        npose_arange = torch.arange(n_poses, dtype=torch.int64, device=device)
        res_is_real_and_not_c_term[npose_arange, n_res - 1] = False

        connected_up_conn_inds = pbt.up_conn_inds[
            block_type_ind64[res_is_real_and_not_c_term]
        ].to(torch.int64)
        connected_down_conn_inds = pbt.down_conn_inds[
            block_type_ind64[res_is_real_and_not_n_term]
        ].to(torch.int64)

        # TO DO: handle termini patches!

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
            0,  # residue id
        ] = nz_res_is_real_and_not_n_term_res_ind
        inter_residue_connections64[
            nz_res_is_real_and_not_c_term_pose_ind,
            nz_res_is_real_and_not_c_term_res_ind,
            connected_up_conn_inds,
            1,  # connection id
        ] = connected_down_conn_inds

        inter_residue_connections64[
            nz_res_is_real_and_not_n_term_pose_ind,
            nz_res_is_real_and_not_n_term_res_ind,
            connected_down_conn_inds,
            0,  # residue id
        ] = nz_res_is_real_and_not_c_term_res_ind
        inter_residue_connections64[
            nz_res_is_real_and_not_n_term_pose_ind,
            nz_res_is_real_and_not_n_term_res_ind,
            connected_down_conn_inds,
            1,  # connection id
        ] = connected_up_conn_inds

        if chain_lengths:
            n_chains = [len(c_lens) for c_lens in chain_lengths]
            max_n_chains_minus1 = max(n_chains) - 1
            n_chains = torch.tensor(n_chains, dtype=torch.int64, device=device)
            chain_lengths_t = torch.full(
                (n_poses, max_n_chains_minus1), -1, dtype=torch.int64
            )
            for i, c_lens in enumerate(chain_lengths):
                for j, l in enumerate(c_lens):
                    if j != len(c_lens) - 1:
                        # we will leave off the last chain from each pose
                        chain_lengths_t[i, j] = l
            chain_lengths_t = chain_lengths_t.to(device)
            cl_real = chain_lengths_t != -1
            # cl_real = cl_real[:, :-1]
            cl_offsets = torch.cumsum(chain_lengths_t, dim=1)
            nz_cl_real_pose_ind, _ = torch.nonzero(cl_real, as_tuple=True)
            n_term_res = cl_offsets[cl_real]
            c_term_res = n_term_res - 1

            cterm_bts = block_type_ind64[nz_cl_real_pose_ind, c_term_res]
            up_conn_for_cterm = pbt.up_conn_inds[cterm_bts].to(torch.int64)

            nterm_bts = block_type_ind64[nz_cl_real_pose_ind, n_term_res]
            down_conn_for_nterm = pbt.down_conn_inds[nterm_bts].to(torch.int64)

            # sentinel out the down connection residue and connection
            inter_residue_connections64[
                nz_cl_real_pose_ind, n_term_res, down_conn_for_nterm, 0:2
            ] = -1
            # sentinel out the up connection residue and connection
            inter_residue_connections64[
                nz_cl_real_pose_ind, c_term_res, up_conn_for_cterm, 0:1
            ] = -1

        return inter_residue_connections64

    @classmethod
    @validate_args
    def _incorporate_extra_connections_into_inter_res_conn_set(
        cls,
        expoly_connections: List[List[Tuple[int, int, int, int]]],
        inter_residue_connections64: Tensor[torch.int64][:, :, :, 2],
    ):
        device = inter_residue_connections64.device
        # a:
        expoly_conn_pose_ind = torch.tensor(
            [i for i, pconn_list in enumerate(expoly_connections) for _ in pconn_list],
            dtype=torch.int64,
            device=device,
        )
        expoly_conns_t = torch.tensor(
            [
                conn_info
                for pconn_list in expoly_connections
                for conn_info in pconn_list
            ],
            dtype=torch.int64,
            device=device,
        )
        expoly_conn1_block_ind = expoly_conns_t[:, 0]
        expoly_conn1_conn_ind = expoly_conns_t[:, 1]
        expoly_conn2_block_ind = expoly_conns_t[:, 2]
        expoly_conn2_conn_ind = expoly_conns_t[:, 3]

        inter_residue_connections64[
            expoly_conn_pose_ind, expoly_conn1_block_ind, expoly_conn1_conn_ind, 0
        ] = expoly_conn2_block_ind
        inter_residue_connections64[
            expoly_conn_pose_ind, expoly_conn1_block_ind, expoly_conn1_conn_ind, 1
        ] = expoly_conn2_conn_ind

        inter_residue_connections64[
            expoly_conn_pose_ind, expoly_conn2_block_ind, expoly_conn2_conn_ind, 0
        ] = expoly_conn1_block_ind
        inter_residue_connections64[
            expoly_conn_pose_ind, expoly_conn2_block_ind, expoly_conn2_conn_ind, 1
        ] = expoly_conn1_conn_ind

    @classmethod
    def _incorporate_inter_residue_connections_into_connectivity_graph(
        cls, inter_residue_connections, pconn_offset, pconn_matrix
    ):
        real_connections = inter_residue_connections[:, :, :, 0] != -1
        (
            nz_real_conn_pose_ind,
            nz_real_conn_block_ind,
            nz_real_conn_conn_ind,
        ) = torch.nonzero(real_connections, as_tuple=True)

        pconn_from = (
            pconn_offset[nz_real_conn_pose_ind, nz_real_conn_block_ind]
            + nz_real_conn_conn_ind
        )
        real_to_block = inter_residue_connections[
            nz_real_conn_pose_ind, nz_real_conn_block_ind, nz_real_conn_conn_ind, 0
        ]
        pconn_to = (
            pconn_offset[nz_real_conn_pose_ind, real_to_block]
            + inter_residue_connections[
                nz_real_conn_pose_ind, nz_real_conn_block_ind, nz_real_conn_conn_ind, 1
            ]
        )

        pconn_matrix[nz_real_conn_pose_ind, pconn_from, pconn_to] = 1

    @classmethod
    @validate_args
    def _calculate_interblock_bondsep_from_connectivity_graph(
        cls,
        pbt,
        block_n_conn: Tensor[torch.int32][:, :],
        pose_n_pconn: Tensor[torch.int32][:],
        pconn_matrix: Tensor[torch.int32][:, :, :],
    ):
        return cls._calculate_interblock_bondsep_from_connectivity_graph_heavy(
            pbt.max_n_conn, pbt.device, block_n_conn, pose_n_pconn, pconn_matrix
        )

    @classmethod
    @validate_args
    def _calculate_interblock_bondsep_from_connectivity_graph_heavy(
        cls,
        pbt_max_n_conn,
        pbt_device,
        block_n_conn: Tensor[torch.int32][:, :],
        pose_n_pconn: Tensor[torch.int32][:],
        pconn_matrix: Tensor[torch.int32][:, :, :],
    ):
        n_poses = block_n_conn.shape[0]
        max_n_blocks = block_n_conn.shape[1]
        max_n_conn = pbt_max_n_conn
        max_n_pconn = pconn_matrix.shape[1]
        assert pconn_matrix.shape[0] == n_poses
        assert pconn_matrix.shape[1] == pconn_matrix.shape[2]

        # the final destination tensor that we will have to carefully write to
        # indexed pose-id x r1 x r2 x c1 x c2
        # but, we will see later, has to be transposed at the r2/c1 dimensions
        # so we can write to it in the correct order
        inter_block_bondsep = torch.full(
            (n_poses, max_n_blocks, max_n_blocks, max_n_conn, max_n_conn),
            MAX_SIG_BOND_SEPARATION,
            dtype=torch.int32,
            device=pbt_device,
        )

        bconn_inds1 = (
            torch.arange(max_n_conn, dtype=torch.int64, device=pbt_device)
            .repeat(n_poses * max_n_blocks * max_n_blocks * max_n_conn)
            .view(n_poses, max_n_blocks, max_n_blocks, max_n_conn, max_n_conn)
        )
        bconn_inds2 = torch.transpose(bconn_inds1, 3, 4)
        # not all entries in the inter_block_bondsep tensor correspond to
        # actual connections; this boolean tensor helps us identify the
        # real entries
        bconn_pair_real = torch.logical_and(
            bconn_inds1 < block_n_conn[:, None, :, None, None],
            bconn_inds2 < block_n_conn[:, :, None, None, None],
        )

        # in order to assign from a tensor ordered pose_id x pconn1 x pconn2
        # we have to make these tensors (temporarily) indexed
        # pose_id x r1 x c1 x r2 x c2
        # later we will re-transpose these dimensions before returning
        # the inter_block_bondsep tensor
        bconn_pair_real = torch.transpose(bconn_pair_real, 2, 3)
        inter_block_bondsep = torch.transpose(inter_block_bondsep, 2, 3)

        pconn_inds1 = (
            torch.arange(max_n_pconn, dtype=torch.int64, device=pbt_device)
            .repeat(n_poses * max_n_pconn)
            .view(n_poses, max_n_pconn, max_n_pconn)
        )
        pconn_inds2 = torch.transpose(pconn_inds1, 1, 2)
        # same deal as with the bconn_pair_real tensor
        pconn_pair_real = torch.logical_and(
            pconn_inds1 < pose_n_pconn[:, None, None],
            pconn_inds2 < pose_n_pconn[:, None, None],
        )

        # invoke the all-pairs-shortest-path algorithm on the connectivity
        # matrices
        cls._shortest_paths_for_connectivity_graph(pconn_matrix)

        # the big assignment!
        # print("inter_block_bondsep", inter_block_bondsep.shape)
        # print("bconn_pair_real", bconn_pair_real.shape, torch.sum(bconn_pair_real))
        # print("pconn_matrix", pconn_matrix.shape)
        # print("pconn_pair_real", pconn_pair_real.shape, torch.sum(pconn_pair_real))

        inter_block_bondsep[bconn_pair_real] = pconn_matrix[pconn_pair_real]

        # now reorder so it's pose-ind x r1 x r2 x c1 x c2
        inter_block_bondsep = torch.transpose(inter_block_bondsep, 2, 3)

        return inter_block_bondsep

    @classmethod
    @validate_args
    def _shortest_paths_for_connectivity_graph(cls, pconn_matrix):
        from tmol.pose.compiled.apsp_ops import stacked_apsp

        stacked_apsp(pconn_matrix, MAX_SIG_BOND_SEPARATION)

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

        # Let's take an example of a single four residue alpha amino acid chain:
        #    "MT[Beta-Ala]Q"
        # (This example has been chosen to  ensure that our logic is sound; if
        # all the backbone types had the same number of atoms, then we might
        # compute the right answer for the wrong reason if we worked only with
        # alpha amino acids, but then fail once we attempt to build this
        # tensor for a mix of alpha- and beta amino acids.)
        #
        # What we want is the connection-point distance tensor of:
        #
        # [ [ [[0,2],[2,0]],   [[3,5],[1,3]], [[6,9],[4,7]], [[10,12],[8,10]] ]
        #   [ [[3,1],[5,3]],   [[0,2],[2,0]], [[3,6],[1,4]], [[7,9],[5,7]]    ]
        #   [ [[6,4],[9,7]],   [[3,1],[6,4]], [[0,3],[3,0]], [[4,6],[1,3]]    ]
        #   [ [[10,8],[12,10], [[7,5],[9,7]], [[4,1],[6,3]], [[0, 2],[2,0]]   ]]
        #
        # Where, e.g., the [0, 2, :, :] matrix [[6,9],[4,7]] says:
        # - N on residue 0 is 6 chemical bonds from N on residue 2
        # - N on residue 0 is 9 chemical bonds from C on residue 2
        # - C on residue 0 is 4 chemical bonds from N on residue 2
        # - C on residue 0 is 7 chemical bonds from C on residue 2
        #
        # What about if we have [Met:NTerm]T[Beta-Ala][GLN:CTerm] ?
        # Then we would want this tensor instead
        # (note that "up" on res 0 and "down" on res 3 are both ind 0)
        # [ [ [[0,-1],[-1,-1]],  [[1,3],[-1,-1]], [[4,7],[-1,-1]], [[8,-1],[-1,-1]]  ]   # noqa: B950
        #   [ [[1,-1],[3,-1]],   [[0,2],[2,0]],   [[3,6],[1,4]],   [[7,-1],[5,-1]]   ]   # noqa: B950
        #   [ [[4,-1],[7,-1]],   [[3,1],[6,4]],   [[0,3],[3,0]],   [[4,-1],[1,-1]]   ]   # noqa: B950
        #   [ [[8,-1],[-1,-1],   [[7,5],[-1,-1]], [[4,1],[-1,-1]], [[0, -1],[-1,-1]] ]]  # noqa: B950
        #
        # So how do we build this tensor? Let's think about it in terms of a
        # single chain. We will add an extra dimension later to talk about
        # multiple chains. Let's start by reading out the length of the down-to-up path
        # separation for each block type. In the former case, A = [2, 2, 3, 2] and in
        # the latter case, A = [0, 2, 3, 0] with zero entries for the residues missing
        # one of the connections. What the two cases will have in common is the subset
        # of residues that contain both up and down connections. Let's focus on the
        # former case and return later to the latter. If we were to add 1 to A, we
        # would have the chemical bond path distance between down-to-down or up-to-up
        # pairs. An exclusive cumulative sum of A+1 would give B = [0, 3, 6, 10]:
        # the down-to-down path distance from residue 0 to residue i.
        # We can do a broadcast subtraction C = B[None,:] - B[:, None]
        # giving
        #
        # [[0,    3,  6, 10],
        #  [-3,   0,  3,  7],
        #  [-6,  -3,  0,  4],
        #  [-10, -7, -4,  0]]
        #
        # so that the absolute value of each entry represents the down-to-down
        # path distance.
        #
        # Then we can compute the output tensor D:
        # D[i, j, down, down] = abs(C[i,j])
        # when i < j, D[i,j,down,up] = C[i,j] + A[j] = abs(C[i,j] + A[j])
        #             D[i,j,up,down] = C[i,j] - A[i]) = abs(C[i,j] - A[i]))
        #             D[i,j, up, up] = C[i,j] - A[i] + A[j]
        # when j < i, D[i,j,down,up] = -1 * C[i,j] - A[j]
        #                            = -1 * (C[i,j] + A[j])
        #                            = abs(C[i,j] + A[j])
        #             D[i,j,up,down] = -1 * C[i,j] + A[i]
        #                            = -1 * (C[i,j] - A[i])
        #                            = abs(C[i,j] - A[i])
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

        # now we need the indices for the down and up connections that correspond to
        # the nz_brr indices
        nz_brr_upconn_1 = bt_up_conn_inds[nz_brr_bt_1].to(torch.int64)
        nz_brr_upconn_2 = bt_up_conn_inds[nz_brr_bt_2].to(torch.int64)
        nz_brr_downconn_1 = bt_down_conn_inds[nz_brr_bt_1].to(torch.int64)
        nz_brr_downconn_2 = bt_down_conn_inds[nz_brr_bt_2].to(torch.int64)

        # finally we can enter the information for these connections into their
        # positions in the output tensor
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
