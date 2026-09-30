import copy

import itertools
from dataclasses import replace

import numpy
import torch
import pandas

from typing import List, Tuple, Optional

from tmol.types import (
    NDArray,
    Tensor,
    validate_args,
)
from tmol.chemical import (
    MAX_SIG_BOND_SEPARATION,
    RefinedResidueType,
    three2one,
)

from tmol.pose import (
    InterBlockBondsep,
    PackedBlockTypes,
    PDBInfo,
    DEFAULT_ATOM_B_FACTOR,
    DEFAULT_ATOM_OCCUPANCY,
    PoseStack,
    ConstraintSet,
    SplitBlockMapping,
)
from tmol.utility.tensor import (
    exclusive_cumsum1d,
    exclusive_cumsum2d,
)
from tmol.utility._device import resolve_device


def _is_leading_run(block_types, pbt) -> bool:
    """Whether block_types begin pbt's active block types, as the same objects."""
    active = pbt.active_block_types
    return len(block_types) <= len(active) and all(
        a is b for a, b in zip(block_types, active)
    )


class PoseStackBuilder:
    """Build heterogeneous pose stacks while preserving chemical-database identity."""

    @staticmethod
    def _widest_packed_block_types(pose_stacks):
        """The packed block types over the largest chemical database.

        Databases only grow by appending residues, whether ligands or metal
        donor forms, so a database whose residues are a leading run of
        another's, as the same objects, names nothing the larger one does not
        mean identically. Any other pair was built from unrelated sources.
        """
        widest = max(
            (ps.packed_block_types for ps in pose_stacks),
            key=lambda pbt: (len(pbt.chem_db.residues), pbt.n_types),
        )
        residues = widest.chem_db.residues
        for ps in pose_stacks:
            chem_db = ps.packed_block_types.chem_db
            if chem_db is widest.chem_db:
                continue
            shared = residues[: len(chem_db.residues)]
            if len(shared) != len(chem_db.residues) or any(
                a is not b for a, b in zip(shared, chem_db.residues)
            ):
                raise ValueError(
                    "pose stacks were built from chemical databases neither of "
                    "which extends the other; build them from one context"
                )
        return widest

    @classmethod
    @validate_args
    def from_poses(
        cls, pose_stacks: List[PoseStack], device: torch.device
    ) -> PoseStack:
        """Combine one or more pose stacks on a common device.

        Args:
            pose_stacks: Pose stacks whose chemical databases are one database
                or extensions of one another.
            device: Device for the combined tensors.

        Returns:
            A padded pose stack containing every input pose in order. Split-
            block mappings are concatenated when present.
        """
        device = resolve_device(device)
        pbt0 = cls._widest_packed_block_types(pose_stacks)
        # a grown generation keeps every earlier block type at its index
        reuse_pbt = all(
            _is_leading_run(ps.packed_block_types.active_block_types, pbt0)
            for ps in pose_stacks
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

        n_poses = sum(len(ps) for ps in pose_stacks)
        ps_offset = exclusive_cumsum1d(
            torch.tensor([len(ps) for ps in pose_stacks], dtype=torch.int64)
        )

        inter_residue_connections = cls._inter_residue_connections_from_pose_stacks(
            packed_block_types, pose_stacks, n_poses, ps_offset, max_n_blocks, device
        )
        inter_block_bondsep = InterBlockBondsep.concatenate(
            [ps.inter_block_bondsep.to(device) for ps in pose_stacks],
            max_n_blocks,
            packed_block_types.max_n_conn,
        )
        block_type_ind = cls._resolve_block_type_ind(
            packed_block_types, pose_stacks, n_poses, ps_offset, max_n_blocks, device
        )

        chain_id = cls._chain_id_from_pose_stacks(
            pose_stacks, n_poses, ps_offset, max_n_blocks, device
        )
        pdb_info = cls._pdb_info_from_pose_stacks(
            pose_stacks, n_poses, ps_offset, max_n_blocks
        )

        # Concatenate the constraint sets
        constraint_set = ConstraintSet.concatenate(
            [ps.constraint_set for ps in pose_stacks],
            from_multiple_pose_stacks=True,
            n_poses=n_poses,
            ps_offset=ps_offset,
        )
        split_block_mapping = cls._split_block_mapping_from_pose_stacks(
            packed_block_types, pose_stacks
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
            block_type_ind=block_type_ind,
            block_type_ind64=i64(block_type_ind),
            chain_id=chain_id,
            chain_id64=i64(chain_id),
            pdb_info=pdb_info,
            constraint_set=constraint_set,
            device=coords.device,
            split_block_mapping=split_block_mapping,
        )

    @staticmethod
    def _split_block_mapping_from_pose_stacks(
        packed_block_types: PackedBlockTypes,
        pose_stacks: List[PoseStack],
    ) -> SplitBlockMapping | None:
        """Concatenate split-block metadata and remap original block types."""
        if not any(
            pose_stack.split_block_mapping is not None
            and pose_stack.split_block_mapping.entries
            for pose_stack in pose_stacks
        ):
            return None

        block_type_index_by_name = {
            block_type.name: index
            for index, block_type in enumerate(packed_block_types.active_block_types)
        }
        combined_entries = []
        pose_offset = 0
        for pose_stack in pose_stacks:
            mapping = pose_stack.split_block_mapping
            if mapping is not None:
                source_block_types = pose_stack.packed_block_types.active_block_types
                combined_entries.extend(
                    replace(
                        entry,
                        pose_ind=pose_offset + entry.pose_ind,
                        orig_block_type_ind=block_type_index_by_name[
                            source_block_types[entry.orig_block_type_ind].name
                        ],
                    )
                    for entry in mapping.entries
                )
            pose_offset += pose_stack.n_poses

        return SplitBlockMapping(entries=tuple(combined_entries))

    @classmethod
    @validate_args
    def from_block_type_names(
        cls,
        packed_block_types: PackedBlockTypes,
        sequences,  # List[List[str]]
        chain_lengths,  # List[List[int]]
    ):
        """Construct a zero-coordinate PoseStack from per-pose block type names.

        A name may carry a non-polymeric connection as "NAME--conn-label", where
        "conn" is the name of an inter-residue connection on that block type and
        "label" pairs up the two partners. E.g. a pose with two disulfides:

        AAAA[CYD--dslf-first]AAA[CYD--dslf-second]AAA ...
        AA[CYD--dslf-second]AAAA[CYD--dslf-first]AAA
        """
        cls._annotate_pbt_w_canonical_aa1lc_lookup(packed_block_types)

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
        # 3) Finally, we search the graph of connection points, weighted by the
        # intra-residue connection distances of the PBT object and joined by the
        # inter_residue_connections64 bonds, for the bond separations below the cap

        # 1
        resolved_expoly_connections = cls._find_connection_pairs_for_residue_subset(
            pbt, sequences, block_type_ind64, expoly_connections
        )

        # 2a
        irc64, chain_id = cls._inter_residue_connections_for_polymeric_monomers(
            pbt, n_poses, max_n_res, real_res, n_res, block_type_ind64, chain_lengths
        )
        inter_residue_connections64 = irc64

        # 2b add in non-polymeric connections (such as disulfides)
        cls._incorporate_extra_connections_into_inter_res_conn_set(
            resolved_expoly_connections, inter_residue_connections64
        )

        # 3
        inter_block_bondsep = cls._inter_block_bondsep_from_connections(
            pbt, block_type_ind64, real_res, inter_residue_connections64
        )

        n_atoms = torch.zeros((n_poses, max_n_res), dtype=torch.int32, device=device)
        n_atoms[real_res] = pbt.n_atoms[block_type_ind64[real_res]]
        block_coord_offset = exclusive_cumsum2d(n_atoms)

        max_n_atoms = torch.max(torch.sum(n_atoms, dim=1)).item()

        max_n_chains = max(len(clens) for clens in chain_lengths)
        chain_ind_to_label = numpy.array(
            [chr(ord("A") + i) for i in range(max_n_chains)], dtype=object
        )
        chain_labels = numpy.full(block_type_ind64.shape, "", dtype=object)
        real_res_np = real_res.cpu().numpy()
        chain_labels[real_res_np] = chain_ind_to_label[
            chain_id.cpu().numpy()[real_res_np]
        ]
        residue_labels = numpy.full(block_type_ind64.shape, -1, dtype=int)
        arange1 = numpy.expand_dims(
            numpy.arange(max_n_res, dtype=int) + 1, axis=0
        ).repeat(n_poses, axis=0)
        residue_labels[real_res_np] = arange1[real_res_np]
        residue_insertion_codes = numpy.full(block_type_ind64.shape, "", dtype=object)
        atom_occupancy = numpy.full(
            (n_poses, max_n_atoms), DEFAULT_ATOM_OCCUPANCY, dtype=numpy.float32
        )
        atom_b_factor = numpy.full(
            (n_poses, max_n_atoms), DEFAULT_ATOM_B_FACTOR, dtype=numpy.float32
        )

        pdb_info = PDBInfo(
            residue_labels=residue_labels,
            residue_insertion_codes=residue_insertion_codes,
            chain_labels=chain_labels,
            atom_occupancy=atom_occupancy,
            atom_b_factor=atom_b_factor,
        )

        return PoseStack(
            packed_block_types=packed_block_types,
            coords=torch.zeros(
                (n_poses, max_n_atoms, 3), dtype=torch.float32, device=device
            ),
            block_coord_offset=block_coord_offset,
            block_coord_offset64=block_coord_offset.to(torch.int64),
            inter_residue_connections=inter_residue_connections64.to(torch.int32),
            inter_residue_connections64=inter_residue_connections64,
            inter_block_bondsep=inter_block_bondsep,
            block_type_ind=block_type_ind64.to(torch.int32),
            block_type_ind64=block_type_ind64,
            chain_id=chain_id,
            chain_id64=chain_id.to(torch.int64),
            pdb_info=pdb_info,
            constraint_set=None,
            device=device,
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
        n_poses: int,
        ps_offsets: Tensor[torch.int64][:],
        max_n_blocks: int,
        device: torch.device,
    ) -> Tensor[torch.int32][:, :, :, 2]:
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
    def _resolve_block_type_ind(
        cls,
        packed_block_types: PackedBlockTypes,
        pose_stacks,  #: List["PoseStack"],
        n_poses: int,
        ps_offsets: Tensor[torch.int64][:],
        max_n_blocks: int,
        device: torch.device,
    ):
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
    def _chain_id_from_pose_stacks(
        cls,
        pose_stacks,  # : List["PoseStack"],
        n_poses: int,
        ps_offsets: Tensor[torch.int64][:],
        max_n_blocks: int,
        device: torch.device,
    ) -> Tensor[torch.int32][:, :]:
        chain_id = torch.full(
            (n_poses, max_n_blocks),
            -1,
            dtype=torch.int32,
            device=device,
        )
        for i, pose_stack in enumerate(pose_stacks):
            offset = ps_offsets[i]
            i_nblocks = pose_stack.chain_id.shape[1]
            chain_id[offset : (offset + len(pose_stack)), :i_nblocks] = (
                pose_stack.chain_id
            )
        return chain_id

    @classmethod
    def _pdb_info_from_pose_stacks(
        cls,
        pose_stacks,  # : List["PoseStack"],
        n_poses: int,
        ps_offsets: Tensor[torch.int64][:],
        max_n_blocks: int,
    ) -> PDBInfo:
        ps_offsets = ps_offsets.cpu().numpy()

        max_n_atoms = max(ps.coords.shape[1] for ps in pose_stacks)

        residue_labels = numpy.full((n_poses, max_n_blocks), 0, dtype=int)
        residue_insertion_codes = numpy.full((n_poses, max_n_blocks), "", dtype=object)
        chain_labels = numpy.full((n_poses, max_n_blocks), "", dtype=object)
        atom_occupancy = numpy.full((n_poses, max_n_atoms), 1.0, dtype=numpy.float32)
        atom_b_factor = numpy.full((n_poses, max_n_atoms), 0.0, dtype=numpy.float32)
        metal_origins = (
            numpy.full((n_poses, max_n_blocks), None, dtype=object)
            if any(ps.pdb_info.metal_origins is not None for ps in pose_stacks)
            else None
        )
        annotations = [ps.pdb_info.residue_annotations for ps in pose_stacks]
        residue_annotations = (
            numpy.zeros((n_poses, max_n_blocks), dtype=annotations[0].dtype)
            if all(a is not None for a in annotations)
            and len({a.dtype for a in annotations}) == 1
            else None
        )

        for i, pose_stack in enumerate(pose_stacks):
            offset = ps_offsets[i]
            i_nblocks = pose_stack.pdb_info.residue_labels.shape[1]
            if pose_stack.pdb_info.metal_origins is not None:
                metal_origins[offset : (offset + len(pose_stack)), :i_nblocks] = (
                    pose_stack.pdb_info.metal_origins
                )
            if residue_annotations is not None:
                residue_annotations[offset : (offset + len(pose_stack)), :i_nblocks] = (
                    pose_stack.pdb_info.residue_annotations
                )
            residue_labels[offset : (offset + len(pose_stack)), :i_nblocks] = (
                pose_stack.pdb_info.residue_labels
            )
            residue_insertion_codes[offset : (offset + len(pose_stack)), :i_nblocks] = (
                pose_stack.pdb_info.residue_insertion_codes
            )
            chain_labels[offset : (offset + len(pose_stack)), :i_nblocks] = (
                pose_stack.pdb_info.chain_labels
            )
            i_natoms = pose_stack.coords.shape[1]
            atom_occupancy[offset : (offset + len(pose_stack)), :i_natoms] = (
                pose_stack.pdb_info.atom_occupancy
            )
            atom_b_factor[offset : (offset + len(pose_stack)), :i_natoms] = (
                pose_stack.pdb_info.atom_b_factor
            )
        return PDBInfo(
            residue_labels=residue_labels,
            residue_insertion_codes=residue_insertion_codes,
            chain_labels=chain_labels,
            atom_occupancy=atom_occupancy,
            atom_b_factor=atom_b_factor,
            metal_origins=metal_origins,
            residue_annotations=residue_annotations,
        )

    @classmethod
    @validate_args
    def _chain_labels_from_pose_stacks(
        cls,
        pose_stacks,  # : List["PoseStack"],
        ps_offsets: Tensor[torch.int64][:],
        max_n_blocks: int,
        device: torch.device,
    ) -> NDArray[object][:, :]:
        n_poses = sum(len(ps) for ps in pose_stacks)
        chain_labels = numpy.full((n_poses, max_n_blocks), "", dtype=object)
        for i, pose_stack in enumerate(pose_stacks):
            offset = ps_offsets[i]
            i_nblocks = pose_stack.chain_labels.shape[1]
            chain_labels[offset : (offset + len(pose_stack)), :i_nblocks] = (
                pose_stack.chain_labels
            )
        return chain_labels

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
        connection atoms: the weights of the graph of chemical bonds from which
        the chemical separation of the connection atoms is found.
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
    ) -> Tuple[Tensor[torch.int64][:, :, :, 2], Tensor[torch.int32][:, :]]:
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
                for j, chain_length in enumerate(c_lens):
                    if j != len(c_lens) - 1:
                        # we will leave off the last chain from each pose
                        chain_lengths_t[i, j] = chain_length
            chain_lengths_t = chain_lengths_t.to(device)
            cl_real = chain_lengths_t != -1
            cl_offsets = torch.cumsum(chain_lengths_t, dim=1)

            # mark the first residue of each chain with a 1, then run inclusive cumsum
            # to get chain IDs
            cl_real_real = (chain_lengths_t != -1) & (cl_offsets <= n_res[:, None])
            cl_real_real_pose_ind, _ = torch.nonzero(cl_real_real, as_tuple=True)
            chain_id = torch.zeros(
                (n_poses, max_n_res), dtype=torch.int32, device=device
            )
            chain_id[cl_real_real_pose_ind, cl_offsets[cl_real_real]] = 1
            chain_id = torch.cumsum(chain_id, dim=1).to(torch.int32)
            chain_id[torch.logical_not(real_res)] = -1

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
        else:
            # everything is in a single chain, so mark the real residues as chain 0
            chain_id = torch.full(
                (n_poses, max_n_res), -1, dtype=torch.int32, device=device
            )
            chain_id[real_res] = 0

        return inter_residue_connections64, chain_id

    @classmethod
    @validate_args
    def _incorporate_extra_connections_into_inter_res_conn_set(
        cls,
        expoly_connections: List[List[Tuple[int, int, int, int]]],
        inter_residue_connections64: Tensor[torch.int64][:, :, :, 2],
    ):
        if not any(expoly_connections):
            return
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
    def _inter_block_bondsep_from_connections(
        cls,
        pbt: PackedBlockTypes,
        block_type_ind64: Tensor[torch.int64][:, :],
        real_blocks: Tensor[torch.bool][:, :],
        inter_residue_connections64: Tensor[torch.int64][:, :, :, 2],
    ) -> InterBlockBondsep:
        """Bond separations between the connections of nearby blocks.

        Searches the bonded graph of the blocks' connections only for
        separations below ``MAX_SIG_BOND_SEPARATION``.
        """
        cls._annotate_pbt_w_intraresidue_connection_atom_distances(pbt)
        counts = torch.zeros_like(block_type_ind64, dtype=torch.int32)
        counts[real_blocks] = pbt.n_conn[block_type_ind64[real_blocks]]
        max_n_conn = pbt.conn_at_intrablock_bond_sep.shape[1]
        intra_separation = torch.full(
            (*block_type_ind64.shape, max_n_conn, max_n_conn),
            MAX_SIG_BOND_SEPARATION,
            dtype=torch.int32,
            device=pbt.device,
        )
        intra_separation[real_blocks] = pbt.conn_at_intrablock_bond_sep[
            block_type_ind64[real_blocks]
        ]
        return InterBlockBondsep.from_bonded_graph(
            counts, intra_separation, inter_residue_connections64
        )
