import torch
import attr
import numpy

from typing import Tuple

from tmol.types.torch import Tensor
from tmol.types.functional import validate_args

from tmol.score.dunbrack.params import (
    SamplingDunbrackDatabaseView,
    DunbrackParamResolver,
)

from tmol.pack.packer_task import PackerTask, ResidueLevelTask
from tmol.system.restypes import RefinedResidueType
from tmol.system.pose import PackedBlockTypes, Poses
from tmol.system.packed import PackedResidueSystem
from tmol.system.score_support import indexed_atoms_for_dihedral


@attr.s(auto_attribs=True)
class ChiSampler:
    def sample_chi_for_poses(self, systems: Poses, task: PackerTask):
        raise NotImplementedError()


@attr.s(auto_attribs=True)
class DunbrackChiSampler:

    dun_param_resolver: DunbrackParamResolver
    sampling_params: SamplingDunbrackDatabaseView

    @property
    def device(self):
        return self.dun_param_resolver.device

    @classmethod
    @validate_args
    def from_database(cls, param_resolver: DunbrackParamResolver):
        return cls(
            dun_param_resolver=param_resolver,
            sampling_params=param_resolver.sampling_db,
        )

    @validate_args
    def annotate_residue_type(self, restype: RefinedResidueType):
        """TEMP TEMP TEMP: assume the dihedrals we care about are phi and psi"""
        if hasattr(restype, "dun_sampler_bbdihe_uaids"):
            return
        # #chi = 2; #atoms in a dihedral = 4; #entries in a uaid  = 3
        uaids = numpy.full((2, 4, 3), -1, dtype=numpy.int32)
        if "phi" in restype.torsion_to_uaids:
            uaids[0] = numpy.array(restype.torsion_to_uaids["phi"], dtype=numpy.int32)
        if "psi" in restype.torsion_to_uaids:
            uaids[1] = numpy.array(restype.torsion_to_uaids["psi"], dtype=numpy.int32)
        setattr(restype, "dun_sampler_bbdihe_uaids", uaids)

    @validate_args
    def annotate_packed_block_types(self, packed_block_types: PackedBlockTypes):
        if hasattr(packed_block_types, "dun_sampler_bbdihe_uaids"):
            return
        uaids = numpy.stack(
            [rt.dun_sampler_bbdihe_uaids for rt in packed_block_types.active_residues]
        )
        # print("uaids")
        # print(uaids.shape)
        # print(uaids)
        uaids = torch.tensor(uaids, dtype=torch.int32, device=self.device)
        setattr(packed_block_types, "dun_sampler_bbdihe_uaids", uaids)

    @validate_args
    def OLD_chi_samples_for_residues(
        self,
        system: PackedResidueSystem,
        coords: Tensor(torch.float32)[:, 3],
        task: PackerTask,
    ) -> Tuple[
        Tensor(torch.int32)[:],
        Tensor(torch.int32)[:],
        Tensor(torch.int32)[:],
        Tensor(torch.float32)[:, :],
    ]:
        dev = coords.device

        all_allowed_restypes = numpy.array(
            [rt for rlt in task.rlts for rt in rlt.allowed_restypes], dtype=object
        )
        # print("all_allowed_restypes", all_allowed_restypes.shape, [rt.name for rt in all_allowed_restypes])
        rt_names = numpy.array(
            [rt.name for rlt in task.rlts for rt in rlt.allowed_restypes], dtype=object
        )
        # print("rt_names", len(rt_names))
        # print(rt_names)
        rt_res = numpy.array(
            [i for i, rlt in enumerate(task.rlts) for rt in rlt.allowed_restypes],
            dtype=numpy.int32,
        )
        # print("rt_res", len(rt_res))

        dun_rot_inds_for_rts = self.dun_param_resolver._indices_from_names(
            self.dun_param_resolver.all_table_indices,
            rt_names[None, :],
            torch.device("cpu"),
        ).squeeze()

        # print("dun_rot_inds_for_rts", dun_rot_inds_for_rts.shape)
        # print(dun_rot_inds_for_rts)

        inds_of_phi_res = indexed_atoms_for_dihedral(system, "phi")
        inds_of_psi_res = indexed_atoms_for_dihedral(system, "psi")
        inds_of_phi_res = numpy.concatenate(
            (
                inds_of_phi_res[:, :1],
                numpy.zeros((inds_of_phi_res.shape[0], 1), dtype=numpy.int32),
                inds_of_phi_res[:, 1:],
            ),
            axis=1,
        )
        inds_of_psi_res = numpy.concatenate(
            (
                inds_of_psi_res[:, :1],
                numpy.ones((inds_of_psi_res.shape[0], 1), dtype=numpy.int32),
                inds_of_psi_res[:, 1:],
            ),
            axis=1,
        )
        join_phi_psi = numpy.concatenate((inds_of_phi_res, inds_of_psi_res), 0)
        dihe_res = join_phi_psi[:, 0]
        dihe_inds = join_phi_psi[:, 1]
        sort_inds = numpy.lexsort((dihe_res, dihe_inds))
        phi_psi = join_phi_psi[sort_inds, :]

        nonzero_dunrot_inds_for_rts = torch.nonzero(dun_rot_inds_for_rts != -1)
        rottable_set_for_buildable_restype = dun_rot_inds_for_rts[
            nonzero_dunrot_inds_for_rts
        ]
        orig_residue_for_buildable_restype = rt_res[
            nonzero_dunrot_inds_for_rts.cpu().numpy()
        ]
        # print("orig_residue_for_buildable_restype", orig_residue_for_buildable_restype.shape)
        # print(orig_residue_for_buildable_restype)
        uniq_res_for_brt, uniq_inds = numpy.unique(
            orig_residue_for_buildable_restype, return_inverse=True
        )

        rottable_set_for_buildable_restype = torch.tensor(
            numpy.concatenate(
                (
                    uniq_inds.reshape(-1, 1),
                    rottable_set_for_buildable_restype.reshape(-1, 1),
                ),
                axis=1,
            ),
            dtype=torch.int32,
            device=dev,
        )

        phi_arange = numpy.arange(inds_of_phi_res.shape[0], dtype=numpy.int32)
        phi_res_inds = numpy.full((len(system.residues),), -1, dtype=numpy.int32)
        phi_res_inds[inds_of_phi_res[:, 0]] = phi_arange

        psi_arange = numpy.arange(inds_of_psi_res.shape[0], dtype=numpy.int32)
        psi_res_inds = numpy.full((len(system.residues),), -1, dtype=numpy.int32)
        psi_res_inds[inds_of_psi_res[:, 0]] = psi_arange

        n_sampling_res = uniq_res_for_brt.shape[0]

        dihedral_atom_inds = numpy.full((2 * n_sampling_res, 4), -1, dtype=numpy.int32)

        # map the residue-numbered list of dihedral angles to their positions in the
        # set of residues that the dunbrack library will provide chi samples for
        phi_reindexed = numpy.full((n_sampling_res, 4), -1, dtype=numpy.int32)
        phi_reindexed[phi_res_inds[uniq_res_for_brt] != -1] = inds_of_phi_res[
            phi_res_inds[uniq_res_for_brt][phi_res_inds[uniq_res_for_brt] != -1], 2:
        ]
        dihedral_atom_inds[
            numpy.arange(2 * n_sampling_res, dtype=int) % 2 == 0
        ] = phi_reindexed

        psi_reindexed = numpy.full((n_sampling_res, 4), -1, dtype=numpy.int32)
        psi_reindexed[psi_res_inds[uniq_res_for_brt] != -1] = inds_of_psi_res[
            psi_res_inds[uniq_res_for_brt][psi_res_inds[uniq_res_for_brt] != -1], 2:
        ]
        dihedral_atom_inds[
            numpy.arange(2 * n_sampling_res, dtype=int) % 2 == 1
        ] = psi_reindexed

        dihedral_atom_inds = torch.tensor(
            dihedral_atom_inds, dtype=torch.int32, device=dev
        )

        ndihe_for_res = torch.full((n_sampling_res,), 2, dtype=torch.int32, device=dev)
        dihedral_offset_for_res = 2 * torch.arange(
            n_sampling_res, dtype=torch.int32, device=dev
        )

        n_brts = nonzero_dunrot_inds_for_rts.shape[0]
        chi_expansion_for_buildable_restype = torch.full(
            (n_brts, 4), 0, dtype=torch.int32, device=dev
        )

        # ok, we'll go to the residue types and look at their protonation state expansions
        # and we'll put that information into the chi_expansions_for_buildable_restype
        # tensor

        # print("non-negative", dun_rot_inds_for_rts != -1)
        # print(dun_rot_inds_for_rts.numpy().shape)
        # print(all_allowed_restypes.shape)
        # sele =
        # print(sele)

        nchi_for_buildable_restype = self.sampling_params.nchi_for_table_set[
            rottable_set_for_buildable_restype[:, 1].to(torch.int64)
        ]

        brts = all_allowed_restypes[dun_rot_inds_for_rts.numpy() != -1]

        non_dunbrack_expansion_counts_for_buildable_restype = torch.full(
            (n_brts, 4), 0, dtype=torch.int32, device=dev
        )
        max_chi_samples = 0
        for i, rt in enumerate(brts):
            # print(i, rt.name)
            for j, rt_scb in enumerate(rt.sidechain_building):
                # print("count", rt.name, j, len(rt_scb.chi_samples))
                for k, rt_scb_chi in enumerate(rt_scb.chi_samples):
                    chi_name = rt_scb_chi.chi_dihedral
                    # weird assumption: chi are named chiX with X an integer
                    assert chi_name[:3] == "chi"
                    chi_ind = int(chi_name[3:]) - 1
                    if chi_ind >= nchi_for_buildable_restype[i]:
                        nchi_for_buildable_restype[i] = chi_ind + 1
                    n_expansions = (
                        1
                        + 2
                        * len(rt_scb_chi.expansions)
                        * chi_expansion_for_buildable_restype[i, chi_ind]
                    )
                    n_samples = len(rt_scb_chi.samples)
                    n_expanded_samples = n_samples * n_expansions
                    max_chi_samples = max(max_chi_samples, n_expanded_samples)
                    non_dunbrack_expansion_counts_for_buildable_restype[
                        i, chi_ind
                    ] = n_expanded_samples
                    # print("non dunbrack expansion counts", i, j, k, rt.name, n_expanded_samples)

        non_dunbrack_expansion_for_buildable_restype = torch.full(
            (n_brts, 4, max_chi_samples), -1, dtype=torch.float32, device=dev
        )

        for i, rt in enumerate(brts):
            for j, rt_scb in enumerate(rt.sidechain_building):
                for k, rt_scb_chi in enumerate(rt_scb.chi_samples):
                    chi_name = rt_scb_chi.chi_dihedral
                    # weird assumption: chi are named chiX with X an integer
                    assert chi_name[:3] == "chi"
                    chi_ind = int(chi_name[3:]) - 1
                    n_expansions = (
                        1
                        + 2
                        * len(rt_scb_chi.expansions)
                        * chi_expansion_for_buildable_restype[i, chi_ind]
                    )
                    n_samples = len(rt_scb_chi.samples)
                    n_expanded_samples = n_samples * n_expansions
                    for l in range(n_samples):
                        # print("  ",l)
                        for m in range(n_expansions):
                            # print("    ", m)
                            if m == 0:
                                non_dunbrack_expansion_for_buildable_restype[
                                    i, chi_ind, n_expansions * l + m
                                ] = rt_scb_chi.samples[l]
                                # print("rt_scb_chi.samples[",l,"]", rt_scb_chi.samples[l])
                            else:
                                expansion = (m - 1) // 2
                                factor = -1 if (m - 1) % 2 == 0 else 1
                                non_dunbrack_expansion_for_buildable_restype[
                                    i, chi_ind, n_expansions * l + m
                                ] = (
                                    rt_scb_chi.samples[l]
                                    + factor * rt_scb_chi.expansions[expansion]
                                )

        # treat all residues as if they are exposed
        prob_cumsum_limit_for_buildable_restype = torch.full(
            (n_brts,), 0.95, dtype=torch.float32, device=dev
        )

        # print("ndihe_for_res")
        # print(ndihe_for_res.shape)
        # print(ndihe_for_res)
        # print("dihedral_offset_for_res")
        # print(dihedral_offset_for_res.shape)
        # print(dihedral_offset_for_res)
        # print("dihedral_atom_inds")
        # print(dihedral_atom_inds.shape)
        # print(dihedral_atom_inds)
        # print("rottable_set_for_buildable_restype")
        # print(rottable_set_for_buildable_restype.shape)
        # print(rottable_set_for_buildable_restype)
        # print("chi_expansion_for_buildable_restype")
        # print(chi_expansion_for_buildable_restype.shape)
        # print(chi_expansion_for_buildable_restype)
        # print("non_dunbrack_expansion_for_buildable_restype")
        # print(non_dunbrack_expansion_for_buildable_restype.shape)
        # print(non_dunbrack_expansion_for_buildable_restype)
        # print("non_dunbrack_expansion_counts_for_buildable_restype")
        # print(non_dunbrack_expansion_counts_for_buildable_restype.shape)
        # print(non_dunbrack_expansion_counts_for_buildable_restype)
        # print("prob_cumsum_limit_for_buildable_restype")
        # print(prob_cumsum_limit_for_buildable_restype.shape)
        # print(prob_cumsum_limit_for_buildable_restype)
        # print("nchi_for_buildable_restype")
        # print(nchi_for_buildable_restype.shape)
        # print(nchi_for_buildable_restype)

        retval = torch.ops.tmol.dun_sample_chi(
            coords,
            self.sampling_params.rotameric_prob_tables,
            self.sampling_params.rotprob_table_sizes,
            self.sampling_params.rotprob_table_strides,
            self.sampling_params.rotameric_mean_tables,
            self.sampling_params.rotameric_sdev_tables,
            self.sampling_params.rotmean_table_sizes,
            self.sampling_params.rotmean_table_strides,
            self.sampling_params.rotameric_meansdev_tableset_offsets,
            self.sampling_params.rotameric_bb_start,
            self.sampling_params.rotameric_bb_step,
            self.sampling_params.rotameric_bb_periodicity,
            self.sampling_params.semirotameric_tables,
            self.sampling_params.semirot_table_sizes,
            self.sampling_params.semirot_table_strides,
            self.sampling_params.semirot_start,
            self.sampling_params.semirot_step,
            self.sampling_params.semirot_periodicity,
            self.sampling_params.rotameric_rotind2tableind,
            self.sampling_params.semirotameric_rotind2tableind,
            self.sampling_params.all_chi_rotind2tableind,
            self.sampling_params.all_chi_rotind2tableind_offsets,
            self.sampling_params.n_rotamers_for_tableset,
            self.sampling_params.n_rotamers_for_tableset_offsets,
            self.sampling_params.sorted_rotamer_2_rotamer,
            self.sampling_params.nchi_for_table_set,
            self.sampling_params.rotwells,
            ndihe_for_res,
            dihedral_offset_for_res,
            dihedral_atom_inds,
            rottable_set_for_buildable_restype,
            chi_expansion_for_buildable_restype,
            non_dunbrack_expansion_for_buildable_restype,
            non_dunbrack_expansion_counts_for_buildable_restype,
            prob_cumsum_limit_for_buildable_restype,
            nchi_for_buildable_restype,
        )

        return tuple(retval)

    @validate_args
    def sample_chi_for_poses(
        self, systems: Poses, task: PackerTask
    ) -> Tuple[
        Tensor(torch.int32)[:, :],  # n_rots_for_brt
        Tensor(torch.int32)[:, :],  # n_rots_for_brt_offsets
        Tensor(torch.int32)[:, :],  # brt_for_rotamer
        Tensor(torch.float32)[:, :, :],  # chi_for_rotamers
    ]:
        dev = systems.coords.device
        n_sys = systems.block_inds.shape[0]
        max_n_blocks = systems.block_inds.shape[1]
        max_n_atoms = systems.coords.shape[2]

        all_allowed_restypes = numpy.array(
            [
                rt
                for one_pose_rlts in task.rlts
                for rlt in one_pose_rlts
                for rt in rlt.allowed_restypes
                if self in rlt.chi_samplers
            ],
            dtype=object,
        )

        n_allowed_per_pose = numpy.array(
            [
                len(rlt.allowed_restypes)
                for one_pose_rlts in task.rlts
                for rlt in one_pose_rlts
                if self in rlt.chi_samplers
            ],
            dtype=numpy.int32,
        )

        rt_names = numpy.array([rt.name for rt in all_allowed_restypes], dtype=object)

        rt_res = numpy.array(
            [
                i * max_n_blocks + j
                for i, one_pose_rlts in enumerate(task.rlts)
                for j, rlt in enumerate(one_pose_rlts)
                for rt in rlt.allowed_restypes
            ],
            dtype=numpy.int32,
        )

        dun_rot_inds_for_rts = self.dun_param_resolver._indices_from_names(
            self.dun_param_resolver.all_table_indices,
            rt_names[None, :],
            torch.device("cpu"),
        ).squeeze()

        inds_of_phi = self.atom_indices_for_backbone_dihedral(systems, 0).reshape(-1, 4)
        inds_of_phi = self.atom_indices_for_backbone_dihedral(systems, 1).reshape(-1, 4)

        phi_psi_inds = torch.hstack(
            (inds_of_phi.reshape(-1, 4), inds_of_psi.reshape(-1, 4))
        )
        phi_psi_inds = phi_psi_inds.reshape(-1, 4)

        nonzero_dunrot_inds_for_rts = torch.nonzero(dun_rot_inds_for_rts != -1)
        rottable_set_for_buildable_restype = dun_rot_inds_for_rts[
            nonzero_dunrot_inds_for_rts
        ]

        # the "indices" of the blocks that the block types we will be building come
        # from, assuming we are colapsing the n_sys x max_n_blocks into a single
        # numbering. We will need to keep this array as it will be used by the
        # caller to understand what block types we are defining samples for.
        # We will shortly be renumbering the residues to talk about only the ones
        # that we will build rotamers for
        orig_residue_for_buildable_restype = rt_res[nonzero_dunrot_inds_for_rts]

        uniq_res_for_brt, uniq_inds = numpy.unique(
            orig_residue_for_buildable_restype, return_inverse=True
        )

        rottable_set_for_buildable_restype = torch.tensor(
            numpy.concatenate(
                (
                    uniq_inds.reshape(-1, 1),
                    rottable_set_for_buildable_restype.reshape(-1),
                ),
                axis=1,
            ),
            dtype=torch.int32,
            device=dev,
        )

        phi_psi_res_inds = numpy.arange(n_sys * max_n_blocks, dtype=numpy.int32)

        n_sampling_res = uniq_res_for_brt.shape[0]

        # map the residue-numbered list of dihedral angles to their positions in
        # the set of residues that the Dunbrack library will provice chi samples for
        dihedral_atom_inds = numpy.full((2 * n_sampling_res, 4), -1, dtype=numpy.int32)
        dihedral_atom_inds[
            2 * numpy.arange(n_sampling_res, dtype=int), :
        ] = inds_of_phi[uniq_res_for_brt, :]
        dihedral_atom_inds[
            2 * numpy.arange(n_sampling_res, dtype=int) + 1, :
        ] = inds_of_psi[uniq_res_for_brt, :]

        ndihe_for_res = torch.full((n_sampling_res,), 2, dtype=torch.int32, device=dev)
        dihedral_offset_for_res = 2 * torch.arange(
            n_sampling_res, dtype=torch.int32, device=dev
        )

        n_brts = nonzero_dunrot_inds_for_res.shape[0]
        ### NOTE: improve logic here for determining max_n_chi
        max_n_chi = 4
        chi_expansion_for_buildable_restype = torch.full(
            (n_brts, max_n_chi), 0, dtype=torch.int32, device=dev
        )

        # ok, we'll go to the residue types and look at their protonation state expansions
        # and we'll put that information into the chi_expansions_for_buildable_restype
        # tensor

        nchi_for_buildable_restype = self.sampling_params.nchi_for_table_set[
            rottable_set_for_buildable_restype[:, 1].to(torch.int64)
        ]

        brts = all_allowed_restypes[dun_rot_inds_for_rts.numpy() != -1]

        non_dunbrack_expansion_counts_for_buildable_restype = torch.zeros(
            (n_brts, 4), dtype=torch.int32, device=dev
        )
        max_chi_samples = 0

        # MOST OF THIS LOGIC SHOULD BE MOVED INTO A SETUP PHASE WITH
        # THE RefinedResidueType
        for i, rt in enumerate(brts):
            for j in rt_chi in enumerate(rt.chi_samples):
                chi_name = rt_scb_chi.chi_dihedral
                assert chi_name[:, 3] == "chi"
                chi_ind = int(chi_name[3:]) - 1
                if chi_ind >= nchi_for_buildable_restype[i]:
                    nchi_for_buildable_restype[i] = chi_ind + 1
                n_expansions = (
                    1
                    + 2
                    * len(rt_chi.expansions)
                    * chi_expansion_for_buildable_restype[i, chi_ind]
                )
                n_samples = len(rt_chi.samples)
                n_example_samples = n_samples * n_expansions
                max_chi_samples = max(max_chi_samples, n_expanded_samples)
                non_dunbrack_expansion_counts_for_buildable_restype[
                    i, chi_ind
                ] = n_expanded_samples
        non_dunbrack_expansion_for_buildable_restype = torch.full(
            (n_brts, max_n_chi, max_chi_samples), -1, dtype=torch.float32, device=dev
        )

        for i, rt in enumerate(brts):
            for j, rt_chi in enumerate(rt.chi_samples):
                chi_name = rt_chi.chi_dihedral
                chi_ind = int(chi_name[3:]) - 1
                n_expansions = (
                    1
                    + 2
                    * len(rt_chi.expansions)
                    * chi_expansions_for_buildable_restype[i, chi_ind]
                )
                n_samples = len(rt_chi.samples)
                n_expanded_samples = n_samples * n_expansions
                for l in range(n_samples):
                    for m in range(n_expansions):
                        if m == 0:
                            non_dunbrack_expansion_for_buildable_restype[
                                i, chi_ind, n_expansions * l + m
                            ] = rt_chi.samples[l]
                        else:
                            expansion = (m - 1) // 2
                            factor = -1 if (m - 1) % 2 == 0 else 1
                            non_dunbrack_expansion_for_buildable_restype[
                                i, chi_ind, n_expansions * l + m
                            ] = (
                                rt_chi.samples[l]
                                + factor * rt_chi.expansions[expansion]
                            )
        # oof

        # treat all residues as if they are exposed
        prob_cumsum_limit_for_buildable_restype = torch.full(
            (n_brts,), 0.95, dtype=torch.float32, device=dev
        )

        retval = torch.ops.tmol.dun_sample_chi(
            systems.coords,
            self.sampling_params.rotameric_prob_tables,
            self.sampling_params.rotprob_table_sizes,
            self.sampling_params.rotprob_table_strides,
            self.sampling_params.rotameric_mean_tables,
            self.sampling_params.rotameric_sdev_tables,
            self.sampling_params.rotmean_table_sizes,
            self.sampling_params.rotmean_table_strides,
            self.sampling_params.rotameric_meansdev_tableset_offsets,
            self.sampling_params.rotameric_bb_start,
            self.sampling_params.rotameric_bb_step,
            self.sampling_params.rotameric_bb_periodicity,
            self.sampling_params.semirotameric_tables,
            self.sampling_params.semirot_table_sizes,
            self.sampling_params.semirot_table_strides,
            self.sampling_params.semirot_start,
            self.sampling_params.semirot_step,
            self.sampling_params.semirot_periodicity,
            self.sampling_params.rotameric_rotind2tableind,
            self.sampling_params.semirotameric_rotind2tableind,
            self.sampling_params.all_chi_rotind2tableind,
            self.sampling_params.all_chi_rotind2tableind_offsets,
            self.sampling_params.n_rotamers_for_tableset,
            self.sampling_params.n_rotamers_for_tableset_offsets,
            self.sampling_params.sorted_rotamer_2_rotamer,
            self.sampling_params.nchi_for_table_set,
            self.sampling_params.rotwells,
            ndihe_for_res,
            dihedral_offset_for_res,
            dihedral_atom_inds,
            rottable_set_for_buildable_restype,
            chi_expansion_for_buildable_restype,
            non_dunbrack_expansion_for_buildable_restype,
            non_dunbrack_expansion_counts_for_buildable_restype,
            prob_cumsum_limit_for_buildable_restype,
            nchi_for_buildable_restype,
        )

    @validate_args
    def atom_indices_for_backbone_dihedral(self, systems: Poses, bb_dihedral_ind: int):
        assert hasattr(systems.packed_block_types, "dun_sampler_bbdihe_uaids")
        n_systems = systems.block_inds.shape[0]
        max_n_blocks = systems.block_inds.shape[1]
        max_n_atoms = systems.coords.shape[2]

        pbts = systems.packed_block_types

        real = torch.nonzero(systems.block_inds >= 0)

        uaids = torch.full(
            (n_systems, max_n_blocks, 4, 3), -1, dtype=torch.int64, device=self.device
        )

        uaids[real[:, 0], real[:, 1], :, :] = pbts.dun_sampler_bbdihe_uaids[
            systems.block_inds[real[:, 0], real[:, 1]].to(torch.int64),
            bb_dihedral_ind,
            :,
            :,
        ].to(torch.int64)

        # what we will return
        dihe_atom_inds = torch.full(
            (n_systems, max_n_blocks, 4), -1, dtype=torch.int32, device=self.device
        )

        # copy over all atom ids from the uaids as if they were resolved; the
        # atoms that are unresolved will be overwritten later
        dihe_atom_inds[:] = uaids[:, :, :, 0]

        inter_res = torch.nonzero(uaids[:, :, :, 1] >= 0)

        print("uaids[inter_res[:, 0], inter_res[:, 1], inter_res[:, 2], 1],")
        print(uaids[inter_res[:, 0], inter_res[:, 1], inter_res[:, 2], 1])

        dest_res = systems.inter_residue_connections[
            inter_res[:, 0],
            inter_res[:, 1],
            uaids[inter_res[:, 0], inter_res[:, 1], inter_res[:, 2], 1],
            0,
        ]
        dest_conn = systems.inter_residue_connections[
            inter_res[:, 0],
            inter_res[:, 1],
            uaids[inter_res[:, 0], inter_res[:, 1], inter_res[:, 2], 1],
            1,
        ].to(torch.int64)

        # now which atom on the downstream residue is the one that
        # the source residue is pointint at
        dihe_atom_inds[inter_res[:, 0], inter_res[:, 1], inter_res[:, 2]] = (
            (inter_res[:, 0] * max_n_atoms * max_n_blocks).to(torch.int32)
            + dest_res * max_n_atoms
            + pbts.atom_downstream_of_conn[
                systems.block_inds[inter_res[:, 0], inter_res[:, 1]].to(torch.int64),
                dest_conn,
                uaids[inter_res[:, 0], inter_res[:, 1], inter_res[:, 2], 2],
            ]
        )

        return dihe_atom_inds
