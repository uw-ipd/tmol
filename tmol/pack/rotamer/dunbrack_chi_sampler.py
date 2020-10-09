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
            assert hasattr(restype, "dun_sampler_chi_defining_atom")
            return

        # #chi = 2; #atoms in a dihedral = 4; #entries in a uaid  = 3
        uaids = numpy.full((2, 4, 3), -1, dtype=numpy.int32)
        if "phi" in restype.torsion_to_uaids:
            uaids[0] = numpy.array(restype.torsion_to_uaids["phi"], dtype=numpy.int32)
        if "psi" in restype.torsion_to_uaids:
            uaids[1] = numpy.array(restype.torsion_to_uaids["psi"], dtype=numpy.int32)

        # ok; lets ask the appropriate library what chi it defines samples for
        # and also incorporate additional chi that the residue type itself
        # defines its own samples for, as long as this is actually a residue type
        # that the dunbrack library handles
        dun_lib_ind = self.dun_param_resolver._indices_from_names(
            self.dun_param_resolver.all_table_indices,
            numpy.array([[restype.name]], dtype=object),
            device=self.device,
        )[0, 0]

        if dun_lib_ind >= 0:

            n_chi = self.dun_param_resolver.scoring_db_aux.nchi_for_table_set[
                dun_lib_ind
            ].item()

            # The second atom (atom 1) in the chi defines the dihedral; this
            # atoms needs to be a member of the residue type, and not unresolved
            chi_names = ["chi%d" % (i + 1) for i in range(n_chi)]
            for chi_name in chi_names:
                assert chi_name in restype.torsion_to_uaids
            chi_defining_atom = numpy.array(
                [restype.torsion_to_uaids[chi_name][1][0] for chi_name in chi_names],
                dtype=numpy.int32,
            )
            assert numpy.all(chi_defining_atom >= 0)
            if restype.chi_samples:
                n_rt_chi_samples = len(restype.chi_samples)
                chi_inds = numpy.array(
                    [int(samp.chi_dihedral[3:]) for samp in restype.chi_samples],
                    dtype=int,
                )
                sort_inds = numpy.argsort(chi_inds)

                addnl_chi_defining_atom = numpy.array(
                    [
                        restype.torsion_to_uaids[restype.chi_samples[ind].chi_dihedral][
                            1
                        ][0]
                        for ind in sort_inds
                    ],
                    dtype=numpy.int32,
                )
                chi_defining_atom = numpy.concatenate(
                    (chi_defining_atom, addnl_chi_defining_atom)
                )
        else:
            chi_defining_atom = numpy.zeros((0,), dtype=numpy.int32)

        setattr(restype, "dun_sampler_bbdihe_uaids", uaids)
        setattr(restype, "dun_sampler_chi_defining_atom", chi_defining_atom)

    @validate_args
    def annotate_packed_block_types(self, packed_block_types: PackedBlockTypes):
        if hasattr(packed_block_types, "dun_sampler_bbdihe_uaids"):
            assert hasattr(packed_block_types, "dun_sampler_chi_defining_atom")
            return

        for rt in packed_block_types.active_residues:
            assert hasattr(rt, "dun_sampler_bbdihe_uaids")
            assert hasattr(rt, "dun_sampler_chi_defining_atom")

        uaids = numpy.stack(
            [rt.dun_sampler_bbdihe_uaids for rt in packed_block_types.active_residues]
        )
        uaids = torch.tensor(uaids, dtype=torch.int32, device=self.device)

        max_n_chi = max(
            rt.dun_sampler_chi_defining_atom.shape[0]
            for rt in packed_block_types.active_residues
        )

        chi_defining_atom = torch.full(
            (packed_block_types.n_types, max_n_chi),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        for i, rt in enumerate(packed_block_types.active_residues):
            chi_defining_atom[
                i, : rt.dun_sampler_chi_defining_atom.shape[0]
            ] = torch.tensor(
                rt.dun_sampler_chi_defining_atom, dtype=torch.int32, device=self.device
            )

        setattr(packed_block_types, "dun_sampler_bbdihe_uaids", uaids)
        setattr(packed_block_types, "dun_sampler_chi_defining_atom", chi_defining_atom)

    @validate_args
    def sample_chi_for_poses(
        self, systems: Poses, task: PackerTask
    ) -> Tuple[
        Tensor(torch.int32)[:, :, :],  # n_rots_for_rt
        Tensor(torch.int32)[:, :, :],  # n_rots_for_rt_offsets
        Tensor(torch.int32)[:, 3],  # rt_for_rotamer
        Tensor(torch.int32)[:, :],  # chi_defining_atom_for_rotamer
        Tensor(torch.float32)[:, :],  # chi_for_rotamers
    ]:
        assert self.device == systems.coords.device
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

        n_allowed_per_pose = torch.tensor(
            [
                len(rlt.allowed_restypes)
                for one_pose_rlts in task.rlts
                for rlt in one_pose_rlts
                if self in rlt.chi_samplers
            ],
            dtype=torch.int32,
            device=self.device,
        )

        rt_names = numpy.array([rt.name for rt in all_allowed_restypes], dtype=object)

        rt_res = torch.tensor(
            [
                i * max_n_blocks + j
                for i, one_pose_rlts in enumerate(task.rlts)
                for j, rlt in enumerate(one_pose_rlts)
                for rt in rlt.allowed_restypes
            ],
            dtype=torch.int32,
            device=self.device,
        )

        dun_rot_inds_for_rts = self.dun_param_resolver._indices_from_names(
            self.dun_param_resolver.all_table_indices,
            rt_names[None, :],
            torch.device("cpu"),
        ).squeeze()

        inds_of_phi = self.atom_indices_for_backbone_dihedral(systems, 0).reshape(-1, 4)
        inds_of_psi = self.atom_indices_for_backbone_dihedral(systems, 1).reshape(-1, 4)

        phi_psi_inds = torch.cat(
            (inds_of_phi.reshape(-1, 4), inds_of_psi.reshape(-1, 4)), dim=1
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

        uniq_res_for_brt, uniq_inds = torch.unique(
            orig_residue_for_buildable_restype, return_inverse=True
        )
        uniq_res_for_brt = uniq_res_for_brt.to(torch.int64)

        rottable_set_for_buildable_restype = torch.tensor(
            torch.cat(
                (
                    uniq_inds.reshape(-1, 1),
                    rottable_set_for_buildable_restype.reshape(-1, 1),
                ),
                dim=1,
            ),
            dtype=torch.int32,
            device=self.device,
        )

        phi_psi_res_inds = numpy.arange(n_sys * max_n_blocks, dtype=numpy.int32)

        n_sampling_res = uniq_res_for_brt.shape[0]

        # map the residue-numbered list of dihedral angles to their positions in
        # the set of residues that the Dunbrack library will provice chi samples for
        dihedral_atom_inds = torch.full(
            (2 * n_sampling_res, 4), -1, dtype=torch.int32, device=self.device
        )
        dihedral_atom_inds[
            2 * torch.arange(n_sampling_res, dtype=torch.int64, device=self.device), :
        ] = inds_of_phi[uniq_res_for_brt, :]
        dihedral_atom_inds[
            2 * torch.arange(n_sampling_res, dtype=torch.int64, device=self.device) + 1,
            :,
        ] = inds_of_psi[uniq_res_for_brt, :]

        ndihe_for_res = torch.full(
            (n_sampling_res,), 2, dtype=torch.int32, device=self.device
        )
        dihedral_offset_for_res = 2 * torch.arange(
            n_sampling_res, dtype=torch.int32, device=self.device
        )

        n_brts = nonzero_dunrot_inds_for_rts.shape[0]
        ### NOTE: improve logic here for determining max_n_chi
        max_n_chi = systems.packed_block_types.dun_sampler_chi_defining_atom.shape[1]
        chi_expansion_for_buildable_restype = torch.full(
            (n_brts, max_n_chi), 0, dtype=torch.int32, device=self.device
        )

        # ok, we'll go to the residue types and look at their protonation state expansions
        # and we'll put that information into the chi_expansions_for_buildable_restype
        # tensor

        nchi_for_buildable_restype = self.sampling_params.nchi_for_table_set[
            rottable_set_for_buildable_restype[:, 1].to(torch.int64)
        ]

        brts = all_allowed_restypes[dun_rot_inds_for_rts.numpy() != -1]

        non_dunbrack_expansion_counts_for_buildable_restype = torch.zeros(
            (n_brts, 4), dtype=torch.int32, device=self.device
        )
        max_chi_samples = 0

        # MOST OF THIS LOGIC SHOULD BE MOVED INTO A SETUP PHASE WITH
        # THE RefinedResidueType
        for i, rt in enumerate(brts):
            for j, rt_chi in enumerate(rt.chi_samples):
                chi_name = rt_chi.chi_dihedral
                assert chi_name[:3] == "chi"
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
                n_expanded_samples = n_samples * n_expansions
                max_chi_samples = max(max_chi_samples, n_expanded_samples)
                non_dunbrack_expansion_counts_for_buildable_restype[
                    i, chi_ind
                ] = n_expanded_samples
        non_dunbrack_expansion_for_buildable_restype = torch.full(
            (n_brts, max_n_chi, max_chi_samples),
            -1,
            dtype=torch.float32,
            device=self.device,
        )

        for i, rt in enumerate(brts):
            for j, rt_chi in enumerate(rt.chi_samples):
                chi_name = rt_chi.chi_dihedral
                chi_ind = int(chi_name[3:]) - 1
                n_expansions = (
                    1
                    + 2
                    * len(rt_chi.expansions)
                    * chi_expansion_for_buildable_restype[i, chi_ind]
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
            (n_brts,), 0.95, dtype=torch.float32, device=self.device
        )

        sampled_chi = self.launch_rotamer_building(
            systems.coords.reshape(-1, 3),
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

        n_rots_for_brt = sampled_chi[0]
        n_rots_for_brt_offsets = sampled_chi[1]
        brt_for_rotamer = sampled_chi[2]
        chi_for_rotamers = sampled_chi[3]

        # Now lets map back to the original set of rts per block type.
        # lots of reindxing below
        max_n_rts = max(
            len(rts.allowed_restypes)
            for one_pose_rlts in task.rlts
            for rts in one_pose_rlts
        )

        rt_global_index = torch.tensor(
            [
                max_n_rts * max_n_blocks * i + max_n_rts * j + k
                for i, one_pose_rlts in enumerate(task.rlts)
                for j, rlt in enumerate(one_pose_rlts)
                for k in range(len(rlt.allowed_restypes))
            ],
            dtype=torch.int64,
            device=self.device,
        )

        rt_real = torch.zeros(
            (n_sys * max_n_blocks * max_n_rts,), dtype=torch.int32, device=self.device
        )
        rt_real[rt_global_index] = 1
        nz_rt_real = torch.nonzero(rt_real)
        global_rt_that_are_brt = nz_rt_real[nonzero_dunrot_inds_for_rts.squeeze(), 0]
        global_rt_ind_for_brt = rt_global_index[nonzero_dunrot_inds_for_rts]

        n_rots_for_rt = torch.zeros(
            (n_sys * max_n_blocks * max_n_rts,), dtype=torch.int32, device=self.device
        )
        n_rots_for_rt_offset = torch.zeros_like(n_rots_for_rt)

        n_rots_for_rt[global_rt_that_are_brt] = n_rots_for_brt
        n_rots_for_rt = n_rots_for_rt.reshape(n_sys, max_n_blocks, max_n_rts)

        n_rots_for_rt_offsets = torch.full(
            (n_sys * max_n_blocks * max_n_rts,),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        n_rots_for_rt_offsets[global_rt_that_are_brt] = n_rots_for_brt_offsets
        n_rots_for_rt_offsets = n_rots_for_rt_offsets.reshape(
            n_sys, max_n_blocks, max_n_rts
        )

        rt_for_rotamer_global = global_rt_ind_for_brt[brt_for_rotamer.to(torch.int64)]

        # div will need to be replaced by floor_div in later versions of torch
        rt_for_rotamer_system = torch.div(
            rt_for_rotamer_global, max_n_blocks * max_n_rts
        )
        rt_for_rotamer_remainder = torch.remainder(
            rt_for_rotamer_global, max_n_blocks * max_n_rts
        )
        rt_for_rotamer_block = torch.div(rt_for_rotamer_remainder, max_n_rts)
        rt_for_rotamer_rt = torch.remainder(rt_for_rotamer_remainder, max_n_rts)
        rt_for_rotamer = torch.cat(
            (rt_for_rotamer_system, rt_for_rotamer_block, rt_for_rotamer_rt), dim=1
        )
        rt_for_rotamer = rt_for_rotamer.to(torch.int32)

        pbt = systems.packed_block_types
        pbt_cda = pbt.dun_sampler_chi_defining_atom
        chi_defining_atom_for_rotamer = torch.full(
            (chi_for_rotamers.shape[0], max_n_chi),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        max_n_chi = min((max_n_chi, pbt_cda.shape[1]))

        pbt_restype_ind_for_rt = torch.tensor(
            pbt.restype_index.get_indexer(
                rt_names[nonzero_dunrot_inds_for_rts.cpu()[:, 0]]
            ),
            dtype=torch.int64,
            device=self.device,
        )

        # chi_defining_atom_for_rotamer[:, :max_n_chi] = pbt_cda[
        #    rottable_set_for_buildable_restype.to(torch.int64), :max_n_chi
        # ]

        chi_defining_atom_for_rotamer[:, :max_n_chi] = pbt_cda[
            pbt_restype_ind_for_rt[brt_for_rotamer.to(torch.int64)], :max_n_chi
        ]

        print("n_rots_for_rt")
        print(n_rots_for_rt.shape)
        print(n_rots_for_rt.dtype)
        print("n_rots_for_rt_offsets")
        print(n_rots_for_rt_offsets.shape)
        print(n_rots_for_rt_offsets.dtype)
        print("rt_for_rotamer")
        print(rt_for_rotamer.shape)
        print(rt_for_rotamer.dtype)
        print("chi_defining_atom_for_rotamer")
        print(chi_defining_atom_for_rotamer.shape)
        print(chi_defining_atom_for_rotamer.dtype)
        print("chi_for_rotamers")
        print(chi_for_rotamers.shape)
        print(chi_for_rotamers.dtype)

        return (
            n_rots_for_rt,
            n_rots_for_rt_offsets,
            rt_for_rotamer,
            chi_defining_atom_for_rotamer,
            chi_for_rotamers,
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

    def launch_rotamer_building(
        self,
        coords,
        ndihe_for_res,
        dihedral_offset_for_res,
        dihedral_atom_inds,
        rottable_set_for_buildable_restype,
        chi_expansion_for_buildable_restype,
        non_dunbrack_expansion_for_buildable_restype,
        non_dunbrack_expansion_counts_for_buildable_restype,
        prob_cumsum_limit_for_buildable_restype,
        nchi_for_buildable_restype,
    ):
        return torch.ops.tmol.dun_sample_chi(
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
