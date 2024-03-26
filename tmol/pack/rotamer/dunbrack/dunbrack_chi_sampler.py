import torch
import attr
import numpy

from typing import Tuple

from tmol.types.torch import Tensor
from tmol.types.array import NDArray
from tmol.types.functional import validate_args

from tmol.score.dunbrack.params import DunbrackParamResolver

from tmol.pack.rotamer.chi_sampler import ChiSampler  # noqa F401

# from tmol.pack.rotamer.dunbrack.compiled import _compiled  # noqa F401
from tmol.pack.packer_task import PackerTask
from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack

# from tmol.system.packed import PackedResidueSystem
# from tmol.system.score_support import indexed_atoms_for_dihedral


@attr.s(auto_attribs=True, slots=True, frozen=True)
class DunSamplerRTCache:
    """Data to store in RefinedResidueType that will be reused
    repeatedly in the creation of the DunSamplerPBTCache
    """

    bbdihe_uaids: NDArray[numpy.int32][2, 4, 3]
    chi_defining_atom: NDArray[numpy.int32][:]
    non_dunbrack_sample_counts: NDArray[numpy.int32][:, 2]
    non_dunbrack_samples: NDArray[numpy.int32][:, 2, :]


@attr.s(auto_attribs=True, slots=True, frozen=True)
class DunSamplerPBTCache:
    """Data needed for chi sampling and for reporting how
    the chi are to be assigned to atoms
    """

    bbdihe_uaids: Tensor[torch.int32][:, 2, 4, 3]
    chi_defining_atom: Tensor[torch.int32][:, :]
    non_dunbrack_sample_counts: Tensor[torch.int32][:, :, 2]
    non_dunbrack_samples: Tensor[torch.int32][:, :, 2, :]

    @property
    def max_n_chi(self):
        return self.chi_defining_atom.shape[1]


# I can't use attr here because the DunbrackParamResolver contains the
# mutable datatype of a Pandas DataFrame. This might make it seem like
# using a DunbrackChiSampler in a set is dangerous; however, the
# ParamResolver is singleton-esq in that, when it is constructed from
# a dunbrack database (identified by the database's path), it is
# memoized. So each database should construct one and only one
# ParamResolver.
# @attr.s(auto_attribs=True, slots=True, frozen=True)
class DunbrackChiSampler:
    dun_param_resolver: DunbrackParamResolver

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return id(self.dun_param_resolver)

    def __init__(self, dun_param_resolver: DunbrackParamResolver):
        self.dun_param_resolver = dun_param_resolver

    @property
    def device(self):
        return self.dun_param_resolver.device

    @classmethod
    @validate_args
    def from_database(cls, param_resolver: DunbrackParamResolver):
        return cls(dun_param_resolver=param_resolver)

    @classmethod
    def sampler_name(cls):
        return "DunbrackChiSampler"

    @validate_args
    def annotate_residue_type(self, restype: RefinedResidueType):
        """TEMP TEMP TEMP: assume the dihedrals we care about are phi and psi"""
        if hasattr(restype, "dun_sampler_cache"):
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
        # that the dunbrack library handles.
        # Strip away any patches and use the "base name" of the residue type to make
        # the "which library should I read from?" decision
        dun_lib_ind = self.dun_param_resolver._indices_from_names(
            self.dun_param_resolver.all_table_indices,
            numpy.array([[restype.base_name]], dtype=object),
            device=self.device,
        )[0, 0]

        if dun_lib_ind >= 0:
            n_chi = self.dun_param_resolver.scoring_db_aux.nchi_for_table_set[
                dun_lib_ind
            ].item()

            n_chi_total = n_chi
            for rt_chi in restype.chi_samples:
                chi_name = rt_chi.chi_dihedral
                assert chi_name[:3] == "chi"
                chi_ind = int(chi_name[3:])
                if chi_ind > n_chi_total:
                    n_chi_total = chi_ind

            # Assumption: the Dunbrack library assigns chi for a contiguous
            # block of chi dihedrals starting with chi1.
            chi_names = ["chi%d" % (i + 1) for i in range(n_chi)]
            for chi_name in chi_names:
                assert chi_name in restype.torsion_to_uaids

            chi_defining_atom = numpy.full(n_chi_total, -1, dtype=numpy.int32)
            # The third atom (atom 2) in the chi defines the dihedral; this
            # atoms needs to be a member of the residue type, and not unresolved
            chi_defining_atom[:n_chi] = [
                restype.torsion_to_uaids[chi_name][2][0] for chi_name in chi_names
            ]

            if restype.chi_samples:
                chi_inds = numpy.array(
                    [int(samp.chi_dihedral[3:]) - 1 for samp in restype.chi_samples],
                    dtype=int,
                )
                sort_inds = numpy.argsort(chi_inds)
                chi_defining_atom[chi_inds[sort_inds]] = [
                    restype.torsion_to_uaids[restype.chi_samples[ind].chi_dihedral][2][
                        0
                    ]
                    for ind in sort_inds
                ]

            non_dunbrack_sample_counts = numpy.zeros(
                (n_chi_total, 2), dtype=numpy.int32
            )
            max_chi_samples = 0
            for rt_chi in restype.chi_samples:
                chi_name = rt_chi.chi_dihedral
                assert chi_name[:3] == "chi"
                chi_ind = int(chi_name[3:]) - 1
                n_expansions = 1 + 2 * len(rt_chi.expansions)
                n_samples = len(rt_chi.samples)
                n_expanded_samples = n_samples * n_expansions
                max_chi_samples = max(max_chi_samples, n_expanded_samples)
                non_dunbrack_sample_counts[chi_ind, 0] = n_samples
                non_dunbrack_sample_counts[chi_ind, 1] = n_expanded_samples

            non_dunbrack_samples = numpy.zeros(
                (n_chi_total, 2, max_chi_samples), dtype=numpy.float32
            )
            for rt_chi in restype.chi_samples:
                chi_name = rt_chi.chi_dihedral
                chi_ind = int(chi_name[3:]) - 1
                n_expansions = 1 + 2 * len(rt_chi.expansions)
                n_samples = len(rt_chi.samples)
                non_dunbrack_samples[chi_ind, 0, :n_samples] = rt_chi.samples
                for i in range(n_samples):
                    for j in range(n_expansions):
                        if j == 0:
                            non_dunbrack_samples[chi_ind, 1, n_expansions * i + j] = (
                                rt_chi.samples[i]
                            )
                        else:
                            expansion = (j - 1) // 2
                            factor = -1 if (j - 1) % 2 == 0 else 1
                            non_dunbrack_samples[chi_ind, 1, n_expansions * i + j] = (
                                rt_chi.samples[i]
                                + factor * rt_chi.expansions[expansion]
                            )

        else:
            chi_defining_atom = numpy.zeros((0,), dtype=numpy.int32)
            non_dunbrack_sample_counts = numpy.zeros((0, 2), dtype=numpy.int32)
            non_dunbrack_samples = numpy.zeros((0, 2, 0), dtype=numpy.int32)

        cache = DunSamplerRTCache(
            bbdihe_uaids=uaids,
            chi_defining_atom=chi_defining_atom,
            non_dunbrack_sample_counts=non_dunbrack_sample_counts,
            non_dunbrack_samples=non_dunbrack_samples,
        )
        setattr(restype, "dun_sampler_cache", cache)

    @validate_args
    def annotate_packed_block_types(self, packed_block_types: PackedBlockTypes):
        if hasattr(packed_block_types, "dun_sampler_cache"):
            # assert hasattr(packed_block_types, "dun_sampler_chi_defining_atom")
            return

        for rt in packed_block_types.active_block_types:
            assert hasattr(rt, "dun_sampler_cache")
            # assert hasattr(rt, "dun_sampler_bbdihe_uaids")
            # assert hasattr(rt, "dun_sampler_chi_defining_atom")

        uaids = numpy.stack(
            [
                rt.dun_sampler_cache.bbdihe_uaids
                for rt in packed_block_types.active_block_types
            ]
        )
        uaids = torch.tensor(uaids, dtype=torch.int32, device=self.device)

        max_n_chi = max(
            rt.dun_sampler_cache.chi_defining_atom.shape[0]
            for rt in packed_block_types.active_block_types
        )
        max_chi_samples = max(
            rt.dun_sampler_cache.non_dunbrack_samples.shape[2]
            for rt in packed_block_types.active_block_types
        )

        chi_defining_atom = torch.full(
            (packed_block_types.n_types, max_n_chi),
            -1,
            dtype=torch.int32,
            device=packed_block_types.device,
        )
        for i, rt in enumerate(packed_block_types.active_block_types):
            chi_defining_atom[i, : rt.dun_sampler_cache.chi_defining_atom.shape[0]] = (
                torch.tensor(
                    rt.dun_sampler_cache.chi_defining_atom,
                    dtype=torch.int32,
                    device=self.device,
                )
            )
        non_dunbrack_sample_counts = torch.full(
            (packed_block_types.n_types, max_n_chi, 2),
            -1,
            dtype=torch.int32,
            device=packed_block_types.device,
        )
        non_dunbrack_samples = torch.full(
            (packed_block_types.n_types, max_n_chi, 2, max_chi_samples),
            -1,
            dtype=torch.float32,
            device=packed_block_types.device,
        )
        for i, rt in enumerate(packed_block_types.active_block_types):
            rt_ndsc = rt.dun_sampler_cache.non_dunbrack_sample_counts
            rt_nds = rt.dun_sampler_cache.non_dunbrack_samples
            non_dunbrack_sample_counts[i, : rt_ndsc.shape[0], :] = torch.tensor(
                rt_ndsc, dtype=torch.int32, device=packed_block_types.device
            )
            if rt_ndsc.shape[0] == 0 or rt_nds.shape[2] == 0:
                continue
            non_dunbrack_samples[i, : rt_ndsc.shape[0], :, : rt_nds.shape[2]] = (
                torch.tensor(
                    rt_nds, dtype=torch.float32, device=packed_block_types.device
                )
            )

        cache = DunSamplerPBTCache(
            bbdihe_uaids=uaids,
            chi_defining_atom=chi_defining_atom,
            non_dunbrack_sample_counts=non_dunbrack_sample_counts,
            non_dunbrack_samples=non_dunbrack_samples,
        )
        setattr(packed_block_types, "dun_sampler_cache", cache)

    @validate_args
    def defines_rotamers_for_rt(self, rt: RefinedResidueType):
        # ugly hack for now:
        if not rt.properties.polymer.is_polymer:
            return False
        if rt.properties.polymer.polymer_type != "amino_acid":
            return False
        if rt.properties.polymer.backbone_type != "alpha":
            return False

        # and then what??
        if rt.base_name == "GLY" or rt.base_name == "ALA":
            return False

        # all amino acids except GLY and ALA?? That feels wrong
        # go with it for now
        return True

    @validate_args
    def first_sc_atoms_for_rt(self, rt: RefinedResidueType) -> Tuple[str, ...]:
        assert self.defines_rotamers_for_rt(rt)
        return ("CB",)

    @validate_args
    def sample_chi_for_poses(
        self, pose_stack: PoseStack, task: PackerTask
    ) -> Tuple[
        Tensor[torch.int32][:],  # n_rots_for_rt
        Tensor[torch.int32][:],  # rt_for_rotamer
        Tensor[torch.int32][:, :],  # chi_defining_atom_for_rotamer
        Tensor[torch.float32][:, :],  # chi_for_rotamers
    ]:
        assert self.device == pose_stack.coords.device
        max_n_blocks = pose_stack.block_type_ind.shape[1]

        dun_allowed_restypes = numpy.array(
            [
                rt
                for one_pose_rlts in task.rlts
                for rlt in one_pose_rlts
                for rt in rlt.allowed_restypes
                if self in rlt.chi_samplers
            ],
            dtype=object,
        )

        # n_allowed_per_pose = torch.tensor(
        #     [
        #         len(rlt.allowed_restypes)
        #         for one_pose_rlts in task.rlts
        #         for rlt in one_pose_rlts
        #         if self in rlt.chi_samplers
        #     ],
        #     dtype=torch.int32,
        #     device=self.device,
        # )

        rt_names = numpy.array([rt.name for rt in dun_allowed_restypes], dtype=object)
        rt_base_names = numpy.array(
            [rt.name.partition(":")[0] for rt in dun_allowed_restypes], dtype=object
        )
        pbt = pose_stack.packed_block_types

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
            rt_base_names[None, :],
            # ??? torch.device("cpu"),
            self.device,
        ).squeeze()

        block_type_ind_for_brt = torch.tensor(
            pbt.restype_index.get_indexer(
                rt_names[dun_rot_inds_for_rts.cpu().numpy() != -1]
            ),
            dtype=torch.int64,
            device=self.device,
        )

        inds_of_phi = self.atom_indices_for_backbone_dihedral(pose_stack, 0).reshape(
            -1, 4
        )
        inds_of_psi = self.atom_indices_for_backbone_dihedral(pose_stack, 1).reshape(
            -1, 4
        )

        # fd unused
        # phi_psi_inds = torch.cat(
        #    (inds_of_phi.reshape(-1, 4), inds_of_psi.reshape(-1, 4)), dim=1
        # )
        # phi_psi_inds = phi_psi_inds.reshape(-1, 4)

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

        # phi_psi_res_inds = numpy.arange(n_sys * max_n_blocks, dtype=numpy.int32)

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

        max_n_chi = pose_stack.packed_block_types.dun_sampler_cache.max_n_chi
        chi_expansion_for_buildable_restype = torch.full(
            (n_brts, max_n_chi), 0, dtype=torch.int32, device=self.device
        )

        # ok, we'll go to the residue types and look at their protonation
        # state expansions aand we'll put that information into the
        # chi_expansions_for_buildable_restype tensor

        sampling_db = self.dun_param_resolver.sampling_db
        nchi_for_buildable_restype = sampling_db.nchi_for_table_set[
            rottable_set_for_buildable_restype[:, 1].to(torch.int64)
        ]

        non_dunbrack_expansion_counts_for_buildable_restype = torch.zeros(
            (n_brts, max_n_chi), dtype=torch.int32, device=self.device
        )
        # max_chi_samples = 0

        # TEMP! Treat everything as exposed (0)
        sc = pbt.dun_sampler_cache
        ndecfbr = sc.non_dunbrack_sample_counts[block_type_ind_for_brt, 0]
        non_dunbrack_expansion_counts_for_buildable_restype = ndecfbr

        # TEMP! Treat everything as exposed (0)
        non_dunbrack_expansion_for_buildable_restype = sc.non_dunbrack_samples[
            block_type_ind_for_brt, 0
        ]

        # treat all residues as if they are exposed
        prob_cumsum_limit_for_buildable_restype = torch.full(
            (n_brts,), 0.95, dtype=torch.float32, device=self.device
        )

        sampled_chi = self.launch_rotamer_building(
            pose_stack.coords.reshape(-1, 3),
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

        return self.package_samples_for_output(
            pbt,
            task,
            block_type_ind_for_brt,
            max_n_chi,
            nonzero_dunrot_inds_for_rts,
            sampled_chi,
        )

    @validate_args
    def atom_indices_for_backbone_dihedral(
        self, pose_stack: PoseStack, bb_dihedral_ind: int
    ):
        assert hasattr(pose_stack.packed_block_types, "dun_sampler_cache")
        n_pose_stack = pose_stack.block_type_ind.shape[0]
        max_n_blocks = pose_stack.block_type_ind.shape[1]
        max_n_atoms = pose_stack.coords.shape[2]

        pbts = pose_stack.packed_block_types

        real = torch.nonzero(pose_stack.block_type_ind >= 0)

        uaids = torch.full(
            (n_pose_stack, max_n_blocks, 4, 3),
            -1,
            dtype=torch.int64,
            device=self.device,
        )

        uaids[real[:, 0], real[:, 1], :, :] = pbts.dun_sampler_cache.bbdihe_uaids[
            pose_stack.block_type_ind[real[:, 0], real[:, 1]].to(torch.int64),
            bb_dihedral_ind,
            :,
            :,
        ].to(torch.int64)

        # what we will return
        dihe_atom_inds = torch.full(
            (n_pose_stack, max_n_blocks, 4), -1, dtype=torch.int32, device=self.device
        )

        # copy over all atom ids from the uaids as if they were resolved; the
        # atoms that are unresolved will be overwritten later
        dihe_atom_inds[:] = uaids[:, :, :, 0]

        inter_res = torch.nonzero(uaids[:, :, :, 1] >= 0)

        dest_res = pose_stack.inter_residue_connections[
            inter_res[:, 0],
            inter_res[:, 1],
            uaids[inter_res[:, 0], inter_res[:, 1], inter_res[:, 2], 1],
            0,
        ]
        dest_conn = pose_stack.inter_residue_connections[
            inter_res[:, 0],
            inter_res[:, 1],
            uaids[inter_res[:, 0], inter_res[:, 1], inter_res[:, 2], 1],
            1,
        ].to(torch.int64)

        # now which atom on the downstream residue is the one that
        # the source residue is pointing at
        dihe_atom_inds[inter_res[:, 0], inter_res[:, 1], inter_res[:, 2]] = (
            (inter_res[:, 0] * max_n_atoms * max_n_blocks).to(torch.int32)
            + dest_res * max_n_atoms
            + pbts.atom_downstream_of_conn[
                pose_stack.block_type_ind[inter_res[:, 0], inter_res[:, 1]].to(
                    torch.int64
                ),
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
        from .compiled import dun_sample_chi

        return dun_sample_chi(
            coords,
            self.dun_param_resolver.sampling_db.rotameric_prob_tables,
            self.dun_param_resolver.sampling_db.rotprob_table_sizes,
            self.dun_param_resolver.sampling_db.rotprob_table_strides,
            self.dun_param_resolver.sampling_db.rotameric_mean_tables,
            self.dun_param_resolver.sampling_db.rotameric_sdev_tables,
            self.dun_param_resolver.sampling_db.rotmean_table_sizes,
            self.dun_param_resolver.sampling_db.rotmean_table_strides,
            self.dun_param_resolver.sampling_db.rotameric_meansdev_tableset_offsets,
            self.dun_param_resolver.sampling_db.rotameric_bb_start,
            self.dun_param_resolver.sampling_db.rotameric_bb_step,
            self.dun_param_resolver.sampling_db.rotameric_bb_periodicity,
            self.dun_param_resolver.sampling_db.semirotameric_tables,
            self.dun_param_resolver.sampling_db.semirot_table_sizes,
            self.dun_param_resolver.sampling_db.semirot_table_strides,
            self.dun_param_resolver.sampling_db.semirot_start,
            self.dun_param_resolver.sampling_db.semirot_step,
            self.dun_param_resolver.sampling_db.semirot_periodicity,
            self.dun_param_resolver.sampling_db.rotameric_rotind2tableind,
            self.dun_param_resolver.sampling_db.semirotameric_rotind2tableind,
            self.dun_param_resolver.sampling_db.all_chi_rotind2tableind,
            self.dun_param_resolver.sampling_db.all_chi_rotind2tableind_offsets,
            self.dun_param_resolver.sampling_db.n_rotamers_for_tableset,
            self.dun_param_resolver.sampling_db.n_rotamers_for_tableset_offsets,
            self.dun_param_resolver.sampling_db.sorted_rotamer_2_rotamer,
            self.dun_param_resolver.sampling_db.nchi_for_table_set,
            self.dun_param_resolver.sampling_db.rotwells,
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
    def package_samples_for_output(
        self,
        pbt: PackedBlockTypes,
        task: PackerTask,
        block_type_ind_for_brt: Tensor[torch.int64][:],
        max_n_chi: int,
        nonzero_dunrot_inds_for_rts: Tensor[torch.int64][:, :],
        sampled_chi,
    ):
        restype_is_allowed_for_dun = torch.tensor(
            [
                True if self in rlt.chi_samplers else False
                for one_pose_rlts in task.rlts
                for rlt in one_pose_rlts
                for rt in rlt.allowed_restypes
            ],
            dtype=torch.uint8,
            device=self.device,
        )
        n_restypes_total = restype_is_allowed_for_dun.shape[0]
        dun_allowed_inds = torch.nonzero(restype_is_allowed_for_dun)[:, 0]
        dun_brt_global_inds = dun_allowed_inds[nonzero_dunrot_inds_for_rts[:, 0]].to(
            self.device
        )

        n_rots_for_rt = torch.zeros(
            (n_restypes_total,), dtype=torch.int32, device=self.device
        )
        n_rots_for_rt[dun_brt_global_inds] = sampled_chi[0]
        n_rots_for_rt_offsets = torch.zeros_like(n_rots_for_rt)
        n_rots_for_rt_offsets[dun_brt_global_inds] = sampled_chi[1]

        # n_rots_for_brt = sampled_chi[0]
        # n_rots_for_brt_offsets = sampled_chi[1]
        brt_for_rotamer = sampled_chi[2]
        chi_for_rotamers = sampled_chi[3]

        # Now lets map back to the original set of rts per block type.
        # lots of reindxing below
        # max_n_rts = max(
        #     len(rts.allowed_restypes)
        #     for one_pose_rlts in task.rlts
        #     for rts in one_pose_rlts
        # )

        rt_global_index = torch.arange(
            n_restypes_total, dtype=torch.int32, device=self.device
        )
        global_rt_ind_for_brt = rt_global_index[nonzero_dunrot_inds_for_rts.squeeze()]

        rt_for_rotamer = global_rt_ind_for_brt[brt_for_rotamer.to(torch.int64)]

        pbt_cda = pbt.dun_sampler_cache.chi_defining_atom
        chi_defining_atom_for_rotamer = torch.full(
            (chi_for_rotamers.shape[0], max_n_chi),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        max_n_chi = min((max_n_chi, pbt_cda.shape[1]))

        # block_type_ind_for_brt_old = torch.tensor(
        #     pbt.restype_index.get_indexer(
        #         rt_names[nonzero_dunrot_inds_for_rts.cpu()[:, 0]]
        #     ),
        #     dtype=torch.int64,
        #     device=self.device,
        # )
        # numpy.testing.assert_equal(
        #     block_type_ind_for_brt_old.cpu().numpy(),
        #     block_type_ind_for_brt.cpu().numpy()
        # )

        chi_defining_atom_for_rotamer[:, :max_n_chi] = pbt_cda[
            block_type_ind_for_brt[brt_for_rotamer.to(torch.int64)], :max_n_chi
        ]

        return (
            n_rots_for_rt,
            rt_for_rotamer,
            chi_defining_atom_for_rotamer,
            chi_for_rotamers,
        )
