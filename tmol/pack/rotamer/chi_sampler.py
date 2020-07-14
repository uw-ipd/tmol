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
from tmol.system.packed import PackedResidueSystem
from tmol.system.score_support import indexed_atoms_for_dihedral



@attr.s(auto_attribs=True)
class ChiSampler:

    dun_param_resolver: DunbrackParamResolver
    sampling_params: SamplingDunbrackDatabaseView

    @classmethod
    @validate_args
    def from_database(cls, param_resolver: DunbrackParamResolver):
        return cls(
            dun_param_resolver=param_resolver,
            sampling_params=param_resolver.sampling_db)

    @validate_args
    def chi_samples_for_residues(
        self,
        system: PackedResidueSystem,
        coords: Tensor(torch.float32)[:,3],
        task: PackerTask
    ) -> Tuple[
        Tensor(torch.int32)[:],
        Tensor(torch.int32)[:],
        Tensor(torch.int32)[:],
        Tensor(torch.float32)[:,:]
    ] :
        dev = coords.device

        rt_names = numpy.array([
            rt.name for rlt in task.rlts
            for rt in rlt.allowed_restypes
        ], dtype=object)
        print("rt_names", len(rt_names))
        print(rt_names)
        rt_res = numpy.array([
            i for i, rlt in enumerate(task.rlts)
            for rt in rlt.allowed_restypes
        ], dtype=numpy.int32)
        print("rt_res", len(rt_res))

        dun_rot_inds_for_rts = self.dun_param_resolver._indices_from_names(
            self.dun_param_resolver.all_table_indices,
            rt_names[None,:],
            torch.device("cpu")
        ).squeeze()
        print("dun_rot_inds_for_rts", dun_rot_inds_for_rts.shape)
        print(dun_rot_inds_for_rts)

        inds_of_phi_res = indexed_atoms_for_dihedral(system, "phi")
        inds_of_psi_res = indexed_atoms_for_dihedral(system, "psi")
        inds_of_phi_res = numpy.concatenate((
            inds_of_phi_res[:,:1],
            numpy.zeros((inds_of_phi_res.shape[0],1),dtype=numpy.int32),
            inds_of_phi_res[:,1:]), axis=1)
        inds_of_psi_res = numpy.concatenate((
            inds_of_psi_res[:,:1],
            numpy.ones((inds_of_psi_res.shape[0],1),dtype=numpy.int32),
            inds_of_psi_res[:,1:]), axis=1)
        join_phi_psi = numpy.concatenate((inds_of_phi_res, inds_of_psi_res), 0)
        dihe_res = join_phi_psi[:, 0]
        dihe_inds = join_phi_psi[:, 1]
        sort_inds = numpy.lexsort((dihe_res, dihe_inds));
        phi_psi = join_phi_psi[sort_inds,:]

        nonzero_dunrot_inds_for_rts = torch.nonzero(dun_rot_inds_for_rts != -1)
        rottable_set_for_buildable_restype = dun_rot_inds_for_rts[nonzero_dunrot_inds_for_rts]
        orig_residue_for_buildable_restype = rt_res[nonzero_dunrot_inds_for_rts.cpu().numpy()]
        print("orig_residue_for_buildable_restype", orig_residue_for_buildable_restype.shape)
        print(orig_residue_for_buildable_restype)
        uniq_res_for_brt, uniq_inds = numpy.unique(orig_residue_for_buildable_restype, return_inverse=True)

        rottable_set_for_buildable_restype = torch.tensor(numpy.concatenate((
            uniq_inds.reshape(-1,1),
            rottable_set_for_buildable_restype.reshape(-1,1)), axis=1),
            dtype=torch.int32, device=dev)

        phi_arange = numpy.arange(inds_of_phi_res.shape[0], dtype=numpy.int32)
        phi_res_inds = numpy.full((len(system.residues),), -1, dtype=numpy.int32)
        phi_res_inds[inds_of_phi_res[:,0]] = phi_arange

        psi_arange = numpy.arange(inds_of_psi_res.shape[0], dtype=numpy.int32)
        psi_res_inds = numpy.full((len(system.residues),), -1, dtype=numpy.int32)
        psi_res_inds[inds_of_psi_res[:,0]] = psi_arange

        n_sampling_res = uniq_res_for_brt.shape[0]
        
        dihedral_atom_inds = numpy.full(
            (2*n_sampling_res,4), -1, dtype=numpy.int32)

        # map the residue-numbered list of dihedral angles to their positions in the
        # set of residues that the dunbrack library will provide chi samples for
        phi_reindexed = numpy.full(
            (n_sampling_res, 4), -1, dtype=numpy.int32)
        phi_reindexed[
            phi_res_inds[uniq_res_for_brt] != -1
        ] = inds_of_phi_res[ phi_res_inds[uniq_res_for_brt][phi_res_inds[uniq_res_for_brt] != -1], 2:]
        dihedral_atom_inds[
            numpy.arange(2*n_sampling_res, dtype=int) % 2 == 0
        ] = phi_reindexed

        psi_reindexed = numpy.full(
            (n_sampling_res, 4), -1, dtype=numpy.int32)
        psi_reindexed[
            psi_res_inds[uniq_res_for_brt] != -1
        ] = inds_of_psi_res[psi_res_inds[uniq_res_for_brt][psi_res_inds[uniq_res_for_brt] != -1], 2:]
        dihedral_atom_inds[
            numpy.arange(2*n_sampling_res, dtype=int) % 2 == 1
        ] = psi_reindexed

        dihedral_atom_inds = torch.tensor(dihedral_atom_inds, dtype=torch.int32, device=dev)

        ndihe_for_res = torch.full((n_sampling_res,), 2, dtype=torch.int32, device=dev)
        dihedral_offset_for_res = 2 * torch.arange(
            n_sampling_res, dtype=torch.int32, device=dev)
            
        
        n_brts = nonzero_dunrot_inds_for_rts.shape[0]
        chi_expansion_for_buildable_restype = torch.full(
            (n_brts, 4), 0, dtype=torch.int32, device=dev)

        # for now, punt on sampling the chi not specified by the dunbrack library
        non_dunbrack_expansion_for_buildable_restype = torch.full(
            (n_brts, 4, 1), 0, dtype=torch.float32, device=dev)
        non_dunbrack_expansion_counts_for_buildable_restype = torch.full(
            (n_brts, 4), 0, dtype=torch.int32, device=dev)

        # treat all residues as if they are exposed
        prob_cumsum_limit_for_buildable_restype = torch.full(
            (n_brts,), 0.95, dtype=torch.float32, device=dev)

        # for now, ignore the chi not specified by the dunbrack library
        nchi_for_buildable_restype = self.sampling_params.nchi_for_table_set[
            rottable_set_for_buildable_restype[:,1].to(torch.int64)]

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
