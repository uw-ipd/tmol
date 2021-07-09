import torch

from .params import ScoringDunbrackDatabaseView, DunbrackParams, DunbrackScratch

from tmol.score.dunbrack.potentials.compiled import score_dun

# Workaround for https://github.com/pytorch/pytorch/pull/15340
# on torch<1.0.1
if "to" in torch.jit.ScriptModule.__dict__:
    delattr(torch.jit.ScriptModule, "to")


class DunbrackScoreModule(torch.jit.ScriptModule):
    def __init__(
        self,
        dundb: ScoringDunbrackDatabaseView,
        params: DunbrackParams,
        scratch: DunbrackScratch,
    ):
        super().__init__()

        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        def _tint(ts):
            return tuple(map(lambda t: t.to(torch.int32), ts))

        def _tfloat(ts):
            return tuple(map(lambda t: t.to(torch.float), ts))

        self.ndihe_for_res = _p(params.ndihe_for_res)
        self.dihedral_offset_for_res = _p(params.dihedral_offset_for_res)
        self.dihedral_atom_inds = _p(params.dihedral_atom_inds)
        self.rottable_set_for_res = _p(params.rottable_set_for_res)
        self.nchi_for_res = _p(params.nchi_for_res)
        self.nrotameric_chi_for_res = _p(params.nrotameric_chi_for_res)
        self.rotres2resid = _p(params.rotres2resid)
        self.prob_table_offset_for_rotresidue = _p(
            params.prob_table_offset_for_rotresidue
        )
        self.rotmean_table_offset_for_residue = _p(
            params.rotmean_table_offset_for_residue
        )
        self.rotind2tableind_offset_for_res = _p(params.rotind2tableind_offset_for_res)
        self.rotameric_chi_desc = _p(params.rotameric_chi_desc)
        self.semirotameric_chi_desc = _p(params.semirotameric_chi_desc)

        self.dihedrals = _p(scratch.dihedrals)
        self.ddihe_dxyz = _p(scratch.ddihe_dxyz)
        self.rotameric_rottable_assignment = _p(scratch.rotameric_rottable_assignment)
        self.semirotameric_rottable_assignment = _p(
            scratch.semirotameric_rottable_assignment
        )

        # self.rotameric_prob_tables = _p(dundb.rotameric_prob_tables)
        self.rotameric_neglnprob_tables = _p(dundb.rotameric_neglnprob_tables)
        self.rotprob_table_sizes = _p(dundb.rotprob_table_sizes)
        self.rotprob_table_strides = _p(dundb.rotprob_table_strides)
        self.rotameric_mean_tables = _p(dundb.rotameric_mean_tables)
        self.rotameric_sdev_tables = _p(dundb.rotameric_sdev_tables)
        self.rotmean_table_sizes = _p(dundb.rotmean_table_sizes)
        self.rotmean_table_strides = _p(dundb.rotmean_table_strides)

        self.rotameric_bb_start = _p(dundb.rotameric_bb_start)
        self.rotameric_bb_step = _p(dundb.rotameric_bb_step)
        self.rotameric_bb_periodicity = _p(dundb.rotameric_bb_periodicity)

        self.rotameric_rotind2tableind = _p(dundb.rotameric_rotind2tableind)
        self.semirotameric_rotind2tableind = _p(dundb.semirotameric_rotind2tableind)

        self.semirotameric_tables = _p(dundb.semirotameric_tables)
        self.semirot_table_sizes = _p(dundb.semirot_table_sizes)
        self.semirot_table_strides = _p(dundb.semirot_table_strides)
        self.semirot_start = _p(dundb.semirot_start)
        self.semirot_step = _p(dundb.semirot_step)
        self.semirot_periodicity = _p(dundb.semirot_periodicity)

    @torch.jit.script_method
    def forward(self, coords):
        return score_dun(
            coords,
            # self.rotameric_prob_tables,
            self.rotameric_neglnprob_tables,
            self.rotprob_table_sizes,
            self.rotprob_table_strides,
            self.rotameric_mean_tables,
            self.rotameric_sdev_tables,
            self.rotmean_table_sizes,
            self.rotmean_table_strides,
            self.rotameric_bb_start,
            self.rotameric_bb_step,
            self.rotameric_bb_periodicity,
            self.semirotameric_tables,
            self.semirot_table_sizes,
            self.semirot_table_strides,
            self.semirot_start,
            self.semirot_step,
            self.semirot_periodicity,
            self.rotameric_rotind2tableind,
            self.semirotameric_rotind2tableind,
            self.ndihe_for_res,
            self.dihedral_offset_for_res,
            self.dihedral_atom_inds,
            self.rottable_set_for_res,
            self.nchi_for_res,
            self.nrotameric_chi_for_res,
            self.rotres2resid,
            self.prob_table_offset_for_rotresidue,
            self.rotind2tableind_offset_for_res,
            self.rotmean_table_offset_for_residue,
            self.rotameric_chi_desc,
            self.semirotameric_chi_desc,
            self.dihedrals,
            self.ddihe_dxyz,
            self.rotameric_rottable_assignment,
            self.semirotameric_rottable_assignment,
        )
