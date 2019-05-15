import torch
from tmol.score.ljlk.params import LJLKParamResolver
from tmol.score.chemical_database import AtomTypeParamResolver

from tmol.database.chemical import ChemicalDatabase
from tmol.database.scoring.ljlk import LJLKDatabase

# Import compiled components to load torch_ops
import tmol.score.lk_ball.potentials.compiled  # noqa

# Workaround for https://github.com/pytorch/pytorch/pull/15340
# on torch<1.0.1
if "to" in torch.jit.ScriptModule.__dict__:
    delattr(torch.jit.ScriptModule, "to")


class _LKBallScoreModule(torch.jit.ScriptModule):
    @classmethod
    def from_database(
        cls,
        chemical_database: ChemicalDatabase,
        ljlk_database: LJLKDatabase,
        device: torch.device,
    ):
        return cls(
            param_resolver=LJLKParamResolver.from_database(
                chemical_database, ljlk_database, device
            ),
            # additional resolution needed for is_hydrogen, acceptor_type
            atom_type_resolver=AtomTypeParamResolver.from_database(
                chemical_database, device
            ),
        )

    def __init__(
        self,
        param_resolver: LJLKParamResolver,
        atom_type_resolver: AtomTypeParamResolver,
    ):
        super().__init__()

        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        def _tfloat(ts):
            return tuple(map(lambda t: t.to(torch.float), ts))

        def _tint(ts):
            return tuple(map(lambda t: t.to(torch.long), ts))

        # Pack parameters into dense tensors. Parameter ordering must match
        # struct layout declared in `potentials/params.hh`.
        self.lkball_type_params = _p(
            torch.stack(
                _tfloat(
                    [
                        param_resolver.type_params.lj_radius,
                        param_resolver.type_params.lk_dgfree,
                        param_resolver.type_params.lk_lambda,
                        param_resolver.type_params.lk_volume,
                        param_resolver.type_params.is_donor,
                        param_resolver.type_params.is_hydroxyl,
                        param_resolver.type_params.is_polarh,
                        param_resolver.type_params.is_acceptor,
                    ]
                ),
                dim=1,
            )
        )

        self.watergen_type_params = _p(
            torch.stack(
                _tint(
                    [
                        atom_type_resolver.params.is_acceptor,
                        atom_type_resolver.params.acceptor_hybridization,
                        atom_type_resolver.params.is_donor,
                        atom_type_resolver.params.is_hydrogen,
                    ]
                ),
                dim=1,
            )
        )

        self.lkball_global_params = _p(
            torch.stack(
                _tfloat(
                    [
                        param_resolver.global_params.lj_hbond_dis,
                        param_resolver.global_params.lj_hbond_OH_donor_dis,
                        param_resolver.global_params.lj_hbond_hdis,
                        param_resolver.global_params.lkb_water_dist,
                    ]
                ),
                dim=1,
            )
        )

        self.watergen_global_params = _p(
            torch.stack(
                _tfloat(
                    [
                        param_resolver.global_params.lkb_water_dist,
                        param_resolver.global_params.lkb_water_angle_sp2,
                        param_resolver.global_params.lkb_water_angle_sp3,
                        param_resolver.global_params.lkb_water_angle_ring,
                    ]
                ),
                dim=1,
            )
        )

        self.watergen_water_tors_sp2 = torch.nn.Parameter(
            param_resolver.global_params.lkb_water_tors_sp2, requires_grad=False
        )
        self.watergen_water_tors_sp3 = torch.nn.Parameter(
            param_resolver.global_params.lkb_water_tors_sp3, requires_grad=False
        )
        self.watergen_water_tors_ring = torch.nn.Parameter(
            param_resolver.global_params.lkb_water_tors_ring, requires_grad=False
        )

        self.heavyatom_mask = torch.nn.Parameter(
            ~atom_type_resolver.params.is_hydrogen, requires_grad=False
        )


class LKBallIntraModule(_LKBallScoreModule):
    @torch.jit.script_method
    def forward(
        self,
        I,
        atom_type_I,
        bonded_path_lengths,
        indexed_bond_bonds,
        indexed_bond_spans,
    ):
        waters_I = torch.ops.tmol.watergen_lkball(
            I,
            atom_type_I,
            indexed_bond_bonds,
            indexed_bond_spans,
            self.watergen_type_params,
            self.watergen_global_params,
            self.watergen_water_tors_sp2,
            self.watergen_water_tors_sp3,
            self.watergen_water_tors_ring,
        )

        I_heavyatom_mask = self.heavyatom_mask[atom_type_I]
        I_idx = torch.nonzero(I_heavyatom_mask)[:, 0]

        return torch.ops.tmol.score_lkball(
            I[I_idx],
            atom_type_I[I_idx],
            waters_I[I_idx],
            I[I_idx],
            atom_type_I[I_idx],
            waters_I[I_idx],
            bonded_path_lengths[I_idx, :][:, I_idx],
            self.lkball_type_params,
            self.lkball_global_params,
        )


class LKBallInterModule(_LKBallScoreModule):
    # @torch.jit.script_method
    def forward(
        self,
        I,
        atom_type_I,
        J,
        atom_type_J,
        bonded_path_lengths,
        indexed_bond_bonds,
        indexed_bond_spans,
    ):
        waters_I = torch.ops.tmol.watergen_lkball(
            I,
            atom_type_I,
            indexed_bond_bonds,
            indexed_bond_spans,
            self.watergen_type_params,
            self.watergen_global_params,
            self.watergen_water_tors_sp2,
            self.watergen_water_tors_sp3,
            self.watergen_water_tors_ring,
        )

        waters_J = torch.ops.tmol.watergen_lkball(
            J,
            atom_type_J,
            indexed_bond_bonds,
            indexed_bond_spans,
            self.watergen_type_params,
            self.watergen_global_params,
            self.watergen_water_tors_sp2,
            self.watergen_water_tors_sp3,
            self.watergen_water_tors_ring,
        )

        I_heavyatom_mask = self.heavyatom_mask[atom_type_I]
        I_idx = torch.nonzero(I_heavyatom_mask)[:, 0]

        J_heavyatom_mask = self.heavyatom_mask[atom_type_J]
        J_idx = torch.nonzero(J_heavyatom_mask)[:, 0]

        V_ij = torch.ops.tmol.score_lkball(
            I[I_idx],
            atom_type_I[I_idx],
            waters_I[I_idx],
            J[J_idx],
            atom_type_J[J_idx],
            waters_J[J_idx],
            bonded_path_lengths[I_idx, :][:, J_idx],
            self.lkball_type_params,
            self.lkball_global_params,
        )

        V_ji = torch.ops.tmol.score_lkball(
            J[J_idx],
            atom_type_J[J_idx],
            waters_J[J_idx],
            I[I_idx],
            atom_type_I[I_idx],
            waters_I[I_idx],
            bonded_path_lengths[I_idx, :][:, J_idx].t(),
            self.lkball_type_params,
            self.lkball_global_params,
        )

        return V_ij + V_ji
