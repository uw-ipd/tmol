from typing import Optional

import torch

from tmol.utility.reactive import reactive_attrs, reactive_property
from tmol.types.functional import validate_args

from tmol.types.torch import Tensor
from tmol.types.array import NDArray

from tmol.database import ParameterDatabase
from tmol.database.scoring import LJLKDatabase

from ..database import ParamDB
from ..device import TorchDevice
from ..interatomic_distance import InteratomicDistanceGraphBase
from ..bonded_atom import BondedAtomScoreGraph
from ..factory import Factory
from ..score_components import ScoreComponent, ScoreComponentClasses, IntraScoreGraph

from .potentials import lj_score, lk_score
from .params import LJLKParamResolver, LJLKTypePairParams


@reactive_attrs
class LJLKIntraParam(IntraScoreGraph):
    @reactive_property
    def atom_pair_inds(target) -> Tensor(torch.long)[:, 3]:
        return target.atom_pair_inds

    @reactive_property
    def atom_pair_dist(target) -> Tensor(torch.float)[:]:
        return target.atom_pair_dist

    @reactive_property
    def ljlk_interaction_weight(target) -> Tensor(torch.float)[:, :, :]:
        return target.ljlk_interaction_weight

    @reactive_property
    def ljlk_atom_pair_params(target) -> LJLKTypePairParams:
        return target.ljlk_atom_pair_params

    @reactive_property
    def param_resolver(target) -> LJLKTypePairParams:
        return target.param_resolver


@reactive_attrs
class LJIntraScore(LJLKIntraParam):
    @reactive_property
    @validate_args
    def lj(
        atom_pair_inds: Tensor(torch.long)[:, 3],
        atom_pair_dist: Tensor(torch.float)[:],
        ljlk_interaction_weight: Tensor(torch.float)[:, :, :],
        ljlk_atom_pair_params: LJLKTypePairParams,
        param_resolver: LJLKParamResolver,
    ):
        gparams = param_resolver.global_params
        pparams = ljlk_atom_pair_params

        assert (atom_pair_inds[:, 0] == 0).all()
        pidx = (atom_pair_inds[:, 1], atom_pair_inds[:, 2])

        assert ljlk_interaction_weight.shape[0] == 1
        ljlk_interaction_weight = ljlk_interaction_weight[0]

        return lj_score(
            # Distance
            dist=atom_pair_dist,
            # Bonded params
            interaction_weight=ljlk_interaction_weight[pidx],
            # Pair params
            lj_sigma=pparams.lj_sigma[pidx],
            lj_switch_slope=pparams.lj_switch_slope[pidx],
            lj_switch_intercept=pparams.lj_switch_intercept[pidx],
            lj_coeff_sigma12=pparams.lj_coeff_sigma12[pidx],
            lj_coeff_sigma6=pparams.lj_coeff_sigma6[pidx],
            lj_spline_y0=pparams.lj_spline_y0[pidx],
            lj_spline_dy0=pparams.lj_spline_dy0[pidx],
            # Global params
            lj_switch_dis2sigma=gparams.lj_switch_dis2sigma,
            spline_start=gparams.spline_start,
            max_dis=gparams.max_dis,
        )

        # split into atr & rep
        # atrE = np.copy(ljE);
        # selector3 = (dists < lj_lk_pair_params["lj_sigma"])
        # atrE[ selector3  ] = -lj_lk_pair_params["lj_wdepth"][ selector3 ]
        # repE = ljE - atrE

        # atrE *= lj_lk_pair_params["weights"]
        # repE *= lj_lk_pair_params["weights"]

    @reactive_property
    def total_lj(lj):
        """total inter-atomic lj"""
        return lj.sum()


@reactive_attrs
class LKIntraScore(LJLKIntraParam):
    @reactive_property
    @validate_args
    def lk(
        atom_pair_inds: Tensor(torch.long)[:, 3],
        atom_pair_dist: Tensor(torch.float)[:],
        ljlk_interaction_weight: Tensor(torch.float)[:, :, :],
        ljlk_atom_pair_params: LJLKTypePairParams,
        param_resolver: LJLKParamResolver,
    ):
        gparams = param_resolver.global_params
        pparams = ljlk_atom_pair_params

        assert (atom_pair_inds[:, 0] == 0).all()
        pidx = (atom_pair_inds[:, 1], atom_pair_inds[:, 2])

        assert ljlk_interaction_weight.shape[0] == 1
        ljlk_interaction_weight = ljlk_interaction_weight[0]

        return lk_score(
            # Distance
            dist=atom_pair_dist,
            # Bonded params
            interaction_weight=ljlk_interaction_weight[pidx],
            # Pair params
            lj_rad1=pparams.lj_rad1[pidx],
            lj_rad2=pparams.lj_rad2[pidx],
            lk_coeff1=pparams.lk_coeff1[pidx],
            lk_coeff2=pparams.lk_coeff2[pidx],
            lk_inv_lambda2_1=pparams.lk_inv_lambda2_1[pidx],
            lk_inv_lambda2_2=pparams.lk_inv_lambda2_2[pidx],
            lk_spline_close_dy1=pparams.lk_spline_close_dy1[pidx],
            lk_spline_close_x0=pparams.lk_spline_close_x0[pidx],
            lk_spline_close_x1=pparams.lk_spline_close_x1[pidx],
            lk_spline_close_y0=pparams.lk_spline_close_y0[pidx],
            lk_spline_close_y1=pparams.lk_spline_close_y1[pidx],
            lk_spline_far_dy0=pparams.lk_spline_far_dy0[pidx],
            lk_spline_far_y0=pparams.lk_spline_far_y0[pidx],
            # Global params
            spline_start=gparams.spline_start,
            max_dis=gparams.max_dis,
        )

    @reactive_property
    def total_lk(lk):
        """total inter-atomic lk"""
        return lk.sum()


@reactive_attrs(auto_attribs=True)
class LJLKScoreGraph(
    InteratomicDistanceGraphBase,
    BondedAtomScoreGraph,
    ScoreComponent,
    ParamDB,
    TorchDevice,
    Factory,
):
    total_score_components = [
        ScoreComponentClasses("lj", intra_container=LJIntraScore, inter_container=None),
        ScoreComponentClasses("lk", intra_container=LKIntraScore, inter_container=None),
    ]

    @staticmethod
    def factory_for(
        val,
        parameter_database: ParameterDatabase,
        ljlk_database: Optional[LJLKDatabase] = None,
        **_,
    ):
        """Overridable clone-constructor.

        Initialize from ``val.ljlk_database`` if possible, otherwise from
        ``parameter_database.scoring.ljlk``.
        """
        if ljlk_database is None:
            if getattr(val, "ljlk_database", None):
                ljlk_database = val.ljlk_database
            else:
                ljlk_database = parameter_database.scoring.ljlk

        return dict(ljlk_database=ljlk_database)

    ljlk_database: LJLKDatabase

    @property
    def component_atom_pair_dist_threshold(self):
        """Expose lj threshold distance for interatomic distance dispatch."""
        return self.ljlk_database.global_parameters.max_dis

    @reactive_property
    @validate_args
    def param_resolver(
        ljlk_database: LJLKDatabase, device: torch.device
    ) -> LJLKParamResolver:
        """Parameter tensor groups and atom-type to parameter resolver."""
        return LJLKParamResolver.from_database(ljlk_database, device)

    @reactive_property
    @validate_args
    def ljlk_interaction_weight(
        bonded_path_length: NDArray("f4")[:, :, :],
        real_atoms: Tensor(bool)[:, :],
        device: torch.device,
    ) -> Tensor(torch.float)[:, :, :]:
        """lj&lk interaction weight, bonded cutoff"""

        bonded_path_length = torch.from_numpy(bonded_path_length).to(device)

        result = bonded_path_length.new_ones(
            bonded_path_length.shape, dtype=torch.float
        )

        real_atoms = real_atoms.to(device=device)

        result[bonded_path_length < 4] = 0
        result[bonded_path_length == 4] = .2
        result = result.where(real_atoms[:, None, :], result.new_full((1,), 0))
        result = result.where(real_atoms[:, :, None], result.new_full((1,), 0))

        return result

    @reactive_property
    @validate_args
    def ljlk_atom_pair_params(
        atom_types: NDArray(object)[:, :], param_resolver: LJLKParamResolver
    ) -> LJLKTypePairParams:
        """Pair parameter tensors for all atoms within system."""
        assert atom_types.shape[0] == 1
        atom_types = atom_types[0]
        return param_resolver[atom_types.reshape((-1, 1)), atom_types.reshape((1, -1))]
