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
from ..total_score import ScoreComponentAttributes, TotalScoreComponentsGraph
from ..interatomic_distance import InteratomicDistanceGraphBase
from ..bonded_atom import BondedAtomScoreGraph
from ..factory import Factory

from .potentials import lj_score, lk_score
from .params import LJLKParamResolver, LJLKTypePairParams


@reactive_attrs(auto_attribs=True)
class LJLKScoreGraph(
    InteratomicDistanceGraphBase,
    BondedAtomScoreGraph,
    TotalScoreComponentsGraph,
    ParamDB,
    TorchDevice,
    Factory,
):
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
    def component_total_score_terms(self):
        """Expose lj/lk as total_score terms."""
        return (
            ScoreComponentAttributes(name="lk", total="total_lk"),
            ScoreComponentAttributes(name="lj", total="total_lj"),
        )

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
        bonded_path_length: NDArray("f4")[:, :],
        atom_types: NDArray(object)[:],
        real_atoms: Tensor(bool)[:],
        device: torch.device,
    ) -> Tensor(torch.float)[:, :]:
        """lj&lk interaction weight, bonded cutoff"""

        bonded_path_length = torch.from_numpy(bonded_path_length).to(device)

        result = bonded_path_length.new_ones(
            bonded_path_length.shape, dtype=torch.float
        )

        result[bonded_path_length < 4] = 0
        result[bonded_path_length == 4] = .2
        result[~real_atoms, :] = 0
        result[:, ~real_atoms] = 0

        return result

    @reactive_property
    @validate_args
    def ljlk_atom_pair_params(
        atom_types: NDArray(object)[:], param_resolver: LJLKParamResolver
    ) -> LJLKTypePairParams:
        """Pair parameter tensors for all atoms within system."""
        return param_resolver[atom_types.reshape((-1, 1)), atom_types.reshape((1, -1))]

    @reactive_property
    @validate_args
    def lj(
        atom_pair_inds: Tensor(torch.long)[2, :],
        atom_pair_dist: Tensor(torch.float)[:],
        ljlk_interaction_weight: Tensor(torch.float)[:, :],
        ljlk_atom_pair_params: LJLKTypePairParams,
        param_resolver: LJLKParamResolver,
    ):
        gparams = param_resolver.global_params
        pparams = ljlk_atom_pair_params
        pidx = (atom_pair_inds[0], atom_pair_inds[1])
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
    @validate_args
    def lk(
        atom_pair_inds: Tensor(torch.long)[2, :],
        atom_pair_dist: Tensor(torch.float)[:],
        ljlk_interaction_weight: Tensor(torch.float)[:, :],
        ljlk_atom_pair_params: LJLKTypePairParams,
        param_resolver: LJLKParamResolver,
    ):
        gparams = param_resolver.global_params
        pparams = ljlk_atom_pair_params
        pidx = (atom_pair_inds[0], atom_pair_inds[1])
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
    def total_lj(lj):
        """total inter-atomic lj"""
        return lj.sum()

    @reactive_property
    def total_lk(lk):
        """total inter-atomic lk"""
        return lk.sum()
