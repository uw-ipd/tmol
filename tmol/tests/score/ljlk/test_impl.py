import torch
import tmol.score.ljlk.params as params
import tmol.database as database
import tmol.system.score_support as score_support
import tmol.score.ljlk.potentials as potentials


def test_ljlk_potential_benchmark(benchmark, ubq_system, torch_device):

    coords = score_support.coords_for_system(
        ubq_system, torch_device, requires_grad=False
    )["coords"]
    atom_types = (
        score_support.bonded_atoms_for_system(ubq_system)["atom_types"]
    )

    db = database.ParameterDatabase.get_default()
    param_resolver = params.LJLKParamResolver.from_database(
        db.scoring.ljlk, torch_device
    )

    ljlk_atom_pair_params = param_resolver[atom_types.reshape((-1, 1)),
                                           atom_types.reshape((1, -1))]

    gparams = param_resolver.global_params
    pparams = ljlk_atom_pair_params
    idx = torch.arange(len(atom_types), device=torch_device, dtype=torch.long)
    pidx = [idx[:, None], idx[None, :]]

    @benchmark
    def lj_total_score():
        atom_pair_dist = (coords[:, None, :] - coords[None, :, :]).norm(dim=-1)
        ljlk_interaction_weight = torch.full_like(atom_pair_dist, 1)

        return float(
            potentials.lj_score(
                # Distance
                dist=atom_pair_dist,

                # Bonded params
                interaction_weight=ljlk_interaction_weight,

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
            ).sum()
        )
