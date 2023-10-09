from pytest import approx

from tmol.system.score_support import score_method_to_even_weights_dict

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.dunbrack import DunbrackScore
from tmol.score.modules.coords import coords_for


def test_dun_baseline_comparison(ubq_system, torch_device):
    score_system = ScoreSystem.build_for(
        ubq_system,
        {DunbrackScore},
        weights=score_method_to_even_weights_dict(DunbrackScore),
        drop_missing_atoms=False,
        requires_grad=False,
        device=torch_device,
    )
    coords = coords_for(ubq_system, score_system)

    intra_container = score_system.intra_forward(coords)

    # Rosetta 3 with -beta:
    # fa_dun_dev 239.723
    # fa_dun_rot 71.703
    # fa_dun_semi 99.705
    #
    # Obs as of 2023/8/3:
    # the discrepency appears to be that residue 1 and residue N are still not getting scored?
    # in R3:
    #                           fa_dun_dev fa_dun_rot fa_dun_semi
    #   MET:NtermProteinFull_1	2.60383    5.11634	  0
    #
    # Fix on 2023/8/21:
    # MET:nterm is now being scored; GLY:cterm would not get scored in any case
    # dunbrack_rot -- 66.5865 --> 70.649; 4.0625
    # dunbrack_rotdev -- 237.8934 --> 240.3100; 2.4166
    assert intra_container["dunbrack_rot"].detach().cpu().numpy() == approx(
        70.6497, rel=1e-3
    )
    assert intra_container["dunbrack_rotdev"].detach().cpu().numpy() == approx(
        240.3100, rel=1e-3
    )
    assert intra_container["dunbrack_semirot"].detach().cpu().numpy() == approx(
        99.6609, rel=1e-3
    )
