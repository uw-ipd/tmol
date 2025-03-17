import torch
import numpy

from tmol.score import _non_memoized_beta2016
from tmol.score.score_function import ScoreFunction
from tmol.score.score_types import ScoreType
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol import (
    pose_stack_from_pdb,
    beta2016_score_function,
    canonical_form_from_pdb,
    default_canonical_ordering,
    default_packed_block_types,
    pose_stack_from_canonical_form,
)


def test_pose_score_smoke(ubq_pdb, default_database, torch_device):
    pose_stack1 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=4)
    pose_stack100 = PoseStackBuilder.from_poses([pose_stack1] * 100, torch_device)

    sfxn = ScoreFunction(default_database, torch_device)
    sfxn.set_weight(ScoreType.fa_ljatr, 1.0)
    sfxn.set_weight(ScoreType.fa_ljrep, 0.55)
    sfxn.set_weight(ScoreType.fa_lk, 0.8)

    scorer = sfxn.render_whole_pose_scoring_module(pose_stack100)

    scores = scorer(pose_stack100.coords)

    assert scores is not None


def test_virtual_residue_scoring(ubq_pdb, torch_device):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)

    # vrt_ratmap = co.restypes_atom_index_mapping["VRT"]
    # print("vrt_ratmap", vrt_ratmap)

    # ATOM     37  N   ILE A   3      26.849  29.656   6.217  1.00  5.87           N
    # ATOM     38  CA  ILE A   3      26.235  30.058   7.497  1.00  5.07           C
    # ATOM     39  C   ILE A   3      26.882  31.428   7.862  1.00  4.01           C

    def pose_stack_of_nres(nres, add_vrt):
        def xyz(x, y, z):
            return torch.tensor((x, y, z), dtype=torch.float32, device=torch_device)

        canonical_form = canonical_form_from_pdb(
            co, ubq_pdb, torch_device, residue_start=0, residue_end=nres
        )
        if add_vrt:
            vrt_co_ind = co.restype_io_equiv_classes.index("VRT")
            # print("vrt_co_ind", vrt_co_ind)
            orig_coords = canonical_form["coords"]
            ocs = orig_coords.shape
            new_coords = torch.full(
                (ocs[0], ocs[1] + 1, ocs[2], ocs[3]),
                numpy.nan,
                dtype=torch.float32,
                device=torch_device,
            )
            new_coords[0, :-1, :, :] = orig_coords
            # Let's put the VRT right in the center of res "2", ILE 3
            new_coords[0, -1, 0, :] = xyz(26.849, 29.656, 6.217)
            new_coords[0, -1, 1, :] = xyz(26.849, 29.656, 6.217) + xyz(1.0, 0.0, 0.0)
            new_coords[0, -1, 2, :] = xyz(26.849, 29.656, 6.217) + xyz(0.0, 1.0, 0.0)
            orig_chain_id = canonical_form["chain_id"]

            ocis = orig_chain_id.shape
            new_chain_id = torch.zeros(
                (ocis[0], ocis[1] + 1), dtype=torch.int32, device=torch_device
            )
            new_chain_id[0, :-1] = orig_chain_id
            new_chain_id[0, -1] = (
                orig_chain_id[0, -1] + 1
            )  # give the vrt res a new chain id

            orig_restypes = canonical_form["res_types"]
            ors = orig_restypes.shape
            new_restypes = torch.full(
                (ors[0], ors[1] + 1), -1, dtype=torch.int32, device=torch_device
            )
            new_restypes[0, :-1] = orig_restypes
            new_restypes[0, -1] = vrt_co_ind

            canonical_form["coords"] = new_coords
            canonical_form["chain_id"] = new_chain_id
            canonical_form["res_types"] = new_restypes

        return pose_stack_from_canonical_form(co, pbt, **canonical_form)

    ps_wo_vrt = PoseStackBuilder.from_poses(
        [pose_stack_of_nres(x, False) for x in [4, 6, 5]], torch_device
    )
    ps_w_vrt = PoseStackBuilder.from_poses(
        [pose_stack_of_nres(x, True) for x in [4, 6, 5]], torch_device
    )

    sfxn = beta2016_score_function(torch_device)
    scorer_wo_vrt = sfxn.render_whole_pose_scoring_module(ps_wo_vrt)
    scores_wo_vrt = scorer_wo_vrt(ps_wo_vrt.coords)

    scorer_w_vrt = sfxn.render_whole_pose_scoring_module(ps_w_vrt)
    scores_w_vrt = scorer_w_vrt(ps_w_vrt.coords)

    unweighted_scores_wo_vrt = scorer_wo_vrt.unweighted_scores(ps_wo_vrt.coords)
    unweighted_scores_w_vrt = scorer_w_vrt.unweighted_scores(ps_w_vrt.coords)

    torch.testing.assert_close(scores_wo_vrt, scores_w_vrt)
    torch.testing.assert_close(unweighted_scores_wo_vrt, unweighted_scores_w_vrt)


def test_score_function_all_score_types(ubq_pdb):
    device = torch.device("cpu")
    ps = pose_stack_from_pdb(ubq_pdb, device)
    sfxn = beta2016_score_function(device)

    wpsm = sfxn.render_whole_pose_scoring_module(ps)
    unweighted_scores = wpsm.unweighted_scores(ps.coords)
    score_types = sfxn.all_score_types()
    unweighted_score_map = {
        st: unweighted_scores[i, :].detach().cpu().numpy()
        for i, st in enumerate(score_types)
    }

    def n(x):
        return numpy.array(x)

    gold_score_map = {
        ScoreType.cart_lengths: n([37.762318]),
        ScoreType.cart_angles: n([183.56915]),
        ScoreType.cart_torsions: n([50.584225]),
        ScoreType.cart_impropers: n([9.430529]),
        ScoreType.cart_hxltorsions: n([47.41971]),
        ScoreType.disulfide: n([0.0]),
        ScoreType.fa_ljatr: n([-417.9582]),
        ScoreType.fa_ljrep: n([240.7147]),
        ScoreType.fa_lk: n([298.27637]),
        ScoreType.fa_elec: n([-136.454]),
        ScoreType.hbond: n([-55.675613]),
        ScoreType.lk_ball_iso: n([422.03955]),
        ScoreType.lk_ball: n([172.19647]),
        ScoreType.lk_bridge: n([1.5785888]),
        ScoreType.lk_bridge_uncpl: n([10.9946]),
        ScoreType.rama: n([-12.743372]),
        ScoreType.omega: n([4.100171]),
        ScoreType.ref: n([-41.275]),
        ScoreType.dunbrack_rot: n([70.64968]),
        ScoreType.dunbrack_rotdev: n([240.31009]),
        ScoreType.dunbrack_semirot: n([99.660904]),
    }
    for st in score_types:
        numpy.testing.assert_allclose(
            unweighted_score_map[st], gold_score_map[st], rtol=1e-4, atol=1e-4
        )


def test_score_function_one_body_terms_getter():
    from tmol.score.backbone_torsion.bb_torsion_energy_term import (
        BackboneTorsionEnergyTerm,
    )
    from tmol.score.cartbonded.cartbonded_energy_term import CartBondedEnergyTerm
    from tmol.score.disulfide.disulfide_energy_term import DisulfideEnergyTerm
    from tmol.score.dunbrack.dunbrack_energy_term import DunbrackEnergyTerm
    from tmol.score.elec.elec_energy_term import ElecEnergyTerm
    from tmol.score.hbond.hbond_energy_term import HBondEnergyTerm
    from tmol.score.ljlk.ljlk_energy_term import LJLKEnergyTerm
    from tmol.score.lk_ball.lk_ball_energy_term import LKBallEnergyTerm
    from tmol.score.ref.ref_energy_term import RefEnergyTerm

    device = torch.device("cpu")
    sfxn = _non_memoized_beta2016(device)
    assert sfxn._one_body_terms_out_of_date

    terms_1b = sfxn.one_body_terms()
    assert not sfxn._one_body_terms_out_of_date

    valid_one_body_terms = [DunbrackEnergyTerm, RefEnergyTerm]
    for term in terms_1b:
        found = False
        for valid_option in valid_one_body_terms:
            if isinstance(term, valid_option):
                found = True
                break
        assert found


def test_score_function_two_body_terms_getter():
    from tmol.score.backbone_torsion.bb_torsion_energy_term import (
        BackboneTorsionEnergyTerm,
    )
    from tmol.score.cartbonded.cartbonded_energy_term import CartBondedEnergyTerm
    from tmol.score.disulfide.disulfide_energy_term import DisulfideEnergyTerm
    from tmol.score.dunbrack.dunbrack_energy_term import DunbrackEnergyTerm
    from tmol.score.elec.elec_energy_term import ElecEnergyTerm
    from tmol.score.hbond.hbond_energy_term import HBondEnergyTerm
    from tmol.score.ljlk.ljlk_energy_term import LJLKEnergyTerm
    from tmol.score.lk_ball.lk_ball_energy_term import LKBallEnergyTerm
    from tmol.score.ref.ref_energy_term import RefEnergyTerm

    device = torch.device("cpu")
    sfxn = _non_memoized_beta2016(device)
    assert sfxn._two_body_terms_out_of_date

    terms_2b = sfxn.two_body_terms()
    assert not sfxn._two_body_terms_out_of_date

    valid_two_body_terms = [
        BackboneTorsionEnergyTerm,
        CartBondedEnergyTerm,
        DisulfideEnergyTerm,
        ElecEnergyTerm,
        HBondEnergyTerm,
        LJLKEnergyTerm,
        LKBallEnergyTerm,
    ]
    for term in terms_2b:
        found = False
        for valid_option in valid_two_body_terms:
            if isinstance(term, valid_option):
                found = True
                break
        assert found


def test_score_function_all_terms_getter():
    from tmol.score.backbone_torsion.bb_torsion_energy_term import (
        BackboneTorsionEnergyTerm,
    )
    from tmol.score.cartbonded.cartbonded_energy_term import CartBondedEnergyTerm
    from tmol.score.disulfide.disulfide_energy_term import DisulfideEnergyTerm
    from tmol.score.dunbrack.dunbrack_energy_term import DunbrackEnergyTerm
    from tmol.score.elec.elec_energy_term import ElecEnergyTerm
    from tmol.score.hbond.hbond_energy_term import HBondEnergyTerm
    from tmol.score.ljlk.ljlk_energy_term import LJLKEnergyTerm
    from tmol.score.lk_ball.lk_ball_energy_term import LKBallEnergyTerm
    from tmol.score.ref.ref_energy_term import RefEnergyTerm

    device = torch.device("cpu")
    sfxn = _non_memoized_beta2016(device)
    assert sfxn._all_terms_out_of_date

    all_terms = sfxn.all_terms()
    assert not sfxn._all_terms_out_of_date

    valid_terms = [
        DunbrackEnergyTerm,
        RefEnergyTerm,
        BackboneTorsionEnergyTerm,
        CartBondedEnergyTerm,
        DisulfideEnergyTerm,
        ElecEnergyTerm,
        HBondEnergyTerm,
        LJLKEnergyTerm,
        LKBallEnergyTerm,
    ]
    for term in all_terms:
        found = False
        for valid_option in valid_terms:
            if isinstance(term, valid_option):
                found = True
                break
        assert found
