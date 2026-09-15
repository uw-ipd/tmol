"""Prepare, persist, score and minimize each noncanonical backbone class.

The fixed-seed, per-lane score regression catches changes at the boundary
between canonical and generated parameters. Parameter replay must additionally
preserve every score lane and coordinate gradient. No independent absolute
gradient reference exists for these generated parameters, so gradients are not
promoted to self-generated goldens.
"""

import numpy as np
import pytest
import torch
import yaml

from tmol.io import atom_array_from_cif, pose_stack_from_biotite
from tmol.ligand import prepare_ligands
from tmol.score import beta2016_score_function
from tmol.tests.data import data_path
from tmol.tests.io.test_atomworks_corpus_regressions import _score_and_minimize

FIXTURES = {
    "alpha_aa": "collagen_hyp_1bkv",
    "nonstandard_aa": "beta_peptide_3c3g",
    "dna": "na_dna_5mc_1d17",
    "nonstandard_na": "na_dna_ttd_1ttd",
}

SCORE_GOLDEN_ATOL = 5e-4


@pytest.mark.parametrize("backbone_class", sorted(FIXTURES))
def test_noncanonical_parameter_replay_and_minimization(
    backbone_class, torch_device, tmp_path
):
    array = atom_array_from_cif(
        data_path("ncaa_fixtures", FIXTURES[backbone_class] + ".cif")
    )
    supplied = array.coord.copy()
    path = tmp_path / "prepared.tmol"
    prepared, _ = prepare_ligands(array, seed=20250828, params_output=str(path))
    restored, _ = prepare_ligands(array, params_files=[str(path)])
    poses, scores, gradients = [], [], []
    for database in (prepared, restored):
        pose, context = pose_stack_from_biotite(
            array, torch_device, param_db=database, no_optH=True, return_context=True
        )
        poses.append(pose)
        active = [
            pose.packed_block_types.active_block_types[i]
            for i in pose.block_type_ind[pose.block_type_ind >= 0].tolist()
        ]
        assert backbone_class in {bt.properties.polymer.backbone_type for bt in active}
        assert torch.isfinite(pose.coords[pose.real_atoms]).all()
        # Use double precision for a single strict cross-device score oracle;
        # float32 full-pose reductions drift by about 1e-3 between CPU and CUDA.
        coords = pose.coords.detach().to(dtype=torch.float64).clone().requires_grad_()
        sfxn = beta2016_score_function(torch_device, param_db=database)
        module = sfxn.render_whole_pose_scoring_module(pose)
        value = module(coords, sum_terms=False, apply_weights=False)
        gradient = torch.autograd.grad(value.sum(), coords)[0]
        assert torch.isfinite(value).all() and torch.isfinite(gradient).all()
        by_name = {
            st.name: float(v.detach().sum())
            for st, v in zip(sfxn.all_score_types(), value)
        }
        assert by_name["gen_torsions"] != 0
        scores.append(value.detach())
        gradients.append(gradient)
    for field in (
        "coords",
        "block_type_ind",
        "block_coord_offset",
        "inter_residue_connections",
    ):
        torch.testing.assert_close(
            getattr(poses[0], field), getattr(poses[1], field), rtol=0, atol=0
        )
    tolerances = {"rtol": 0, "atol": 0} if torch_device.type == "cpu" else {}
    torch.testing.assert_close(scores[0], scores[1], **tolerances)
    torch.testing.assert_close(gradients[0], gradients[1], **tolerances)
    score_golden = yaml.safe_load(data_path("noncanonical_scores.yaml").read_text())[
        backbone_class
    ]
    assert score_golden.keys() == by_name.keys()
    moved = {
        name: {
            "golden": score_golden[name],
            "observed": value,
            "delta": value - score_golden[name],
        }
        for name, value in by_name.items()
        if not np.isclose(value, score_golden[name], rtol=0, atol=SCORE_GOLDEN_ATOL)
    }
    assert not moved, moved
    _score_and_minimize(poses[1], context, max_iter=20)
    np.testing.assert_array_equal(array.coord, supplied)
