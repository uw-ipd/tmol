"""Execute the custom-interface tutorial on real saved model predictions."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from tmol.score import beta2016_score_function
from tmol.io import default_canonical_ordering, canonical_form_from_pose_stack


@pytest.mark.parametrize("source", ["openfold", "rf2_preserve", "rf2_rebuild"])
def test_tutorial_model_to_score_gradient(source, torch_device):
    root = Path(__file__).parents[3]
    spec = importlib.util.spec_from_file_location(
        "model_input_tutorial", root / "docs/examples/model_inputs.py"
    )
    tutorial = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tutorial)
    data = root / "tmol/tests/data"
    if source == "openfold":
        prediction = torch.load(
            data / "openfold/openfold_ubq_and_sumo.pt", map_location=torch_device
        )
        coords = prediction["positions"].detach().clone().requires_grad_()
        prediction["positions"] = coords
        pose = tutorial.openfold_example(prediction)
        expected_poses = 2
    else:
        prediction = torch.load(
            data / "rosettafold2/ubiquitin.pt", map_location=torch_device
        )
        layout = json.loads((data / "rosettafold2/input_layout.json").read_text())
        coords = prediction["xyz"].detach().clone().requires_grad_()
        prediction["xyz"] = coords
        pose = tutorial.rf2_example(
            prediction, layout["num2aa"], layout["aa2long"], hydrogens=source[4:]
        )
        expected_poses = 1
    assert pose.n_poses == expected_poses
    assert torch.isfinite(pose.coords[pose.real_atoms]).all()
    ordering = default_canonical_ordering()
    form = canonical_form_from_pose_stack(ordering, pose)
    ca_slots = torch.tensor(
        [
            ordering.restypes_atom_index_mapping[name].get("CA", -1)
            for name in ordering.restype_io_equiv_classes
        ],
        device=torch_device,
    )
    batch, residue = torch.nonzero(form.res_types >= 0, as_tuple=True)
    actual_ca = form.coords[batch, residue, ca_slots[form.res_types[batch, residue]]]
    expected_ca = (
        coords[-1, batch, residue, 1] if source == "openfold" else coords[residue, 1]
    )
    torch.testing.assert_close(actual_ca, expected_ca, atol=0, rtol=0)
    score = beta2016_score_function(torch_device).render_whole_pose_scoring_module(pose)
    energy = score(pose.coords)
    energy.sum().backward()
    assert torch.isfinite(energy).all()
    assert torch.isfinite(coords.grad).all()
    assert torch.any(coords.grad != 0)
    if source == "rf2_rebuild":
        assert torch.count_nonzero(coords.grad[:, 14:]) == 0
