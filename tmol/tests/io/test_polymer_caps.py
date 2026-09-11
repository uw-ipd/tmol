"""Prepared terminal caps must build an intact, scoreable polymer chain."""

import pytest
import torch

from tmol.io import pose_stack_from_cif
from tmol.score import beta2016_score_function
from tmol.tests.data import data_path


@pytest.mark.parametrize("cap", ["nh2", "nme"])
def test_capped_peptide_builds_bonds_scores_and_gradients(cap, torch_device):
    pose, context = pose_stack_from_cif(
        data_path("ncaa_fixtures") / f"capped_peptide_ace_{cap}.cif",
        torch_device,
        prepare_ligands=True,
        ligand_seed=20260909,
        no_optH=True,
        return_context=True,
    )
    types = [
        pose.packed_block_types.active_block_types[int(i)]
        for i in pose.block_type_ind[0]
    ]
    assert [t.name for t in types] == ["ACE", "ALA", cap.upper()]
    assert types[0].down_connection_ind == -1
    assert types[2].up_connection_ind == -1
    for left in range(2):
        right = left + 1
        up, down = types[left].up_connection_ind, types[right].down_connection_ind
        assert pose.inter_residue_connections[0, left, up].tolist() == [right, down]
        assert pose.inter_residue_connections[0, right, down].tolist() == [left, up]
    assert "OXT" not in types[1].atom_to_idx

    def xyz(block, name):
        index = int(pose.block_coord_offset[0, block]) + types[block].atom_to_idx[name]
        return pose.coords[0, index]

    elements = {
        a.name: a.element for a in context.parameter_database.chemical.atom_types
    }
    element = {a.name: elements[a.atom_type] for a in types[2].atoms}
    neighbors = {b for a, b, *_ in types[2].bonds if a == "N"}
    neighbors.update(a for a, b, *_ in types[2].bonds if b == "N")
    hydrogens = sorted(n for n in neighbors if element[n] == "H")
    carbons = sorted(n for n in neighbors if element[n] == "C")
    assert len(hydrogens) == (2 if cap == "nh2" else 1)
    nitrogen, carbon = xyz(2, "N"), xyz(1, "C")
    third = xyz(2, carbons[0]) if carbons else xyz(1, "CA")
    normal = torch.linalg.cross(carbon - nitrogen, third - nitrogen)
    normal = normal / normal.norm()
    for hydrogen in hydrogens:
        vector = xyz(2, hydrogen) - nitrogen
        assert 0.95 < float(vector.norm()) < 1.1
        assert abs(float(vector.dot(normal))) < 0.02
    score = beta2016_score_function(torch_device, param_db=context.parameter_database)
    module = score.render_whole_pose_scoring_module(pose)
    coords = pose.coords.detach().clone().requires_grad_(True)
    values = module(coords)
    values.sum().backward()
    assert torch.isfinite(values).all()
    assert torch.isfinite(coords.grad).all()
