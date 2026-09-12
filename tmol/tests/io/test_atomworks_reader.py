"""Exercise the optional shared parser without duplicating CIF completion."""

from pathlib import Path

import numpy as np
import pytest
import torch

from tmol.io import atom_array_from_cif, pose_stack_from_cif
from tmol.score import beta2016_score_function

pytest.importorskip("atomworks")

DATA = Path(__file__).parents[1] / "data"


def test_atomworks_completion_preserves_entirely_unresolved_ligand():
    array = atom_array_from_cif(
        DATA / "atomworks_regressions/unresolved_unl.cif", reader="atomworks"
    )
    ligand = array[array.res_name == "UNL"]
    assert len(ligand) == 28
    assert np.isnan(ligand.coord).all()
    assert ligand.bonds.get_bond_count() > 0


def test_label_template_substitution_cannot_silently_erase_unknown_atom():
    with pytest.raises(ValueError, match="XYZ"):
        atom_array_from_cif(
            DATA / "atomworks_regressions/unknown_heavy_atom_1a8o.cif",
            reader="atomworks",
        )


def test_leaving_group_completion_cannot_silently_erase_observed_phosphate():
    # The 8OG template flags both OP2 and OP3 as possible leaving atoms.
    # One polymer linkage must not silently remove both branches. Until the
    # shared parser resolves this ambiguity, the adapter reports the loss.
    with pytest.raises(ValueError, match="8OG.OP2"):
        atom_array_from_cif(
            DATA / "ncaa_fixtures/na_dna_8og_183d.cif", reader="atomworks"
        )


@pytest.mark.parametrize(
    "fixture", ["capped_peptide_ace_nh2.cif", "beta_peptide_3c3g.cif"]
)
def test_shared_parser_builds_and_scores_general_chemistry(fixture):
    pose, context = pose_stack_from_cif(
        DATA / "ncaa_fixtures" / fixture,
        torch.device("cpu"),
        reader="atomworks",
        prepare_ligands=True,
        ligand_seed=20260909,
        no_optH=True,
        return_context=True,
    )
    assert torch.isfinite(pose.coords[pose.real_atoms]).all()
    score = beta2016_score_function(pose.device, param_db=context.parameter_database)
    coords = pose.coords.detach().clone().requires_grad_()
    energy = score.render_whole_pose_scoring_module(pose)(coords)
    energy.sum().backward()
    assert torch.isfinite(energy).all()
    assert torch.isfinite(coords.grad).all()


@pytest.mark.parametrize("state", ["HD1", "HE2", "both", "none"])
def test_reader_preserves_observed_histidine_tautomer_evidence(tmp_path, state):
    from biotite.structure import info
    from biotite.structure.io import pdbx

    source = info.residue("HIS")
    source.chain_id[:] = "A"
    source.res_id[:] = 1
    removed = {"HD1": ["HE2"], "HE2": ["HD1"], "both": [], "none": ["HD1", "HE2"]}[
        state
    ]
    source = source[~np.isin(source.atom_name, removed)]
    file = pdbx.CIFFile()
    pdbx.set_structure(file, source)
    path = tmp_path / "histidine.cif"
    file.write(path)
    parsed = atom_array_from_cif(path, reader="atomworks")
    for name in ("HD1", "HE2"):
        observed = source.atom_name == name
        retained = parsed.atom_name == name
        assert bool(retained.any()) == bool(observed.any())
        if observed.any():
            np.testing.assert_allclose(
                parsed.coord[retained], source.coord[observed], atol=0.001
            )
