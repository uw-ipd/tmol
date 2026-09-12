"""Exercise the optional shared parser without duplicating CIF completion."""

from pathlib import Path

import numpy as np
import pytest
import torch

from tmol.io import atom_array_from_cif, pose_stack_from_cif
from tmol.score import beta2016_score_function

pytest.importorskip("atomworks")

DATA = Path(__file__).parents[1] / "data"


@pytest.mark.parametrize("reader", ["tmol", "atomworks"])
def test_file_hydrogen_policy_through_scoring(tmp_path, ubq_pdb, reader):
    from biotite.structure.io import pdbx
    from tmol.io import (
        biotite_from_pose_stack,
        pose_stack_from_pdb,
        pose_stack_from_biotite,
    )

    device = torch.device("cpu")
    source = biotite_from_pose_stack(pose_stack_from_pdb(ubq_pdb, device))
    index = np.flatnonzero((source.res_name == "ALA") & (source.element == "H"))[0]
    source.coord[index] += [0.2, 0.1, -0.1]
    file = pdbx.CIFFile()
    pdbx.set_structure(file, source)
    path = tmp_path / "hydrogens.cif"
    file.write(path)
    for policy in ("preserve", "rebuild"):
        array = atom_array_from_cif(path, reader=reader, hydrogen_policy=policy)
        selected = (
            (array.res_id == source.res_id[index])
            & (array.atom_name == source.atom_name[index])
            & (array.chain_id == source.chain_id[index])
        )
        if policy == "preserve":
            np.testing.assert_allclose(
                array.coord[selected], source.coord[index][None], atol=0.001
            )
        else:
            assert not np.isfinite(array.coord[selected]).all(axis=-1).any()
        pose = pose_stack_from_biotite(array, device, no_optH=True)
        coords = pose.coords.detach().clone().requires_grad_()
        energy = beta2016_score_function(device).render_whole_pose_scoring_module(pose)(
            coords
        )
        energy.sum().backward()
        assert torch.isfinite(energy).all()
        assert torch.isfinite(coords.grad).all()


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
    path = DATA / "ncaa_fixtures/na_dna_8og_183d.cif"
    native = atom_array_from_cif(path)
    shared = atom_array_from_cif(path, reader="atomworks")
    for array in (native, shared):
        nucleotide = array[array.res_name == "8OG"]
        assert "OP2" in nucleotide.atom_name
        assert "OP3" not in nucleotide.atom_name
    np.testing.assert_array_equal(
        shared.coord[(shared.res_name == "8OG") & (shared.atom_name == "OP2")],
        native.coord[(native.res_name == "8OG") & (native.atom_name == "OP2")],
    )


@pytest.mark.parametrize(
    "fixture",
    [
        "ncaa_fixtures/capped_peptide_ace_nh2.cif",
        "ncaa_fixtures/beta_peptide_3c3g.cif",
        "ncaa_fixtures/na_dna_8og_183d.cif",
        "atomworks_regressions/acetylated_peptide_1j8z.cif",
    ],
)
def test_shared_parser_builds_and_scores_general_chemistry(fixture):
    pose, context = pose_stack_from_cif(
        DATA / fixture,
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
