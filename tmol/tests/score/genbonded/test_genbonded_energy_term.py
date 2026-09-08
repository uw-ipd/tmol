import pytest
import torch

from tmol.database import ParameterDatabase
from tmol.score.genbonded import GenBondedEnergyTerm

# one fixture per way a noncanonical residue reaches the scoring path; ubq and
#    friends score no gen_torsions at all, so they pin nothing here
# generate with fixed seed
_CONFORMER_SEED = 20250828

NCAA_FIXTURES = {
    "alpha_aa": "collagen_hyp_1bkv",
    "nonstandard_aa": "beta_peptide_3c3g",
    "dna": "na_dna_5mc_1d17",
    "nonstandard_na": "na_dna_ttd_1ttd",
}


def test_smoke(default_database, torch_device: torch.device):
    genbonded_energy = GenBondedEnergyTerm(
        param_db=default_database, device=torch_device
    )

    assert genbonded_energy.device == torch_device


def test_annotate_restypes(
    fresh_default_packed_block_types, default_database, torch_device
):
    genbonded_energy = GenBondedEnergyTerm(
        param_db=default_database, device=torch_device
    )
    pbt = fresh_default_packed_block_types

    for bt in pbt.active_block_types:
        genbonded_energy.setup_block_type(bt)
        assert hasattr(bt, "genbonded_intra_subgraphs")
        assert hasattr(bt, "genbonded_intra_params")
        assert hasattr(bt, "genbonded_atom_type_hierarchy")

    genbonded_energy.setup_packed_block_types(pbt)
    assert hasattr(pbt, "genbonded_atom_is_rosetta")
    assert hasattr(pbt, "genbonded_conn_scored_elsewhere")
    assert pbt.genbonded_atom_is_rosetta.device == torch_device


def test_canonical_protein_scores_no_gen_torsions(
    ubq_pdb, default_database, torch_device
):
    """Every torsion of a canonical protein belongs to the Rosetta terms.

    The generic database can match a canonical peptide junction once the
    Rosetta types carry generic fall-backs, so a non-zero score here means the
    partition has sprung a leak rather than that a parameter changed.
    """
    from tmol.io import pose_stack_from_pdb
    from tmol.score import ScoreFunction
    from tmol.score._score_types import ScoreType

    pose_stack = pose_stack_from_pdb(ubq_pdb, torch_device)
    sfxn = ScoreFunction(default_database, torch_device)
    sfxn.set_weight(ScoreType.gen_torsions, 1.0)

    scores = sfxn.render_whole_pose_scoring_module(pose_stack)(pose_stack.coords)
    assert scores.sum().item() == 0.0


@pytest.mark.parametrize("backbone_class", sorted(NCAA_FIXTURES))
def test_gradcheck_on_noncanonical_backbones(backbone_class, torch_device):
    """Derivatives against finite differences, on each backbone class.

    The inter-block gate returns early on two branches, and a branch that skips
    a torsion in the forward pass while its derivative still contributes shows
    up here and nowhere else.
    """
    from tmol.io import atom_array_from_cif, pose_stack_from_biotite
    from tmol.ligand import prepare_ligands
    from tmol.score import ScoreFunction
    from tmol.score._score_types import ScoreType
    from tmol.tests.data import data_path

    stem = NCAA_FIXTURES[backbone_class]
    param_db = ParameterDatabase.get_default()
    structure = atom_array_from_cif(data_path("ncaa_fixtures") / (stem + ".cif"))
    prepared, _ordering = prepare_ligands(
        structure, param_db=param_db, seed=_CONFORMER_SEED
    )
    pose_stack = pose_stack_from_biotite(structure, torch_device, param_db=prepared)

    sfxn = ScoreFunction(prepared, torch_device)
    sfxn.set_weight(ScoreType.gen_torsions, 1.0)
    module = sfxn.render_whole_pose_scoring_module(pose_stack)

    coords = pose_stack.coords.detach().clone().to(torch.float64).requires_grad_(True)
    total = module(coords).sum()
    assert total.item() != 0.0, "%s scores no gen_torsions" % stem
    total.backward()
    analytic = coords.grad.detach().clone()

    # a handful of atoms, spread through the pose rather than all from one end
    n_atoms = coords.shape[1]
    step = max(1, n_atoms // 8)
    eps = 1e-4
    for atom in range(0, n_atoms, step):
        for axis in range(3):
            if analytic[0, atom, axis] == 0.0:
                continue
            shifted = coords.detach().clone()
            shifted[0, atom, axis] += eps
            up = module(shifted).sum().item()
            shifted[0, atom, axis] -= 2 * eps
            down = module(shifted).sum().item()
            numeric = (up - down) / (2 * eps)
            assert numeric == pytest.approx(
                float(analytic[0, atom, axis]), abs=1e-3, rel=1e-3
            ), "%s atom %d axis %d" % (stem, atom, axis)
