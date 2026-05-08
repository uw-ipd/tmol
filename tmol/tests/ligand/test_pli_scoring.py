"""Protein-ligand interface scoring tests.

Scores every complex PDB in the protein_ligand_test folder and prints the
total energy from both whole-pose scoring and block-pair scoring.
"""

from pathlib import Path

import pytest

PLI_DIR = Path(__file__).parent.parent / "data" / "protein_ligand_test"
PLI_CASES = sorted(p.name for p in PLI_DIR.glob("*_complex*.pdb"))


class TestPLIScoring:
    """Score each protein-ligand complex in protein_ligand_test and print the total energy."""

    @pytest.fixture(params=PLI_CASES)
    def pli_pdb(self, request):
        return PLI_DIR / request.param

    def test_score(self, pli_pdb, torch_device):
        import biotite.structure
        import biotite.structure.io

        from tmol.database import ParameterDatabase
        from tmol.io.pose_stack_from_biotite import pose_stack_from_biotite
        from tmol.score import beta2016_score_function

        bt_struct = biotite.structure.io.load_structure(str(pli_pdb))
        if isinstance(bt_struct, biotite.structure.AtomArrayStack):
            bt_struct = bt_struct[0]

        param_db = ParameterDatabase.get_default()
        pose_stack = pose_stack_from_biotite(bt_struct, torch_device, param_db=param_db)

        sfxn = beta2016_score_function(torch_device, param_db=param_db)

        # --- Whole-pose scoring ---
        whole_scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
        unweighted = whole_scorer.unweighted_scores(pose_stack.coords)
        weights = sfxn.weights_tensor()
        score_types = sfxn.all_score_types()
        total_whole = sum(
            float(weights[i]) * float(unweighted[i, 0]) for i in range(len(score_types))
        )

        # --- Block-pair scoring ---
        # scorer(coords) returns weighted, summed-over-terms tensor of shape
        # [n_poses, max_n_blocks, max_n_blocks]; sum over block pairs for total.
        bp_scorer = sfxn.render_block_pair_scoring_module(pose_stack)
        bp_scores = bp_scorer(pose_stack.coords)  # [n_poses, n_blocks, n_blocks]
        total_bp = float(bp_scores[0].sum())

        print(
            f"\n{pli_pdb.name}:"
            f"  whole_pose = {total_whole:.4f}"
            f"  block_pair = {total_bp:.4f}"
        )
