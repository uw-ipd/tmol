"""Regression tests for DUD benchmark ligands.

``TestDUDScoring`` injects the golden ``.tmol`` params (Rosetta reference) and
scores the input pose with tmol, comparing per-term energies against the
pre-generated Rosetta ``.sc`` files.

Param-generation parity (SMILES/mol2 -> Rosetta ``.params``) is covered by the
DUD-80 suite (``test_ligand_pipeline.py::TestGroundTruthRegression`` and
``scripts/ligand_prep/validate_dud80.py``); the CIF-generated-param path is
intentionally not scored here (it is not the validated guanfeng pipeline).
"""

from pathlib import Path

import pytest
import torch  # noqa: F401  (used via torch_device fixture)

DUD_DIR = Path(__file__).parent.parent / "data" / "dud_ligands"
DUD_CASES = [
    ("ada", name)
    for name in [
        "ZINC02169852",
        "ZINC03814293",
        "ZINC03814294",
        "ZINC03814298",
        "ZINC03814300",
        "ZINC03814305",
        "ZINC03814301",
        "ZINC03814297",
        "ZINC03814303",
        "ZINC01614355",
    ]
] + [
    ("comt", name)
    for name in [
        "ZINC00330141",
        "ZINC00392003",
        "ZINC00021789",
        "ZINC03801154",
        "ZINC03814480",
    ]
]


def _param_db_with_ligand_prep(prep):
    from tmol.database import ParameterDatabase
    from tmol.ligand.registry import inject_ligand_preparations

    return inject_ligand_preparations(ParameterDatabase.get_default(), [prep])


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------


# Terms in score.sc we compare against tmol (values are already weighted).
# For a single isolated ligand, Rosetta's intra-residue terms correspond to
# tmol's full pair terms (no inter-residue contributions exist).
_ROS_TERMS = {
    "fa_intra_atr_xover4": "fa_ljatr",
    "fa_intra_rep_xover4": "fa_ljrep",
    "fa_intra_sol_xover4": "fa_lk",
    "fa_intra_elec": "fa_elec",
    "gen_bonded": "gen_torsions",
}


def _rosetta_score(sc_path: Path) -> dict[str, float]:
    """Parse a pre-generated score.sc file; values are already weighted."""
    with open(sc_path) as f:
        lines = [l for l in f if l.startswith("SCORE:")]
    header = lines[0].split()[1:]
    values = lines[1].split()[1:]
    scores = {}
    for h, v in zip(header, values):
        try:
            scores[h] = float(v)
        except ValueError:
            scores[h] = v
    return scores


class TestDUDScoring:
    """Score each DUD ligand from golden ``.tmol`` params vs Rosetta ``.sc``.

    Parameters are read straight from the golden ``.tmol`` file via
    ``load_params_file`` (Rosetta-reference params), so this isolates scoring
    correctness from param generation.
    """

    @pytest.fixture(params=DUD_CASES, ids=[f"{t}_{n}" for t, n in DUD_CASES])
    def dud_scoring_data(self, request):
        target, name = request.param
        cif_path = DUD_DIR / target / f"{name}.cif"
        tmol_path = DUD_DIR / target / f"{name}.tmol"
        in_pdb = DUD_DIR / target / f"{name}_in.pdb"

        return {
            "name": name,
            "target": target,
            "cif_path": cif_path,
            "tmol_path": tmol_path,
            "in_pdb": in_pdb,
        }

    @staticmethod
    def _score_against_rosetta(dud_scoring_data, param_db, torch_device):
        """Score the input pose with ``param_db`` and diff vs Rosetta ``.sc``."""
        import biotite.structure
        import biotite.structure.io

        from tmol.io.pose_stack_from_biotite import pose_stack_from_biotite
        from tmol.score import beta2016_score_function

        bt_struct = biotite.structure.io.load_structure(str(dud_scoring_data["in_pdb"]))
        if isinstance(bt_struct, biotite.structure.AtomArrayStack):
            bt_struct = bt_struct[0]

        pose_stack = pose_stack_from_biotite(
            bt_struct,
            torch_device,
            param_db=param_db,
        )

        sfxn = beta2016_score_function(torch_device, param_db=param_db)
        scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
        unweighted = scorer.unweighted_scores(pose_stack.coords)
        weights = sfxn.weights_tensor()

        score_types = sfxn.all_score_types()
        tmol_weighted = {
            st.name: float(weights[i]) * float(unweighted[i, 0])
            for i, st in enumerate(score_types)
        }

        # --- Rosetta reference scores from pre-generated .sc file ---
        sc_path = (
            DUD_DIR / dud_scoring_data["target"] / f"{dud_scoring_data['name']}.sc"
        )
        ros_scores = _rosetta_score(sc_path)

        mismatches = []
        for ros_term, tmol_term in _ROS_TERMS.items():
            ros_val = float(ros_scores.get(ros_term, 0.0))
            tmol_val = tmol_weighted.get(tmol_term, 0.0)
            if abs(tmol_val - ros_val) >= 1e-3:
                mismatches.append(
                    f"  {ros_term} (tmol {tmol_term}): "
                    f"tmol={tmol_val:.4f}, ros={ros_val:.4f}, "
                    f"diff={tmol_val - ros_val:+.4f}"
                )
        assert not mismatches, (
            f"Per-term score mismatch (>1e-3) for {dud_scoring_data['name']}:\n"
            + "\n".join(mismatches)
        )

    def test_score_tmol(self, dud_scoring_data, torch_device):
        """Parameters read from the golden ``.tmol`` file (Rosetta reference)."""
        from tmol.ligand.params_file import load_params_file

        preps = load_params_file(dud_scoring_data["tmol_path"])
        assert (
            len(preps) == 1
        ), f"{dud_scoring_data['tmol_path']}: expected one residue, got {len(preps)}"
        param_db = _param_db_with_ligand_prep(preps[0])

        self._score_against_rosetta(dud_scoring_data, param_db, torch_device)
