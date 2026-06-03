"""Protein-ligand interface scoring tests.

For each ``*_complex*.pdb`` in ``protein_ligand_test``, prepare the ligand via
``cif_inputs/<target>.ligand.cif`` (Dimorphite @ pH 7.4, ``charge_mode="mmff94"``),
inject params, score with tmol, and compare to Rosetta ``<stem>.sc``.
"""

from pathlib import Path

import pytest
import torch

from tmol.score.score_utils import calculate_block_pair_ddg

from tmol.tests.ligand.test_dud_ligands import (
    _param_db_with_ligand_prep,
    _rosetta_score,
)
from tmol.tests.ligand.test_pli_tmol_equivalence import (
    prepare_pli_ligand_from_cif,
)

PLI_DIR = Path(__file__).parent.parent / "data" / "protein_ligand_test"
PLI_CASES = [
    "ace_complex_nometals.pdb",
    "ache_complex.pdb",
    "ada_complex_nometals.pdb",
    "ampc_complex.pdb",
    "ar_complex.pdb",
    "cdk2_complex.pdb",
    "cox1_complex.pdb",
    "cox2_complex.pdb",
    "egfr_complex.pdb",
    "er_agonist_complex.pdb",
    "er_antagonist_complex.pdb",
    "fgfr1_complex.pdb",
    "fxa_complex.pdb",
    "gr_complex.pdb",
    "hivrt_complex.pdb",
    "hmga_complex.pdb",
    "hsp90_complex.pdb",
    "mr_complex.pdb",
    "na_complex.pdb",
    "p38_complex.pdb",
    "parp_complex.pdb",
    "pde5_complex_nometals.pdb",
    "pdgfrb_complex.pdb",
    "pr_complex.pdb",
    "rxr_complex.pdb",
    "src_complex.pdb",
    "tk_complex.pdb",
    "trypsin_complex.pdb",
    "vegfr2_complex.pdb",
]

# Pack+minimize is expensive and only run on the first few cases
PLI_MINPACK_CASES = PLI_CASES[:4]


# Per-row mapping: (label, rosetta_terms, tmol_terms). Each side is summed
# and compared. The Rosetta values in score.sc are already weighted; tmol
# values are weighted via sfxn.weights_tensor() before aggregation.
_PLI_TERM_ROWS = [
    ("fa_ljatr", ("fa_atr", "fa_intra_atr_xover4"), ("fa_ljatr",)),
    ("fa_ljrep", ("fa_rep", "fa_intra_rep_xover4"), ("fa_ljrep",)),
    ("fa_lk", ("fa_sol", "fa_intra_sol_xover4"), ("fa_lk",)),
    ("fa_elec", ("fa_elec", "fa_intra_elec"), ("fa_elec",)),
    ("lk_ball_iso", ("lk_ball_iso",), ("lk_ball_iso",)),
    ("lk_ball", ("lk_ball",), ("lk_ball",)),
    ("lk_bridge", ("lk_ball_bridge",), ("lk_bridge",)),
    ("lk_bridge_uncpl", ("lk_ball_bridge_uncpl",), ("lk_bridge_uncpl",)),
    ("hbond", ("hbond_sr_bb", "hbond_lr_bb", "hbond_bb_sc", "hbond_sc"), ("hbond",)),
    ("dunbrack_rot", ("fa_dun_rot",), ("dunbrack_rot",)),
    ("dunbrack_rotdev", ("fa_dun_dev",), ("dunbrack_rotdev",)),
    ("dunbrack_semirot", ("fa_dun_semi",), ("dunbrack_semirot",)),
    ("rama", ("rama_prepro", "p_aa_pp"), ("rama",)),
    ("omega", ("omega",), ("omega",)),
    ("disulfide", ("dslf_fa13",), ("disulfide",)),
    ("ref", ("ref",), ("ref",)),
    ("gen_bonded", ("gen_bonded",), ("gen_torsions",)),
]


# Targets whose ligand contains an sp2 acceptor.
# tmol and rosetta differ in B0 definitions:
#   - tmol uses the base's first bonded heavy neighbor
#   - Rosetta uses the base's ICOOR parent
# Loosen tolerance in these cases for hbond geom dependant terms
_SP2_ACCEPTOR_TARGETS = {
    "ace",
    "ache",
    "ada",
    "ampc",
    "ar",
    "cox1",
    "cox2",
    "fgfr1",
    "fxa",
    "gr",
    "hivrt",
    "hmga",
    "mr",
    "na",
    "p38",
    "parp",
    "pde5",
    "pdgfrb",
    "pr",
    "rxr",
    "src",
    "tk",
    "trypsin",
    "vegfr2",
}
_SP2_RELAXED_TERMS = {"lk_ball", "lk_bridge", "lk_bridge_uncpl", "hbond"}


def _target_for_complex(pdb_name: str) -> str:
    stem = pdb_name[: -len(".pdb")]
    for suf in ("_complex_nometals", "_complex"):
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def _pli_param_db_from_pipeline(target: str):
    prep = prepare_pli_ligand_from_cif(target)
    return _param_db_with_ligand_prep(prep)


class TestPLIScoring:
    """Score each protein-ligand complex using the CIF MMFF94 pipeline."""

    @pytest.fixture(params=PLI_CASES)
    def pli_pdb(self, request):
        return PLI_DIR / request.param

    @pytest.fixture(params=PLI_MINPACK_CASES)
    def tmol_minpack_pdb(self, request):
        return PLI_DIR / request.param

    @staticmethod
    def _write_scored_cif(pose_stack, canonical_ordering, out_name):
        """Write ``pose_stack`` to a CIF file named ``out_name`` in the cwd."""
        import biotite.structure
        from biotite.structure.io.pdbx import CIFFile, set_structure

        from tmol.io.pose_stack_from_biotite import biotite_from_pose_stack

        out = biotite_from_pose_stack(pose_stack, co=canonical_ordering)
        if isinstance(out, biotite.structure.AtomArrayStack):
            out = out[0]
        cif = CIFFile()
        set_structure(cif, out)
        out_path = PLI_DIR / out_name
        cif.write(str(out_path))
        print(f"  wrote structure -> {out_path}")

    def _dg_vs_rosetta(
        self,
        pli_pdb,
        param_db,
        torch_device_gpu,
        label_prefix="",
        threshold=None,
        threshold_sp2_acc=None,
    ):
        """Score the raw complex and compare block-pair dG with Rosetta.

        Args:
            pli_pdb: Path to the PDB file.
            param_db: Parameter database with ligand params injected.
            torch_device_gpu: Torch device.
            label_prefix: Label prefix for output naming (e.g. ".tmol", ".cif", ".mol2").
            threshold: If not None, fail the test when any term's absolute delta
                between tmol and Rosetta exceeds this value.
        """
        import biotite.structure
        import biotite.structure.io

        from tmol.io.pose_stack_from_biotite import pose_stack_from_biotite
        from tmol.score import beta2016_score_function

        target = _target_for_complex(pli_pdb.name)
        sc_path = PLI_DIR / f"{pli_pdb.stem}.sc"

        print(f"\n=== {pli_pdb.name}  (target={target}) [{label_prefix}/nomin] ===")

        bt_struct = biotite.structure.io.load_structure(str(pli_pdb))
        if isinstance(bt_struct, biotite.structure.AtomArrayStack):
            bt_struct = bt_struct[0]

        pose_stack, context = pose_stack_from_biotite(
            bt_struct,
            torch_device_gpu,
            param_db=param_db,
            prepare_ligands=False,
            no_optH=True,
            return_context=True,
        )

        sfxn = beta2016_score_function(torch_device_gpu, param_db=param_db)
        score_types = sfxn.all_score_types()

        mask = torch.zeros(
            (1, pose_stack.max_n_blocks),
            dtype=torch.bool,
            device=torch_device_gpu,
        )
        mask[0][-1] = True

        ddg, scored_pose_stack = calculate_block_pair_ddg(
            pose_stack,
            mask,
            sfxn=sfxn,
            sum_terms=False,
            minimize=False,
            pack=False,
            database=param_db,
            return_pose_stack=True,
        )

        # Dump the scored structure, keyed by example id: e.g. "ace.tmol.nomin.cif".
        self._write_scored_cif(
            scored_pose_stack,
            context.canonical_ordering,
            f"{target}{label_prefix}.nomin.cif",
        )

        tmol_scores = {
            key.name: val.item() for key, val in zip(score_types, ddg.squeeze(0))
        }

        ros_scores = _rosetta_score(sc_path) if sc_path.exists() else {}

        data = []
        failing_terms = []
        if not sc_path.exists():
            print(f"  (no Rosetta .sc at {sc_path.name} -- run run_rosetta_score.sh)")

        # sp2 acceptors have a different B0 resolution strategy in tmol than in Rosetta
        # by design.  Relax the tolerance for these cases
        relax_sp2 = target in _SP2_ACCEPTOR_TARGETS

        def _term_threshold(label):
            if threshold is None:
                return None
            if (
                threshold_sp2_acc is not None
                and relax_sp2
                and label in _SP2_RELAXED_TERMS
            ):
                return threshold_sp2_acc
            return threshold

        print(f"  {'term':<18} {'tmol':>12} {'rosetta':>12} {'diff':>12}")
        for label, ros_terms, tmol_terms in _PLI_TERM_ROWS:
            t = sum(tmol_scores.get(n, 0.0) for n in tmol_terms)
            rosetta = sum(float(ros_scores.get("dG_" + n, 0.0)) for n in ros_terms)
            delta = abs(t - rosetta)
            term_thresh = _term_threshold(label)
            # With a tolerance set print the offending rows only
            if term_thresh is None or delta > term_thresh:
                print(f"  {label:<18} {t:12.4f} {rosetta:12.4f} {delta:+12.4f}")
            data += [
                (
                    target,
                    label_prefix,
                    "nomin",
                    label,
                    t,
                    rosetta,
                    delta,
                )
            ]
            if term_thresh is not None and delta > term_thresh:
                failing_terms.append((label, t, rosetta, delta))

        if failing_terms:
            msg_lines = [
                f"{len(failing_terms)} term(s) exceeded the delta threshold "
                f"(threshold={threshold}):"
            ]
            msg_lines.append(
                f"  {'term':<18} {'tmol':>12} {'rosetta':>12} {'delta':>12}"
            )
            for label, t, rosetta, delta in failing_terms:
                msg_lines.append(
                    f"  {label:<18} {t:12.4f} {rosetta:12.4f} {delta:+12.4f}"
                )
            pytest.fail("\n".join(msg_lines))

    def test_compare_dg_score_with_rosetta_tmol(self, pli_pdb, torch_device_gpu):
        """Compare Rosetta dG scores with tmol scores using .tmol file params."""
        from tmol.database import ParameterDatabase
        from tmol.ligand.params_file import inject_params_file

        target = _target_for_complex(pli_pdb.name)
        tmol_path = PLI_DIR / f"{target}.xtal-lig.mmff94.tmol"

        param_db = inject_params_file(ParameterDatabase.get_default(), tmol_path)

        self._dg_vs_rosetta(
            pli_pdb,
            param_db,
            torch_device_gpu,
            label_prefix=".tmol",
            threshold=1e-2,
            threshold_sp2_acc=0.2,
        )

    def test_compare_dg_score_with_rosetta_cif(self, pli_pdb, torch_device_gpu):
        """Compare Rosetta dG scores with tmol scores using .cif file params."""
        from tmol.database import ParameterDatabase
        from tmol.ligand import prepare_ligand_from_cif

        target = _target_for_complex(pli_pdb.name)
        ligand_cif = PLI_DIR / "cif_inputs" / f"{target}.ligand.cif"

        extended_db, _ = prepare_ligand_from_cif(
            str(ligand_cif),
            param_db=ParameterDatabase.get_default(),
            ph=7.4,
        )

        self._dg_vs_rosetta(
            pli_pdb,
            extended_db,
            torch_device_gpu,
            label_prefix=".cif",
            threshold=1e-2,
            threshold_sp2_acc=0.2,
        )

    def test_compare_dg_score_with_rosetta_mol2(self, pli_pdb, torch_device_gpu):
        """Compare Rosetta dG scores with tmol scores using .mol2 file params."""
        from tmol.database import ParameterDatabase
        from tmol.ligand import prepare_ligand_from_mol2

        target = _target_for_complex(pli_pdb.name)
        ligand_mol2 = PLI_DIR / f"{target}.lig.mol2"

        extended_db, _ = prepare_ligand_from_mol2(
            str(ligand_mol2),
            param_db=ParameterDatabase.get_default(),
            ph=7.4,
        )

        self._dg_vs_rosetta(
            pli_pdb,
            extended_db,
            torch_device_gpu,
            label_prefix=".mol2",
            threshold=1e-2,
            threshold_sp2_acc=0.2,
        )

    def _packmin_lowers_energy(
        self, pli_pdb, param_db, torch_device_gpu, label_prefix=""
    ):
        """Pack + minimize the complex and assert the total energy drops.

        No reference comparison: pack+min must not raise the total weighted
        whole-pose energy. The packed/minimized structure is dumped as
        ``<target><label_prefix>.packmin.cif``.
        """
        import biotite.structure
        import biotite.structure.io

        from tmol.io.pose_stack_from_biotite import pose_stack_from_biotite
        from tmol.score import beta2016_score_function

        target = _target_for_complex(pli_pdb.name)
        print(f"\n=== {pli_pdb.name}  (target={target}) [{label_prefix}/packmin] ===")

        bt_struct = biotite.structure.io.load_structure(str(pli_pdb))
        if isinstance(bt_struct, biotite.structure.AtomArrayStack):
            bt_struct = bt_struct[0]

        pose_stack, context = pose_stack_from_biotite(
            bt_struct,
            torch_device_gpu,
            param_db=param_db,
            prepare_ligands=False,
            no_optH=False,
            return_context=True,
        )

        sfxn = beta2016_score_function(torch_device_gpu, param_db=param_db)

        def total_weighted_energy(ps):
            scorer = sfxn.render_whole_pose_scoring_module(ps)
            unweighted = scorer.unweighted_scores(ps.coords)
            weights = sfxn.weights_tensor()
            return sum(
                float(weights[i]) * float(unweighted[i, 0]) for i in range(len(weights))
            )

        total_before = total_weighted_energy(pose_stack)

        mask = torch.zeros(
            (1, pose_stack.max_n_blocks),
            dtype=torch.bool,
            device=torch_device_gpu,
        )
        mask[0][-1] = True

        _, scored_pose_stack = calculate_block_pair_ddg(
            pose_stack,
            mask,
            sfxn=sfxn,
            sum_terms=False,
            minimize=True,
            pack=True,
            database=param_db,
            return_pose_stack=True,
        )

        total_after = total_weighted_energy(scored_pose_stack)

        self._write_scored_cif(
            scored_pose_stack,
            context.canonical_ordering,
            f"{target}{label_prefix}.packmin.cif",
        )

        print(f"  total energy before packmin: {total_before:12.4f}")
        print(f"  total energy after  packmin: {total_after:12.4f}")
        print(f"  delta:                       {total_after - total_before:+12.4f}")

        assert total_after < total_before, (
            f"pack+minimize did not lower the total energy for {target}: "
            f"before={total_before:.4f} after={total_after:.4f}"
        )

    def test_packmin_lowers_energy_tmol(self, tmol_minpack_pdb, torch_device_gpu):
        """Pack+minimize must lower the total energy (tmol params, first 4 cases)."""
        from tmol.database import ParameterDatabase
        from tmol.ligand.params_file import inject_params_file

        target = _target_for_complex(tmol_minpack_pdb.name)
        tmol_path = PLI_DIR / f"{target}.xtal-lig.mmff94.tmol"

        param_db = inject_params_file(ParameterDatabase.get_default(), tmol_path)

        self._packmin_lowers_energy(
            tmol_minpack_pdb, param_db, torch_device_gpu, label_prefix=".tmol"
        )
