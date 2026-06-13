"""Protein-ligand interface scoring tests.

For each ``*_complex*.pdb`` in ``protein_ligand_test``, prepare the ligand via
``cif_inputs/<target>.ligand.cif`` (Dimorphite @ pH 7.4, ``charge_mode="mmff94"``),
inject params, score with tmol, and compare to Rosetta ``<stem>.sc``.
"""

from pathlib import Path

import pytest
import torch

from tmol.score.score_utils import calculate_block_pair_ddg

from tmol.tests.ligand.test_dud_ligands import _rosetta_score

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


class TestPLIScoring:
    """Score each protein-ligand complex using the CIF MMFF94 pipeline."""

    @pytest.fixture(params=PLI_CASES)
    def pli_pdb(self, request):
        return PLI_DIR / request.param

    @staticmethod
    def _write_scored_cif(pose_stack, canonical_ordering, out_name: str) -> None:
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
        torch_device,
        label_prefix: str = "",
        threshold=None,
        threshold_sp2_acc=None,
    ) -> None:
        """Score the raw complex and compare block-pair dG with Rosetta.

        Args:
            pli_pdb: Path to the PDB file.
            param_db: Parameter database with ligand params injected.
            torch_device: Torch device.
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
            torch_device,
            param_db=param_db,
            prepare_ligands=False,
            no_optH=True,
            return_context=True,
        )

        sfxn = beta2016_score_function(torch_device, param_db=param_db)
        score_types = sfxn.all_score_types()

        mask = torch.zeros(
            (1, pose_stack.max_n_blocks),
            dtype=torch.bool,
            device=torch_device,
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

    def test_compare_dg_score_with_rosetta_tmol(self, pli_pdb, torch_device) -> None:
        """Compare Rosetta dG scores with tmol scores using .tmol file params."""
        from tmol.database import ParameterDatabase
        from tmol.ligand.params_file import inject_params_file

        target = _target_for_complex(pli_pdb.name)
        tmol_path = PLI_DIR / f"{target}.xtal-lig.mmff94.tmol"

        param_db = inject_params_file(ParameterDatabase.get_default(), tmol_path)

        self._dg_vs_rosetta(
            pli_pdb,
            param_db,
            torch_device,
            label_prefix=".tmol",
            threshold=1e-2,
            threshold_sp2_acc=0.2,
        )

    # Two representative targets whose ligand parameters are fully regenerated
    # from the .lig.mol2 entry point -- including a from-scratch MMFF94 charge
    # recompute -- and reproduce the golden .tmol exactly.
    @pytest.mark.parametrize("target", ["cox1", "hmga"])
    def test_pipeline_params_match_golden_tmol(self, target) -> None:
        """Generated-param parity: mol2 pipeline output vs the golden ``.tmol``.

        The ``test_compare_dg_score_with_rosetta_tmol`` cases inject the frozen
        golden ``.tmol`` and check *scoring*; this instead exercises the
        *parameter-generation* pipeline that produced those goldens, in the same
        spirit as the DUD-80 ``TestGroundTruthRegression``. It prepares a
        ``LigandPreparation`` straight from ``<target>.lig.mol2`` (the mol2 entry
        point that backs ``prepare_ligand_from_mol2``) with
        ``charge_mode="mmff94"`` so the MMFF94 partial charges are *recomputed*
        from scratch rather than copied from the input -- the charges, atom
        types, and bond graph are therefore all generated by tmol. It then
        compares the result against the checked-in golden
        ``<target>.xtal-lig.mmff94.tmol``. The comparison is graph-isomorphic, so
        atom-name differences are tolerated, and asserts the atom set, tmol atom
        types, bond graph, and partial charges (within 0.05) all match -- i.e.
        the live pipeline still reproduces the reference parameters that the
        Rosetta dG goldens are scored from.
        """
        from tmol.ligand.detect import nonstandard_residue_info_from_mol2
        from tmol.ligand.preparation import prepare_single_ligand
        from tmol.ligand.params_file import load_params_file
        from tmol.ligand.equivalence import compare_ligand_preparations

        ligand_mol2 = PLI_DIR / f"{target}.lig.mol2"
        golden_tmol = PLI_DIR / f"{target}.xtal-lig.mmff94.tmol"

        lig = nonstandard_residue_info_from_mol2(str(ligand_mol2), "LG1")
        generated = prepare_single_ligand(lig, ph=7.4, charge_mode="mmff94")
        reference = load_params_file(golden_tmol)[0]

        result = compare_ligand_preparations(
            generated, reference, charge_tolerance=0.05
        )
        assert result.is_equivalent, (
            f"{target}: pipeline params diverge from golden .tmol on "
            + ", ".join(k for k, ok in result.checks.items() if not ok)
            + f"\ndetails={result.details}"
        )
