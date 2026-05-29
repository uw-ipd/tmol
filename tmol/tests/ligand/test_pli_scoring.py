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
    prepare_pli_ligand_mmff94_from_cif,
)

PLI_DIR = Path(__file__).parent.parent / "data" / "protein_ligand_test"
PLI_CASES = sorted(p.name for p in PLI_DIR.glob("*_complex*.pdb"))


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


def _target_for_complex(pdb_name: str) -> str:
    stem = pdb_name[: -len(".pdb")]
    for suf in ("_complex_nometals", "_complex"):
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def _pli_param_db_from_pipeline(target: str):
    prep = prepare_pli_ligand_mmff94_from_cif(target)
    return _param_db_with_ligand_prep(prep)


class TestPLIScoring:
    """Score each protein-ligand complex using the CIF MMFF94 pipeline."""

    # Per-fixture expected-failure lists
    _XFAIL_TMOL = ["ace"]
    _XFAIL_CIF = [
        "ache",
        "ada",
        "ampc",
        "ar",
        "cdk2",
        "er",
        "fgfr1",
        "fxa",
        "hivrt",
        "hmga",
        "mr",
        "na",
        "p38",
        "parp",
        "pde5",
        "pdgfrb",
        "pr",
        "src",
        "tk",
        "vegfr2",
    ]
    _XFAIL_MOL2 = [
        "ache",
        "ada",
        "ampc",
        "ar",
        "cdk2",
        "er",
        "fgfr1",
        "fxa",
        "hivrt",
        "hmga",
        "mr",
        "na",
        "p38",
        "parp",
        "pde5",
        "pdgfrb",
        "pr",
        "src",
        "tk",
        "trypsin",
        "vegfr2",
    ]

    @staticmethod
    def _xfail_pdb(request, xfail_substrings, label):
        """If request.param contains any substring from xfail_substrings, mark as xfail."""
        for sub in xfail_substrings:
            if request.param.startswith(sub):
                request.applymarker(
                    pytest.mark.xfail(
                        reason=f"Expected failure for PDB '{sub}' ({label})"
                    )
                )
                return

    @pytest.fixture(params=PLI_CASES)
    def tmol_pli_pdb(self, request):
        self._xfail_pdb(request, self._XFAIL_TMOL, "tmol")
        return PLI_DIR / request.param

    @pytest.fixture(params=PLI_CASES)
    def cif_pli_pdb(self, request):
        self._xfail_pdb(request, self._XFAIL_CIF, "cif")
        return PLI_DIR / request.param

    @pytest.fixture(params=PLI_CASES)
    def mol2_pli_pdb(self, request):
        self._xfail_pdb(request, self._XFAIL_MOL2, "mol2")
        return PLI_DIR / request.param

    @pytest.fixture(params=[False, True], ids=["nomin", "packmin"])
    def packmin(self, request):
        return request.param

    def test_score(self, pli_pdb, torch_device):
        import biotite.structure
        import biotite.structure.io

        from tmol.io.pose_stack_from_biotite import pose_stack_from_biotite
        from tmol.score import beta2016_score_function

        target = _target_for_complex(pli_pdb.name)
        sc_path = PLI_DIR / f"{pli_pdb.stem}.sc"
        param_db = _pli_param_db_from_pipeline(target)

        bt_struct = biotite.structure.io.load_structure(str(pli_pdb))
        if isinstance(bt_struct, biotite.structure.AtomArrayStack):
            bt_struct = bt_struct[0]

        pose_stack = pose_stack_from_biotite(
            bt_struct, torch_device, param_db=param_db, prepare_ligands=False
        )

        sfxn = beta2016_score_function(torch_device, param_db=param_db)
        scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
        unweighted = scorer.unweighted_scores(pose_stack.coords)
        weights = sfxn.weights_tensor()
        score_types = sfxn.all_score_types()
        tmol_weighted: dict[str, float] = {}
        for i, st in enumerate(score_types):
            tmol_weighted[st.name] = tmol_weighted.get(st.name, 0.0) + float(
                weights[i]
            ) * float(unweighted[i, 0])

        ros_scores = _rosetta_score(sc_path) if sc_path.exists() else {}

        tmol_total = sum(tmol_weighted.values())
        ros_total = float(ros_scores.get("total_score", 0.0))

        print(f"\n=== {pli_pdb.name}  (target={target}) ===")
        if not sc_path.exists():
            print(f"  (no Rosetta .sc at {sc_path.name} -- run run_rosetta_score.sh)")
        print(f"  {'term':<18} {'tmol':>12} {'rosetta':>12} {'diff':>12}")
        for label, ros_terms, tmol_terms in _PLI_TERM_ROWS:
            t = sum(tmol_weighted.get(n, 0.0) for n in tmol_terms)
            r = sum(float(ros_scores.get(n, 0.0)) for n in ros_terms)
            print(f"  {label:<18} {t:12.4f} {r:12.4f} {t - r:+12.4f}")
        print(
            f"  {'TOTAL':<18} {tmol_total:12.4f} {ros_total:12.4f} "
            f"{tmol_total - ros_total:+12.4f}"
        )

    def _dg_vs_rosetta(
        self,
        pli_pdb,
        param_db,
        torch_device,
        label_prefix="",
        packmin=False,
        threshold=None,
    ):
        """Score the complex and compare block-pair dG with Rosetta.

        Args:
            pli_pdb: Path to the PDB file.
            param_db: Parameter database with ligand params injected.
            torch_device: Torch device.
            label_prefix: Label prefix for output naming (e.g. ".tmol", ".cif", ".mol2").
            packmin: If True, compute scores after minimization+packing; otherwise raw.
            threshold: If not None, fail the test when any term's absolute delta
                between tmol and Rosetta exceeds this value.
        """
        import biotite.structure
        import biotite.structure.io

        from tmol.io.pose_stack_from_biotite import pose_stack_from_biotite
        from tmol.score import beta2016_score_function

        target = _target_for_complex(pli_pdb.name)
        sc_path = PLI_DIR / f"{pli_pdb.stem}.sc"

        min_label = "packmin" if packmin else "nomin"
        print(
            f"\n=== {pli_pdb.name}  (target={target}) [{label_prefix}/{min_label}] ==="
        )

        bt_struct = biotite.structure.io.load_structure(str(pli_pdb))
        if isinstance(bt_struct, biotite.structure.AtomArrayStack):
            bt_struct = bt_struct[0]

        pose_stack = pose_stack_from_biotite(
            bt_struct, torch_device, param_db=param_db, prepare_ligands=False
        )

        sfxn = beta2016_score_function(torch_device, param_db=param_db)
        score_types = sfxn.all_score_types()

        mask = torch.zeros(
            (1, pose_stack.max_n_blocks),
            dtype=torch.bool,
            device=torch_device,
        )
        mask[0][-1] = True

        ddg = calculate_block_pair_ddg(
            pose_stack,
            mask,
            sfxn=sfxn,
            sum_terms=False,
            minimize=packmin,
            pack=packmin,
            database=param_db,
        )

        tmol_scores = {
            key.name: val.item() for key, val in zip(score_types, ddg.squeeze(0))
        }

        ros_scores = _rosetta_score(sc_path) if sc_path.exists() else {}

        data = []
        failing_terms = []
        if not sc_path.exists():
            print(f"  (no Rosetta .sc at {sc_path.name} -- run run_rosetta_score.sh)")
        print(f"  {'term':<18} {'tmol':>12} {'rosetta':>12} {'diff':>12}")
        for label, ros_terms, tmol_terms in _PLI_TERM_ROWS:
            t = sum(tmol_scores.get(n, 0.0) for n in tmol_terms)
            rosetta = sum(float(ros_scores.get("dG_" + n, 0.0)) for n in ros_terms)
            delta = abs(t - rosetta)
            print(f"  {label:<18} {t:12.4f} {rosetta:12.4f} {delta:+12.4f}")
            data += [
                (
                    target,
                    label_prefix,
                    min_label,
                    label,
                    t,
                    rosetta,
                    delta,
                )
            ]
            if threshold is not None and delta > threshold:
                failing_terms.append((label, t, rosetta, delta))

        # import csv
        # with open(f"{target}{label_prefix}.{min_label}.table", "w") as f:
        # writer = csv.writer(
        # f, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        # )
        # for row in data:
        # writer.writerow(row)

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

    def test_compare_dg_score_with_rosetta_tmol(
        self, tmol_pli_pdb, packmin, torch_device
    ):
        """Compare Rosetta dG scores with tmol scores using .tmol file params."""
        from tmol.database import ParameterDatabase
        from tmol.ligand.params_file import inject_params_file

        target = _target_for_complex(tmol_pli_pdb.name)
        tmol_path = PLI_DIR / f"{target}.xtal-lig.mmff94.tmol"

        param_db = inject_params_file(ParameterDatabase.get_default(), tmol_path)

        self._dg_vs_rosetta(
            tmol_pli_pdb,
            param_db,
            torch_device,
            label_prefix=".tmol",
            packmin=packmin,
            threshold=1e-2,
        )

    def test_compare_dg_score_with_rosetta_cif(
        self, cif_pli_pdb, packmin, torch_device
    ):
        """Compare Rosetta dG scores with tmol scores using .cif file params."""
        from tmol.database import ParameterDatabase
        from tmol.ligand import prepare_ligand_from_cif

        target = _target_for_complex(cif_pli_pdb.name)
        ligand_cif = PLI_DIR / "cif_inputs" / f"{target}.ligand.cif"

        extended_db, _ = prepare_ligand_from_cif(
            str(ligand_cif),
            param_db=ParameterDatabase.get_default(),
            ph=7.4,
        )

        self._dg_vs_rosetta(
            cif_pli_pdb, extended_db, torch_device, label_prefix=".cif", packmin=packmin
        )

    def test_compare_dg_score_with_rosetta_mol2(
        self, mol2_pli_pdb, packmin, torch_device
    ):
        """Compare Rosetta dG scores with tmol scores using .mol2 file params."""
        from tmol.database import ParameterDatabase
        from tmol.ligand import prepare_ligand_from_mol2

        target = _target_for_complex(mol2_pli_pdb.name)
        ligand_mol2 = PLI_DIR / f"{target}.lig.mol2"

        extended_db, _ = prepare_ligand_from_mol2(
            str(ligand_mol2),
            param_db=ParameterDatabase.get_default(),
            ph=7.4,
        )

        self._dg_vs_rosetta(
            mol2_pli_pdb,
            extended_db,
            torch_device,
            label_prefix=".mol2",
            packmin=packmin,
        )
