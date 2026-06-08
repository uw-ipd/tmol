"""Regression tests for DUD benchmark ligands.

1. ``TestDudCifGoldenEquivalence`` — CIF input, Dimorphite @ pH 7.4,
   ``charge_mode="mmff94"``, full equivalence vs golden ``.tmol``.

2. ``TestDudCifSkipProtonationGoldenEquivalence`` — CIF input protonation
   (explicit H, no Dimorphite) + ``charge_mode="mmff94"`` vs golden ``.tmol``.

3. ``TestDUDScoring`` — same CIF pipeline as (1), score vs Rosetta ``.sc``.
"""

from pathlib import Path

import attr

import numpy as np
import pytest
import torch  # noqa: F401  (used via torch_device_gpu fixture)

from tmol.ligand.equivalence import compare_ligand_preparations

DUD_DIR = Path(__file__).parent.parent / "data" / "dud_ligands"
CHARGE_TOLERANCE = 0.05
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


# ---------------------------------------------------------------------------
# CIF -> NonStandardResidueInfo
# ---------------------------------------------------------------------------


def _cif_value_order_to_biotite_bond_type(value_order: str, aromatic_flag: str) -> int:
    import biotite.structure as struc

    order = str(value_order).strip().upper()
    is_aromatic = str(aromatic_flag).strip().upper() == "Y"

    if is_aromatic:
        if order == "SING":
            return int(struc.BondType.AROMATIC_SINGLE)
        if order == "DOUB":
            return int(struc.BondType.AROMATIC_DOUBLE)
        if order == "TRIP":
            return int(struc.BondType.AROMATIC_TRIPLE)
        return int(struc.BondType.AROMATIC)

    if order == "SING":
        return int(struc.BondType.SINGLE)
    if order == "DOUB":
        return int(struc.BondType.DOUBLE)
    if order == "TRIP":
        return int(struc.BondType.TRIPLE)
    if order == "AROM":
        return int(struc.BondType.AROMATIC)
    return int(struc.BondType.SINGLE)


def _cif_to_nonstandard_residue_info(cif_path: Path, res_name: str):
    """Build a ``NonStandardResidueInfo`` from a DUD CIF.

    Reads atoms + bonds via biotite's pdbx round-trip and pulls the
    AM1-BCC partial charges out of the custom ``_atom_site.partial_charge``
    column we wrote alongside the standard fields. The result has the
    same shape as what ``detect_nonstandard_residues`` would emit for
    an unknown residue code, so ``prepare_single_ligand`` can consume
    it without further setup.
    """
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx

    from tmol.ligand.detect import NonStandardResidueInfo

    cif = pdbx.CIFFile.read(str(cif_path))
    # ``extra_fields=['charge']`` pulls in the integer formal charges
    # written by our mol2-to-cif step; without them RDKit cannot
    # perceive aromaticity for charged-N rings on the round-trip.
    arr = pdbx.get_structure(cif, model=1, include_bonds=True, extra_fields=["charge"])
    if isinstance(arr, struc.AtomArrayStack):
        arr = arr[0]

    arr.res_name = np.array([res_name] * len(arr), dtype=arr.res_name.dtype)

    atom_site = cif.block["atom_site"]
    atom_names = list(atom_site["label_atom_id"].as_array())
    partial_charges = {
        name: float(q)
        for name, q in zip(atom_names, atom_site["partial_charge"].as_array(float))
    }
    # Custom per-atom aromatic flag (Y/N) carried by the CIF — bonds are
    # stored as plain Kekulé SINGLE/DOUBLE so this column is the only
    # signal that distinguishes a benzene ring from an isolated C=C.
    aromatic_chars = atom_site["tmol_aromatic"].as_array()
    arr.set_annotation(
        "tmol_aromatic",
        np.array([str(v) == "Y" for v in aromatic_chars], dtype=bool),
    )
    # Source mol2 atom-type subtype (``ar``, ``2``, ``cat``, ``3``,
    # ``pl3`` …). Drives the CR-vs-CD carbon-typing decision: aromatic
    # carbons tagged ``ar`` get CR, ``2``/``cat`` get CD/CD1, etc.
    if "tmol_source_subtype" in atom_site:
        arr.set_annotation(
            "tmol_source_subtype",
            np.array([str(v) for v in atom_site["tmol_source_subtype"].as_array()]),
        )

    # Rebuild bonds from _chem_comp_bond to avoid parser fallback bond code 0.
    if "chem_comp_bond" in cif.block:
        bond_site = cif.block["chem_comp_bond"]
        atom_id_1 = [str(v) for v in bond_site["atom_id_1"].as_array()]
        atom_id_2 = [str(v) for v in bond_site["atom_id_2"].as_array()]
        value_order = [str(v) for v in bond_site["value_order"].as_array()]
        if "pdbx_aromatic_flag" in bond_site:
            aromatic_flags = [
                str(v) for v in bond_site["pdbx_aromatic_flag"].as_array()
            ]
        else:
            aromatic_flags = ["N"] * len(value_order)

        name_to_idx = {name: i for i, name in enumerate(atom_names)}
        bonds = []
        for a1, a2, order, aromatic_flag in zip(
            atom_id_1, atom_id_2, value_order, aromatic_flags
        ):
            if a1 not in name_to_idx or a2 not in name_to_idx:
                continue
            bonds.append(
                (
                    name_to_idx[a1],
                    name_to_idx[a2],
                    _cif_value_order_to_biotite_bond_type(order, aromatic_flag),
                )
            )
        if bonds:
            arr.bonds = struc.BondList(len(arr), np.asarray(bonds, dtype=np.int32))

    return NonStandardResidueInfo(
        res_name=res_name,
        ccd_type="UNKNOWN",
        atom_names=tuple(atom_names),
        elements=tuple(str(e) for e in arr.element),
        coords=arr.coord.copy(),
        atom_array=arr,
        ccd_smiles=None,
        covalently_linked=False,
        partial_charges=partial_charges,
    )


def _cif_input_for_golden_pipeline(cif_path: Path, res_name: str):
    """CIF geometry/bonds only; Dimorphite + MMFF run in ``prepare_single_ligand``."""
    info = _cif_to_nonstandard_residue_info(cif_path, res_name)
    return attr.evolve(info, partial_charges=None, skip_protonation=False)


def prepare_dud_ligand_mmff94_from_cif(cif_path: Path, res_name: str = "LG1"):
    """CIF → Dimorphite @ pH 7.4 → MMFF94 (shared by equivalence and scoring tests)."""
    from tmol.ligand import prepare_single_ligand

    info = _cif_input_for_golden_pipeline(cif_path, res_name)
    return prepare_single_ligand(info, ph=7.4, charge_mode="mmff94")


def prepare_dud_ligand_input_protonation_mmff94_from_cif(
    cif_path: Path, res_name: str = "LG1"
):
    """CIF mol2 protonation (explicit H) → MMFF94 charges, no Dimorphite."""
    from tmol.ligand import prepare_single_ligand

    info = _cif_to_nonstandard_residue_info(cif_path, res_name)
    info = attr.evolve(info, partial_charges=None, skip_protonation=True)
    return prepare_single_ligand(info, charge_mode="mmff94")


class TestSkipProtonationPreservesInput:
    def test_explicit_hydrogens_and_names(self):
        from tmol.ligand import prepare_single_ligand

        cif_path = DUD_DIR / "ada" / "ZINC01614355.cif"
        info = _cif_to_nonstandard_residue_info(cif_path, "LG1")
        info = attr.evolve(info, partial_charges=None, skip_protonation=True)

        prep = prepare_single_ligand(info, charge_mode="mmff94")
        out_names = {a.name for a in prep.residue_type.atoms}
        in_names = set(info.atom_names)

        assert len(out_names) == len(in_names)
        assert out_names == in_names
        assert "H19" in out_names
        assert len(prep.partial_charges) == len(in_names)


def _param_db_with_ligand_prep(prep):
    from tmol.database import ParameterDatabase
    from tmol.ligand.registry import inject_ligand_preparations

    return inject_ligand_preparations(ParameterDatabase.get_default(), [prep])


@pytest.mark.slow
class TestDudCifGoldenEquivalence:
    """CIF → Dimorphite @ pH 7.4 → MMFF94 → golden ``.tmol``."""

    @pytest.fixture(params=DUD_CASES, ids=[f"{t}_{n}" for t, n in DUD_CASES])
    def prep_pair(self, request):
        from tmol.ligand.params_file import load_params_file

        target, name = request.param
        cif_path = DUD_DIR / target / f"{name}.cif"
        tmol_path = DUD_DIR / target / f"{name}.tmol"

        prep_cif = prepare_dud_ligand_mmff94_from_cif(cif_path, "LG1")

        preps_tmol = load_params_file(tmol_path)
        assert (
            len(preps_tmol) == 1
        ), f"{tmol_path}: expected one residue, got {len(preps_tmol)}"
        prep_tmol = preps_tmol[0]

        equivalence = compare_ligand_preparations(
            prep_cif,
            prep_tmol,
            charge_tolerance=CHARGE_TOLERANCE,
        )
        return {"name": name, "equivalence": equivalence}

    @staticmethod
    def _format_check_error(prep_pair, check: str) -> str:
        details = prep_pair["equivalence"].details.get(check)
        return f"{check} mismatch for {prep_pair['name']} (CIF pipeline vs .tmol): {details}"

    def test_atom_set(self, prep_pair):
        assert prep_pair["equivalence"].checks["atom_set"], self._format_check_error(
            prep_pair, "atom_set"
        )

    def test_atom_types(self, prep_pair):
        assert prep_pair["equivalence"].checks["atom_types"], self._format_check_error(
            prep_pair, "atom_types"
        )

    def test_bonds(self, prep_pair):
        assert prep_pair["equivalence"].checks["bonds"], self._format_check_error(
            prep_pair, "bonds"
        )

    def test_partial_charges(self, prep_pair):
        assert prep_pair["equivalence"].checks[
            "partial_charges"
        ], self._format_check_error(prep_pair, "partial_charges")

    def test_cartbonded_params(self, prep_pair):
        assert prep_pair["equivalence"].checks[
            "cartbonded_params"
        ], self._format_check_error(prep_pair, "cartbonded_params")


@pytest.mark.slow
class TestDudCifSkipProtonationGoldenEquivalence:
    """CIF mol2 protonation (``skip_protonation``) + MMFF94 vs golden ``.tmol``.

    Topology tests (atom set, types, bonds, cartbonded) compare against the
    checked-in reference files. ``test_partial_charges`` compares MMFF94 to
    those references too; they still store mol2 AM1-BCC charges, so charge
    mismatches are expected unless you regenerate goldens with
    ``prepare_dud_ligand_input_protonation_mmff94_from_cif``.
    """

    @pytest.fixture(params=DUD_CASES, ids=[f"{t}_{n}" for t, n in DUD_CASES])
    def prep_pair(self, request):
        from tmol.ligand.params_file import load_params_file

        target, name = request.param
        cif_path = DUD_DIR / target / f"{name}.cif"
        tmol_path = DUD_DIR / target / f"{name}.tmol"

        prep_cif = prepare_dud_ligand_input_protonation_mmff94_from_cif(cif_path, "LG1")

        preps_tmol = load_params_file(tmol_path)
        assert (
            len(preps_tmol) == 1
        ), f"{tmol_path}: expected one residue, got {len(preps_tmol)}"
        prep_tmol = preps_tmol[0]

        equivalence = compare_ligand_preparations(
            prep_cif,
            prep_tmol,
            charge_tolerance=CHARGE_TOLERANCE,
        )
        return {"name": name, "equivalence": equivalence}

    @staticmethod
    def _format_check_error(prep_pair, check: str) -> str:
        details = prep_pair["equivalence"].details.get(check)
        return (
            f"{check} mismatch for {prep_pair['name']} "
            f"(skip-protonation CIF pipeline vs .tmol): {details}"
        )

    def test_atom_set(self, prep_pair):
        assert prep_pair["equivalence"].checks["atom_set"], self._format_check_error(
            prep_pair, "atom_set"
        )

    def test_atom_types(self, prep_pair):
        assert prep_pair["equivalence"].checks["atom_types"], self._format_check_error(
            prep_pair, "atom_types"
        )

    def test_bonds(self, prep_pair):
        assert prep_pair["equivalence"].checks["bonds"], self._format_check_error(
            prep_pair, "bonds"
        )

    def test_partial_charges(self, prep_pair):
        assert prep_pair["equivalence"].checks[
            "partial_charges"
        ], self._format_check_error(prep_pair, "partial_charges")

    def test_cartbonded_params(self, prep_pair):
        assert prep_pair["equivalence"].checks[
            "cartbonded_params"
        ], self._format_check_error(prep_pair, "cartbonded_params")


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
    """Score each DUD ligand and compare per-term against Rosetta ``.sc``.

    Two param-db sources are exercised:

    * ``test_score_injected`` — parameters read straight from the golden
      ``.tmol`` file via ``load_params_file`` (Rosetta-reference params).
    * ``test_score_generated`` — parameters generated from the input
      AtomArray through the CIF → Dimorphite → MMFF94 pipeline.
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
    def _score_against_rosetta(dud_scoring_data, param_db, torch_device_gpu):
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
            torch_device_gpu,
            param_db=param_db,
        )

        sfxn = beta2016_score_function(torch_device_gpu, param_db=param_db)
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

    def test_score_tmol(self, dud_scoring_data, torch_device_gpu):
        """Parameters read from the golden ``.tmol`` file (Rosetta reference)."""
        from tmol.ligand.params_file import load_params_file

        preps = load_params_file(dud_scoring_data["tmol_path"])
        assert (
            len(preps) == 1
        ), f"{dud_scoring_data['tmol_path']}: expected one residue, got {len(preps)}"
        param_db = _param_db_with_ligand_prep(preps[0])

        self._score_against_rosetta(dud_scoring_data, param_db, torch_device_gpu)

    def test_score_cif(self, dud_scoring_data, torch_device_gpu):
        """Parameters generated from the AtomArray via the CIF pipeline."""
        prep = prepare_dud_ligand_mmff94_from_cif(dud_scoring_data["cif_path"], "LG1")
        param_db = _param_db_with_ligand_prep(prep)

        self._score_against_rosetta(dud_scoring_data, param_db, torch_device_gpu)
