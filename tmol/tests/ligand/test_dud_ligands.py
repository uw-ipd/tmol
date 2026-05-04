"""Regression tests for the ligand pipeline against DUD benchmark ligands.

Compares our SMILES-based pipeline (RDKit mol from mol2, protonation,
MMFF94 charges, OB atom typing, residue building) against Rosetta's
mol2genparams reference .params files for drug-like molecules from the
DUD (Directory of Useful Decoys) dataset.
"""

from pathlib import Path

import cattr
import pytest
import torch
import yaml

from tmol.tests.ligand.test_ligand_pipeline import _parse_reference_params

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


class TestDUDRegression:
    """Validate pipeline against DUD benchmark ligands with Rosetta reference params."""

    CHARGE_TOLERANCE = 0.05

    @pytest.fixture(params=DUD_CASES, ids=[f"{t}_{n}" for t, n in DUD_CASES])
    def dud_data(self, request):
        """Load a DUD mol2, run pipeline, and load reference params."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from tmol.ligand.atom_typing import assign_tmol_atom_types
        from tmol.ligand.mol3d import compute_mmff94_charges
        from tmol.ligand.rdkit_mol import protonate_ligand_mol
        from tmol.ligand.residue_builder import build_residue_type

        target, name = request.param
        mol2_path = DUD_DIR / target / f"{name}.mol2"
        ref_params_path = DUD_DIR / target / f"{name}.params"

        ref = _parse_reference_params(ref_params_path)

        mol = Chem.MolFromMol2File(str(mol2_path), removeHs=True, sanitize=True)
        if mol is None:
            pytest.skip(f"RDKit cannot parse {mol2_path}")

        protonated = protonate_ligand_mol(mol, ph=7.4)
        try:
            protonated = AllChem.AssignBondOrdersFromTemplate(protonated, mol)
        except Exception:
            pass
        try:
            Chem.SanitizeMol(protonated)
        except Exception:
            pass
        protonated = Chem.AddHs(protonated, addCoords=True)
        charges_by_idx = compute_mmff94_charges(protonated)
        atom_types = assign_tmol_atom_types(protonated)
        charges = {
            at.atom_name: charges_by_idx[at.index]
            for at in atom_types
            if at.index in charges_by_idx
        }
        restype = build_residue_type(protonated, name, atom_types)

        ref_atom_types = {name: atype for name, atype, _ in ref["atoms"]}
        ref_charges = {name: charge for name, _, charge in ref["atoms"]}

        return {
            "name": name,
            "target": target,
            "ref": ref,
            "ref_atom_types": ref_atom_types,
            "ref_charges": ref_charges,
            "atom_types": atom_types,
            "charges": charges,
            "restype": restype,
        }

    def test_atom_count_matches(self, dud_data):
        """Heavy atom count must match reference."""
        ref_heavy = sum(
            1 for n, _, _ in dud_data["ref"]["atoms"] if not n.startswith("H")
        )
        actual_heavy = sum(
            1 for a in dud_data["restype"].atoms if not a.name.startswith("H")
        )
        assert actual_heavy == ref_heavy, (
            f"Heavy atom count mismatch for {dud_data['name']}: "
            f"got {actual_heavy}, expected {ref_heavy}"
        )

    def test_atom_types_match(self, dud_data):
        """Atom types must match reference for shared atoms."""
        ref_types = dud_data["ref_atom_types"]
        actual = {a.name: a.atom_type for a in dud_data["restype"].atoms}
        for name, expected_type in ref_types.items():
            if name not in actual:
                continue
            assert actual[name] == expected_type, (
                f"Type mismatch for {name} in {dud_data['name']}: "
                f"got {actual[name]}, expected {expected_type}"
            )

    def test_bond_count_heavy(self, dud_data):
        """Heavy-atom bond count must match reference."""
        ref_bonds_heavy = {
            pair
            for pair, order, ring in dud_data["ref"]["bond_types"]
            if all(not a.startswith("H") for a in pair)
        }
        actual_bonds_heavy = {
            frozenset([b[0], b[1]])
            for b in dud_data["restype"].bonds
            if not b[0].startswith("H") and not b[1].startswith("H")
        }
        assert len(actual_bonds_heavy) == len(ref_bonds_heavy), (
            f"Heavy bond count mismatch for {dud_data['name']}: "
            f"got {len(actual_bonds_heavy)}, expected {len(ref_bonds_heavy)}"
        )

    def test_charges_within_tolerance(self, dud_data):
        """Charges for shared atoms must be within tolerance of reference."""
        ref_charges = dud_data["ref_charges"]
        actual = dud_data["charges"]
        compared = 0
        for atom_name, ref_q in ref_charges.items():
            if atom_name not in actual:
                continue
            compared += 1
            assert abs(actual[atom_name] - ref_q) < self.CHARGE_TOLERANCE, (
                f"Charge mismatch for {atom_name} in {dud_data['name']}: "
                f"got {actual[atom_name]:.4f}, expected {ref_q:.4f}"
            )
        assert (
            compared > 0
        ), f"No shared atoms to compare charges for {dud_data['name']}"


# ---------------------------------------------------------------------------
# .tmol file loader
# ---------------------------------------------------------------------------


def _load_tmol_file(path):
    """Load a .tmol YAML and return data ready for inject_residue_params.

    Returns:
        (residues, partial_charges, cartbonded) where
        - residues: list[RawResidueType]
        - partial_charges: {res_name: {atom: charge}}
        - cartbonded: {res_name: CartRes}
    """
    from tmol.database.chemical import RawResidueType, normalize_bond_tuples
    from tmol.database.scoring.cartbonded import CartRes

    with open(path) as f:
        raw = yaml.safe_load(f)

    res_list = raw.get("chemical", {}).get("residues", [])
    normalize_bond_tuples({"residues": res_list})
    residues = [cattr.structure(r, RawResidueType) for r in res_list]

    ec_raw = raw.get("elec", {}).get("atom_charge_parameters", [])
    partial_charges: dict[str, dict[str, float]] = {}
    for entry in ec_raw:
        partial_charges.setdefault(entry["res"], {})[entry["atom"]] = entry["charge"]

    cb_raw = raw.get("cartbonded", {}).get("residue_params", {})
    cartbonded = {
        name: cattr.structure(params, CartRes) for name, params in cb_raw.items()
    }

    return residues, partial_charges, cartbonded


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------


# Terms in score.sc we compare against tmol (values are already weighted).
_ROS_TERMS = {
    "fa_intra_atr_xover4",
    "fa_intra_rep_xover4",
    "fa_intra_sol_xover4",
    "fa_intra_elec",
    "gen_bonded",
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
    """Load Rosetta-reference .tmol params into tmol and score each ligand."""

    @pytest.fixture(params=DUD_CASES, ids=[f"{t}_{n}" for t, n in DUD_CASES])
    def dud_scoring_data(self, request):
        target, name = request.param
        tmol_path = DUD_DIR / target / f"{name}.tmol"
        in_pdb = DUD_DIR / target / f"{name}_in.pdb"

        if not tmol_path.exists():
            pytest.skip(f"No .tmol file: {tmol_path}")
        if not in_pdb.exists():
            pytest.skip(f"No _in.pdb: {in_pdb}")

        return {
            "name": name,
            "target": target,
            "tmol_path": tmol_path,
            "in_pdb": in_pdb,
        }

    def test_score(self, dud_scoring_data, torch_device):
        import biotite.structure
        import biotite.structure.io

        from tmol.database import ParameterDatabase, inject_residue_params
        from tmol.io.pose_stack_from_biotite import pose_stack_from_biotite
        from tmol.score import beta2016_score_function

        residues, partial_charges, cartbonded = _load_tmol_file(
            dud_scoring_data["tmol_path"]
        )
        param_db = inject_residue_params(
            ParameterDatabase.get_default(),
            residue_types=residues,
            partial_charges=partial_charges,
            cartbonded_params=cartbonded,
        )

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
        total = sum(
            float(weights[i]) * float(unweighted[i, 0]) for i in range(len(score_types))
        )

        # --- Rosetta reference scores from pre-generated .sc file ---
        sc_path = (
            DUD_DIR / dud_scoring_data["target"] / f"{dud_scoring_data['name']}.sc"
        )
        if not sc_path.exists():
            pytest.skip(f"No reference scores: {sc_path}")
        total_ros = _rosetta_score(sc_path).get("total_score", 0.0)

        assert abs(total - total_ros) < 1.0, (
            f"Total score mismatch for {dud_scoring_data['name']}: "
            f"tmol={total:.4f}, ros={total_ros:.4f}, diff={total - total_ros:.4f}"
        )
