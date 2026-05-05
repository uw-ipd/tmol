"""Regression tests for DUD benchmark ligands.

Two regression layers:

1. ``TestCifToTmolEquivalence`` — drives the AtomArray pipeline from each
   ligand's ``.cif`` and asserts the resulting ``LigandPreparation``
   matches Frank's reference ``.tmol`` (loaded via ``load_params_file``)
   on atoms, atom types, bonds (including the 4-tuple ring flag),
   partial charges, and cartbonded parameters.

2. ``TestDUDScoring`` — loads the ``.tmol`` directly into a
   ``ParameterDatabase`` and confirms scoring the matching ``_in.pdb``
   pose reproduces the pre-generated Rosetta total in the ``.sc`` file.
"""

from pathlib import Path

import cattr
import numpy as np
import pytest
import torch  # noqa: F401  (used via torch_device fixture)
import yaml

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
# CIF -> NonStandardResidueInfo
# ---------------------------------------------------------------------------


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
    arr = pdbx.get_structure(
        cif, model=1, include_bonds=True, extra_fields=["charge"]
    )
    if isinstance(arr, struc.AtomArrayStack):
        arr = arr[0]

    arr.res_name = np.array([res_name] * len(arr), dtype=arr.res_name.dtype)

    atom_site = cif.block["atom_site"]
    atom_names = list(atom_site["label_atom_id"].as_array())
    partial_charges = {
        name: float(q)
        for name, q in zip(
            atom_names, atom_site["partial_charge"].as_array(float)
        )
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


# ---------------------------------------------------------------------------
# CIF -> LigandPreparation == .tmol -> LigandPreparation
# ---------------------------------------------------------------------------


def _is_heavy(name: str) -> bool:
    """True iff this atom name belongs to a heavy atom (not a hydrogen).

    H-naming convention diverges between the two paths (the AtomArray
    pipeline produces ``HC1``/``HN1``/``HO1`` from ``_classify_H`` while
    Frank's ``.tmol`` uses sequential ``H1``..``Hn``). The chemistry
    we're regressing on is the heavy-atom skeleton + atom types + bond
    schema + cartbonded geometry, so we filter Hs out of the structural
    comparisons. Names are compared as plain ``str`` to neutralise
    ``numpy.str_`` artifacts from the CIF-load path.
    """
    return not str(name).startswith("H")


def _cartres_heavy_key_set(params, kind):
    """Canonical, order-normalised key set for a CartRes parameter list.

    The two paths can emit the same physical bond/angle/improper with
    different atom orderings. Each kind has its own normalisation:

    - ``"length"`` — frozenset of the two endpoints.
    - ``"angle"``  — center atom (atm2) fixed; endpoints sorted.
    - ``"improper"`` — sorted 4-tuple of atom names. The central sp2
      atom isn't always at the same position in the two
      representations, so we sort everything and rely on the per-group
      uniqueness in both sources.
    """
    keys = set()
    if kind == "length":
        for p in params:
            a, b = str(p.atm1), str(p.atm2)
            if _is_heavy(a) and _is_heavy(b):
                keys.add(frozenset([a, b]))
    elif kind == "angle":
        for p in params:
            a1, c, a3 = str(p.atm1), str(p.atm2), str(p.atm3)
            if all(_is_heavy(n) for n in (a1, c, a3)):
                lo, hi = sorted([a1, a3])
                keys.add((lo, c, hi))
    elif kind == "improper":
        for p in params:
            names = [str(p.atm1), str(p.atm2), str(p.atm3), str(p.atm4)]
            if all(_is_heavy(n) for n in names):
                keys.add(tuple(sorted(names)))
    else:
        raise ValueError(f"Unknown cartbonded group kind: {kind}")
    return keys


class TestCifToTmolEquivalence:
    """The AtomArray pipeline must produce the same LigandPreparation
    that ``load_params_file`` produces from Frank's reference ``.tmol``."""

    CHARGE_TOLERANCE = 0.05

    @pytest.fixture(params=DUD_CASES, ids=[f"{t}_{n}" for t, n in DUD_CASES])
    def prep_pair(self, request):
        from tmol.ligand import prepare_single_ligand
        from tmol.ligand.params_file import load_params_file

        target, name = request.param
        cif_path = DUD_DIR / target / f"{name}.cif"
        tmol_path = DUD_DIR / target / f"{name}.tmol"

        info = _cif_to_nonstandard_residue_info(cif_path, "LG1")
        prep_cif = prepare_single_ligand(info, ph=7.4)

        preps_tmol = load_params_file(tmol_path)
        assert len(preps_tmol) == 1, (
            f"{tmol_path}: expected one residue, got {len(preps_tmol)}"
        )
        prep_tmol = preps_tmol[0]

        return {"name": name, "cif": prep_cif, "tmol": prep_tmol}

    def test_atom_set(self, prep_pair):
        """Same set of (heavy_atom_name, atom_type) pairs."""
        cif_atoms = {
            (str(a.name), a.atom_type)
            for a in prep_pair["cif"].residue_type.atoms
            if _is_heavy(a.name)
        }
        tmol_atoms = {
            (str(a.name), a.atom_type)
            for a in prep_pair["tmol"].residue_type.atoms
            if _is_heavy(a.name)
        }
        assert cif_atoms == tmol_atoms, (
            f"Heavy atom set mismatch for {prep_pair['name']}:\n"
            f"  only in cif:  {sorted(cif_atoms - tmol_atoms)}\n"
            f"  only in tmol: {sorted(tmol_atoms - cif_atoms)}"
        )

    def test_atom_types(self, prep_pair):
        """For heavy atoms shared by name, atom_type must match."""
        cif_types = {
            str(a.name): a.atom_type
            for a in prep_pair["cif"].residue_type.atoms
            if _is_heavy(a.name)
        }
        tmol_types = {
            str(a.name): a.atom_type
            for a in prep_pair["tmol"].residue_type.atoms
            if _is_heavy(a.name)
        }
        mismatches = [
            (n, cif_types[n], tmol_types[n])
            for n in cif_types.keys() & tmol_types.keys()
            if cif_types[n] != tmol_types[n]
        ]
        assert not mismatches, (
            f"Atom type mismatches for {prep_pair['name']}:\n"
            + "\n".join(f"  {n}: cif={c}, tmol={t}" for n, c, t in mismatches)
        )

    def test_bonds(self, prep_pair):
        """Heavy-atom bonds with bond_type and is_in_ring (4-tuple)."""

        def keyset(bonds):
            out = set()
            for a, b, bond_type, *rest in bonds:
                a, b = str(a), str(b)
                if not (_is_heavy(a) and _is_heavy(b)):
                    continue
                ring = bool(rest[0]) if rest else False
                out.add((frozenset([a, b]), bond_type, ring))
            return out

        cif_bonds = keyset(prep_pair["cif"].residue_type.bonds)
        tmol_bonds = keyset(prep_pair["tmol"].residue_type.bonds)
        assert cif_bonds == tmol_bonds, (
            f"Heavy bond set mismatch for {prep_pair['name']}:\n"
            f"  only in cif:  {cif_bonds - tmol_bonds}\n"
            f"  only in tmol: {tmol_bonds - cif_bonds}"
        )

    def test_partial_charges(self, prep_pair):
        """Per-atom partial charges agree within tolerance for shared names."""
        cif_q = prep_pair["cif"].partial_charges
        tmol_q = prep_pair["tmol"].partial_charges
        shared = cif_q.keys() & tmol_q.keys()
        assert shared, f"No shared atom names for {prep_pair['name']}"
        bad = [
            (n, cif_q[n], tmol_q[n])
            for n in shared
            if abs(cif_q[n] - tmol_q[n]) >= self.CHARGE_TOLERANCE
        ]
        assert not bad, (
            f"Partial-charge mismatch (>{self.CHARGE_TOLERANCE}) for "
            f"{prep_pair['name']}:\n"
            + "\n".join(
                f"  {n}: cif={c:+.4f}, tmol={t:+.4f}, diff={c - t:+.4f}"
                for n, c, t in bad
            )
        )

    def test_cartbonded_params(self, prep_pair):
        """Heavy-atom length / angle / improper atom-tuples (order-normalised)."""
        cif_cb = prep_pair["cif"].cartbonded_params
        tmol_cb = prep_pair["tmol"].cartbonded_params
        groups = [
            ("length_parameters", "length"),
            ("angle_parameters", "angle"),
            ("improper_parameters", "improper"),
        ]
        diffs = []
        for attr_name, kind in groups:
            cif_keys = _cartres_heavy_key_set(getattr(cif_cb, attr_name), kind)
            tmol_keys = _cartres_heavy_key_set(getattr(tmol_cb, attr_name), kind)
            if cif_keys != tmol_keys:
                diffs.append(
                    f"  {attr_name}: only in cif {cif_keys - tmol_keys}, "
                    f"only in tmol {tmol_keys - cif_keys}"
                )
        assert not diffs, (
            f"Cartbonded parameter set mismatch for {prep_pair['name']}:\n"
            + "\n".join(diffs)
        )


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
        total_ros = _rosetta_score(sc_path).get("total_score", 0.0)

        assert abs(total - total_ros) < 1.0, (
            f"Total score mismatch for {dud_scoring_data['name']}: "
            f"tmol={total:.4f}, ros={total_ros:.4f}, diff={total - total_ros:.4f}"
        )
