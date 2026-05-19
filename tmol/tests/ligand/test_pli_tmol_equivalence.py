"""Regression tests for protein_ligand_test ligand `.tmol` equivalence.

For each generated ligand CIF in `tests/data/protein_ligand_test/cif_inputs`,
this test:
1) builds `NonStandardResidueInfo` from explicit-bond CIF data,
2) runs the RDKit atom-typing pipeline (`prepare_single_ligand`), and
3) compares against the target `<name>.xtal-lig.mmff94.tmol` reference.
"""

from pathlib import Path

import pytest

from tmol.tests.ligand.build_pli_ligand_cifs import ensure_pli_ligand_cifs
from tmol.tests.ligand.test_dud_ligands import (
    _cartres_heavy_key_set,
    _cif_to_nonstandard_residue_info,
    _is_heavy,
)

PLI_DIR = Path(__file__).parent.parent / "data" / "protein_ligand_test"
PLI_CIF_DIR = PLI_DIR / "cif_inputs"
_TMOL_SUFFIX = ".xtal-lig.mmff94.tmol"
PLI_CASES = sorted(
    p.name[: -len(_TMOL_SUFFIX)] for p in PLI_DIR.glob(f"*{_TMOL_SUFFIX}")
)


class TestPLITmolEquivalence:
    """CIF-derived ligand prep must match reference `.tmol` ligand prep."""

    CHARGE_TOLERANCE = 0.05

    @pytest.fixture(params=PLI_CASES)
    def prep_pair(self, request):
        from tmol.ligand import prepare_single_ligand
        from tmol.ligand.params_file import load_params_file

        target = request.param
        tmol_path = PLI_DIR / f"{target}.xtal-lig.mmff94.tmol"
        ensure_pli_ligand_cifs(PLI_CIF_DIR)
        cif_path = PLI_CIF_DIR / f"{target}.ligand.cif"

        preps_tmol = load_params_file(tmol_path)
        assert (
            len(preps_tmol) == 1
        ), f"{tmol_path.name}: expected one residue, got {len(preps_tmol)}"
        prep_tmol = preps_tmol[0]

        info = _cif_to_nonstandard_residue_info(cif_path, prep_tmol.residue_type.name)
        prep_cif = prepare_single_ligand(info, ph=7.4)

        return {
            "name": target,
            "case_name": target,
            "cif": prep_cif,
            "tmol": prep_tmol,
        }

    def test_atom_set(self, prep_pair):
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
        only_cif = sorted(cif_atoms - tmol_atoms)
        only_tmol = sorted(tmol_atoms - cif_atoms)
        assert cif_atoms == tmol_atoms, (
            f"Heavy atom set mismatch for {prep_pair['case_name']}:\n"
            f"  only in cif-pipeline: {only_cif}\n"
            f"  only in tmol-ref:     {only_tmol}"
        )

    def test_atom_types(self, prep_pair):
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
        assert (
            not mismatches
        ), f"Atom type mismatches for {prep_pair['case_name']}:\n" + "\n".join(
            f"  {n}: cif={c}, tmol={t}" for n, c, t in mismatches
        )

    def test_bonds(self, prep_pair):
        aromatic_equiv = frozenset({"AROMATIC", "SINGLE", "DOUBLE"})

        def keyset(bonds):
            out = set()
            for a, b, bond_type, *rest in bonds:
                a, b = str(a), str(b)
                if not (_is_heavy(a) and _is_heavy(b)):
                    continue
                ring = bool(rest[0]) if rest else False
                out.add((frozenset([a, b]), bond_type, ring))
            return out

        def is_delocalized(pair, btype, ring, aromatic_atoms):
            if btype == "AROMATIC":
                return True
            if ring and btype in aromatic_equiv:
                return True
            if pair & aromatic_atoms and btype in aromatic_equiv:
                return True
            return False

        cif_bonds = keyset(prep_pair["cif"].residue_type.bonds)
        tmol_bonds = keyset(prep_pair["tmol"].residue_type.bonds)
        all_bonds = cif_bonds | tmol_bonds

        aromatic_atoms: set[str] = set()
        for pair, btype, _ring in all_bonds:
            if btype == "AROMATIC":
                aromatic_atoms.update(pair)

        delocalized_pairs: set[frozenset] = set()
        for pair, btype, ring in all_bonds:
            if is_delocalized(pair, btype, ring, aromatic_atoms):
                delocalized_pairs.add(pair)

        def normalize(bond_set):
            out = set()
            for pair, btype, ring in bond_set:
                if pair in delocalized_pairs and btype in aromatic_equiv:
                    out.add((pair, "DELOCALIZED", ring))
                else:
                    out.add((pair, btype, ring))
            return out

        cif_norm = normalize(cif_bonds)
        tmol_norm = normalize(tmol_bonds)
        only_cif = sorted(cif_norm - tmol_norm)
        only_tmol = sorted(tmol_norm - cif_norm)
        assert cif_norm == tmol_norm, (
            f"Heavy bond set mismatch for {prep_pair['case_name']}:\n"
            f"  only in cif-pipeline: {set(only_cif)}\n"
            f"  only in tmol-ref:     {set(only_tmol)}"
        )

    def test_partial_charges(self, prep_pair):
        cif_q = prep_pair["cif"].partial_charges
        tmol_q = prep_pair["tmol"].partial_charges
        shared = cif_q.keys() & tmol_q.keys()
        assert shared, f"No shared atom names for {prep_pair['case_name']}"
        bad = [
            (n, cif_q[n], tmol_q[n])
            for n in shared
            if abs(cif_q[n] - tmol_q[n]) >= self.CHARGE_TOLERANCE
        ]
        assert not bad, (
            f"Partial-charge mismatch (>{self.CHARGE_TOLERANCE}) for "
            f"{prep_pair['case_name']}:\n"
            + "\n".join(
                f"  {n}: cif={c:+.4f}, tmol={t:+.4f}, diff={c - t:+.4f}"
                for n, c, t in bad
            )
        )

    def test_cartbonded_params(self, prep_pair):
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
            f"Cartbonded parameter set mismatch for {prep_pair['case_name']}:\n"
            + "\n".join(diffs)
        )


def test_prepare_single_ligand_rejects_topology_only_single_bonds():
    import biotite.structure as struc
    import biotite.structure.io

    from tmol.io.pose_stack_from_biotite import canonical_ordering_for_biotite
    from tmol.ligand import prepare_single_ligand
    from tmol.ligand.detect import detect_nonstandard_residues

    pdb_path = PLI_DIR / "ace_complex_nometals.pdb"
    bt_struct = biotite.structure.io.load_structure(str(pdb_path), include_bonds=True)
    if isinstance(bt_struct, struc.AtomArrayStack):
        bt_struct = bt_struct[0]

    ligands = detect_nonstandard_residues(bt_struct, canonical_ordering_for_biotite())
    lg1 = next(l for l in ligands if l.res_name == "LG1")

    with pytest.raises(
        ValueError,
        match=(
            "topology-only SINGLE bonds|"
            "PDB ligand chemistry inference is unsupported|"
            "unsupported bond type codes"
        ),
    ):
        prepare_single_ligand(lg1, ph=7.4)
