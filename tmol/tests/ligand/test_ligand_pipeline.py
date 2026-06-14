"""Integration tests for the ligand preparation pipeline.

Tests the full pipeline from CIF structure with ligands through detection,
SMILES perception, protonation, 3D generation, atom typing, residue
building, and database registration. Includes ground truth regression tests
against reference pipeline outputs.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tmol.io.pose_stack_from_biotite import canonical_ordering_for_biotite
from tmol.ligand import prepare_ligands, prepare_single_ligand
from tmol.ligand.detect import detect_nonstandard_residues
from tmol.ligand.parity_manifest import load_parity_manifest
from tmol.ligand.dimorphite_dl import protonate_mol_variants
from tmol.ligand.params_io import read_params_file, write_params_file
from tmol.ligand.registry import clear_cache
from tmol.ligand.registry import inject_ligand_preparations

PLI_CIF_INPUT_DIR = (
    Path(__file__).parent.parent / "data" / "protein_ligand_test" / "cif_inputs"
)
PLI_DATA_DIR = Path(__file__).parent.parent / "data" / "protein_ligand_test"


@pytest.fixture(autouse=True)
def _clear_ligand_cache():
    clear_cache()
    yield
    clear_cache()


class TestDetectFromCIF:
    """Verify detection of non-standard residues in real CIF files."""

    @pytest.fixture
    def canonical_ordering(self):
        return canonical_ordering_for_biotite()

    def test_no_ligands_in_ubq(self, biotite_1ubq, canonical_ordering) -> None:
        """A pure-protein structure yields no non-standard residues."""
        assert len(detect_nonstandard_residues(biotite_1ubq, canonical_ordering)) == 0

    def test_detects_i4b_in_184l(self, cif_184l_with_i4b, canonical_ordering) -> None:
        """The I4B ligand in 184L is detected with coords and CCD type."""
        ligands = detect_nonstandard_residues(cif_184l_with_i4b, canonical_ordering)
        i4b = {lig.res_name: lig for lig in ligands}.get("I4B")
        assert i4b is not None
        assert "NON-POLYMER" in i4b.ccd_type.upper()
        assert i4b.coords.shape == (len(i4b.atom_names), 3)

    def test_detects_pse_with_partial_occupancy(
        self, cif_1a25_with_pse, canonical_ordering
    ) -> None:
        """The partial-occupancy PSE ligand is still detected."""
        ligands = detect_nonstandard_residues(cif_1a25_with_pse, canonical_ordering)
        assert any(lig.res_name == "PSE" for lig in ligands)

    def test_detects_authoritative_partial_charges_from_load_structure(
        self, canonical_ordering
    ) -> None:
        """Authoritative CIF partial charges are surfaced and skip protonation."""
        import biotite.structure
        import biotite.structure.io

        cif_path = PLI_CIF_INPUT_DIR / "ada.ligand.cif"
        bt_struct = biotite.structure.io.load_structure(
            str(cif_path),
            model=1,
            include_bonds=True,
            extra_fields=["partial_charge"],
        )
        if isinstance(bt_struct, biotite.structure.AtomArrayStack):
            bt_struct = bt_struct[0]

        ligands = detect_nonstandard_residues(bt_struct, canonical_ordering)
        lg1 = next(lig for lig in ligands if lig.res_name == "LG1")
        assert lg1.partial_charges is not None
        assert len(lg1.partial_charges) == len(lg1.atom_names)
        assert lg1.skip_protonation is True


class TestFullPipeline:
    """End-to-end: CIF -> detect -> prepare -> register -> verify."""

    @pytest.fixture
    def param_db(self):
        from tmol.database import ParameterDatabase

        return ParameterDatabase.get_default()

    def test_hem_metal_ligand_skipped(self, cif_155c_with_hem, param_db) -> None:
        """HEM contains Fe; metal-containing ligands are unsupported and skipped."""
        param_db, new_co = prepare_ligands(cif_155c_with_hem, param_db=param_db)

        assert "HEM" not in {r.name for r in param_db.chemical.residues}

    def test_pse_partial_occupancy(self, cif_1a25_with_pse, param_db) -> None:
        """Ligand with partial occupancy (PSE, 0.56) still prepares."""
        param_db, new_co = prepare_ligands(cif_1a25_with_pse, param_db=param_db)

        assert "PSE" in {r.name for r in param_db.chemical.residues}

    def test_inject_ligand_preparations_is_idempotent(
        self, cif_184l_with_i4b, param_db
    ) -> None:
        """Re-injecting an already-registered ligand preparation is a no-op."""
        ligands = detect_nonstandard_residues(
            cif_184l_with_i4b, canonical_ordering_for_biotite()
        )
        i4b = next(l for l in ligands if l.res_name == "I4B")
        prep = prepare_single_ligand(i4b, ph=7.4)

        n_before = len(param_db.chemical.residues)
        extended_db = inject_ligand_preparations(param_db, [prep])
        assert len(extended_db.chemical.residues) > n_before
        assert any(r.name == "I4B" for r in extended_db.chemical.residues)

        # Re-injecting the same preparation is a no-op (residue already
        # registered) — same number of residues, same database identity.
        again = inject_ligand_preparations(extended_db, [prep])
        assert len(again.chemical.residues) == len(extended_db.chemical.residues)

    def test_ubq_passes_through_unchanged(self, biotite_1ubq, param_db) -> None:
        """A ligand-free structure leaves the parameter database unchanged."""
        n_before = len(param_db.chemical.residues)
        param_db, _ = prepare_ligands(biotite_1ubq, param_db=param_db)
        assert len(param_db.chemical.residues) == n_before


class TestLigandScoringData:
    """Verify that scoring databases are correctly populated for ligands."""

    def test_ljlk_halogen_params_exist(self) -> None:
        """All halogen types (aromatic and non-aromatic) have LJLK params."""
        from tmol.database import ParameterDatabase

        param_db = ParameterDatabase.get_default()
        ljlk_names = {p.name for p in param_db.scoring.ljlk.atom_type_parameters}
        for halogen in ["F", "Cl", "Br", "I", "FR", "ClR", "BrR", "IR"]:
            assert halogen in ljlk_names, f"Missing LJLK params for {halogen}"


def test_prepare_ligands_missing_ligand_atom_fails(
    cif_184l_with_i4b, torch_device
) -> None:
    """Ligands with missing atoms are unsupported; loading must fail."""
    import numpy

    from tmol.database import ParameterDatabase
    from tmol.io.pose_stack_from_biotite import pose_stack_from_biotite

    bt = cif_184l_with_i4b.copy()
    ligand_atoms = numpy.nonzero(bt.res_name == "I4B")[0]
    if ligand_atoms.shape[0] == 0:
        pytest.skip("Could not find I4B ligand atoms to remove")

    keep_mask = numpy.ones(bt.array_length(), dtype=bool)
    keep_mask[ligand_atoms[0]] = False
    bt_ligand_missing = bt[keep_mask]

    with pytest.raises(Exception):
        pose_stack_from_biotite(
            bt_ligand_missing,
            torch_device,
            prepare_ligands=True,
            param_db=ParameterDatabase.get_default(),
        )


def test_ddg_from_cif_complex_with_onthefly_ligand_prep(
    cif_1a25_with_pse, torch_device
) -> None:
    """End-to-end protein-ligand ddG straight from a CIF complex.

    Exercises the path a user hits when they only have a structure (a biotite
    AtomArray loaded from a CIF whose ligand carries explicit bond orders via a
    ``_chem_comp_bond`` block) and no preprocessed ``.tmol``/``.params`` file:
    ``pose_stack_from_biotite(prepare_ligands=True)`` generates the ligand
    parameters on the fly, and ``calculate_block_pair_ddg`` then scores the
    protein-ligand interface. The score function must be built from the
    ligand-extended parameter database returned in the build context, otherwise
    the freshly minted ligand block type has no scoring parameters.
    """
    from tmol.io.pose_stack_from_biotite import pose_stack_from_biotite
    from tmol.score import beta2016_score_function
    from tmol.score.score_utils import calculate_block_pair_ddg

    pose_stack, context = pose_stack_from_biotite(
        cif_1a25_with_pse,
        torch_device,
        prepare_ligands=True,
        no_optH=True,
        return_context=True,
    )

    # Locate the ligand (PSE) block(s) via the canonical form rather than
    # assuming a fixed position in the pose.
    co = context.canonical_ordering
    res_types = context.canonical_form.res_types[0]
    n_blocks = pose_stack.max_n_blocks

    ligand_mask = torch.zeros((1, n_blocks), dtype=torch.bool, device=torch_device)
    block_names = []
    for block_idx in range(min(n_blocks, res_types.shape[0])):
        res_type_id = int(res_types[block_idx])
        if res_type_id < 0:
            continue
        name = co.restype_io_equiv_classes[res_type_id]
        block_names.append(name)
        if name == "PSE":
            ligand_mask[0, block_idx] = True

    assert (
        "PSE" in block_names
    ), f"ligand PSE not found among pose blocks: {block_names}"
    assert ligand_mask.any(), "no PSE ligand block was masked"

    sfxn = beta2016_score_function(torch_device, param_db=context.parameter_database)
    ddg = calculate_block_pair_ddg(
        pose_stack,
        ligand_mask,
        sfxn=sfxn,
        minimize=False,
        pack=False,
        database=context.parameter_database,
    )

    assert torch.isfinite(ddg).all(), f"non-finite ddG from on-the-fly path: {ddg}"


class TestParamsRoundtrip:
    """Write a prepared ligand to .params and read it back."""

    def test_i4b_params_roundtrip(self, tmp_path, cif_184l_with_i4b) -> None:
        """A prepared ligand written to .params reloads with matching topology."""
        co = canonical_ordering_for_biotite()
        ligands = detect_nonstandard_residues(cif_184l_with_i4b, co)
        i4b = next(lig for lig in ligands if lig.res_name == "I4B")
        prep = prepare_single_ligand(i4b)
        restype = prep.residue_type

        path = tmp_path / "I4B.params"
        write_params_file(prep, path, format="rosetta")
        loaded = read_params_file(path)

        assert loaded.name == "I4B"
        assert len(loaded.atoms) == len(restype.atoms)
        assert len(loaded.bonds) == len(restype.bonds)
        assert len(loaded.icoors) == len(restype.icoors)
        assert prep.atom_type_elements is not None
        assert len(prep.atom_type_elements) > 0

    def test_params_roundtrip_preserves_bond_types(
        self, tmp_path, cif_184l_with_i4b
    ) -> None:
        """Bond types and ring flags survive a .params write/read roundtrip."""
        co = canonical_ordering_for_biotite()
        ligands = detect_nonstandard_residues(cif_184l_with_i4b, co)
        i4b = next(lig for lig in ligands if lig.res_name == "I4B")
        prep = prepare_single_ligand(i4b)
        restype = prep.residue_type

        path = tmp_path / "I4B_bondtypes.params"
        write_params_file(prep, path, format="rosetta")
        loaded = read_params_file(path)

        assert all(len(b) == 4 for b in loaded.bonds)
        assert {(a, b, t, r) for a, b, t, r in loaded.bonds} == {
            (a, b, t, r) for a, b, t, r in restype.bonds
        }


def test_collect_new_atom_types_strict_mode_errors(default_database) -> None:
    """Strict atom typing raises on an unknown element mapping."""
    from tmol.ligand.registry import collect_new_atom_types

    residue = SimpleNamespace(
        name="UNK",
        atoms=(SimpleNamespace(name="X1", atom_type="ZZZ"),),
    )
    with pytest.raises(ValueError, match="Unknown element mapping"):
        collect_new_atom_types(
            default_database.chemical,
            residue,
            atom_type_elements={},
            strict_atom_types=True,
        )


def test_protonate_mol_variants_produces_valid_mol() -> None:
    """Protonating a molecule yields RDKit-parseable variant SMILES."""
    from rdkit import Chem

    input_smiles = "CC(=O)ON"
    mol = Chem.MolFromSmiles(input_smiles)
    assert mol is not None
    mol_variants = protonate_mol_variants(
        mol,
        min_ph=7.4,
        max_ph=7.4,
        pka_precision=0.1,
        max_variants=128,
        silent=True,
    )
    assert mol_variants
    result_smi = Chem.MolToSmiles(mol_variants[0], isomericSmiles=True)
    assert Chem.MolFromSmiles(result_smi) is not None


def test_prepare_single_ligand_uses_index_mapping_before_graph(
    cif_184l_with_i4b, monkeypatch
) -> None:
    """Direct-Mol preparation uses index mapping and skips graph matching."""
    import tmol.ligand.preparation as ligand_preparation

    ligands = detect_nonstandard_residues(
        cif_184l_with_i4b, canonical_ordering_for_biotite()
    )
    i4b = next(l for l in ligands if l.res_name == "I4B")

    def _fail_graph_match(*_args, **_kwargs):
        raise AssertionError("Graph matching should not be called in direct Mol path")

    monkeypatch.setattr(
        ligand_preparation, "_rename_atoms_to_cif_by_graph", _fail_graph_match
    )
    prep = prepare_single_ligand(i4b, ph=7.4)
    atom_names = {a.name for a in prep.residue_type.atoms}
    assert "C3'" in atom_names
    assert "C4'" in atom_names
    assert "C2'" in atom_names
    assert "C3'" in prep.partial_charges


def test_prepare_ligand_from_cif_helper_loads_reference_fixture() -> None:
    """The CIF helper prepares a ligand and registers it as LG1."""
    from tmol.database import ParameterDatabase
    from tmol.ligand import prepare_ligand_from_cif

    cif_path = PLI_CIF_INPUT_DIR / "ada.ligand.cif"
    param_db, _ = prepare_ligand_from_cif(
        str(cif_path),
        param_db=ParameterDatabase.get_default(),
    )
    assert any(rt.name == "LG1" for rt in param_db.chemical.residues)


def test_prepare_ligand_from_mol2_helper_loads_reference_fixture() -> None:
    """The MOL2 helper prepares a ligand and registers it as LG1."""
    from tmol.database import ParameterDatabase
    from tmol.ligand import prepare_ligand_from_mol2

    mol2_path = PLI_DATA_DIR / "ace.lig.mol2"
    param_db, _ = prepare_ligand_from_mol2(
        str(mol2_path),
        param_db=ParameterDatabase.get_default(),
    )
    assert any(rt.name == "LG1" for rt in param_db.chemical.residues)


def test_missing_authoritative_charges_reports_cif_loading_guidance(
    monkeypatch, torch_device
) -> None:
    """Missing authoritative charges raise an error with CIF-loading guidance."""
    import biotite.structure
    import biotite.structure.io
    import numpy
    import tmol.ligand.mol3d as mol3d

    from tmol.database import ParameterDatabase
    from tmol.io.pose_stack_from_biotite import pose_stack_from_biotite

    def _force_mmff_failure(_mol):
        raise RuntimeError("forced-mmff-failure")

    monkeypatch.setattr(mol3d, "compute_mmff94_charges", _force_mmff_failure)

    cif_path = PLI_CIF_INPUT_DIR / "ada.ligand.cif"
    bt_struct = biotite.structure.io.load_structure(
        str(cif_path),
        model=1,
        include_bonds=True,
    )
    if isinstance(bt_struct, biotite.structure.AtomArrayStack):
        bt_struct = bt_struct[0]

    bt_struct.set_annotation(
        "partial_charge",
        numpy.full(bt_struct.array_length(), numpy.nan, dtype=numpy.float64),
    )

    with pytest.raises(RuntimeError) as exc:
        pose_stack_from_biotite(
            bt_struct,
            torch_device,
            prepare_ligands=True,
            param_db=ParameterDatabase.get_default(),
        )

    msg = str(exc.value)
    assert "include_bonds=True" in msg
    assert "extra_fields=['partial_charge']" in msg
    assert "ligand_params_files" in msg


def _parse_reference_params(path: Path) -> dict:
    """Parse a Rosetta .params file into the legacy regression dict.

    Delegates to the shared :mod:`tmol.ligand.params_reference` parser so the
    regression suite and the parity harness use one implementation. The dict
    shape (atoms, bond_types, cut_bonds, chis, proton_chis, nbr_atom,
    icoor_topology) is preserved for existing callers.
    """
    from tmol.ligand.params_reference import as_legacy_dict, parse_reference_params

    return as_legacy_dict(parse_reference_params(path))


def _load_smi_file(path: Path, name: str) -> str:
    """Load a SMILES for a given molecule name from a .smi file.

    Accepts both tab-separated and whitespace-separated formats:
      <SMILES><ws><name>
    """
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[-1] == name:
                return parts[0]
    raise ValueError(f"Molecule {name!r} not found in {path}")


class TestGroundTruthRegression:
    """Validate pipeline against Rosetta mol2genparams reference outputs.

    The reference .params files were generated by Rosetta's mol2genparams.py
    (from guangfeng/ligand_prep). Our atom typing must produce identical
    atom names, atom types, charges, bond topology, and ICOOR tree structure.
    """

    GROUND_TRUTH_DIR = Path(__file__).parent.parent / "data" / "ligand_ground_truth"
    CHARGE_TOLERANCE = 0.01

    @pytest.fixture(params=load_parity_manifest(), ids=lambda e: e.name)
    def ref_data(self, request):
        """Load reference data and run our pipeline for comparison.

        Parametrized from the parity manifest seed entries, so adding a
        manifest molecule adds a regression case rather than editing a literal
        list. Per-entry settings (``sample_proton_chi``) drive preparation.
        """
        from rdkit import Chem

        from tmol.ligand.atom_typing import assign_tmol_atom_types
        from tmol.ligand.mol3d import compute_mmff94_charges
        from tmol.ligand.residue_builder import build_residue_type
        from tmol.ligand.rdkit_mol import protonate_ligand_mol

        entry = request.param
        name = entry.name

        input_smi = entry.input_smiles
        expected_prot_smi = entry.expected_prot_smiles
        ref = _parse_reference_params(entry.params)

        rdkit_mol = Chem.MolFromSmiles(input_smi)
        protonated = protonate_ligand_mol(rdkit_mol, ph=7.4)
        protonated = Chem.AddHs(protonated, addCoords=False)
        prot_smi = Chem.MolToSmiles(Chem.RemoveHs(protonated), isomericSmiles=True)
        charges_by_idx = compute_mmff94_charges(protonated)
        atom_types, typing_state = assign_tmol_atom_types(protonated, return_state=True)
        charges = {
            at.atom_name: charges_by_idx[at.index]
            for at in atom_types
            if at.index in charges_by_idx
        }
        restype = build_residue_type(
            protonated,
            name,
            atom_types,
            typing_state=typing_state,
            sample_proton_chi=entry.sample_proton_chi,
        )

        return {
            "name": name,
            "input_smiles": input_smi,
            "expected_prot_smiles": expected_prot_smi,
            "actual_prot_smiles": prot_smi,
            "ref": ref,
            "atom_types": atom_types,
            "charges": charges,
            "restype": restype,
        }

    def test_chi_axes_match(self, ref_data) -> None:
        """Emitted CHI torsions cover the same rotatable axes as ref .params.

        Semantic parity: compare the unordered set of central {b, c} heavy-atom
        name pairs (CHI numbering / quad text are not asserted).
        """
        ref_axes = {
            frozenset((quad[1], quad[2]))
            for (_num, quad, _biaryl) in ref_data["ref"]["chis"]
        }
        emit_axes = {
            frozenset((t.b.atom, t.c.atom)) for t in ref_data["restype"].torsions
        }
        assert emit_axes == ref_axes, (
            f"CHI axis mismatch for {ref_data['name']}: "
            f"only-in-ref={sorted(tuple(sorted(s)) for s in ref_axes - emit_axes)}, "
            f"only-in-emit={sorted(tuple(sorted(s)) for s in emit_axes - ref_axes)}"
        )

    def test_proton_chi_samples_match(self, ref_data) -> None:
        """PROTON_CHI sample sets and EXTRA expansions match ref .params."""
        ref = ref_data["ref"]
        restype = ref_data["restype"]

        chi_axis_by_num = {
            num: frozenset((quad[1], quad[2])) for (num, quad, _b) in ref["chis"]
        }
        ref_by_axis = {}
        for line in ref["proton_chis"]:
            toks = line.split()
            num = int(toks[1])
            si = toks.index("SAMPLES")
            k = int(toks[si + 1])
            samples = tuple(float(x) for x in toks[si + 2 : si + 2 + k])
            extra = (
                tuple(float(x) for x in toks[toks.index("EXTRA") + 1 :])
                if "EXTRA" in toks
                else ()
            )
            ref_by_axis[chi_axis_by_num[num]] = (samples, extra)

        axis_by_chiname = {
            t.name: frozenset((t.b.atom, t.c.atom)) for t in restype.torsions
        }
        emit_by_axis = {
            axis_by_chiname[cs.chi_dihedral]: (cs.samples, cs.expansions)
            for cs in restype.chi_samples
        }

        assert set(emit_by_axis) == set(ref_by_axis), (
            f"PROTON_CHI axis-set mismatch for {ref_data['name']}: "
            f"ref={sorted(tuple(sorted(s)) for s in ref_by_axis)}, "
            f"emit={sorted(tuple(sorted(s)) for s in emit_by_axis)}"
        )
        for axis, (r_samples, r_extra) in ref_by_axis.items():
            e_samples, e_expansions = emit_by_axis[axis]
            assert sorted(e_samples) == sorted(r_samples), (
                f"{ref_data['name']} samples mismatch on {tuple(sorted(axis))}: "
                f"ref={r_samples} emit={e_samples}"
            )
            # Rosetta EXTRA "1 20" -> one expansion of 20 deg -> (20.0,);
            # EXTRA "0" -> no expansion -> ().
            expected = (r_extra[1],) if (r_extra and r_extra[0] >= 1) else ()
            assert tuple(e_expansions) == expected, (
                f"{ref_data['name']} EXTRA mismatch on {tuple(sorted(axis))}: "
                f"ref EXTRA={r_extra} -> expected {expected}, emit={e_expansions}"
            )

    def test_protonation_matches(self, ref_data) -> None:
        """dimorphite_dl protonation must produce the expected SMILES."""
        assert ref_data["actual_prot_smiles"] == ref_data["expected_prot_smiles"], (
            f"Protonation mismatch for {ref_data['name']}: "
            f"got {ref_data['actual_prot_smiles']!r}, "
            f"expected {ref_data['expected_prot_smiles']!r}"
        )

    def test_atom_count_matches(self, ref_data) -> None:
        """Total atom count must match reference params."""
        ref_count = len(ref_data["ref"]["atoms"])
        actual_count = len(ref_data["restype"].atoms)
        assert actual_count == ref_count, (
            f"Atom count mismatch for {ref_data['name']}: "
            f"got {actual_count}, expected {ref_count}"
        )

    def test_atom_types_match(self, ref_data) -> None:
        """Each atom's name and Rosetta type must match the reference."""
        ref_atoms = ref_data["ref"]["atoms"]
        actual_atoms = [(a.name, a.atom_type) for a in ref_data["restype"].atoms]
        ref_name_type = [(name, atype) for name, atype, _ in ref_atoms]

        assert actual_atoms == ref_name_type, (
            f"Atom type mismatch for {ref_data['name']}:\n"
            f"  got:      {actual_atoms}\n"
            f"  expected: {ref_name_type}"
        )

    def test_charges_match(self, ref_data) -> None:
        """MMFF94 partial charges must match reference within tolerance."""
        ref_charges = {name: charge for name, _, charge in ref_data["ref"]["atoms"]}
        actual_charges = ref_data["charges"]

        for atom_name, ref_q in ref_charges.items():
            actual_q = actual_charges.get(atom_name)
            assert (
                actual_q is not None
            ), f"Missing charge for {atom_name} in {ref_data['name']}"
            assert abs(actual_q - ref_q) < self.CHARGE_TOLERANCE, (
                f"Charge mismatch for {atom_name} in {ref_data['name']}: "
                f"got {actual_q:.4f}, expected {ref_q:.4f}"
            )

    def test_bond_topology_matches(self, ref_data) -> None:
        """Bond pairs must match reference (order-independent)."""
        actual_bonds = set()
        for a, b, *_ in ref_data["restype"].bonds:
            actual_bonds.add(frozenset([a, b]))

        ref_bonds = set()
        for pair, _order, _ring in ref_data["ref"]["bond_types"]:
            ref_bonds.add(pair)

        assert actual_bonds == ref_bonds, (
            f"Bond topology mismatch for {ref_data['name']}:\n"
            f"  missing: {ref_bonds - actual_bonds}\n"
            f"  extra:   {actual_bonds - ref_bonds}"
        )

    def test_bond_count_matches(self, ref_data) -> None:
        """Bond count must match reference."""
        ref_count = len(ref_data["ref"]["bond_types"])
        actual_count = len(ref_data["restype"].bonds)
        assert actual_count == ref_count, (
            f"Bond count mismatch for {ref_data['name']}: "
            f"got {actual_count}, expected {ref_count}"
        )

    def test_icoor_completeness(self, ref_data) -> None:
        """Every atom must appear in the ICOOR tree."""
        actual_icoors = ref_data["restype"].icoors
        actual_atoms = ref_data["restype"].atoms
        icoor_names = {ic.name for ic in actual_icoors}
        atom_names = {a.name for a in actual_atoms}
        assert icoor_names == atom_names, (
            f"ICOOR atom set mismatch for {ref_data['name']}:\n"
            f"  missing from icoor: {atom_names - icoor_names}\n"
            f"  extra in icoor: {icoor_names - atom_names}"
        )
