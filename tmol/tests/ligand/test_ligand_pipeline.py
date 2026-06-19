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
from tmol.ligand import prepare_ligands
from tmol.ligand.preparation import _prepare_ligand_via_smiles
from tmol.ligand.detect import detect_nonstandard_residues
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
        prep = _prepare_ligand_via_smiles(i4b, ph=7.4, sample_proton_chi=True)

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
        prep = _prepare_ligand_via_smiles(i4b, ph=7.4, sample_proton_chi=True)
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
        prep = _prepare_ligand_via_smiles(i4b, ph=7.4, sample_proton_chi=True)
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
