import biotite.structure
from biotite.structure.io.pdbx import CIFFile, set_structure
import pytest
import torch

from tmol.io.pose_stack_from_biotite import (
    build_context_from_biotite,
    canonical_form_from_biotite,
    pose_stack_from_biotite,
    biotite_from_pose_stack,
)
from tmol.tests.data import load_cif

_CI_CIF_CODES = [
    "1UBQ",  # small, clean protein
    "1R21",  # NMR ensemble (multi-model)
    "1BL8",  # missing sidechains
    "6H9V",  # OXT edge case
    "3N0I",  # OXT edge case, multi-chain
    "6C4C",  # non-standard hydrogen naming
    "3AA0",  # Crash on export with mismatch of CO and PBT n-atoms
]


@pytest.mark.parametrize("pdb_code", _CI_CIF_CODES)
def test_load_score_roundtrip_cif(pdb_code, tmp_path):
    """Load a local CIF, build PoseStack, score, and re-export."""
    device = torch.device("cpu")

    from tmol import beta2016_score_function

    bt_struct = load_cif(pdb_code)
    if isinstance(bt_struct, biotite.structure.AtomArrayStack):
        bt_struct = bt_struct[0]

    pose_stack = pose_stack_from_biotite(bt_struct, device)
    sfxn = beta2016_score_function(device)
    scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
    scores = scorer.unweighted_scores(pose_stack.coords)

    assert not torch.any(torch.isnan(scores))
    assert not torch.any(torch.isinf(scores))

    bio = biotite_from_pose_stack(pose_stack)
    out_file = CIFFile()
    set_structure(out_file, bio)
    out_file.write(tmp_path / f"{pdb_code}_roundtrip.cif")


def test_build_context_from_biotite_smoke(biotite_1ubq, torch_device):
    context = build_context_from_biotite(biotite_1ubq, torch_device=torch_device)
    assert context.packed_block_types.device.type == torch_device.type


def test_canonical_form_from_biotite_smoke(biotite_1r21, torch_device):
    canonical_form_from_biotite(biotite_1r21, torch_device=torch_device)


def test_pose_stack_from_biotite_1ubq_smoke(biotite_1ubq, torch_device):
    pose_stack_from_biotite(biotite_1ubq, torch_device=torch_device)


# 1ubq with one residue's 3LC changed to ERR to test a non-recognized residue type
def test_pose_stack_from_biotite_1ubq_err_smoke(biotite_1ubq_err, torch_device):
    starts = biotite.structure.get_residue_starts(biotite_1ubq_err)
    bt = biotite_1ubq_err[starts[-5] : starts[-1]]
    pose_stack_from_biotite(bt, torch_device=torch_device)


def test_pose_stack_from_biotite_1ubq_cif_smoke(biotite_1ubq_cif, torch_device):
    pose_stack_from_biotite(biotite_1ubq_cif, torch_device=torch_device)


def test_pose_stack_from_and_to_biotite_1ubq_smoke(biotite_1ubq, torch_device):
    pose_stack = pose_stack_from_biotite(biotite_1ubq, torch_device=torch_device)
    biotite_from_pose_stack(pose_stack)


def test_pose_stack_from_and_to_biotite_1ubq_no_opth_smoke(biotite_1ubq, torch_device):
    pose_stack = pose_stack_from_biotite(
        biotite_1ubq, torch_device=torch_device, no_optH=True
    )
    biotite_from_pose_stack(pose_stack)


def test_pose_stack_from_and_to_biotite_multiple_poses_smoke(
    biotite_1r21, torch_device
):
    pose_stack = pose_stack_from_biotite(biotite_1r21, torch_device=torch_device)
    biotite_from_pose_stack(pose_stack)


def test_canonical_form_multipose_metadata_propagation(biotite_1r21, torch_device):
    cf = canonical_form_from_biotite(biotite_1r21, torch_device=torch_device)
    assert cf.atom_b_factor is not None
    assert cf.atom_occupancy is not None
    assert cf.atom_b_factor.shape[0] == biotite_1r21.stack_depth()
    assert cf.atom_occupancy.shape[0] == biotite_1r21.stack_depth()
    assert float(cf.atom_b_factor[1].sum()) > 0.0
    assert float(cf.atom_occupancy[1].sum()) > 0.0


def test_pose_stack_from_biotite_1ubq_slice_smoke(biotite_1ubq, torch_device):
    starts = biotite.structure.get_residue_starts(biotite_1ubq)
    bt = biotite_1ubq[0 : starts[30]]
    pose_stack_from_biotite(bt, torch_device=torch_device)


def test_pose_stack_from_biotite_n_term_smoke(biotite_1r21, torch_device):
    starts = biotite.structure.get_residue_starts(biotite_1r21)
    bt = biotite_1r21[0][0 : starts[3]]
    pose_stack_from_biotite(bt, torch_device=torch_device)


def test_pose_stack_from_biotite_c_term_smoke(biotite_1r21, torch_device):
    starts = biotite.structure.get_residue_starts(biotite_1r21)
    bt = biotite_1r21[0][starts[-5] : starts[-1]]
    pose_stack_from_biotite(bt, torch_device=torch_device)


def test_pose_stack_from_biotite_his_d_smoke(biotite_1r21, torch_device):
    starts = biotite.structure.get_residue_starts(biotite_1r21)
    bt = biotite_1r21[0][starts[52] : starts[55]]
    pose_stack_from_biotite(bt, torch_device=torch_device)


def test_pose_stack_from_biotite_missing_sidechain_smoke(biotite_1bl8, torch_device):
    bt = biotite_1bl8
    pose_stack = pose_stack_from_biotite(bt, torch_device=torch_device)
    biotite_from_pose_stack(pose_stack)


def test_pose_stack_from_biotite_missing_single_sidechain_smoke(
    biotite_1bl8, torch_device
):
    starts = biotite.structure.get_residue_starts(biotite_1bl8)
    bt = biotite_1bl8[starts[0] : starts[6]]
    pose_stack = pose_stack_from_biotite(bt, torch_device=torch_device)
    biotite_from_pose_stack(pose_stack)


@pytest.mark.parametrize("entry", ["build_context", "pose_stack"])
@pytest.mark.parametrize("sample_proton_chi", [True, False])
def test_sample_proton_chi_forwarded_to_prepare_ligands(
    entry, sample_proton_chi, torch_device, monkeypatch
):
    # An explicit sample_proton_chi setting must reach
    # tmol.ligand.prepare_ligands from the integrated pose-build path. Spy on
    # prepare_ligands to capture the kwarg, then short-circuit before the heavy
    # canonical-form build.
    import tmol.ligand

    captured: dict = {}

    class _Stop(Exception):
        pass

    def fake_prepare_ligands(structure, **kwargs):
        captured.update(kwargs)
        raise _Stop

    monkeypatch.setattr(tmol.ligand, "prepare_ligands", fake_prepare_ligands)

    structure = biotite.structure.AtomArray(1)
    func = (
        build_context_from_biotite
        if entry == "build_context"
        else pose_stack_from_biotite
    )

    with pytest.raises(_Stop):
        func(
            structure,
            torch_device,
            prepare_ligands=True,
            sample_proton_chi=sample_proton_chi,
        )
    assert captured.get("sample_proton_chi") is sample_proton_chi


@pytest.mark.parametrize("entry", ["build_context", "pose_stack"])
def test_sample_proton_chi_enabled_by_default(entry, torch_device, monkeypatch):
    import tmol.ligand

    captured: dict = {}

    class _Stop(Exception):
        pass

    def fake_prepare_ligands(structure, **kwargs):
        captured.update(kwargs)
        raise _Stop

    monkeypatch.setattr(tmol.ligand, "prepare_ligands", fake_prepare_ligands)

    structure = biotite.structure.AtomArray(1)
    func = (
        build_context_from_biotite
        if entry == "build_context"
        else pose_stack_from_biotite
    )

    with pytest.raises(_Stop):
        func(structure, torch_device, prepare_ligands=True)
    assert captured.get("sample_proton_chi") is True


def test_sample_proton_chi_integrated_pose_build_behavior(torch_device):
    # End-to-end behavior of the gate through the integrated pose-build path
    # (forwarding alone cannot prove this):
    #   - sample_proton_chi=False: the pose builds with finite ligand
    #     coordinates, and the prepared LG1 residue carries torsions but no
    #     chi_samples;
    #   - sample_proton_chi=True: the full pose build remains finite and the
    #     prepared LG1 residue gains proton chi_samples.
    import pathlib

    import biotite.structure
    import biotite.structure.io

    from tmol.database import ParameterDatabase
    from tmol.ligand.registry import clear_cache

    cif_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "data"
        / "protein_ligand_test"
        / "cif_inputs"
        / "ace.ligand.cif"
    )
    bt_struct = biotite.structure.io.load_structure(
        str(cif_path), model=1, include_bonds=True, extra_fields=["partial_charge"]
    )
    if isinstance(bt_struct, biotite.structure.AtomArrayStack):
        bt_struct = bt_struct[0]

    def _lg1(context):
        return next(
            rt
            for rt in context.parameter_database.chemical.residues
            if rt.name == "LG1"
        )

    # Explicit opt-out: a full pose builds NaN-free; LG1 has torsions, no
    # chi_samples.
    clear_cache()
    pose_stack, context = pose_stack_from_biotite(
        bt_struct,
        torch_device,
        prepare_ligands=True,
        sample_proton_chi=False,
        param_db=ParameterDatabase.get_default(),
        return_context=True,
    )
    assert torch.isfinite(pose_stack.coords[pose_stack.real_atoms]).all()
    lg1_default = _lg1(context)
    assert lg1_default.torsions  # heavy + proton-chi torsions always emitted
    assert lg1_default.chi_samples == ()  # explicit opt-out suppresses samples

    # Default-on: the full pose remains finite and the prepared LG1 residue
    # carries proton chi_samples.
    clear_cache()
    pose_on, context_on = pose_stack_from_biotite(
        bt_struct,
        torch_device,
        prepare_ligands=True,
        param_db=ParameterDatabase.get_default(),
        return_context=True,
    )
    assert torch.isfinite(pose_on.coords[pose_on.real_atoms]).all()
    lg1_on = _lg1(context_on)
    assert lg1_on.torsions
    assert lg1_on.chi_samples

    clear_cache()


def test_sample_proton_chi_ligand_build_from_mol2(torch_device):
    # Parallel to the CIF-source test above, but sources LG1 from the Tripos
    # mol2 (ace.lig.mol2). The mol2 encodes the carboxylates correctly (O.co2 /
    # C.2 sybyl types => C(=O)[O-]), whereas ace.ligand.cif declares those C-O
    # bonds as SING/SING and over-protonates the carboxyls. Both go through the
    # same unified build; the mol2's correct bonds must not yield hydroxyl H on
    # the carboxylate oxygens.
    import pathlib

    from tmol.database import ParameterDatabase
    from tmol.ligand.detect import nonstandard_residue_info_from_mol2
    from tmol.ligand.registry import clear_cache

    mol2_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "data"
        / "protein_ligand_test"
        / "ace.lig.mol2"
    )
    # Reuse the mol2 reader for its correct bond orders, then run the same
    # biotite build path as the CIF test (which re-derives charges via MMFF).
    bt_struct = nonstandard_residue_info_from_mol2(
        str(mol2_path), res_name="LG1"
    ).atom_array

    clear_cache()
    pose_on, context_on = pose_stack_from_biotite(
        bt_struct,
        torch_device,
        prepare_ligands=True,
        param_db=ParameterDatabase.get_default(),
        return_context=True,
    )
    assert torch.isfinite(pose_on.coords[pose_on.real_atoms]).all()

    lg1 = next(
        rt for rt in context_on.parameter_database.chemical.residues if rt.name == "LG1"
    )
    # No hydrogen bonded to any oxygen: this ligand has only carboxylate/amide/
    # amine chemistry (no genuine hydroxyls), so any H-O bond is spurious
    # carboxylate over-protonation.
    chem = context_on.parameter_database.chemical
    element_of_type = {at.name: at.element for at in chem.atom_types}
    type_of_atom = {a.name: a.atom_type for a in lg1.atoms}

    def _element(name):
        return element_of_type[type_of_atom[name]]

    h_on_o = [
        (b[0], b[1])
        for b in lg1.bonds
        if {_element(b[0]), _element(b[1])} == {"H", "O"}
    ]
    assert not h_on_o, f"spurious hydroxyl H (carboxylate over-protonation): {h_on_o}"
    clear_cache()
