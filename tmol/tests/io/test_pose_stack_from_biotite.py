import warnings

import numpy
import biotite.structure
from biotite.structure.io.pdbx import CIFFile, set_structure
import pytest
import torch

from tmol.io import (
    atom_array_from_cif,
    build_context_from_biotite,
    canonical_form_from_biotite,
    pose_stack_from_biotite,
    pose_stack_from_cif,
    biotite_from_pose_stack,
)
from tmol.io._pose_stack_from_biotite import _renumbered_for_cif
from tmol.tests.data import data_path, load_cif

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


def _score_and_gradient(pose_stack, device):
    from tmol import beta2016_score_function

    coords = pose_stack.coords.detach().clone().requires_grad_(True)
    scorer = beta2016_score_function(device).render_whole_pose_scoring_module(
        pose_stack
    )
    score = scorer(coords)
    score.sum().backward()
    return score.detach(), coords.grad


@pytest.mark.parametrize("hydrogen_offset", [0.0, 0.173])
def test_default_cif_roundtrip_preserves_authored_hydrogens_and_score(
    biotite_1ubq, tmp_path, torch_device, hydrogen_offset
):
    starts = biotite.structure.get_residue_starts(biotite_1ubq)
    initial = pose_stack_from_biotite(
        biotite_1ubq[: starts[3]], torch_device=torch_device
    )
    authored = biotite_from_pose_stack(initial)
    hydrogens = authored.element == "H"
    assert hydrogens.any()
    authored.coord[hydrogens] += numpy.array(
        [hydrogen_offset, -0.119 * bool(hydrogen_offset), 0.087 * bool(hydrogen_offset)]
    )
    authored.coord[:] = numpy.round(authored.coord, 3)

    before = pose_stack_from_biotite(authored, torch_device=torch_device)
    before_array = biotite_from_pose_stack(before)
    numpy.testing.assert_array_equal(before_array.atom_name, authored.atom_name)
    numpy.testing.assert_array_equal(before_array.coord, authored.coord)

    cif = CIFFile()
    set_structure(cif, before_array)
    path = tmp_path / "hydrogen-roundtrip.cif"
    cif.write(path)
    parsed = atom_array_from_cif(path)
    after = pose_stack_from_cif(path, torch_device)
    after_array = biotite_from_pose_stack(after)

    numpy.testing.assert_array_equal(parsed.atom_name, before_array.atom_name)
    numpy.testing.assert_array_equal(after_array.atom_name, before_array.atom_name)
    numpy.testing.assert_array_equal(after_array.element, before_array.element)
    numpy.testing.assert_allclose(after_array.coord, before_array.coord, atol=5e-4)
    torch.testing.assert_close(after.coords, before.coords, rtol=0, atol=5e-4)

    before_score, before_grad = _score_and_gradient(before, torch_device)
    after_score, after_grad = _score_and_gradient(after, torch_device)
    torch.testing.assert_close(after_score, before_score, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(after_grad, before_grad, rtol=1e-5, atol=1e-5)


def test_default_cif_roundtrip_preserves_histidine_tautomer_hydrogen(
    tmp_path, torch_device
):
    from biotite.structure import info

    authored = info.residue("HIS")
    authored.chain_id[:] = "A"
    authored.res_id[:] = 1
    authored = authored[authored.atom_name != "HE2"]
    hd1 = authored.atom_name == "HD1"
    assert hd1.sum() == 1
    authored.coord[hd1] += numpy.array([0.173, -0.119, 0.087])
    authored.coord[:] = numpy.round(authored.coord, 3)

    before = pose_stack_from_biotite(authored, torch_device)
    before_array = biotite_from_pose_stack(before)
    assert "HD1" in before_array.atom_name
    assert "HE2" not in before_array.atom_name
    numpy.testing.assert_array_equal(
        before_array.coord[before_array.atom_name == "HD1"], authored.coord[hd1]
    )

    cif = CIFFile()
    set_structure(cif, before_array)
    path = tmp_path / "histidine-roundtrip.cif"
    cif.write(path)
    after = pose_stack_from_cif(path, torch_device)
    after_array = biotite_from_pose_stack(after)
    assert "HD1" in after_array.atom_name
    assert "HE2" not in after_array.atom_name
    numpy.testing.assert_allclose(
        after_array.coord[after_array.atom_name == "HD1"],
        before_array.coord[before_array.atom_name == "HD1"],
        atol=5e-4,
    )


def test_build_context_from_biotite_smoke(biotite_1ubq, torch_device):
    context = build_context_from_biotite(biotite_1ubq, torch_device=torch_device)
    assert context.packed_block_types.device.type == torch_device.type


def test_canonical_form_from_biotite_smoke(biotite_1r21, torch_device):
    canonical_form_from_biotite(biotite_1r21, torch_device=torch_device)


def test_pose_stack_from_biotite_1ubq_smoke(biotite_1ubq, torch_device):
    pose_stack_from_biotite(biotite_1ubq, torch_device=torch_device)


def test_complete_protein_skips_na_sampler_setup(
    biotite_1ubq, torch_device, monkeypatch
):
    from tmol.pack.rotamer import NaChiRotamerSampler

    def unexpected_na_sampler(*_args, **_kwargs):
        raise AssertionError("NA sampler is unnecessary for a complete protein")

    monkeypatch.setattr(NaChiRotamerSampler, "from_database", unexpected_na_sampler)
    pose_stack_from_biotite(biotite_1ubq, torch_device=torch_device)


def test_default_pose_builds_reuse_packing_setup(biotite_1ubq, torch_device):
    torch.manual_seed(0)
    first, context = pose_stack_from_biotite(
        biotite_1ubq,
        torch_device=torch_device,
        return_context=True,
    )
    score_function = context._packing_score_function
    dunbrack_sampler = context._dunbrack_sampler

    torch.manual_seed(0)
    second, second_context = pose_stack_from_biotite(
        biotite_1ubq,
        torch_device=torch_device,
        return_context=True,
    )

    assert second_context is context
    assert context._packing_score_function is score_function
    assert context._dunbrack_sampler is dunbrack_sampler
    assert torch.equal(torch.isnan(first.coords), torch.isnan(second.coords))
    finite = torch.isfinite(first.coords) & torch.isfinite(second.coords)
    assert torch.equal(first.coords[finite], second.coords[finite])


def test_standard_only_ligand_preparation_reuses_default_context(
    biotite_1ubq, torch_device
):
    default_context = build_context_from_biotite(
        biotite_1ubq, torch_device=torch_device
    )
    ligand_aware_context = build_context_from_biotite(
        biotite_1ubq,
        torch_device=torch_device,
        prepare_ligands=True,
    )

    assert ligand_aware_context is default_context


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


@pytest.mark.parametrize("n_models", [1, 3, 23])
def test_pose_stack_from_and_to_biotite_multiple_poses(
    biotite_1r21, torch_device, n_models
):
    from tmol import beta2016_score_function

    source = biotite_1r21[:n_models]
    canonical = canonical_form_from_biotite(source, torch_device)
    single = canonical_form_from_biotite(source[0], torch_device)
    torch.testing.assert_close(canonical.coords[0], single.coords[0], equal_nan=True)
    pose = pose_stack_from_biotite(source, torch_device, no_optH=True)
    restored = biotite_from_pose_stack(pose)
    restored_coords = restored.coord
    if restored_coords.ndim == 2:
        restored_coords = restored_coords[None]
    assert restored_coords.shape[0] == n_models
    coords = pose.coords.detach().clone().requires_grad_()
    scores = beta2016_score_function(torch_device).render_whole_pose_scoring_module(
        pose
    )(coords)
    scores.sum().backward()
    assert torch.isfinite(scores).all()
    assert torch.isfinite(coords.grad).all()


def test_canonical_form_multipose_metadata_propagation(biotite_1r21, torch_device):
    import attr
    from tmol.io._pose_stack_from_biotite import (
        biotite_from_canonical_form,
        canonical_ordering_for_biotite,
    )

    cf = canonical_form_from_biotite(biotite_1r21, torch_device=torch_device)
    co = canonical_ordering_for_biotite()
    assert cf.atom_b_factor is not None
    assert cf.atom_occupancy is not None
    assert cf.atom_b_factor.shape[0] == biotite_1r21.stack_depth()
    assert cf.atom_occupancy.shape[0] == biotite_1r21.stack_depth()
    assert float(cf.atom_b_factor[1].sum()) > 0.0
    assert float(cf.atom_occupancy[1].sum()) > 0.0
    cf.residue_insertion_codes[:, 0] = "A"
    cf.chain_labels[:] = "long_author_chain"
    first_type = co.restype_io_equiv_classes[int(cf.res_types[0, 0])]
    missing = co.restypes_atom_index_mapping[first_type]["CB"]
    cf.coords[0, 0, missing] = float("nan")
    cf.coords.requires_grad_()
    restored = biotite_from_canonical_form(cf, co)
    assert restored.stack_depth() == biotite_1r21.stack_depth()
    assert set(restored.chain_id) == {"long_author_chain"}
    assert restored.ins_code[0] == "A"
    missing_row = numpy.flatnonzero(
        (restored.res_id == restored.res_id[0]) & (restored.atom_name == "CB")
    )[0]
    assert numpy.isnan(restored.coord[0, missing_row]).all()
    numpy.testing.assert_array_equal(
        restored.coord[1, missing_row], cf.coords[1, 0, missing].detach().cpu().numpy()
    )
    canonical = canonical_form_from_biotite(restored, torch_device, co=co)
    numpy.testing.assert_array_equal(
        canonical.residue_insertion_codes, cf.residue_insertion_codes
    )
    numpy.testing.assert_array_equal(canonical.chain_labels, cf.chain_labels)
    torch.testing.assert_close(canonical.coords, cf.coords, equal_nan=True)

    # An AtomArrayStack has shared annotations; unequal models cannot be exported
    # without losing their occupancy or B-factor information.
    for field in ("atom_b_factor", "atom_occupancy"):
        changed = getattr(cf, field).copy()
        changed[1, 0, 0] += 1
        with pytest.raises(ValueError, match="different metadata"):
            biotite_from_canonical_form(attr.evolve(cf, **{field: changed}), co)
    empty = attr.evolve(cf, coords=torch.full_like(cf.coords, float("nan")))
    empty_array = biotite_from_canonical_form(empty, co)
    assert empty_array.array_length() == 0
    empty_canonical = canonical_form_from_biotite(empty_array, torch_device, co=co)
    assert empty_canonical.res_types.shape == (biotite_1r21.stack_depth(), 0)


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


def test_ligand_proton_chi_samples_build_finite_coords(torch_device):
    # Proton-chi samples are always emitted, and a pose built from a ligand
    # that carries them has finite coordinates: the sampled hydrogens are not
    # left as unbuilt DOFs.
    import pathlib

    import biotite.structure
    import biotite.structure.io

    from tmol.database import ParameterDatabase

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

    carbon = numpy.flatnonzero(
        (bt_struct.res_name == "LG1") & (bt_struct.element == "C")
    )[0]
    assert "QZ" not in bt_struct.atom_name
    bt_struct.atom_name[carbon] = "QZ"

    # This file supplies the whole molecule under a code of its own, which the
    # component dictionary defines as an unrelated one, so it is taken as given.
    pose_stack, context = pose_stack_from_biotite(
        bt_struct,
        torch_device,
        prepare_ligands=True,
        param_db=ParameterDatabase.get_default(),
        return_context=True,
        use_ccd=False,
    )
    assert torch.isfinite(pose_stack.coords[pose_stack.real_atoms]).all()
    lg1 = next(
        rt for rt in context.parameter_database.chemical.residues if rt.name == "LG1"
    )
    assert lg1.torsions
    assert lg1.chi_samples
    exported = biotite_from_pose_stack(pose_stack, context.canonical_ordering)
    assert exported.element[exported.atom_name == "QZ"].tolist() == ["C"]
    rebuilt = pose_stack_from_biotite(
        exported,
        torch_device,
        context=context,
        no_optH=True,
        trust_hydrogen_names=True,
    )
    torch.testing.assert_close(rebuilt.coords, pose_stack.coords, rtol=0, atol=0)

    from tmol.io import canonical_form_from_pose_stack, pose_stack_from_canonical_form

    canonical = canonical_form_from_pose_stack(context.canonical_ordering, pose_stack)
    canonical.coords.requires_grad_()
    trusted = pose_stack_from_canonical_form(
        context.canonical_ordering,
        context.packed_block_types,
        **canonical.as_dict(),
        trust_hydrogen_names=True,
    )
    trusted.coords.sum().backward()
    observed = torch.isfinite(canonical.coords)
    torch.testing.assert_close(
        canonical.coords.grad[observed], torch.ones_like(canonical.coords[observed])
    )


def test_ligand_build_from_mol2_bond_orders(torch_device):
    # Parallel to the CIF-source test above, but sources LG1 from the Tripos
    # mol2 (ace.lig.mol2). The mol2 encodes the carboxylates correctly (O.co2 /
    # C.2 sybyl types => C(=O)[O-]), whereas ace.ligand.cif declares those C-O
    # bonds as SING/SING and over-protonates the carboxyls. Both go through the
    # same unified build; the mol2's correct bonds must not yield hydroxyl H on
    # the carboxylate oxygens.
    import pathlib

    from tmol.database import ParameterDatabase
    from tmol.ligand import nonstandard_residue_info_from_mol2

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

    pose_on, context_on = pose_stack_from_biotite(
        bt_struct,
        torch_device,
        prepare_ligands=True,
        param_db=ParameterDatabase.get_default(),
        return_context=True,
        # a mol2 supplies the whole molecule; its residue code means nothing
        # outside the file, so the component dictionary must not be consulted
        use_ccd=False,
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


CAP_FIXTURES = ("capped_peptide_ace_nme", "capped_peptide_ace_nh2")


@pytest.mark.parametrize("stem", CAP_FIXTURES)
def test_a_capped_peptide_builds_a_pose_stack(stem, torch_device):
    """A terminal cap is a whole residue type, not a variant of one.

    Its single connection is the chain's, so it carries no terminus patch to
    read a name off of, and for NH2 the residue is too small to frame its own
    hydrogens. Both are resolved against the chain it sits in.
    """
    from tmol.database import ParameterDatabase
    from tmol.io import atom_array_from_cif, pose_stack_from_biotite
    from tmol.tests.data import data_path

    structure = atom_array_from_cif(data_path("ncaa_fixtures") / f"{stem}.cif")
    pose_stack = pose_stack_from_biotite(
        structure,
        torch_device,
        prepare_ligands=True,
        param_db=ParameterDatabase.get_default(),
    )

    assert pose_stack.n_poses == 1
    assert int(pose_stack.max_n_blocks) == 3
    # pose_stack_from_biotite raises on NaN coordinates, so reaching here means
    #    every atom of the cap was placed
    real = pose_stack.block_type_ind64[0] >= 0
    names = [
        pose_stack.packed_block_types.active_block_types[int(i)].name
        for i in pose_stack.block_type_ind64[0][real]
    ]
    assert names[0] == "ACE"
    assert names[-1] == ("NME" if stem.endswith("nme") else "NH2")


def test_an_amide_cap_places_its_hydrogens_in_the_amide_plane(torch_device):
    """NH2's hydrogens have no in-residue reference; the partner supplies it."""
    from tmol.database import ParameterDatabase
    from tmol.io import atom_array_from_cif, pose_stack_from_biotite
    from tmol.io._pose_stack_from_biotite import biotite_from_pose_stack
    from tmol.tests.data import data_path

    structure = atom_array_from_cif(
        data_path("ncaa_fixtures") / "capped_peptide_ace_nh2.cif"
    )
    pose_stack, context = pose_stack_from_biotite(
        structure,
        torch_device,
        prepare_ligands=True,
        param_db=ParameterDatabase.get_default(),
        return_context=True,
    )
    arr = biotite_from_pose_stack(pose_stack, context.canonical_ordering)
    names = [str(n) for n in arr.atom_name]
    resids = [int(r) for r in arr.res_id]

    def coord(resid, name):
        for i, (r, n) in enumerate(zip(resids, names)):
            if r == resid and n == name:
                return arr.coord[i]
        raise AssertionError(f"{name} not found in residue {resid}")

    last = max(resids)
    N, H1, H2 = (coord(last, n) for n in ("N", "H1", "H2"))
    C, CA = (coord(last - 1, n) for n in ("C", "CA"))

    def angle(a, b, c):
        u, v = a - b, c - b
        cos = numpy.dot(u, v) / (numpy.linalg.norm(u) * numpy.linalg.norm(v))
        return numpy.degrees(numpy.arccos(numpy.clip(cos, -1.0, 1.0)))

    def dihedral(p0, p1, p2, p3):
        b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
        b1 = b1 / numpy.linalg.norm(b1)
        v = b0 - numpy.dot(b0, b1) * b1
        w = b2 - numpy.dot(b2, b1) * b1
        return numpy.degrees(
            numpy.arctan2(numpy.dot(numpy.cross(b1, v), w), numpy.dot(v, w))
        )

    assert numpy.linalg.norm(N - H1) == pytest.approx(1.02, abs=0.05)
    assert numpy.linalg.norm(N - H2) == pytest.approx(1.02, abs=0.05)
    for a, b, c in ((H1, N, H2), (C, N, H1), (C, N, H2)):
        assert angle(a, b, c) == pytest.approx(120.0, abs=1.0)

    # trans and cis to the partner's own substituent: what makes it planar
    assert abs(dihedral(CA, C, N, H1)) == pytest.approx(180.0, abs=1.0)
    assert abs(dihedral(CA, C, N, H2)) == pytest.approx(0.0, abs=1.0)


def test_missing_mainchain_coords_are_reported_entry_major():
    """Every absent (pose, mainchain atom) is named, entry by entry."""
    from tmol.io._pose_stack_from_biotite import (
        Atom37MappingError,
        _validate_effective_mainchain_coords,
    )

    coords = torch.zeros((3, 2, 4, 3), dtype=torch.float32)
    entries = ((0, 1, "A:1:N"), (1, 2, "A:2:CA"))
    _validate_effective_mainchain_coords(coords, entries)

    coords[2, 0, 1] = torch.nan
    coords[0, 1, 2] = torch.nan
    coords[2, 1, 2] = torch.inf
    with pytest.raises(Atom37MappingError) as err:
        _validate_effective_mainchain_coords(coords, entries)
    reported = str(err.value)
    assert (
        reported.index("pose=2 residue=A:1:N")
        < reported.index("pose=0 residue=A:2:CA")
        < reported.index("pose=2 residue=A:2:CA")
    )


def test_partly_absent_mainchain_triplets_are_still_missing():
    """A mainchain atom needs all three coordinates, not just one."""
    from tmol.io._pose_stack_from_biotite import (
        Atom37MappingError,
        _validate_effective_mainchain_coords,
    )

    coords = torch.zeros((1, 1, 2, 3), dtype=torch.float32)
    coords[0, 0, 0, 1] = torch.nan
    with pytest.raises(Atom37MappingError, match="pose=0 residue=A:1:N"):
        _validate_effective_mainchain_coords(coords, ((0, 0, "A:1:N"),))


@pytest.mark.parametrize(
    "path",
    [
        ("metal_fixtures", "zn_tetrahedral_3ks3.cif.gz"),
        ("atomworks_regressions", "plp_enzyme_7mkv.cif"),
        ("atomworks_regressions", "isopeptide_2rm9.cif.gz"),
        ("atomworks_regressions", "repeated_glycans_6mub.cif.gz"),
    ],
    ids=lambda p: p[1].split(".")[0],
)
def test_export_rebuilds_the_pose_exactly(path):
    """An exported structure carries every bond its pose needs to be rebuilt."""
    structure = atom_array_from_cif(data_path(*path))
    structure = structure[structure.res_name != "HOH"]
    device = torch.device("cpu")
    pose_stack, context = pose_stack_from_biotite(
        structure, device, prepare_ligands=True, no_optH=True, return_context=True
    )
    exported = biotite_from_pose_stack(pose_stack, context.canonical_ordering)
    rebuilt = pose_stack_from_biotite(exported, device, context=context, no_optH=True)

    torch.testing.assert_close(rebuilt.block_type_ind64, pose_stack.block_type_ind64)
    torch.testing.assert_close(
        rebuilt.inter_residue_connections64, pose_stack.inter_residue_connections64
    )
    torch.testing.assert_close(rebuilt.coords, pose_stack.coords, equal_nan=True)


def _residues(ids, chains, ins=None):
    array = biotite.structure.AtomArray(len(ids))
    array.res_id = numpy.array(ids)
    array.chain_id = numpy.array(chains)
    array.res_name = numpy.array(["ALA"] * len(ids))
    array.atom_name = numpy.array(["CA"] * len(ids))
    if ins is not None:
        array.ins_code = numpy.array(ins)
    return array


def test_cif_export_keeps_numbering_it_can_carry():
    array = _residues(
        [-3, -2, 0, 5, 5, 6, 101], ["A"] * 7, ["", "", "", "", "A", "", ""]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _renumbered_for_cif(array) is array


def test_cif_export_shifts_a_chain_numbered_through_minus_one():
    array = _residues([-3, -1, 0, 1, 7], ["A", "A", "A", "A", "B"])
    with pytest.warns(UserWarning, match="chain A .* residue id of -1"):
        out = _renumbered_for_cif(array)
    assert out.res_id.tolist() == [1, 3, 4, 5, 7]


def test_cif_export_renumbers_a_chain_whose_numbering_decreases():
    array = _residues([10, 11, 11, 5, 3], ["A"] * 5, ["", "", "B", "", ""])
    with pytest.warns(UserWarning, match="chain A .* decreases"):
        out = _renumbered_for_cif(array)
    assert out.res_id.tolist() == [1, 2, 3, 4, 5]
    assert out.ins_code.tolist() == [""] * 5


def _heavy_atom_mask(pose_stack):
    pbt = pose_stack.packed_block_types
    element = {at.name: at.element for at in pbt.chem_db.atom_types}
    mask = torch.zeros(pose_stack.coords.shape[:2], dtype=torch.bool)
    for pose, block in zip(*torch.nonzero(pose_stack.block_type_ind64 >= 0).T.tolist()):
        bt = pbt.active_block_types[pose_stack.block_type_ind64[pose, block]]
        offset = int(pose_stack.block_coord_offset64[pose, block])
        for i, atom in enumerate(bt.atoms):
            mask[pose, offset + i] = element[atom.atom_type] not in ("H", "Vr")
    return mask


@pytest.mark.parametrize("route", ["memory", "cif"])
@pytest.mark.parametrize(
    "path",
    [
        ("atomworks_regressions", "isopeptide_2rm9.cif.gz"),
        ("atomworks_regressions", "repeated_glycans_6mub.cif.gz"),
        ("atomworks_regressions", "plp_enzyme_7mkv.cif"),
        ("atomworks_regressions", "retinyl_lysine_4xxj.cif"),
        ("metal_fixtures", "mg_rna_aptamer_7eoh.cif.gz"),
        ("metal_fixtures", "fe_rubredoxin_30oh.cif.gz"),
        ("metal_fixtures", "heme_myoglobin_5yce.cif.gz"),
        ("metal_fixtures", "sf4_ferredoxin_2fdn.cif.gz"),
        ("metal_fixtures", "sf4_ferredoxin_1fdn.cif.gz"),
        ("covalent_fixtures", "lactam_cyclic_7ag5.cif"),
    ],
    ids=lambda p: p[1].split(".")[0],
)
@pytest.mark.filterwarnings("ignore:Renumbering chain")
def test_export_rebuilds_without_its_context(path, route, tmp_path):
    """An exported structure alone carries the chemistry to rebuild its pose."""
    structure = atom_array_from_cif(data_path(*path))
    structure = structure[structure.res_name != "HOH"]
    device = torch.device("cpu")
    pose_stack, context = pose_stack_from_biotite(
        structure, device, prepare_ligands=True, return_context=True
    )
    exported = biotite_from_pose_stack(pose_stack, context.canonical_ordering)
    if route == "cif":
        cif = CIFFile()
        set_structure(cif, exported, include_bonds=True)
        cif.write(str(tmp_path / "out.cif"))
        exported = atom_array_from_cif(str(tmp_path / "out.cif"))
    rebuilt = pose_stack_from_biotite(exported, device, prepare_ligands=True)

    names = [bt.name for bt in pose_stack.packed_block_types.active_block_types]
    rebuilt_names = [bt.name for bt in rebuilt.packed_block_types.active_block_types]
    assert [
        rebuilt_names[i] if i >= 0 else None for i in rebuilt.block_type_ind64[0]
    ] == [names[i] if i >= 0 else None for i in pose_stack.block_type_ind64[0]]
    torch.testing.assert_close(
        rebuilt.inter_residue_connections64, pose_stack.inter_residue_connections64
    )
    heavy = _heavy_atom_mask(pose_stack)
    torch.testing.assert_close(rebuilt.coords[heavy], pose_stack.coords[heavy])
