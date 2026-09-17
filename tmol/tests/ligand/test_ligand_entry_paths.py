"""End-to-end coverage for the single-ligand entry points and params I/O.

Exercises the mol2 and SMILES preparation entry points, the Rosetta/.tmol
params writers and the ``.tmol`` loader (happy path plus its validation
branches), and runs a chemically diverse set of SMILES through the full
detect -> protonate -> 3D mol2 -> atom-typing -> residue-build pipeline so the
typing branches for aromatics, heterocycles, charged groups and ring amidines
are covered by a realistic workflow.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tmol.tests.data import data_path
from tmol.database import ParameterDatabase

DATA = data_path()
MOL2_DIR = DATA / "ligand_test" / "ligand_ground_truth" / "mol2"


def _score_and_minimize_ligand(pose, database):
    import torch
    from tmol.score import beta2016_score_function

    coords = pose.coords.detach().clone().requires_grad_()
    module = beta2016_score_function(
        pose.device, param_db=database
    ).render_whole_pose_scoring_module(pose)
    energy = module(coords).sum()
    assert torch.isfinite(energy)
    assert torch.isfinite(torch.autograd.grad(energy, coords)[0]).all()
    before = float(energy.detach())
    opt = torch.optim.LBFGS([coords], max_iter=10, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = module(coords).sum()
        loss.backward()
        return loss

    opt.step(closure)
    assert torch.isfinite(coords).all()
    assert float(module(coords).sum().detach()) <= before + 1e-3


def _smallest_mol2() -> Path:
    mol2s = sorted(MOL2_DIR.glob("*.mol2"), key=lambda p: p.stat().st_size)
    assert mol2s, "expected ground-truth mol2 fixtures"
    return mol2s[0]


# --------------------------------------------------------------------------- #
# mol2 entry path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "fixture",
    ["er_agonist", "fgfr1", "trypsin", "na", "ammonia", "carbonate", "phosphate"],
)
def test_prepare_ligand_from_mol2_registers_residue(fixture, torch_device, tmp_path):
    import torch
    from tmol.io import pose_stack_from_biotite
    from tmol.ligand import prepare_ligand_from_mol2, write_params_from_mol2
    from tmol.ligand._detect import nonstandard_residue_info_from_mol2

    path = DATA / "protein_ligand_test" / f"{fixture}.lig.mol2"
    if fixture == "ammonia":
        path = tmp_path / "incomplete_hydrogens.mol2"
        path.write_text(
            "@<TRIPOS>MOLECULE\nammonia\n1 0 0 0 0\nSMALL\nUSER_CHARGES\n"
            "@<TRIPOS>ATOM\n1 N 1.0 2.0 3.0 N.3 1 LIG 0.0\n"
            "@<TRIPOS>UNITY_ATOM_ATTR\n1 1\ncharge 0\n"
        )
    if fixture == "carbonate":
        path = tmp_path / "delocalized_carbonate.mol2"
        path.write_text(
            "@<TRIPOS>MOLECULE\ncarbonate\n4 3 0 0 0\nSMALL\nNO_CHARGES\n"
            "@<TRIPOS>ATOM\n1 C 0 0 0 C.2 1 CO3\n2 O1 1.25 0 0 O.co2 1 CO3\n"
            "3 O2 -0.625 1.083 0 O.co2 1 CO3\n4 O3 -0.625 -1.083 0 O.co2 1 CO3\n"
            "@<TRIPOS>UNITY_ATOM_ATTR\n3 1\ncharge -1\n4 1\ncharge -1\n"
            "@<TRIPOS>BOND\n1 1 2 ar\n2 1 3 ar\n3 1 4 ar\n"
        )
    if fixture == "phosphate":
        path = tmp_path / "delocalized_phosphate.mol2"
        path.write_text(
            "@<TRIPOS>MOLECULE\nethylphosphate\n7 6 0 0 0\nSMALL\nNO_CHARGES\n"
            "@<TRIPOS>ATOM\n1 C1 -3 0 0 C.3 1 LIG\n2 C2 -1.5 0 0 C.3 1 LIG\n"
            "3 O -0.5 1 0 O.3 1 LIG\n4 P 1 1 0 P.3 1 LIG\n"
            "5 O1 1.5 2.4 0 O.co2 1 LIG\n6 O2 1.5 0 1 O.co2 1 LIG\n"
            "7 O3 1.5 0 -1 O.co2 1 LIG\n"
            "@<TRIPOS>UNITY_ATOM_ATTR\n6 1\ncharge -1\n7 1\ncharge -1\n"
            "@<TRIPOS>BOND\n1 1 2 1\n2 2 3 1\n3 3 4 1\n"
            "4 4 5 ar\n5 4 6 ar\n6 4 7 ar\n"
        )
    if fixture == "er_agonist":
        lines = path.read_text().splitlines()
        renamed = iter(("long_carbon_label", "long_carbon_label", "long_carbon_label2"))
        section = ""
        for i, line in enumerate(lines):
            fields = line.split()
            if line.startswith("@<TRIPOS>"):
                section = line
            elif (
                section == "@<TRIPOS>ATOM"
                and len(fields) >= 6
                and fields[5].startswith("C.")
            ):
                replacement = next(renamed, None)
                if replacement is not None:
                    fields[1] = replacement
                    lines[i] = " ".join(fields)
        path = tmp_path / "colliding_names.mol2"
        path.write_text("\n".join(lines) + "\n")
    info = nonstandard_residue_info_from_mol2(path, res_name="LG1")
    if fixture == "phosphate":
        assert info.atom_array.charge.sum() == -2
        phosphorus = int((info.atom_array.element == "P").nonzero()[0][0])
        _, orders = info.atom_array.bonds.get_bonds(phosphorus)
        assert sorted(orders) == [1, 1, 1, 2]
    if fixture == "carbonate":
        assert info.atom_array.charge.tolist() == [0, 0, -1, -1]
        assert sorted(info.atom_array.bonds.as_array()[:, 2]) == [1, 1, 2]
    if fixture == "na":
        assert info.atom_array.charge[info.atom_array.atom_name == "NH2"] == 1
        assert info.atom_array.charge.sum() == 0
    param_db, ordering = prepare_ligand_from_mol2(path, res_name="LG1", seed=17)
    assert len(set(info.atom_names)) == len(info.atom_names)
    names = set(ordering.restypes_ordered_atom_names["LG1"])
    assert set(info.atom_array.atom_name[info.atom_array.element != "H"]) <= names
    if info.skip_protonation:
        charges = {
            p.atom: p.charge
            for p in param_db.scoring.elec.atom_charge_parameters
            if p.res == "LG1"
        }
        for atom in info.atom_array[info.atom_array.element != "H"]:
            assert charges[atom.atom_name] == info.partial_charges[atom.atom_name]
    pose = pose_stack_from_biotite(
        info.atom_array, torch_device, param_db=param_db, no_optH=True
    )
    if fixture == "ammonia":
        bt = pose.packed_block_types.active_block_types[int(pose.block_type_ind[0, 0])]
        nitrogen = pose.coords[0, bt.atom_to_idx["N"]]
        torch.testing.assert_close(
            nitrogen, nitrogen.new_tensor([1, 2, 3]), rtol=0, atol=0
        )
        hydrogen = pose.coords[
            0, [i for i, atom in enumerate(bt.atoms) if atom.name != "N"]
        ]
        assert hydrogen.shape == (3, 3)
        lengths = (hydrogen - nitrogen).norm(dim=-1)
        assert torch.all((lengths > 0.9) & (lengths < 1.2))
    _score_and_minimize_ligand(pose, param_db)
    output = tmp_path / "ligand.tmol"
    write_params_from_mol2(path, output, res_name="LG1", seed=17)
    assert output.is_file()
    if fixture == "na":
        # Inferred guanidinium charge may repair this old Tripos file, but a
        # contradictory explicitly declared charge must not be overwritten.
        contradictory = tmp_path / "declared_neutral_guanidinium.mol2"
        contradictory.write_text(
            path.read_text() + "\n@<TRIPOS>UNITY_ATOM_ATTR\n23 1\ncharge 0\n"
        )
        with pytest.raises(ValueError, match="sanitizable chemical graph"):
            prepare_ligand_from_mol2(contradictory)


def test_partially_protonated_atom_array_builds_complete_ligand(torch_device):
    import numpy as np
    import torch

    from tmol.io import pose_stack_from_biotite
    from tmol.ligand._detect import nonstandard_residue_info_from_mol2

    source = nonstandard_residue_info_from_mol2(
        DATA / "protein_ligand_test" / "ace.lig.mol2", res_name="LG1"
    ).atom_array
    hydrogen_indices = np.flatnonzero(source.element == "H")
    assert len(hydrogen_indices) > 1
    partial = source[np.arange(len(source)) != hydrogen_indices[0]]
    partial_hydrogens = int(np.count_nonzero(partial.element == "H"))

    pose, context = pose_stack_from_biotite(
        partial,
        torch_device,
        prepare_ligands=True,
        param_db=ParameterDatabase.get_default(),
        return_context=True,
        use_ccd=False,
        ligand_seed=17,
    )
    residue = next(
        residue
        for residue in context.parameter_database.chemical.residues
        if residue.name == "LG1"
    )
    elements = {
        atom_type.name: atom_type.element
        for atom_type in context.parameter_database.chemical.atom_types
    }
    prepared_hydrogens = sum(elements[atom.atom_type] == "H" for atom in residue.atoms)
    assert prepared_hydrogens > partial_hydrogens
    assert torch.isfinite(pose.coords[pose.real_atoms]).all()
    block_type = pose.packed_block_types.active_block_types[
        int(pose.block_type_ind[0, 0])
    ]
    assert block_type.n_atoms == len(residue.atoms)


@pytest.mark.parametrize(
    "smiles, expected_charge",
    [("CCOP(=O)(O)O", -2), ("O=P(O)(O)CC(=O)NCCNC(=O)CP(=O)(O)O", -4)],
    ids=["phosphate", "2fzc_eop_phosphonate"],
)
def test_neutralized_mol2_protonation_workflow(
    smiles, expected_charge, torch_device, tmp_path
):
    import torch
    from atomworks.io.utils.io_utils import to_cif_file
    from tmol.io import pose_stack_from_biotite
    from tmol.ligand import (
        load_params_file,
        prepare_ligand_from_cif,
        prepare_ligand_from_mol2,
        write_params_from_mol2,
    )
    from tmol.ligand._detect import nonstandard_residue_info_from_mol2
    from tmol.ligand._openbabel_compat import _build_charged_3d_mol2_mol

    # Reproduce neutralized phosphate inputs, including the EOP chemistry in
    # 2FZC; this is not the unavailable original PDBbind addH MOL2 file.
    path = tmp_path / "neutralized_addH.mol2"
    path.write_text(_build_charged_3d_mol2_mol(smiles, seed=17).write("mol2"))
    source = nonstandard_residue_info_from_mol2(path, res_name="LG1")
    assert source.skip_protonation  # Auto preserves complete, prepared input.
    assert source.atom_array.charge.sum() == 0
    assert sum(source.partial_charges.values()) == pytest.approx(0, abs=2e-4)
    heavy = source.atom_array[source.atom_array.element != "H"]
    cif_path = tmp_path / "neutralized.cif"
    to_cif_file(source.atom_array, cif_path, ccd_entries={"LG1": source.atom_array})
    cif_database, _ = prepare_ligand_from_cif(cif_path, res_name="LG1", seed=17)

    damaged = tmp_path / "untrusted_addH.mol2"
    lines = path.read_text().splitlines()
    lines[4] = "MULLIKEN_CHARGES"
    damaged.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="authoritative partial charges"):
        prepare_ligand_from_mol2(damaged, mode="keep")
    with pytest.raises(ValueError, match="MOL2 mode"):
        prepare_ligand_from_mol2(path, mode="unknown")
    for input_path, options, charge in (
        (path, {"mode": "auto"}, 0),
        (path, {"mode": "keep"}, 0),
        (path, {"mode": "regenerate"}, expected_charge),
        (damaged, {}, expected_charge),
    ):
        database, _ = prepare_ligand_from_mol2(
            input_path, res_name="LG1", seed=17, **options
        )
        charges = {
            row.atom: row.charge
            for row in database.scoring.elec.atom_charge_parameters
            if row.res == "LG1"
        }
        assert sum(charges.values()) == pytest.approx(charge, abs=2e-4)
        if options.get("mode") == "regenerate":
            assert charges == {
                row.atom: row.charge
                for row in cif_database.scoring.elec.atom_charge_parameters
                if row.res == "LG1"
            }
            assert next(
                r for r in database.chemical.residues if r.name == "LG1"
            ) == next(r for r in cif_database.chemical.residues if r.name == "LG1")
        if charge == 0:
            assert sorted(charges.values()) == sorted(source.partial_charges.values())
            for name in heavy.atom_name:
                assert charges[name] == source.partial_charges[name]
        # Generated ligand types rebuild source H by default, including names
        # that now belong to a different hydrogen after protonation.
        pose = pose_stack_from_biotite(
            source.atom_array, torch_device, param_db=database, no_optH=True
        )
        bt = pose.packed_block_types.active_block_types[int(pose.block_type_ind[0, 0])]
        assert bt.n_atoms == len(source.atom_names) + charge
        phosphate_oxygens = set()
        for index in (source.atom_array.element == "P").nonzero()[0]:
            neighbors, _ = source.atom_array.bonds.get_bonds(index)
            phosphate_oxygens.update(
                source.atom_array.atom_name[i]
                for i in neighbors
                if source.atom_array.element[i] == "O"
            )
        heavy_names = set(heavy.atom_name)
        phosphate_h = sum(
            (a in phosphate_oxygens and b not in heavy_names)
            or (b in phosphate_oxygens and a not in heavy_names)
            for a, b, *_ in bt.bonds
        )
        assert phosphate_h == (0 if charge else -expected_charge)
        observed = pose.coords[0, [bt.atom_to_idx[name] for name in heavy.atom_name]]
        torch.testing.assert_close(
            observed, observed.new_tensor(heavy.coord), atol=0, rtol=0
        )
        _score_and_minimize_ligand(pose, database)
        output = tmp_path / "ligand.tmol"
        write_params_from_mol2(input_path, output, res_name="LG1", seed=17, **options)
        loaded = load_params_file(output)[0]
        assert loaded.partial_charges == charges
        assert set(loaded.partial_charges) == {atom.name for atom in bt.atoms}


def test_mol2_rejects_undefined_bonds_and_unparameterized_chemistry(tmp_path):
    from tmol.ligand import prepare_ligand_from_mol2, write_params_from_mol2

    methane = tmp_path / "methane.mol2"
    methane.write_text(
        "@<TRIPOS>MOLECULE\nmethane\n1 0 0 0 0\nSMALL\nUSER_CHARGES\n"
        "@<TRIPOS>ATOM\n1 C 0.0 0.0 0.0 C.3 1 LIG 0.0\n"
    )
    for operation in (
        lambda: prepare_ligand_from_mol2(methane),
        lambda: write_params_from_mol2(methane, tmp_path / "methane.tmol"),
    ):
        with pytest.raises(ValueError, match="no CS4 atom type"):
            operation()
    for fixture, message in (
        ("fxa", "Undefined MOL2 bond order 'un'"),
        ("parp", "do not define a sanitizable chemical graph"),
    ):
        path = DATA / "protein_ligand_test" / f"{fixture}.lig.mol2"
        with pytest.raises(ValueError, match=message):
            prepare_ligand_from_mol2(path)
        with pytest.raises(ValueError, match=message):
            write_params_from_mol2(path, tmp_path / "ligand.tmol")


def test_prepare_ligand_from_cif_preserves_selected_model(torch_device, tmp_path):
    import gzip
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx
    import numpy as np
    import torch
    from tmol.io import pose_stack_from_biotite
    from tmol.ligand import _ligand_info_from_cif, prepare_ligand_from_cif

    cif = pdbx.CIFFile.read(DATA / "ligand_cif_fixtures" / "vww.bonds_present.cif")
    original = pdbx.get_structure(cif, model=1, include_bonds=True)
    shifted = original.copy()
    shifted.coord += 100
    pdbx.set_structure(cif, struc.stack([original, shifted]))
    path = tmp_path / "two_models.cif.gz"
    with gzip.open(path, "wt") as stream:
        cif.write(stream)
    info = _ligand_info_from_cif(path, "LONG_LIGAND")
    assert info.atom_names == tuple(original.atom_name)
    assert np.all(info.atom_array.res_name == "LONG_LIGAND")
    np.testing.assert_array_equal(info.coords, original.coord)
    database, ordering = prepare_ligand_from_cif(path, res_name="LONG_LIGAND")
    assert set(info.atom_names) <= set(
        ordering.restypes_ordered_atom_names["LONG_LIGAND"]
    )
    pose = pose_stack_from_biotite(
        info.atom_array, torch_device, param_db=database, no_optH=True
    )
    bt = pose.packed_block_types.active_block_types[int(pose.block_type_ind[0, 0])]
    observed = pose.coords[0, [bt.atom_to_idx[name] for name in info.atom_names]]
    torch.testing.assert_close(
        observed, observed.new_tensor(original.coord), atol=0, rtol=0
    )
    _score_and_minimize_ligand(pose, database)
    shifted.res_id += 1
    pdbx.set_structure(cif, struc.concatenate([original, shifted]))
    multiple = tmp_path / "two_residues.cif"
    cif.write(multiple)
    with pytest.raises(ValueError, match="exactly one residue"):
        prepare_ligand_from_cif(multiple)


def test_nonstandard_residue_info_from_mol2_block_roundtrips() -> None:
    from tmol.ligand import (
        nonstandard_residue_info_from_mol2,
        nonstandard_residue_info_from_mol2_block,
    )

    mol2_path = _smallest_mol2()
    from_file = nonstandard_residue_info_from_mol2(mol2_path, res_name="LG1")
    from_block = nonstandard_residue_info_from_mol2_block(
        mol2_path.read_text(), res_name="LG1"
    )
    assert from_file.atom_names == from_block.atom_names
    assert from_file.elements == from_block.elements


# --------------------------------------------------------------------------- #
# write_params_from_mol2 + params round-trip + .tmol loader branches
# --------------------------------------------------------------------------- #
def _single_prep():
    from tmol.ligand import nonstandard_residue_info_from_mol2
    from tmol.ligand import prepare_single_ligand

    info = nonstandard_residue_info_from_mol2(_smallest_mol2(), res_name="LG1")
    return prepare_single_ligand(info)


def test_write_params_from_mol2_both_formats(tmp_path) -> None:
    from tmol.ligand import write_params_from_mol2

    rosetta = tmp_path / "lig.params"
    write_params_from_mol2(str(_smallest_mol2()), str(rosetta), res_name="LG1")
    assert rosetta.exists() and rosetta.stat().st_size > 0

    tmol_out = tmp_path / "lig.tmol"
    write_params_from_mol2(str(_smallest_mol2()), str(tmol_out), res_name="LG1")
    assert tmol_out.exists() and tmol_out.stat().st_size > 0


def test_tmol_params_roundtrip_and_inject(tmp_path) -> None:
    from tmol.ligand import (
        inject_params_file,
        inject_params_files,
        load_params_file,
    )
    from tmol.ligand import write_params_file

    prep = _single_prep()
    tmol_file = tmp_path / "lig.tmol"
    write_params_file([prep], str(tmol_file))

    loaded = load_params_file(tmol_file)
    assert len(loaded) == 1
    assert loaded[0].residue_type.name == prep.residue_type.name
    loaded[0].partial_charges.clear()
    assert load_params_file(tmol_file)[0].partial_charges

    injected = inject_params_file(ParameterDatabase.get_default(), tmol_file)
    assert any(r.name == prep.residue_type.name for r in injected.chemical.residues)

    injected_multi = inject_params_files(ParameterDatabase.get_default(), [tmol_file])
    assert any(
        r.name == prep.residue_type.name for r in injected_multi.chemical.residues
    )


def test_tmol_loader_accepts_minor_version_difference(tmp_path) -> None:
    from dataclasses import replace
    import yaml
    from tmol.ligand import load_params_file
    from tmol.ligand import write_params_file

    prep = replace(_single_prep(), atom_type_elements=None)
    tmol_file = tmp_path / "lig.tmol"
    write_params_file([prep], str(tmol_file))
    load_params_file(tmol_file)

    payload = yaml.safe_load(tmol_file.read_text())
    payload["version"] = "1.9"
    tmol_file.write_text(yaml.safe_dump(payload))
    loaded = load_params_file(tmol_file)
    assert loaded and loaded[0].residue_type.name == prep.residue_type.name


def test_tmol_loader_warns_when_no_charges(tmp_path) -> None:
    import yaml

    from tmol.ligand import load_params_file
    from tmol.ligand import write_params_file

    prep = _single_prep()
    tmol_file = tmp_path / "lig.tmol"
    write_params_file([prep], str(tmol_file))

    doc = yaml.safe_load(tmol_file.read_text())
    doc["elec"] = {"atom_charge_parameters": []}
    tmol_file.write_text(yaml.safe_dump(doc))

    loaded = load_params_file(tmol_file)
    assert loaded
    assert all(c == 0.0 for c in loaded[0].partial_charges.values()) or (
        loaded[0].partial_charges == {}
    )


@pytest.mark.parametrize(
    "content, match",
    [
        ("- a\n- b\n", "Expected mapping"),
        ("chemical: {}\n", "no 'version' field"),
        ('version: "99.0"\nchemical: {}\n', "incompatible"),
        ('version: "1.0"\nresidues: []\n', "deprecated flat schema"),
    ],
)
def test_tmol_loader_rejects_bad_files(tmp_path, content, match) -> None:
    from tmol.ligand import load_params_file

    bad = tmp_path / "bad.tmol"
    bad.write_text(content)
    with pytest.raises(ValueError, match=match):
        load_params_file(bad)


# --------------------------------------------------------------------------- #
# SMILES entry path over a chemically diverse set (atom-typing breadth)
# --------------------------------------------------------------------------- #
_DIVERSE_SMILES = {
    "benzene": "c1ccccc1",
    "pyridine": "c1ccncc1",
    "pyrrole": "c1cc[nH]c1",
    "furan": "c1ccoc1",
    "thiophene": "c1ccsc1",
    "imidazole": "c1c[nH]cn1",
    "benzoic_acid": "O=C(O)c1ccccc1",
    "benzamide": "O=C(N)c1ccccc1",
    "methanesulfonamide": "CS(=O)(=O)N",
    "nitrobenzene": "O=[N+]([O-])c1ccccc1",
    "aminopyridine": "Nc1ccccn1",
    "trifluoromethylbenzene": "FC(F)(F)c1ccccc1",
    "chlorobromobenzene": "Clc1ccc(Br)cc1",
    "acetanilide": "CC(=O)Nc1ccccc1",
    "cyclopropanecarboxamide": "NC(=O)C1CC1",
    "ethanolamine": "NCCO",
}


@pytest.mark.parametrize("name", sorted(_DIVERSE_SMILES))
def test_prepare_ligand_from_smiles_registers(name: str) -> None:
    """Each diverse ligand prepares and registers via the SMILES path."""
    from tmol.ligand import prepare_ligand_from_smiles

    smiles = _DIVERSE_SMILES[name]
    param_db, _ = prepare_ligand_from_smiles(
        smiles,
        param_db=ParameterDatabase.get_default(),
        res_name="LG1",
    )
    residue = next((r for r in param_db.chemical.residues if r.name == "LG1"), None)
    assert residue is not None, f"{name} ({smiles}) did not register"
    assert len(residue.atoms) > 0


# --------------------------------------------------------------------------- #
# prepare_ligands over real ligand CIFs (detection loop, SMILES candidate
# selection, CIF atom renaming, atom typing breadth, params output)
# --------------------------------------------------------------------------- #
CIF_INPUTS = DATA / "protein_ligand_test" / "cif_inputs"

# A diverse subset of the DUD-derived ligand CIFs: varied ring systems,
# heteroatoms, halogens and charged groups, to exercise the typing/rename
# branches of the full CIF -> params path.
_CIF_LIGANDS = ["ada", "cdk2", "cox2", "hivrt", "src", "egfr"]


def _load_full_array(cif_path: Path):
    from tmol.io import atom_array_from_cif

    # a single-ligand file supplying a whole molecule under a code of its own
    return atom_array_from_cif(cif_path, use_ccd=False)


@pytest.mark.parametrize("name", _CIF_LIGANDS)
def test_prepare_ligand_from_cif_inputs(name: str) -> None:
    """Diverse real ligand CIFs prepare end-to-end via the unified path."""
    from tmol.ligand import prepare_ligand_from_cif

    cif = CIF_INPUTS / f"{name}.ligand.cif"
    param_db, _ = prepare_ligand_from_cif(
        str(cif), param_db=ParameterDatabase.get_default()
    )
    # At least one new (non-canonical) residue should have been registered.
    default_names = {r.name for r in ParameterDatabase.get_default().chemical.residues}
    new_names = [
        r.name for r in param_db.chemical.residues if r.name not in default_names
    ]
    assert new_names, f"{name}: no new ligand residue registered"


def test_prepare_ligands_writes_params_output(tmp_path) -> None:
    """prepare_ligands over a ligand AtomArray writes a reusable .tmol file."""
    from tmol.ligand import prepare_ligands

    arr = _load_full_array(CIF_INPUTS / "ada.ligand.cif")
    out = tmp_path / "out.tmol"
    param_db, ordering = prepare_ligands(
        arr,
        param_db=ParameterDatabase.get_default(),
        params_output=str(out),
        # a ligand file supplying a whole molecule under a code of its own
        use_ccd=False,
    )
    assert out.exists() and out.stat().st_size > 0
    assert ordering is not None


def test_prepare_ligands_accepts_single_model_stack() -> None:
    """An AtomArrayStack with one model is accepted (and default db resolved)."""
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx

    from tmol.ligand import prepare_ligands

    cif = pdbx.CIFFile.read(str(CIF_INPUTS / "ada.ligand.cif"))
    stack = pdbx.get_structure(cif, include_bonds=True, extra_fields=["charge"])
    if not isinstance(stack, struc.AtomArrayStack):
        stack = struc.stack([stack])
    assert len(stack) == 1
    # No param_db passed -> default resolved internally.
    # a ligand file supplying a whole molecule under a code of its own
    param_db, _ = prepare_ligands(stack, use_ccd=False)
    assert param_db is not None


def test_prepare_ligands_rejects_multi_model_stack() -> None:
    """An AtomArrayStack with multiple models is rejected with a clear error."""
    import biotite.structure as struc

    from tmol.ligand import prepare_ligands

    arr = _load_full_array(CIF_INPUTS / "ada.ligand.cif")
    stack = struc.stack([arr, arr])
    with pytest.raises(TypeError, match="single AtomArray"):
        prepare_ligands(stack, param_db=ParameterDatabase.get_default())


def test_prepare_ligands_strict_raises_on_unpreparable_ligand() -> None:
    """A ligand that cannot yield a SMILES raises under strict_ligands=True."""
    from tmol.ligand import LigandPreparationError
    from tmol.ligand import prepare_ligands

    # The parp ligand CIF has bond orders the unified SMILES path can't use, so
    # on-the-fly preparation fails -- strict mode must surface that loudly.
    arr = _load_full_array(CIF_INPUTS / "parp.ligand.cif")
    with pytest.raises(LigandPreparationError):
        prepare_ligands(
            arr, param_db=ParameterDatabase.get_default(), strict_ligands=True
        )


def test_prepare_ligands_lenient_skips_unpreparable_ligand() -> None:
    """The same ligand is skipped with a warning under strict_ligands=False."""
    from tmol.ligand import prepare_ligands

    arr = _load_full_array(CIF_INPUTS / "parp.ligand.cif")
    param_db, ordering = prepare_ligands(
        arr, param_db=ParameterDatabase.get_default(), strict_ligands=False
    )
    # Lenient mode returns a (possibly unchanged) database rather than raising.
    assert param_db is not None
    assert ordering is not None


@pytest.mark.parametrize("strict_ligands", [True, False])
def test_prepare_ligands_honors_lenient_mode_for_ligand_preparation_errors(
    monkeypatch, strict_ligands
) -> None:
    """A LigandPreparationError from a ligand is skipped when not strict.

    Preparation raises this directly when a conjugated residue needs chemistry
    tmol does not support. Lenient mode must drop it like any other unknown
    rather than crash.
    """
    from tmol.ligand import LigandPreparationError, prepare_ligands
    from tmol.ligand import _preparation

    def fail(*args, **kwargs):
        raise LigandPreparationError("unsupported element in conjugated partner")

    monkeypatch.setattr(_preparation, "_prepare_ligand_via_smiles", fail)
    monkeypatch.setattr(_preparation, "prepare_polymer_residue", fail)

    arr = _load_full_array(CIF_INPUTS / "parp.ligand.cif")
    if strict_ligands:
        with pytest.raises(LigandPreparationError):
            prepare_ligands(
                arr, param_db=ParameterDatabase.get_default(), strict_ligands=True
            )
        return
    param_db, ordering = prepare_ligands(
        arr, param_db=ParameterDatabase.get_default(), strict_ligands=False
    )
    assert param_db is not None
    assert ordering is not None


@pytest.mark.parametrize("strict_ligands", [True, False])
def test_prepare_ligands_rejects_incomplete_generated_chemistry(
    monkeypatch, caplog, strict_ligands
) -> None:
    """Authored heavy atoms missing from generated chemistry raise or skip."""
    import logging
    from dataclasses import replace

    import attr

    from tmol.ligand import LigandPreparationError, prepare_ligands
    from tmol.ligand import _preparation

    arr = _load_full_array(CIF_INPUTS / "ada.ligand.cif")
    source_atom_names = arr.atom_name.copy()
    original_prepare = _preparation.prepare_single_ligand

    def omit_c1(*args, **kwargs):
        prep = original_prepare(*args, **kwargs)
        assert any(atom.name == "C1" for atom in prep.residue_type.atoms)
        residue_type = attr.evolve(
            prep.residue_type,
            atoms=tuple(atom for atom in prep.residue_type.atoms if atom.name != "C1"),
        )
        return replace(prep, residue_type=residue_type)

    monkeypatch.setattr(_preparation, "prepare_single_ligand", omit_c1)
    base = ParameterDatabase.get_default()

    if strict_ligands:
        with pytest.raises(LigandPreparationError, match=r"LG1.*C1"):
            prepare_ligands(
                arr, param_db=base, strict_ligands=True, use_ccd=False, seed=1234
            )
    else:
        with caplog.at_level(logging.WARNING, logger=_preparation.__name__):
            prepared, _ = prepare_ligands(
                arr, param_db=base, strict_ligands=False, use_ccd=False, seed=1234
            )
        assert prepared is base
        assert "Skipping LG1" in caplog.text
        assert "C1" in caplog.text

    assert (arr.atom_name == source_atom_names).all()


def test_prepare_ligands_with_params_files_skips_reprep(tmp_path) -> None:
    """A residue supplied via params_files is already known, so no re-prep."""
    from tmol.ligand import nonstandard_residue_info_from_mol2
    from tmol.ligand import write_params_file
    from tmol.ligand import prepare_ligands, prepare_single_ligand

    # Build a .tmol for the mol2 ligand named LG1.
    info = nonstandard_residue_info_from_mol2(_smallest_mol2(), res_name="LG1")
    prep = prepare_single_ligand(info)
    tmol_file = tmp_path / "lg1.tmol"
    write_params_file([prep], str(tmol_file))

    arr = _load_full_array(CIF_INPUTS / "ada.ligand.cif")
    param_db, _ = prepare_ligands(
        arr,
        param_db=ParameterDatabase.get_default(),
        params_files=[str(tmol_file)],
    )
    assert any(r.name == "LG1" for r in param_db.chemical.residues)
