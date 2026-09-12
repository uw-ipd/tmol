"""Prepared attachments must retain their chemistry across params export."""

import attr
from dataclasses import replace
import pytest
import torch
import yaml

from tmol.database import ParameterDatabase
from tmol.io import atom_array_from_cif, pose_stack_from_biotite
from tmol.ligand import prepare_ligands, load_params_file, write_params_file
from tmol.ligand._registry import inject_ligand_preparations
from tmol.score import beta2016_score_function
from tmol.tests.data import data_path
from tmol.tests.pack.test_conjugated_group_packing import FIXTURES


@pytest.mark.parametrize("fixture", sorted(FIXTURES))
@pytest.mark.parametrize("bundle_copies", [1, 2])
def test_conjugate_params_roundtrip(tmp_path, fixture, torch_device, bundle_copies):
    array = atom_array_from_cif(
        data_path("covalent_fixtures", FIXTURES[fixture] + ".cif")
    )
    base = ParameterDatabase.get_default()
    path = tmp_path / "conjugate.tmol"
    prepared, _ = prepare_ligands(
        array, param_db=base, seed=20250828, params_output=str(path)
    )
    # Distinct files with overlapping definitions must coalesce too; simply
    # avoiding repeated reads of one path does not establish that contract.
    paths = [path]
    for i in range(1, bundle_copies):
        other = tmp_path / f"conjugate_{i}.tmol"
        other.write_bytes(path.read_bytes())
        paths.append(other)
    loaded, _ = prepare_ligands(array, param_db=base, params_files=paths)
    assert len(loaded.chemical.residues) == len(prepared.chemical.residues)
    before = {r.name: r for r in prepared.chemical.residues}
    after = {r.name: r for r in loaded.chemical.residues}
    assert before.keys() == after.keys()
    for name, residue in before.items():
        for field in attr.fields(type(residue)):
            assert getattr(residue, field.name) == getattr(after[name], field.name), (
                name,
                field.name,
                getattr(residue, field.name),
                getattr(after[name], field.name),
            )

    def charges(db):
        return {
            (p.res, p.atom): p.charge for p in db.scoring.elec.atom_charge_parameters
        }

    assert charges(prepared) == charges(loaded)
    assert (
        prepared.scoring.cartbonded.residue_params
        == loaded.scoring.cartbonded.residue_params
    )
    poses = [
        pose_stack_from_biotite(array, torch_device, param_db=db, no_optH=True)
        for db in (prepared, loaded)
    ]
    for field in ("coords", "inter_residue_connections", "block_coord_offset"):
        torch.testing.assert_close(getattr(poses[0], field), getattr(poses[1], field))
    names = [
        [
            pose.packed_block_types.active_block_types[t].name
            for t in pose.block_type_ind[0].tolist()
        ]
        for pose in poses
    ]
    assert names[0] == names[1]
    values, gradients = [], []
    for db, pose in zip((prepared, loaded), poses):
        module = beta2016_score_function(
            torch_device, param_db=db
        ).render_whole_pose_scoring_module(pose)
        coords = pose.coords.detach().clone().requires_grad_(True)
        score = module(coords, sum_terms=False, apply_weights=False)
        values.append(score.detach())
        gradients.append(torch.autograd.grad(score.sum(), coords)[0])
    torch.testing.assert_close(values[0], values[1])
    torch.testing.assert_close(gradients[0], gradients[1])

    # The source ligand may already be installed before its attachment bundle.
    preps = load_params_file(path)
    source_names = {p.residue_type.name for p in preps}
    partial = [
        replace(
            p,
            adds_patches=tuple(
                v
                for v in p.adds_patches
                if set(v.applies_to.base_names or ()) <= source_names
            ),
            variant_partial_charges={
                name: charges
                for name, charges in (p.variant_partial_charges or {}).items()
                if name.partition(":")[0] in source_names
            },
            connection_params=(),
        )
        for p in preps
    ]
    existing = inject_ligand_preparations(base, partial)
    enriched = inject_ligand_preparations(existing, preps)
    assert {r.name: r for r in enriched.chemical.residues} == after
    assert charges(enriched) == charges(loaded)
    assert inject_ligand_preparations(enriched, preps) is enriched


@pytest.mark.parametrize("preinstalled", [False, True])
def test_connection_params_bundle_scores_after_reload(
    tmp_path, ubq_pdb, default_database, torch_device, preinstalled
):
    from tmol.tests.ligand.test_ligand_entry_paths import _single_prep
    from tmol.tests.score.common import pose_stack_from_pdb_and_resnums
    from tmol.tests.score.cartbonded.test_explicit_connection_parameters import (
        peptide_record,
        scoring_term,
    )
    from tmol.database import inject_residue_params

    pose = pose_stack_from_pdb_and_resnums(ubq_pdb, torch_device, [(24, 26)])
    record = peptide_record(pose)
    plain = _single_prep()
    prep = replace(plain, connection_params=(record,))
    path = tmp_path / "connections.tmol"
    write_params_file(prep, path, format="tmol")
    loaded = load_params_file(path)
    assert loaded[0].connection_params == (record,)
    assert yaml.safe_load(path.read_text())["version"] == "2.0"
    base = (
        inject_ligand_preparations(default_database, [plain])
        if preinstalled
        else default_database
    )
    db = inject_ligand_preparations(base, loaded)
    reference_db = inject_residue_params(
        default_database, [], connection_params=(record,)
    )
    results = []
    for database in (db, reference_db):
        module = scoring_term(pose, database).render_whole_pose_scoring_module(pose)
        coords = pose.coords.detach().clone().requires_grad_(True)
        score = module(coords)
        results.append((score.detach(), torch.autograd.grad(score.sum(), coords)[0]))
    torch.testing.assert_close(results[0], results[1])
    assert inject_ligand_preparations(db, loaded) is db


def test_repeated_exports_do_not_retain_per_record_classes(tmp_path):
    from tmol.tests.ligand.test_ligand_entry_paths import _single_prep
    from tmol.ligand._params_io import _CompactDumper

    prep = _single_prep()
    path = tmp_path / "repeated.tmol"
    write_params_file(prep, path, format="tmol")
    registered = len(_CompactDumper.yaml_representers)
    for _ in range(10):
        write_params_file(prep, path, format="tmol")
    assert len(_CompactDumper.yaml_representers) == registered


@pytest.mark.parametrize("section", ["patches", "charges", "connections", "bonded"])
def test_shared_metadata_without_residue_is_not_silently_dropped(tmp_path, section):
    import cattr
    import yaml
    from tmol.database.scoring import ConnectionCartRes

    payload = {"version": "1.0", "chemical": {"residues": []}}
    if section == "patches":
        payload["chemical"]["adds_patches"] = [
            cattr.unstructure(ParameterDatabase.get_default().chemical.variants[0])
        ]
    elif section == "charges":
        payload["elec"] = {
            "atom_charge_parameters": [
                {"res": "ASN:conj_ND2", "atom": "ND2", "charge": -0.2}
            ]
        }
    elif section == "connections":
        payload["cartbonded"] = {
            "connection_params": [
                cattr.unstructure(ConnectionCartRes("A", "up", "B", "down", (), ()))
            ]
        }
    else:
        from tmol.ligand._params_file import _empty_cartres

        payload["cartbonded"] = {
            "residue_params": {"LYS:conj_NZ": cattr.unstructure(_empty_cartres())}
        }
    path = tmp_path / "empty.tmol"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="must define a residue"):
        load_params_file(path)


def test_conflicting_patch_name_rejected():
    from tmol.tests.ligand.test_ligand_entry_paths import _single_prep

    base = ParameterDatabase.get_default()
    patch = base.chemical.variants[0]
    conflicting = attr.evolve(patch, display_name=patch.display_name + "_changed")
    prep = replace(_single_prep(), adds_patches=(conflicting,))
    with pytest.raises(ValueError, match="Conflicting patch definition"):
        inject_ligand_preparations(base, [prep])
