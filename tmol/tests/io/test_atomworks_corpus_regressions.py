"""Difficult, unmodified AtomWorks fixtures and their explicit input contracts."""

from pathlib import Path

import attr
import json
import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import pytest
import torch

from tmol.io import atom_array_from_cif, pose_stack_from_biotite, pose_stack_from_cif
from tmol.io._pose_stack_from_biotite import (
    _map_atoms_to_canonical,
    canonical_form_from_biotite,
    canonical_ordering_for_biotite,
)
from tmol.score import beta2016_score_function

DATA = Path(__file__).parents[1] / "data" / "atomworks_regressions"


def test_partial_sugar_ring_completion_preserves_chemical_identity():
    from rdkit import Chem
    from tmol.ligand import prepare_ligands
    from tmol.ligand._conjugate_model import capped_conjugate_models

    inventories = []
    for reader in ("tmol", "atomworks"):
        array = atom_array_from_cif(
            DATA / "partial_sugar_rings_2msb.cif.gz", reader=reader
        )
        sugars = [
            r
            for r in struc.residue_iter(array)
            if r.res_name[0] in ("NAG", "BMA", "MAN")
        ]
        inventories.append([frozenset(r.atom_name) for r in sugars])
        partial = sugars[-1]
        assert partial.res_name[0] == "MAN"
        assert len(partial) == 11
        assert {"C1", "C2", "C3", "C4", "C5", "O5"} <= set(partial.atom_name)
        assert set(partial.atom_name[np.isfinite(partial.coord).all(axis=-1)]) == {"C1"}
        bonds = {
            frozenset((str(partial.atom_name[a]), str(partial.atom_name[b])))
            for a, b, _ in partial.bonds.as_array()
        }
        ring = ("C1", "C2", "C3", "C4", "C5", "O5", "C1")
        assert all(frozenset(pair) in bonds for pair in zip(ring[:-1], ring[1:]))
        template = array._custom_ccd_registry["MAN"]
        stereo = dict(zip(template.atom_name, template.stereo))
        assert [stereo[name] for name in ("C1", "C2", "C3", "C4", "C5")] == [
            "S",
            "S",
            "S",
            "S",
            "R",
        ]
        # Metals are excluded only after checking they have no covalent bonds.
        metal = np.isin(array.element, ["CA"])
        assert not metal[array.bonds.as_array()[:, :2]].any()
        supported = array[~metal & (array.res_name != "HOH")]
        supplied = supported.coord.copy()
        template_before = template.copy()
        db, _ = prepare_ligands(supported, seed=20260909)
        starts = struc.get_residue_starts(supported, add_exclusive_stop=True)
        last_man = max(
            i for i in range(len(starts) - 1) if supported.res_name[starts[i]] == "MAN"
        )
        start, stop = starts[last_man : last_man + 2]
        centers = {
            i
            for i in range(start, stop)
            if supported.atom_name[i] in ("C1", "C2", "C3", "C4", "C5")
        }
        assigned = set()
        for model in capped_conjugate_models(supported, db.chemical):
            for local, source_index in enumerate(model.source_atom_indices):
                if source_index in centers:
                    assert (
                        model.molecule.GetAtomWithIdx(local).GetChiralTag()
                        != Chem.ChiralType.CHI_UNSPECIFIED
                    )
                    assigned.add(source_index)
        assert assigned == centers
        np.testing.assert_array_equal(supported.coord, supplied)
        assert template.equal_annotations(template_before)
        np.testing.assert_array_equal(template.coord, template_before.coord)
    assert inventories[0] == inventories[1]


@pytest.mark.parametrize("reader", ["tmol", "atomworks"])
def test_decreasing_water_author_ids_preserve_full_input(reader, torch_device):
    from tmol.io import build_context_from_biotite

    array = atom_array_from_cif(
        DATA / "decreasing_water_author_ids_5xnl.cif.gz", reader=reader
    )
    assert int(np.isfinite(array.coord).all(axis=-1).sum()) == 98986
    waters = array[array.res_name == "HOH"]
    assert len(waters) == struc.get_residue_count(waters) == 1076
    # Parse the whole photosystem; score protein A with metal cofactors deferred.
    chain = "A" if reader == "tmol" else "E"  # Author A is label E in this CIF.
    protein = array[(array.chain_id == chain) & struc.filter_amino_acids(array)]
    assert struc.get_residue_count(protein) > 300
    context = build_context_from_biotite(protein, torch_device)
    pose = pose_stack_from_biotite(protein, torch_device, context=context, no_optH=True)
    _score_and_minimize(pose, context)


@pytest.mark.parametrize("reader", ["tmol", "atomworks"])
def test_af3_cyclic_peptide_resolves_leaving_atoms_and_minimizes(reader, torch_device):
    from tmol.io import build_context_from_biotite

    path = DATA / "af3_cyclic_peptide_7ubd.cif"
    array = atom_array_from_cif(path, reader=reader)
    site = pdbx.CIFFile.read(path).block["atom_site"]
    retained = {
        (str(c), int(r), str(n)): xyz
        for c, r, n, xyz in zip(
            array.chain_id, array.res_id, array.atom_name, array.coord
        )
    }
    observed = np.stack(
        [site[axis].as_array(float) for axis in ("Cartn_x", "Cartn_y", "Cartn_z")],
        axis=1,
    )
    for chain, residue, name, expected in zip(
        site["auth_asym_id"].as_array(str),
        site["auth_seq_id"].as_array(int),
        site["label_atom_id"].as_array(str),
        observed,
    ):
        if name != "OXT":
            np.testing.assert_allclose(
                retained[chain, residue, name], expected, atol=1e-6
            )
    context = build_context_from_biotite(
        array, torch_device, prepare_ligands=True, ligand_seed=20260909
    )
    pose = pose_stack_from_biotite(array, torch_device, context=context, no_optH=True)
    assert int((pose.block_type_ind >= 0).sum()) == 8
    assert all(
        "OXT" not in pose.packed_block_types.active_block_types[int(i)].atom_to_idx
        for i in pose.block_type_ind[0]
    )
    assert int((pose.inter_residue_connections[..., 0] >= 0).sum()) == 16
    _assert_all_source_connections(pose, array)
    _score_and_minimize(pose, context)


@pytest.mark.parametrize("reader", ["tmol", "atomworks"])
@pytest.mark.parametrize("assembly_id", [None, "1", "2", "copies"])
def test_terminal_nucleoside_keeps_its_backbone_and_minimizes(
    torch_device, reader, assembly_id, tmp_path
):
    path = DATA / "terminal_nucleotide_145d.cif"
    if assembly_id == "copies":
        # Two separated copies exercise instance identity and coordinate transforms.
        cif = pdbx.CIFFile.read(path)
        operations = cif.block["pdbx_struct_oper_list"]
        columns = {
            name: np.repeat(operations[name].as_array(str), 2) for name in operations
        }
        columns["id"] = ["1", "2"]
        columns["vector[1]"] = ["0", "40"]
        cif.block["pdbx_struct_oper_list"] = pdbx.CIFCategory(columns)
        cif.block["pdbx_struct_assembly_gen"] = pdbx.CIFCategory(
            {
                "assembly_id": ["copies"],
                "oper_expression": ["(1,2)"],
                "asym_id_list": ["A"],
            }
        )
        path = tmp_path / "copies.cif"
        cif.write(path)
    pose, context = pose_stack_from_cif(
        path,
        torch_device,
        reader=reader,
        assembly_id=assembly_id,
        prepare_ligands=True,
        ligand_seed=20260909,
        no_optH=True,
        return_context=True,
    )
    blocks = 24 if assembly_id is None else 12
    assert int((pose.block_type_ind >= 0).sum()) == blocks
    types = [
        pose.packed_block_types.active_block_types[int(i)]
        for i in pose.block_type_ind[0]
    ]
    assert all(bt.properties.polymer.backbone_type == "dna" for bt in types)
    assert types[0].name == "MCY:na5prime"
    # Each strand has six residues; proximity adds no conjugations.
    assert int((pose.inter_residue_connections[..., 0] >= 0).sum()) == blocks // 6 * 10
    assert not any("conj_" in bt.name for bt in types)
    array = atom_array_from_cif(path, reader=reader, assembly_id=assembly_id)
    array = array[array.res_name != "HOH"]
    _assert_all_source_connections(pose, array)
    if assembly_id == "copies":
        for chain in ("A",):
            first = array[array.chain_iid == f"{chain}_1"]
            second = array[array.chain_iid == f"{chain}_2"]
            np.testing.assert_array_equal(first.atom_name, second.atom_name)
            np.testing.assert_allclose(
                second.coord, first.coord + [40, 0, 0], atol=1e-5
            )
        # Direct AtomWorks arrays keep the original chain labels alongside IIDs.
        array.chain_id = np.array([name.split("_")[0] for name in array.chain_iid])
        original_chains = array.chain_id.copy()
        canonical = canonical_form_from_biotite(
            array, torch_device, co=context.canonical_ordering
        )
        assert torch.unique(canonical.chain_id).numel() == 2
        direct = pose_stack_from_biotite(
            array, torch_device, context=context, no_optH=True
        )
        np.testing.assert_array_equal(array.chain_id, original_chains)
        torch.testing.assert_close(direct.block_type_ind, pose.block_type_ind)
        torch.testing.assert_close(
            direct.inter_residue_connections, pose.inter_residue_connections
        )
        torch.testing.assert_close(direct.coords, pose.coords)
    _, minimized = _score_and_minimize(pose, context, max_iter=100)
    if assembly_id is not None:
        # Selecting an assembly removes the alternative duplex's severe overlaps.
        # Its phosphodiester bonds must remain intact after unconstrained relaxation.
        offsets = pose.block_coord_offset[0].tolist()
        for block, row in enumerate(pose.inter_residue_connections[0].tolist()):
            for conn, (partner, port) in enumerate(row):
                if partner <= block:
                    continue
                a = offsets[block] + int(types[block].ordered_connection_atoms[conn])
                b = offsets[partner] + int(
                    types[partner].ordered_connection_atoms[port]
                )
                length = (minimized.coords[0, a] - minimized.coords[0, b]).norm()
                assert 1.3 < float(length) < 2.0


def _score_and_minimize(pose, context, max_iter=10):
    from tmol.optimization import CartesianSfxnNetwork, LBFGS_Armijo

    assert torch.isfinite(pose.coords[pose.real_atoms]).all()
    score = beta2016_score_function(pose.device, param_db=context.parameter_database)
    network = CartesianSfxnNetwork(score, pose)
    initial = network().detach()
    optimizer = LBFGS_Armijo(
        network.parameters(), max_iter=max_iter, segment_ids=network.segment_ids
    )

    def closure():
        optimizer.zero_grad()
        energy = network()
        energy.sum().backward()
        assert torch.isfinite(energy).all()
        assert torch.isfinite(network.masked_coords.grad).all()
        return energy

    optimizer.step(closure)
    assert torch.all(closure().detach() < initial)
    return initial, network.pose_stack_from_dofs()


@pytest.mark.parametrize("reader", ["tmol", "atomworks"])
def test_macrocycle_preserves_every_bond_across_residue_order(reader, torch_device):
    from tmol.io import build_context_from_biotite, pose_stack_from_biotite

    array = atom_array_from_cif(DATA / "macrocycle_1xvk.cif", reader=reader)
    # Free magnesium is deferred; waters follow the constructor's usual policy.
    array = array[(np.char.upper(array.element) != "MG") & (array.res_name != "HOH")]
    context = build_context_from_biotite(
        array, torch_device, prepare_ligands=True, ligand_seed=20260909
    )
    records = context.parameter_database.scoring.cartbonded.connection_params
    assert any({r.connection1, r.connection2} == {"up", "conj_OG"} for r in records)
    assert all(p.K == 300 for r in records for p in r.length_parameters)
    assert all(p.K == 80 for r in records for p in r.angle_parameters)
    residue = next(r for r in context.restype_set.residue_types if r.name == "QUI")
    assert {"N1", "C2", "O1"} <= set(residue.atom_to_idx)
    assert np.isfinite(residue.compute_ideal_coords()).all()
    starts = struc.get_residue_starts(array, add_exclusive_stop=True)
    assert len(starts) - 1 == 18
    energies = []
    for order in (np.arange(18), np.arange(18)[::-1]):
        indices = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in order])
        pose = pose_stack_from_biotite(
            array[indices], torch_device, context=context, no_optH=True
        )
        _assert_all_source_connections(pose, array[indices])
        energies.append(_score_and_minimize(pose, context)[0])
    torch.testing.assert_close(energies[0], energies[1], atol=0.002, rtol=1e-5)


def _assert_all_source_connections(pose, array):
    starts = struc.get_residue_starts(array, add_exclusive_stop=True)
    assert int((pose.block_type_ind >= 0).sum()) == len(starts) - 1
    atom_res = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
    expected = {
        frozenset(
            (
                (int(atom_res[a]), str(array.atom_name[a])),
                (int(atom_res[b]), str(array.atom_name[b])),
            )
        )
        for a, b, _ in array.bonds.as_array()
        if atom_res[a] != atom_res[b]
    }
    types = [
        pose.packed_block_types.active_block_types[int(i)]
        for i in pose.block_type_ind[0]
    ]
    actual = {
        frozenset(
            (
                (block, types[block].connections[conn].atom),
                (partner, types[partner].connections[port].atom),
            )
        )
        for block, row in enumerate(pose.inter_residue_connections[0].cpu().tolist())
        for conn, (partner, port) in enumerate(row)
        if partner >= 0
    }
    assert actual == expected


@pytest.mark.parametrize("reader", ["tmol", "atomworks"])
@pytest.mark.parametrize(
    "fixture, shared_type",
    [
        ("repeated_glycans_6mub", None),
        ("repeated_partner_glycans_1ivo", "NAG:conj_C1"),
        ("repeated_partner_glycans_1hge", "NAG:conj_C1:conj_O4"),
        ("terminal_asj_glycans_1iau", "NAG:conj_C1"),
    ],
)
def test_repeated_glycans_share_transferable_attachment_targets(
    reader, torch_device, fixture, shared_type
):
    from tmol.io import build_context_from_biotite
    from tmol.ligand._connection_params import generate_conjugate_connection_params

    array = atom_array_from_cif(DATA / f"{fixture}.cif.gz", reader=reader)
    array = array[array.res_name != "HOH"]
    metals = np.isin(np.char.upper(array.element), ("ZN", "NA", "MG", "CA"))
    assert not metals[array.bonds.as_array()[:, :2]].any()
    array = array[~metals]
    context = build_context_from_biotite(
        array, torch_device, prepare_ligands=True, ligand_seed=20260909
    )
    database = context.parameter_database
    if fixture == "terminal_asj_glycans_1iau":
        terminal_atoms = context.canonical_ordering.termini_patch_added_atoms
        assert {"OD1", "OD2"} <= set(terminal_atoms["ASJ", "cterm"])
        assert {"OD1", "OD2"}.isdisjoint(terminal_atoms["ASP", "cterm"])
    records = database.scoring.cartbonded.connection_params
    if shared_type is None:
        assert any(
            r.block_type1 == "MAN:conj_C1" and r.block_type2 == "MAN:conj_C1:conj_O2"
            for r in records
        )
    else:
        targets = {
            p.x0
            for r in records
            if (r.block_type1, r.connection1) == (shared_type, "conj_C1")
            or (r.block_type2, r.connection2) == (shared_type, "conj_C1")
            for p in r.length_parameters
        }
        assert len(targets) > 1  # Pair-specific targets must not collapse together.
    assert all(p.K == 300 for r in records for p in r.length_parameters)
    assert all(p.K == 80 for r in records for p in r.angle_parameters)
    starts = struc.get_residue_starts(array, add_exclusive_stop=True)
    reversed_array = array[
        np.concatenate([np.arange(a, b) for a, b in zip(starts[-2::-1], starts[:0:-1])])
    ]
    # Both repeated sites and separately prepared poses must agree, independent
    # of encounter order or random seed. Supplied records cannot mask regeneration.
    regenerated = generate_conjugate_connection_params(reversed_array, database, seed=1)
    assert len(regenerated) == len(records)
    for generated, installed in zip(regenerated, records):
        metadata = json.loads(installed.provenance)
        correction = metadata.pop("local_conjugate")
        assert correction["charge_model"] == "conserved-patched-residue-v1"
        assert metadata == json.loads(generated.provenance)
        assert attr.evolve(installed, provenance="") == attr.evolve(
            generated, provenance=""
        )
    energies, identities, coordinates = [], [], []
    for source in (array, reversed_array):
        # Require the declared graph exactly in this regression. The corpus
        # runner separately exercises the default disulfide-inference policy.
        pose = pose_stack_from_biotite(
            source,
            torch_device,
            context=context,
            no_optH=True,
            find_additional_disulfides=False,
        )
        # The full input includes entirely unresolved protein residues. The
        # constructor excludes them; it must retain every observed residue and
        # every glycan, including all their declared attachment bonds.
        starts = struc.get_residue_starts(source, add_exclusive_stop=True)
        retained = np.concatenate(
            [
                np.full(b - a, np.isfinite(source.coord[a:b]).any() or source.hetero[a])
                for a, b in zip(starts[:-1], starts[1:])
            ]
        )
        assert not np.isfinite(source.coord[~retained]).any()
        assert not source.hetero[~retained].any()
        _assert_all_source_connections(pose, source[retained])
        blocks = [
            (pose.packed_block_types.active_block_types[t], offset)
            for t, offset in zip(
                pose.block_type_ind[0].tolist(), pose.block_coord_offset[0].tolist()
            )
        ]
        if fixture == "terminal_asj_glycans_1iau":
            retained_source = source[retained]
            asj = []
            for block, residue in zip(blocks, struc.residue_iter(retained_source)):
                bt, offset = block
                observed = residue[np.isfinite(residue.coord).all(axis=-1)]
                assert len(set(observed.atom_name)) == len(observed)
                indices = [offset + bt.atom_to_idx[str(n)] for n in observed.atom_name]
                np.testing.assert_allclose(
                    pose.coords[0, indices].detach().cpu(), observed.coord, atol=1e-6
                )
                if residue.res_name[0] == "ASJ":
                    assert "cterm" in bt.name and "conj_C" in bt.name
                    assert {"OD1", "OD2"} <= set(bt.atom_to_idx)
                    asj.append(bt)
            assert len(asj) == 1
        if source is reversed_array:
            blocks.reverse()
        identities.append([bt.name for bt, _ in blocks])
        coordinates.append(
            torch.cat(
                [
                    pose.coords[0, offset : offset + bt.n_atoms].detach()
                    for bt, offset in blocks
                ]
            )
        )
        energies.append(_score_and_minimize(pose, context)[0])
    assert identities[0] == identities[1]
    torch.testing.assert_close(coordinates[0], coordinates[1], atol=1e-4, rtol=0)
    torch.testing.assert_close(energies[0], energies[1], atol=0.002, rtol=1e-5)


def test_unknown_heavy_atom_is_not_silently_deleted():
    # The fixture deliberately conflicts: label XYZ versus author CG. The
    # native reader legitimately uses CG; the label-based view must reject XYZ.
    cif = pdbx.CIFFile.read(DATA / "unknown_heavy_atom_1a8o.cif")
    array = pdbx.get_structure(cif, model=1, use_author_fields=False)
    assert np.count_nonzero(array.atom_name == "XYZ") == 1
    with pytest.raises(ValueError, match="Heavy atoms.*ASP.*XYZ"):
        canonical_form_from_biotite(array, torch.device("cpu"))


def test_duplicate_canonical_atoms_are_rejected(torch_device):
    array = struc.AtomArray(6)
    array.res_name[:] = "ALA"
    array.res_id[:] = 1
    array.atom_name = ["N", "CA", "C", "O", "CB", "HB1"]
    array.element = ["N", "C", "C", "O", "C", "H"]
    array.coord[:] = np.arange(18).reshape(6, 3)
    for name, alias in (("CA", "CA"), ("HB1", "1HB")):
        duplicate = array[array.atom_name == name]
        duplicate.atom_name[:] = alias
        with pytest.raises(ValueError, match="Multiple input atoms map to canonical"):
            canonical_form_from_biotite(array + duplicate, torch_device)
    second = array.copy()
    second.ins_code[:] = "A"
    canonical = canonical_form_from_biotite(array + second, torch_device)
    assert canonical.res_types.shape == (1, 2)
    assert torch.isfinite(canonical.coords).all(-1).sum() == 2 * len(array)


def test_author_named_view_preserves_the_conflicting_label_atom_coordinate():
    path = DATA / "unknown_heavy_atom_1a8o.cif"
    source = pdbx.CIFFile.read(path).block["atom_site"]
    selected = source["label_atom_id"].as_array(str) == "XYZ"
    expected = np.column_stack(
        [
            source[name].as_array(float)[selected]
            for name in ("Cartn_x", "Cartn_y", "Cartn_z")
        ]
    )
    array = atom_array_from_cif(path)
    actual = array[(array.res_id == 152) & (array.atom_name == "CG")].coord
    np.testing.assert_allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize("element", ["H", "D"])
def test_unrecognized_hydrogen_names_can_be_rebuilt(element):
    mask, atoms, residues = _map_atoms_to_canonical(
        canonical_ordering_for_biotite(),
        np.array([0, 0]),
        ["ALA", "ALA"],
        ["CA", "extra_H"],
        ["C", element],
    )
    np.testing.assert_array_equal(mask, [True, False])
    assert len(atoms) == len(residues) == 1


@pytest.mark.parametrize("reader", ["tmol", "atomworks"])
@pytest.mark.parametrize(
    "kind,n_hydrogens,bond_type", [("amine", 1, "SINGLE"), ("imine", 0, "DOUBLE")]
)
def test_generated_amine_attachment_uses_bonded_hydrogen_count(
    reader, torch_device, kind, n_hydrogens, bond_type
):
    from tmol.io import build_context_from_biotite
    from tmol.ligand._conjugation_patches import _hydrogens_on
    from tmol.tests.ligand.test_local_conjugate_params import _charges

    array = atom_array_from_cif(
        DATA / f"generated_{kind}_attachment.cif", reader=reader
    )
    context = build_context_from_biotite(
        array, torch_device, prepare_ligands=True, ligand_seed=20260913
    )
    db = context.parameter_database
    residues = {r.name: r for r in db.chemical.residues}
    name = str(array.res_name[array.atom_name == "N1"][0])
    base, patched = residues[name], residues[name + ":conj_N1"]
    assert len(_hydrogens_on(base, "N1", db.chemical)) == 3
    assert len(_hydrogens_on(patched, "N1", db.chemical)) == n_hydrogens
    assert next(c.type for c in patched.connections if c.name == "conj_N1") == bond_type
    assert next(a.atom_type for a in patched.atoms if a.name == "N1") == (
        "Nad" if kind == "amine" else "Nim"
    )
    if bond_type == "DOUBLE":
        for rt in residues.values():
            for connection in rt.connections:
                if connection.type == "DOUBLE":
                    assert all(
                        connection.name not in (t.b.connection, t.c.connection)
                        for t in rt.torsions
                    )
    assert sum(_charges(db, patched).values()) == pytest.approx(
        sum(_charges(db, base).values()), abs=1e-8
    )
    pose = pose_stack_from_biotite(array, torch_device, context=context, no_optH=True)
    _assert_all_source_connections(pose, array)
    _score_and_minimize(pose, context)


@pytest.mark.parametrize("reader", ["tmol", "atomworks"])
def test_schiff_base_reports_missing_covalent_partner_backbone(reader):
    # The double bond can be prepared, but this source has no LYS N/CA/C
    # coordinates. Construction must not discard its covalent partner.
    with pytest.raises(
        ValueError, match="Cannot discard an incomplete or unsupported residue"
    ):
        pose_stack_from_cif(
            DATA / "schiff_base_double_bond.cif",
            torch.device("cpu"),
            reader=reader,
            prepare_ligands=True,
            ligand_seed=20260909,
            no_optH=True,
        )


@pytest.mark.parametrize("reader", ["tmol", "atomworks"])
def test_conflicting_myristate_connections_are_reported(reader, torch_device):
    array = atom_array_from_cif(
        DATA / "conflicting_myristate_1aym.cif.gz", reader=reader
    )
    # The complete source has a free zinc ion; metal parameters are out of scope.
    array = array[array.res_name != "ZN"]
    before = array.bonds.as_array().copy()
    with pytest.raises(ValueError, match=r"MYR\.C1.*multiple declared partners") as exc:
        pose_stack_from_biotite(
            array,
            torch_device,
            prepare_ligands=True,
            ligand_seed=20250828,
            no_optH=True,
        )
    assert ".N (" in str(exc.value) and ".CA (" in str(exc.value)
    np.testing.assert_array_equal(array.bonds.as_array(), before)


def test_entirely_unresolved_ligand_keeps_its_chemical_identity():
    array = atom_array_from_cif(DATA / "unresolved_unl.cif")
    ligand = array[array.res_name == "UNL"]
    assert len(ligand) == 28
    assert np.isnan(ligand.coord).all()
    assert ligand.bonds.get_bond_count() > 0
    # There is no placement anchor. Do not manufacture a positioned ligand.
    with pytest.raises(RuntimeError, match="UNL|missing"):
        pose_stack_from_cif(
            DATA / "unresolved_unl.cif",
            torch.device("cpu"),
            prepare_ligands=True,
            ligand_seed=20260909,
            no_optH=True,
        )


@pytest.mark.parametrize(
    "filename", ["conditional_generation.cif", "acetylated_peptide_1j8z.cif"]
)
def test_backbone_only_and_crosslinked_modified_peptides_build_and_score(
    filename, torch_device
):
    path = DATA / filename
    pose, context = pose_stack_from_cif(
        path,
        torch_device,
        prepare_ligands=True,
        ligand_seed=20260909,
        no_optH=True,
        return_context=True,
    )
    assert torch.isfinite(pose.coords[pose.real_atoms]).all()
    coords = pose.coords.detach().clone().requires_grad_()
    score = beta2016_score_function(torch_device, param_db=context.parameter_database)
    energy = score.render_whole_pose_scoring_module(pose)(coords)
    energy.sum().backward()
    assert torch.isfinite(energy).all()
    assert torch.isfinite(coords.grad).all()
    if filename == "acetylated_peptide_1j8z.cif":
        types = [
            pose.packed_block_types.active_block_types[int(i)]
            for i in pose.block_type_ind[0]
        ]
        assert types[4].name.split(":")[0] == "BCX"
        assert types[4].properties.polymer.is_polymer
        assert (
            int(pose.inter_residue_connections[0, 3, types[3].up_connection_ind, 0])
            == 4
        )
        assert (
            int(pose.inter_residue_connections[0, 4, types[4].up_connection_ind, 0])
            == 5
        )
