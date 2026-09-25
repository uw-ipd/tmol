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


def assert_metal_bonds_are_coordination(array, metal):
    """A metal is declared bonded only by coordination, never covalently."""
    bonds = array.bonds.as_array()
    touches = metal[bonds[:, :2]].any(axis=1)
    assert (bonds[touches, 2] == struc.BondType.COORDINATION).all()


def test_partial_sugar_rings_construct_score_and_minimize(torch_device, monkeypatch):
    from tmol.io import build_context_from_biotite
    from tmol.io.details import _build_missing_nonpolymer_atoms as completion
    from rdkit import Chem
    from tmol.ligand._conjugate_model import capped_conjugate_models

    array = atom_array_from_cif(DATA / "partial_sugar_rings_2msb.cif.gz")
    sugars = [
        r for r in struc.residue_iter(array) if r.res_name[0] in ("NAG", "BMA", "MAN")
    ]
    partial = sugars[-1]
    assert partial.res_name[0] == "MAN"
    assert set(partial.atom_name) == {
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "O2",
        "O3",
        "O4",
        "O5",
        "O6",
    }
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
    template_before = template.copy()
    metal = array.element == "CA"
    assert_metal_bonds_are_coordination(array, metal)
    array = array[~metal & (array.res_name != "HOH")]
    supplied = array.coord.copy()
    context = build_context_from_biotite(
        array, torch_device, prepare_ligands=True, ligand_seed=20260909
    )
    starts = struc.get_residue_starts(array, add_exclusive_stop=True)
    last_man = max(
        i for i in range(len(starts) - 1) if array.res_name[starts[i]] == "MAN"
    )
    start, stop = starts[last_man : last_man + 2]
    centers = {
        i
        for i in range(start, stop)
        if array.atom_name[i] in ("C1", "C2", "C3", "C4", "C5")
    }
    assigned = set()
    for model in capped_conjugate_models(array, context.parameter_database.chemical):
        for local, source_index in enumerate(model.source_atom_indices):
            if source_index in centers:
                assert (
                    model.molecule.GetAtomWithIdx(local).GetChiralTag()
                    != Chem.ChiralType.CHI_UNSPECIFIED
                )
                assigned.add(source_index)
    assert assigned == centers
    assert template.equal_annotations(template_before)
    np.testing.assert_array_equal(template.coord, template_before.coord)
    original = completion.build_missing_nonpolymer_atoms
    calls = []

    def record(*args):
        result = original(*args)
        calls.append((args, result))
        return result

    monkeypatch.setattr(completion, "build_missing_nonpolymer_atoms", record)
    pose = pose_stack_from_biotite(array, torch_device, context=context, no_optH=True)
    np.testing.assert_array_equal(array.coord, supplied)
    assert len(calls) == 1
    (pbt, coords, targets, offsets, types, connections), completed = calls[0]
    changed = torch.isnan(coords).any(-1) & torch.isfinite(completed).all(-1)
    assert int(changed.sum()) == 48
    finite = torch.isfinite(coords).all(-1)
    torch.testing.assert_close(completed[finite], coords[finite], rtol=0, atol=0)

    # Both rings keep every internal distance, including the closure bond, and
    # every heavy-atom tetrahedral center keeps its generated handedness.
    anchors = set()
    for pi, bi in torch.nonzero(targets.any(-1)).tolist():
        bt = pbt.active_block_types[int(types[pi, bi])]
        offset = int(offsets[pi, bi])
        ring = [bt.atom_to_idx[n] for n in ("C1", "C2", "C3", "C4", "C5", "O5")]
        ideal = pose.coords.new_tensor(bt.ideal_coords[bt.at_to_icoor_ind])
        xyz = pose.coords[pi, offset : offset + bt.n_atoms]
        torch.testing.assert_close(
            torch.cdist(xyz[ring], xyz[ring]),
            torch.cdist(ideal[ring], ideal[ring]),
            atol=1e-4,
            rtol=1e-5,
        )
        for atom in range(bt.n_atoms):
            neighbors = sorted({b for a, b in bt.bond_indices if a == atom})
            heavy = [
                n
                for n in neighbors
                if not bool(pbt.atom_is_hydrogen[int(types[pi, bi]), n])
            ]
            if len(heavy) == 3:
                before = torch.linalg.det(ideal[heavy] - ideal[atom])
                if abs(float(before)) > 0.1:
                    assert before * torch.linalg.det(xyz[heavy] - xyz[atom]) > 0
        for ai in (
            torch.nonzero(
                torch.isfinite(coords[pi, offset : offset + bt.n_atoms]).all(-1)
            )
            .flatten()
            .tolist()
        ):
            anchors.add((pi, offset + ai))
        for ci in range(len(bt.connections)):
            other, port = connections[pi, bi, ci].tolist()
            if other >= 0:
                for sep in (0, 1):
                    ai = int(
                        pbt.atom_downstream_of_conn[int(types[pi, other]), port, sep]
                    )
                    if ai >= 0:
                        anchors.add((pi, int(offsets[pi, other]) + ai))

    # Check the actual reconstruction graph with finite differences through
    # the observed atoms, including each connection's external references.
    indices = tuple(torch.tensor(sorted(anchors), device=torch_device).T)
    base = coords.detach().double()
    values = base[indices].clone().requires_grad_()

    def finish(values):
        return original(
            pbt, base.index_put(indices, values), targets, offsets, types, connections
        )[changed]

    assert torch.autograd.gradcheck(finish, (values,), fast_mode=True)
    rotation = base.new_tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    moved = base @ rotation.T + base.new_tensor([12, -4, 7])
    actual = original(pbt, moved, targets, offsets, types, connections)
    torch.testing.assert_close(
        actual[changed], finish(values) @ rotation.T + base.new_tensor([12, -4, 7])
    )

    # A collinear input cannot orient either ring. Removing the partner's
    # downstream reference also leaves the singly anchored MAN unresolved.
    collinear = base.clone()
    collinear[finite] = 0
    unresolved = original(pbt, collinear, targets, offsets, types, connections)
    assert torch.isnan(unresolved[changed]).all()
    for pi, bi in torch.nonzero(targets.any(-1)).tolist():
        bt = pbt.active_block_types[int(types[pi, bi])]
        if bt.name != "MAN:conj_C1":
            continue
        other, port = connections[pi, bi, bt.connection_to_cidx["conj_C1"]].tolist()
        ai = int(pbt.atom_downstream_of_conn[int(types[pi, other]), port, 1])
        absent = base.clone()
        absent[pi, int(offsets[pi, other]) + ai] = float("nan")
        unresolved = original(pbt, absent, targets, offsets, types, connections)
        offset = int(offsets[pi, bi])
        assert torch.isnan(unresolved[pi, offset + bt.atom_to_idx["C2"]]).all()
    _, minimized = _score_and_minimize(pose, context, max_iter=100)
    assert torch.isfinite(minimized.coords[minimized.real_atoms]).all()


@pytest.mark.parametrize("pdb,sugar,count", [("1en2", "NAG", 4), ("4ndz", "GLC", 10)])
def test_terminal_and_linked_glycans_construct_score_and_minimize(
    torch_device, pdb, sugar, count
):
    from tmol.io import build_context_from_biotite
    from tmol.tests.ligand.test_local_conjugate_params import _charges

    array = atom_array_from_cif(DATA / f"terminal_and_linked_glycans_{pdb}.cif.gz")
    metal = np.isin(np.char.upper(array.element), ("ZN", "NA", "MG", "CA"))
    assert_metal_bonds_are_coordination(array, metal)
    array = array[~metal & (array.res_name != "HOH")]
    supplied = array.coord.copy()
    context = build_context_from_biotite(
        array, torch_device, prepare_ligands=True, ligand_seed=20260909
    )
    db = context.parameter_database
    residues = {r.name: r for r in db.chemical.residues}
    base_charge = sum(_charges(db, residues[sugar]).values())
    assert "O1" in {a.name for a in residues[sugar].atoms}
    records = db.scoring.cartbonded.connection_params
    assert records
    assert all(p.K == 300 for r in records for p in r.length_parameters)
    assert all(p.K == 80 for r in records for p in r.angle_parameters)
    pose = pose_stack_from_biotite(
        array,
        torch_device,
        context=context,
        no_optH=True,
        find_additional_disulfides=False,
    )
    np.testing.assert_array_equal(array.coord, supplied)
    starts = struc.get_residue_starts(array, add_exclusive_stop=True)
    # 4NDZ also has five partly resolved protein termini lacking C. The
    # current backbone contract excludes them as well as unresolved residues;
    # this glycan regression makes those remaining exclusions explicit.
    retained, partial = [], 0
    for a, b in zip(starts[:-1], starts[1:]):
        observed_names = set(
            array.atom_name[a:b][np.isfinite(array.coord[a:b]).all(-1)]
        )
        missing_c = not array.hetero[a] and "C" not in observed_names
        partial += bool(observed_names) and missing_c
        retained.extend([not missing_c] * (b - a))
    retained = np.asarray(retained)
    assert partial == (0 if pdb == "1en2" else 5)
    assert not array.hetero[~retained].any()
    _assert_all_source_connections(pose, array[retained])
    glycans, terminal = 0, 0
    stereocenters = []
    for bi, residue in enumerate(struc.residue_iter(array[retained])):
        bt = pose.packed_block_types.active_block_types[int(pose.block_type_ind[0, bi])]
        offset = int(pose.block_coord_offset[0, bi])
        if residue.res_name[0] != sugar:
            continue
        # Packing can rebuild incomplete protein sidechains; these glycans
        # must keep every supplied heavy-atom coordinate exactly.
        observed = residue[
            np.isfinite(residue.coord).all(axis=-1)
            & ~np.isin(residue.element, ("H", "D"))
        ]
        indices = [offset + bt.atom_to_idx[str(n)] for n in observed.atom_name]
        np.testing.assert_array_equal(
            pose.coords[0, indices].detach().cpu(), observed.coord
        )
        glycans += 1
        has_o1 = "O1" in residue.atom_name
        terminal += has_o1
        assert ("O1" in bt.atom_to_idx) == has_o1
        assert ("conj_C1" in bt.connection_to_cidx) != has_o1
        rt = residues[bt.name]
        assert sum(_charges(db, rt).values()) == pytest.approx(base_charge, abs=1e-8)
        names = {a.name for a in rt.atoms} | {c.name for c in rt.connections}
        assert all(
            {ic.parent, ic.grand_parent, ic.great_grand_parent} <= names
            for ic in rt.icoors
            if ic.name in names
        )
        # A generated terminal oxygen must keep the anomeric handedness. All
        # observed ring centers also retain the prepared conformer's chirality.
        ideal = bt.ideal_coords[bt.at_to_icoor_ind]
        xyz = pose.coords[0, offset : offset + bt.n_atoms].detach().cpu().numpy()
        for center in ("C1", "C2", "C3", "C4", "C5"):
            ai = bt.atom_to_idx[center]
            heavy = sorted(
                {
                    b
                    for a, b in bt.bond_indices
                    if a == ai
                    and not bool(
                        pose.packed_block_types.atom_is_hydrogen[
                            int(pose.block_type_ind[0, bi]), b
                        ]
                    )
                }
            )
            if len(heavy) == 3:
                stereocenters.append(
                    (
                        offset + ai,
                        np.asarray(heavy) + offset,
                        np.linalg.det(ideal[heavy] - ideal[ai]),
                    )
                )
                assert (
                    np.linalg.det(ideal[heavy] - ideal[ai])
                    * np.linalg.det(xyz[heavy] - xyz[ai])
                    > 0
                )
    assert glycans == count
    assert terminal == (1 if pdb == "1en2" else 5)
    _, minimized = _score_and_minimize(
        pose, context, max_iter=100 if torch_device.type == "cuda" else 10
    )

    xyz = minimized.coords[0].detach().cpu().numpy()
    for center, neighbors, handedness in stereocenters:
        assert handedness * np.linalg.det(xyz[neighbors] - xyz[center]) > 0


def test_decreasing_water_author_ids_preserve_full_input(torch_device):
    from tmol.io import build_context_from_biotite

    array = atom_array_from_cif(DATA / "decreasing_water_author_ids_5xnl.cif.gz")
    assert int(np.isfinite(array.coord).all(axis=-1).sum()) == 98986
    waters = array[array.res_name == "HOH"]
    assert len(waters) == struc.get_residue_count(waters) == 1076
    # Parse the whole photosystem; score protein A with metal cofactors deferred.
    chain = "A"  # Preserve the author chain ID.
    protein = array[(array.chain_id == chain) & struc.filter_amino_acids(array)]
    assert struc.get_residue_count(protein) > 300
    context = build_context_from_biotite(protein, torch_device)
    pose = pose_stack_from_biotite(protein, torch_device, context=context, no_optH=True)
    _score_and_minimize(pose, context)


def test_af3_cyclic_peptide_resolves_leaving_atoms_and_minimizes(torch_device):
    from tmol.io import build_context_from_biotite

    path = DATA / "af3_cyclic_peptide_7ubd.cif"
    array = atom_array_from_cif(path)
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


@pytest.mark.parametrize("assembly_id", [None, "1", "2", "copies"])
def test_terminal_nucleoside_keeps_its_backbone_and_minimizes(
    torch_device, assembly_id, tmp_path
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
    array = atom_array_from_cif(path, assembly_id=assembly_id)
    array = array[array.res_name != "HOH"]
    _assert_all_source_connections(pose, array)
    if assembly_id == "copies":
        declared = atom_array_from_cif(path, assembly_id=assembly_id, use_ccd=False)
        known = pose_stack_from_biotite(
            declared, torch_device, context=context, no_optH=True
        )
        torch.testing.assert_close(known.block_type_ind, pose.block_type_ind)
        torch.testing.assert_close(
            known.inter_residue_connections, pose.inter_residue_connections
        )
        torch.testing.assert_close(known.coords, pose.coords)
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
    # The asymmetric unit contains overlapping alternative duplexes. Its large
    # clashes need a fixed budget: relative-energy convergence can stop too early.
    # The budget is the smallest that still resolves every clash below; it is the
    # most expensive minimization in this file, so do not raise it casually.
    _, minimized = _score_and_minimize(
        pose,
        context,
        max_iter=600 if assembly_id is None else 100,
        fixed_iterations=assembly_id is None,
    )
    offsets = pose.block_coord_offset[0].tolist()
    for block, row in enumerate(pose.inter_residue_connections[0].tolist()):
        for conn, (partner, port) in enumerate(row):
            if partner <= block:
                continue
            a = offsets[block] + int(types[block].ordered_connection_atoms[conn])
            b = offsets[partner] + int(types[partner].ordered_connection_atoms[port])
            length = (minimized.coords[0, a] - minimized.coords[0, b]).norm()
            assert 1.3 < float(length) < 2.0


def _score_and_minimize(pose, context, max_iter=10, fixed_iterations=False):
    from tmol.optimization import CartesianSfxnNetwork, LBFGS_Armijo

    assert torch.isfinite(pose.coords[pose.real_atoms]).all()
    score = beta2016_score_function(pose.device, param_db=context.parameter_database)
    network = CartesianSfxnNetwork(score, pose)
    initial = network().detach()
    optimizer = LBFGS_Armijo(
        network.parameters(),
        max_iter=max_iter,
        segment_ids=network.segment_ids,
        fixed_iterations=fixed_iterations,
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


def test_macrocycle_preserves_every_bond_across_residue_order(torch_device):
    from tmol.io import build_context_from_biotite, pose_stack_from_biotite

    array = atom_array_from_cif(DATA / "macrocycle_1xvk.cif")
    # Free magnesium is deferred; waters follow the constructor's usual policy.
    array = array[(np.char.upper(array.element) != "MG") & (array.res_name != "HOH")]
    context = build_context_from_biotite(
        array, torch_device, prepare_ligands=True, ligand_seed=20260909
    )
    records = context.parameter_database.scoring.cartbonded.connection_params
    assert any({r.connection1, r.connection2} == {"conj_C", "conj_OG"} for r in records)
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
    torch_device, fixture, shared_type
):
    from tmol.io import build_context_from_biotite
    from tmol.ligand._connection_params import generate_conjugate_connection_params

    array = atom_array_from_cif(DATA / f"{fixture}.cif.gz")
    array = array[array.res_name != "HOH"]
    metals = np.isin(np.char.upper(array.element), ("ZN", "NA", "MG", "CA"))
    assert_metal_bonds_are_coordination(array, metals)
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


@pytest.mark.parametrize(
    "kind,n_hydrogens,bond_type", [("amine", 1, "SINGLE"), ("imine", 0, "DOUBLE")]
)
def test_generated_amine_attachment_uses_bonded_hydrogen_count(
    torch_device, kind, n_hydrogens, bond_type
):
    from tmol.io import build_context_from_biotite
    from tmol.ligand._conjugation_patches import _hydrogens_on
    from tmol.tests.ligand.test_local_conjugate_params import _charges

    array = atom_array_from_cif(DATA / f"generated_{kind}_attachment.cif")
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


def test_schiff_base_reports_missing_covalent_partner_backbone():
    # The double bond can be prepared, but this source has no LYS N/CA/C
    # coordinates. Construction must not discard its covalent partner.
    with pytest.raises(
        ValueError, match="Cannot discard an incomplete or unsupported residue"
    ):
        pose_stack_from_cif(
            DATA / "schiff_base_double_bond.cif",
            torch.device("cpu"),
            prepare_ligands=True,
            ligand_seed=20260909,
            no_optH=True,
        )


def test_conflicting_myristate_connections_are_reported(torch_device):
    array = atom_array_from_cif(DATA / "conflicting_myristate_1aym.cif.gz")
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
    _score_and_minimize(pose, context)
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


def test_chromophore_imine_junction_retains_generated_restoring_forces(torch_device):
    """3SVU's acyl-imine junction must not collapse during relaxation."""
    path = DATA.parent / "ncaa_fixtures" / "chromophore_nrq_3svu.cif"
    array = atom_array_from_cif(path)
    chromophore = array[array.res_name == "NRQ"]
    names = chromophore.atom_name
    assert any(
        {names[a], names[b]} == {"N1", "CA1"} and order == int(struc.BondType.DOUBLE)
        for a, b, order in chromophore.bonds.as_array()
    )
    pose, context = pose_stack_from_biotite(
        array,
        torch_device,
        prepare_ligands=True,
        ligand_seed=20260914,
        no_optH=True,
        return_context=True,
    )
    assert int((pose.block_type_ind >= 0).sum()) == 5
    params = context.parameter_database.scoring.cartbonded.residue_params["NRQ"]
    bond = next(p for p in params.length_parameters if {p.atm1, p.atm2} == {"N1", "+C"})
    assert bond.K == 300 and 1.3 < bond.x0 < 1.5
    angles = [p for p in params.angle_parameters if "+C" in (p.atm1, p.atm2, p.atm3)]
    assert len(angles) == 3 and all(p.K == 80 for p in angles)
    _, relaxed = _score_and_minimize(pose, context, max_iter=200, fixed_iterations=True)
    assert torch.equal(
        relaxed.inter_residue_connections, pose.inter_residue_connections
    )
    pbt = relaxed.packed_block_types
    for block, index in enumerate(relaxed.block_type_ind[0].tolist()):
        bt = pbt.active_block_types[index]
        if bt.base_name != "NRQ":
            continue
        partner, port = relaxed.inter_residue_connections[
            0, block, bt.connection_to_cidx["down"]
        ].tolist()
        neighbor = pbt.active_block_types[int(relaxed.block_type_ind[0, partner])]
        a = int(relaxed.block_coord_offset[0, block]) + bt.atom_to_idx["N1"]
        b = int(relaxed.block_coord_offset[0, partner]) + int(
            neighbor.ordered_connection_atoms[port]
        )
        length = float((relaxed.coords[0, a] - relaxed.coords[0, b]).norm())
        assert abs(length - bond.x0) < 0.08


@pytest.mark.parametrize(
    "fixture,blocks,missing",
    [
        ("missing_ribose_oxygen_6dp5", 5, "O2'"),
        ("missing_sugar_carbon_7n5v", 2, "C2'"),
        ("retinyl_lysine_4xxj", 3, None),
        ("missing_proline_ring_7no8", 3, "CD"),
        ("orphan_hydrogen_9ewf", 1, None),
    ],
)
def test_random_cif_completion_and_relax(fixture, blocks, missing, torch_device):
    from tmol.io import build_context_from_biotite
    from tmol.kinematics import CartesianMoveMap, FoldForest
    from tmol.optimization import run_cart_min
    from tmol.pack import PackerPalette
    from tmol.relax import fast_relax

    array = atom_array_from_cif(DATA / f"{fixture}.cif")
    if fixture == "orphan_hydrogen_9ewf":
        neighbors, _ = array.bonds.get_all_bonds()
        assert ((array.element == "H") & (neighbors[:, 0] < 0)).any()
    if missing:
        assert np.isnan(array.coord[array.atom_name == missing]).any()
    context = build_context_from_biotite(
        array, torch_device, prepare_ligands=True, ligand_seed=20260909
    )
    pose = pose_stack_from_biotite(array, torch_device, context=context, no_optH=True)
    assert int((pose.block_type_ind >= 0).sum()) == blocks
    if fixture == "missing_proline_ring_7no8":
        bt = pose.packed_block_types.active_block_types[int(pose.block_type_ind[0, 0])]
        assert bt.name == "PRO:nterm"
        for hydrogen in ("H2", "H3"):
            distance = (
                pose.coords[0, bt.atom_to_idx[hydrogen]]
                - pose.coords[0, bt.atom_to_idx["N"]]
            ).norm()
            assert 0.9 < float(distance) < 1.1
    _assert_all_source_connections(pose, array)
    if missing:
        for bi, residue in enumerate(struc.residue_iter(array)):
            bt = pose.packed_block_types.active_block_types[
                int(pose.block_type_ind[0, bi])
            ]
            offset = int(pose.block_coord_offset[0, bi])
            observed = residue[np.isfinite(residue.coord).all(axis=-1)]
            indices = [offset + bt.atom_to_idx[name] for name in observed.atom_name]
            np.testing.assert_allclose(
                pose.coords[0, indices].cpu(), observed.coord, atol=1e-5
            )
    elif fixture == "retinyl_lysine_4xxj":
        bt = pose.packed_block_types.active_block_types[int(pose.block_type_ind[0, 1])]
        assert bt.base_name == "LYR" and bt.properties.polymer.is_polymer
        assert any(
            {a, b} == {"NZ", "C1"} and order == "SINGLE" for a, b, order, _ in bt.bonds
        )
    else:
        bt = pose.packed_block_types.active_block_types[int(pose.block_type_ind[0, 0])]
        heavy = array[array.element != "H"]
        assert set(heavy.atom_name) <= set(bt.atom_to_idx)
        heavy = heavy[np.isfinite(heavy.coord).all(axis=1)]
        torch.testing.assert_close(
            pose.coords[0, [bt.atom_to_idx[name] for name in heavy.atom_name]],
            pose.coords.new_tensor(heavy.coord),
        )
        assert {atom.name for atom in bt.atoms} == {
            name for a, b, *_ in bt.bonds for name in (a, b)
        }
    _score_and_minimize(pose, context)
    score = beta2016_score_function(torch_device, param_db=context.parameter_database)
    relaxed = fast_relax(
        pose,
        score,
        PackerPalette(),
        CartesianMoveMap(),
        FoldForest.reasonable_fold_forest(pose),
        num_repeats=1,
        schedule=[1.0],
        min_fn=lambda current, sfxn, **kw: run_cart_min(
            current, sfxn, optimizer_kwargs={"max_iter": 100}
        ),
    )
    assert torch.equal(
        relaxed.inter_residue_connections, pose.inter_residue_connections
    )
    coords = relaxed.coords.detach().requires_grad_(True)
    energy = score.render_whole_pose_scoring_module(relaxed)(coords).sum()
    assert (
        torch.isfinite(energy)
        and torch.isfinite(torch.autograd.grad(energy, coords)[0]).all()
    )
    if missing:
        for bi, ti in enumerate(relaxed.block_type_ind[0].tolist()):
            bt = relaxed.packed_block_types.active_block_types[ti]
            offset = int(relaxed.block_coord_offset[0, bi])
            for a, b, *_ in bt.bonds:
                if (
                    missing in (a, b)
                    and not a.startswith("H")
                    and not b.startswith("H")
                ):
                    length = torch.linalg.vector_norm(
                        coords[0, offset + bt.atom_to_idx[a]]
                        - coords[0, offset + bt.atom_to_idx[b]]
                    )
                    assert 1.2 < float(length.detach()) < 1.8


def test_full_9ewf_conjugate_survives_export_and_minimization(torch_device, tmp_path):
    """Exercise the complete 9EWF adduct containing the orphan-H ligand."""
    from tmol.io import build_context_from_biotite
    from tmol.ligand import prepare_ligands

    array = atom_array_from_cif(DATA / "full_conjugate_9ewf.cif.gz")
    array = array[array.res_name != "HOH"]
    source_coords = array.coord.copy()
    source_bonds = array.bonds.as_array().copy()
    neighbors, _ = array.bonds.get_all_bonds()
    orphan = (array.atom_name == "H10A") & (neighbors[:, 0] < 0)
    assert orphan.sum() == 2

    output = tmp_path / "full_9ewf.tmol"
    database, _ = prepare_ligands(array, seed=20260914, params_output=output)
    np.testing.assert_array_equal(array.coord, source_coords)
    np.testing.assert_array_equal(array.bonds.as_array(), source_bonds)
    neighbors, _ = array.bonds.get_all_bonds()
    assert ((array.atom_name == "H10A") & (neighbors[:, 0] < 0)).sum() == 2

    context = build_context_from_biotite(array, torch_device, param_db=database)
    pose = pose_stack_from_biotite(array, torch_device, context=context, no_optH=True)
    assert int((pose.block_type_ind >= 0).sum()) == 1040
    _assert_all_source_connections(pose, array)

    starts = struc.get_residue_starts(array, add_exclusive_stop=True)
    types = pose.packed_block_types.active_block_types
    for block, (start, stop) in enumerate(zip(starts[:-1], starts[1:])):
        source = array[start:stop]
        observed = (source.element != "H") & np.isfinite(source.coord).all(axis=-1)
        bt = types[int(pose.block_type_ind[0, block])]
        offset = int(pose.block_coord_offset[0, block])
        indices = [offset + bt.atom_to_idx[name] for name in source.atom_name[observed]]
        torch.testing.assert_close(
            pose.coords[0, indices],
            pose.coords.new_tensor(source.coord[observed]),
            atol=1e-5,
            rtol=0,
        )
        if source.res_name[0] == "A1H7V":
            assert "H10A" not in bt.atom_to_idx
            assert {atom.name for atom in bt.atoms} == {
                name for a, b, *_ in bt.bonds for name in (a, b)
            }

    loaded = build_context_from_biotite(
        array,
        torch_device,
        prepare_ligands=True,
        ligand_params_files=[str(output)],
    )
    restored = pose_stack_from_biotite(
        array, torch_device, context=loaded, no_optH=True
    )
    torch.testing.assert_close(restored.coords, pose.coords)
    torch.testing.assert_close(restored.block_type_ind, pose.block_type_ind)
    torch.testing.assert_close(
        restored.inter_residue_connections, pose.inter_residue_connections
    )
    _, minimized = _score_and_minimize(pose, context, max_iter=3)
    assert torch.equal(
        minimized.inter_residue_connections, pose.inter_residue_connections
    )


@pytest.mark.parametrize(
    "fixture,components",
    [
        ("sulfur_attachments_3t14", {"H2S", "S2H"}),
        ("phosphate_attachment_8ch1", {"VDF"}),
    ],
)
def test_small_attachment_frames_and_minimization(fixture, components, torch_device):
    from tmol.io import build_context_from_biotite
    from tmol.ligand._fragmentation import _full_ideal_coords

    array = atom_array_from_cif(DATA / f"{fixture}.cif.gz")
    if "VDF" in components:
        # Exercise the generator length fallback when the attachment is unresolved.
        array.coord[(array.res_name == "VDF") & (array.atom_name == "OP3")] = np.nan
    metals = np.isin(np.char.upper(array.element), ("ZN", "NA", "MG", "CA"))
    assert_metal_bonds_are_coordination(array, metals)
    # Retain every resolved non-water residue, including the complete adducts.
    starts = struc.get_residue_starts(array, add_exclusive_stop=True)
    resolved = np.logical_or.reduceat(np.isfinite(array.coord).all(-1), starts[:-1])
    array = array[
        np.repeat(resolved, np.diff(starts)) & (array.res_name != "HOH") & ~metals
    ]
    context = build_context_from_biotite(
        array, torch_device, prepare_ligands=True, ligand_seed=20260909
    )
    pose = pose_stack_from_biotite(array, torch_device, context=context, no_optH=True)
    _assert_all_source_connections(pose, array)
    types = pose.packed_block_types.active_block_types
    attached = [
        types[i]
        for i in pose.block_type_ind[0].tolist()
        if i >= 0 and types[i].base_name in components
    ]
    assert {bt.base_name for bt in attached} == components
    for bt in attached:
        assert any(c.name.startswith("conj_") for c in bt.connections)
        ideal = _full_ideal_coords(bt)
        assert all(np.isfinite(x).all() for x in ideal.values())
        if bt.base_name == "VDF":
            neighbors = {
                b if a == "OP3" else a for a, b, *_ in bt.bonds if "OP3" in (a, b)
            }
            assert neighbors == {"P"}  # OP3 has no H; its other bond crosses blocks.
            frame = next(ic for ic in bt.icoors if ic.name == "conj_OP3")
            targets = {
                p.x0
                for r in context.parameter_database.scoring.cartbonded.connection_params
                for p in r.angle_parameters
                if p.atm1.lstrip("+") == "P" and p.atm2.lstrip("+") == "OP3"
            }
            assert len(targets) == 1
            assert np.pi - frame.theta == pytest.approx(targets.pop(), abs=1e-8)
            lengths = {
                p.x0
                for r in context.parameter_database.scoring.cartbonded.connection_params
                for p in r.length_parameters
                if "OP3" in (p.atm1.lstrip("+"), p.atm2.lstrip("+"))
            }
            assert len(lengths) == 1
            assert frame.d == pytest.approx(lengths.pop(), abs=1e-8)
        else:
            for a, b, *_ in bt.bonds:
                if a.startswith("H") or b.startswith("H"):
                    assert 1.2 < np.linalg.norm(ideal[a] - ideal[b]) < 1.5
    _, minimized = _score_and_minimize(pose, context)
    assert torch.equal(
        minimized.inter_residue_connections, pose.inter_residue_connections
    )


@pytest.mark.parametrize(
    "code,acyl,carbon,oxygen,departed,links",
    [
        ("2rm9", "GLU", "CD", "OE1", "OE2", 1),
        ("6n0a", "ASN", "CG", "OD1", "ND2", 4),
        ("1hxq", "U5P", "P", "O1P", "O3P", 2),
    ],
)
def test_sidechain_substitutions_retain_chemistry(
    code, acyl, carbon, oxygen, departed, links, torch_device, tmp_path
):
    from tmol.io import build_context_from_biotite
    from tmol.ligand import prepare_ligands
    from tmol.ligand._conjugation_patches import _hydrogens_on
    from tmol.tests.ligand.test_local_conjugate_params import _charges

    prefix = "phosphohistidine" if code == "1hxq" else "isopeptide"
    array = atom_array_from_cif(DATA / f"{prefix}_{code}.cif.gz")
    metals = np.isin(np.char.upper(array.element), ("CA", "ZN", "FE"))
    assert_metal_bonds_are_coordination(array, metals)
    starts = struc.get_residue_starts(array, add_exclusive_stop=True)
    backbone = np.isin(array.atom_name, ("N", "CA", "C"))
    observed = np.isfinite(array.coord).all(-1)
    resolved = (np.add.reduceat(backbone & observed, starts[:-1]) == 3) | (
        array.res_name[starts[:-1]] == "U5P"
    )
    array = array[
        np.repeat(resolved, np.diff(starts)) & ~metals & (array.res_name != "HOH")
    ]
    output = tmp_path / "isopeptide.tmol"
    database, _ = prepare_ligands(array, seed=20260914, params_output=output)
    context = build_context_from_biotite(array, torch_device, param_db=database)
    pose = pose_stack_from_biotite(array, torch_device, context=context, no_optH=True)
    _assert_all_source_connections(pose, array)
    types = pose.packed_block_types.active_block_types
    attached = [types[i] for i in pose.block_type_ind[0].tolist() if i >= 0]
    acyl_types = [
        bt
        for bt in attached
        if bt.base_name == acyl and any(c.atom == carbon for c in bt.connections)
    ]
    partner, site, hydrogens, generic = (
        ("HIS", "NE2", 0, "NG2") if acyl == "U5P" else ("LYS", "NZ", 1, "Nad")
    )
    partners = [
        bt
        for bt in attached
        if bt.base_name == partner and any(c.atom == site for c in bt.connections)
    ]
    assert len(acyl_types) == len(partners) == links
    base = {rt.name: rt for rt in database.chemical.residues}
    for bt in acyl_types:
        assert departed not in bt.atom_to_idx
        assert any(
            {a, b} == {carbon, oxygen} and order == "DOUBLE"
            for a, b, order, *_ in bt.bonds
        )
    for bt in partners:
        assert len(_hydrogens_on(bt, site, database.chemical)) == hydrogens
        assert (
            next(a.genbonded_type or a.atom_type for a in bt.atoms if a.name == site)
            == generic
        )
        if not hydrogens:
            physical = next(a.atom_type for a in bt.atoms if a.name == site)
            traits = next(t for t in database.chemical.atom_types if t.name == physical)
            assert not traits.is_donor and not traits.is_acceptor
    for bt in (*acyl_types, *partners):
        assert sum(_charges(database, bt).values()) == pytest.approx(
            sum(_charges(database, base[bt.base_name]).values()), abs=1e-8
        )
        assert np.isfinite(bt.compute_ideal_coords()).all()
    loaded = build_context_from_biotite(
        array, torch_device, prepare_ligands=True, ligand_params_files=[str(output)]
    )
    restored = pose_stack_from_biotite(
        array, torch_device, context=loaded, no_optH=True
    )
    _assert_all_source_connections(restored, array)
    torch.testing.assert_close(restored.coords, pose.coords)
    _, minimized = _score_and_minimize(pose, context)
    assert torch.equal(
        minimized.inter_residue_connections, pose.inter_residue_connections
    )


def test_ester_and_thioester_contexts_survive_reuse_and_export(torch_device, tmp_path):
    from tmol.io import build_context_from_biotite
    from tmol.ligand import prepare_ligands
    from tmol.pack._packer_task import PackerPalette

    array = atom_array_from_cif(DATA / "attachment_contexts_8trb.cif.gz")
    metals = np.isin(np.char.upper(array.element), ("NA", "ZN"))
    assert_metal_bonds_are_coordination(array, metals)
    starts = struc.get_residue_starts(array, add_exclusive_stop=True)
    backbone = np.isin(array.atom_name, ("N", "CA", "C")) & np.isfinite(
        array.coord
    ).all(-1)
    resolved = (
        np.add.reduceat(backbone, starts[:-1]) == 3
    ) | ~struc.filter_canonical_amino_acids(array)[starts[:-1]]
    array = array[
        np.repeat(resolved, np.diff(starts)) & ~metals & (array.res_name != "HOH")
    ]
    output = tmp_path / "contexts.tmol"
    database, _ = prepare_ligands(array, seed=20260914, params_output=output)
    context = build_context_from_biotite(array, torch_device, param_db=database)
    starts = struc.get_residue_starts(array, add_exclusive_stop=True)
    reverse = array[
        np.concatenate([np.arange(a, b) for a, b in zip(starts[-2::-1], starts[:0:-1])])
    ]
    energies, original_blocks = [], []
    for source in (array, reverse):
        pose = pose_stack_from_biotite(
            source,
            torch_device,
            context=context,
            no_optH=True,
            find_additional_disulfides=False,
        )
        _assert_all_source_connections(pose, source)
        types = pose.packed_block_types.active_block_types
        plm = [
            i for i in pose.block_type_ind[0].tolist() if types[i].base_name == "PLM"
        ]
        assert len({types[i].conjugation_context for i in plm}) == 2
        for i in plm:
            bt = types[i]
            ((site, partner, partner_atom),) = bt.conjugation_context
            assert site == "C1"
            assert "O1" not in bt.atom_to_idx
            assert any(
                {a, b} == {"C1", "O2"} and order == "DOUBLE"
                for a, b, order, *_ in bt.bonds
            )
            assert (partner, partner_atom) in (("SER", "OG"), ("CYS", "SG"))
            assert next(a.atom_type for a in bt.atoms if a.name == "O2") == (
                "Oal" if partner == "SER" else "OG2"
            )
        counts, choices, _ = PackerPalette().block_types_from_original(
            pose.packed_block_types, pose.block_type_ind.to(torch.int64)
        )
        for block, original in enumerate(pose.block_type_ind[0].tolist()):
            if original in plm:
                for candidate in choices[0, block, : int(counts[0, block])].tolist():
                    assert (
                        types[candidate].conjugation_context
                        == types[original].conjugation_context
                    )
        blocks = [types[i] for i in pose.block_type_ind[0].tolist()]
        if source is array:
            original_blocks = blocks
            original_coords = [
                pose.coords[0, offset : offset + bt.n_atoms].clone()
                for bt, offset in zip(blocks, pose.block_coord_offset[0].tolist())
            ]
        else:
            assert [bt.name for bt in blocks] == [
                bt.name for bt in original_blocks[::-1]
            ]
            # Missing sidechains may be packed differently. Test reordered
            # scoring on the same conformation without relaxing its tolerance.
            assert torch.isfinite(pose.coords).all()
            pose = attr.evolve(
                pose, coords=torch.cat(original_coords[::-1]).unsqueeze(0)
            )
        energies.append(_score_and_minimize(pose, context)[0])
    torch.testing.assert_close(energies[0], energies[1], atol=0.002, rtol=1e-5)
    restored_context = build_context_from_biotite(
        array, torch_device, prepare_ligands=True, ligand_params_files=[str(output)]
    )
    restored = pose_stack_from_biotite(
        reverse,
        torch_device,
        context=restored_context,
        no_optH=True,
        find_additional_disulfides=False,
    )
    _assert_all_source_connections(restored, reverse)
    assert [
        restored.packed_block_types.active_block_types[i].name
        for i in restored.block_type_ind[0].tolist()
    ] == [types[i].name for i in pose.block_type_ind[0].tolist()]
    for original, loaded in zip(
        original_blocks[::-1],
        [
            restored.packed_block_types.active_block_types[i]
            for i in restored.block_type_ind[0].tolist()
        ],
    ):
        assert (original.conjugation_context, original.atoms, original.icoors) == (
            loaded.conjugation_context,
            loaded.atoms,
            loaded.icoors,
        )
    restored = attr.evolve(restored, coords=pose.coords.clone())
    energy, _ = _score_and_minimize(restored, restored_context)
    torch.testing.assert_close(energy, energies[-1], atol=0.002, rtol=1e-5)
    # Reuse both contexts for a separate input containing only one attachment.
    ends = array.bonds.as_array()[:, :2]
    pair = next(pair for pair in ends if set(array.res_name[pair]) == {"PLM", "CYS"})
    residues = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
    subset = array[np.isin(residues, residues[pair])]
    reused = pose_stack_from_biotite(
        subset, torch_device, context=context, no_optH=True
    )
    _assert_all_source_connections(reused, subset)
    plm = next(
        reused.packed_block_types.active_block_types[i]
        for i in reused.block_type_ind[0].tolist()
        if reused.packed_block_types.active_block_types[i].base_name == "PLM"
    )
    assert plm.conjugation_context == (("C1", "CYS", "SG"),)
    assert next(a.atom_type for a in plm.atoms if a.name == "O2") == "OG2"
    _score_and_minimize(reused, context)
