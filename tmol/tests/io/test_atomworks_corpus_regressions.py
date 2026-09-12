"""Difficult, unmodified AtomWorks fixtures and their explicit input contracts."""

from pathlib import Path

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


def test_terminal_nucleoside_keeps_its_backbone_and_minimizes(torch_device):
    pose, context = pose_stack_from_cif(
        DATA / "terminal_nucleotide_145d.cif",
        torch_device,
        prepare_ligands=True,
        ligand_seed=20260909,
        no_optH=True,
        return_context=True,
    )
    assert int((pose.block_type_ind >= 0).sum()) == 24
    types = [
        pose.packed_block_types.active_block_types[int(i)]
        for i in pose.block_type_ind[0]
    ]
    assert all(bt.properties.polymer.backbone_type == "dna" for bt in types)
    assert types[0].name == "MCY:na5prime"
    # Four six-residue strands: proximity between strands adds no conjugations.
    assert int((pose.inter_residue_connections[..., 0] >= 0).sum()) == 40
    assert not any("conj_" in bt.name for bt in types)
    _score_and_minimize(pose, context)


def _score_and_minimize(pose, context):
    from tmol.optimization import CartesianSfxnNetwork, LBFGS_Armijo

    assert torch.isfinite(pose.coords[pose.real_atoms]).all()
    score = beta2016_score_function(pose.device, param_db=context.parameter_database)
    network = CartesianSfxnNetwork(score, pose)
    initial = network().detach()
    optimizer = LBFGS_Armijo(
        network.parameters(), max_iter=10, segment_ids=network.segment_ids
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
    return initial


@pytest.mark.parametrize("reader", ["tmol", "atomworks"])
def test_macrocycle_preserves_every_bond_across_residue_order(reader, torch_device):
    from tmol.io import build_context_from_biotite, pose_stack_from_biotite

    array = atom_array_from_cif(DATA / "macrocycle_1xvk.cif", reader=reader)
    # Free magnesium is deferred; waters follow the constructor's usual policy.
    array = array[(np.char.upper(array.element) != "MG") & (array.res_name != "HOH")]
    context = build_context_from_biotite(
        array, torch_device, prepare_ligands=True, ligand_seed=20260909
    )
    residue = next(r for r in context.restype_set.residue_types if r.name == "QUI")
    assert {"N1", "C2", "O1"} <= set(residue.atom_to_idx)
    assert np.isfinite(residue.compute_ideal_coords()).all()
    starts = struc.get_residue_starts(array, add_exclusive_stop=True)
    assert len(starts) - 1 == 18
    atom_res = np.repeat(np.arange(18), np.diff(starts))
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
    assert len(expected) == 18
    energies = []
    for order in (np.arange(18), np.arange(18)[::-1]):
        indices = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in order])
        pose = pose_stack_from_biotite(
            array[indices], torch_device, context=context, no_optH=True
        )
        types = [
            pose.packed_block_types.active_block_types[int(i)]
            for i in pose.block_type_ind[0]
        ]
        actual = set()
        for block, connections in enumerate(
            pose.inter_residue_connections[0].cpu().tolist()
        ):
            for conn, (partner, port) in enumerate(connections):
                if partner >= 0:
                    actual.add(
                        frozenset(
                            (
                                (
                                    int(order[block]),
                                    types[block].connections[conn].atom,
                                ),
                                (
                                    int(order[partner]),
                                    types[partner].connections[port].atom,
                                ),
                            )
                        )
                    )
        assert actual == expected
        energies.append(_score_and_minimize(pose, context))
    torch.testing.assert_close(energies[0], energies[1], atol=0.002, rtol=1e-5)


def test_unknown_heavy_atom_is_not_silently_deleted():
    # The fixture deliberately conflicts: label XYZ versus author CG. The
    # native reader legitimately uses CG; the label-based view must reject XYZ.
    cif = pdbx.CIFFile.read(DATA / "unknown_heavy_atom_1a8o.cif")
    array = pdbx.get_structure(cif, model=1, use_author_fields=False)
    assert np.count_nonzero(array.atom_name == "XYZ") == 1
    with pytest.raises(ValueError, match="Heavy atoms.*ASP.*XYZ"):
        canonical_form_from_biotite(array, torch.device("cpu"))


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


def test_schiff_base_cannot_lose_its_incomplete_lysine_partner():
    with pytest.raises(ValueError, match="discard.*covalent"):
        pose_stack_from_cif(
            DATA / "schiff_base_double_bond.cif",
            torch.device("cpu"),
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
