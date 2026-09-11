"""Generic lookup references must not transfer canonical torsion ownership."""

import attr
import pytest
import torch
import yaml

from tmol.chemical import ResidueTypeSet
from tmol.score.genbonded import GenBondedEnergyTerm


def _lysine(default_database, references):
    raw = next(r for r in default_database.chemical.residues if r.name == "LYS")
    return ResidueTypeSet._refine(
        attr.evolve(
            raw,
            atoms=tuple(
                attr.evolve(a, genbonded_type=references.get(a.name)) for a in raw.atoms
            ),
        )
    )


def test_lookup_reference_preserves_canonical_chi_ownership(default_database):
    term = GenBondedEnergyTerm(default_database, torch.device("cpu"))
    bt = _lysine(default_database, {"CD": "CS2", "CE": "CS2", "NZ": "Nam"})
    quad = tuple(bt.atom_to_idx[name] for name in ("CG", "CD", "CE", "NZ"))
    # This axis would acquire a nonzero generic torsion if reference types
    # determined ownership, adding to the existing LYS chi4 Dunbrack term.
    entry = term.gen_database.find_torsion_params(
        *(term.get_atom_chem_type(bt, i) for i in quad), 1, False
    )
    assert entry is not None and any((entry.k1, entry.k2, entry.k3, entry.k4))
    assert term.resolve_torsion_params(bt, [quad])[0] == []


@pytest.mark.parametrize("physical_center", ["Nbb", "Nad"])
def test_intra_improper_uses_physical_center_ownership(
    default_database, physical_center
):
    from types import SimpleNamespace
    from tmol.database.chemical import Atom

    bt = SimpleNamespace(
        atoms=(
            Atom("N", physical_center, "Nad"),
            Atom("C", "CNH2", "CDp"),
            Atom("CA", "CH2", "CS2"),
            Atom("H", "Hpol", "HN"),
        )
    )
    term = GenBondedEnergyTerm(default_database, torch.device("cpu"))
    kept, params = term.resolve_improper_params(bt, [(0, 1, 2, 3)])
    assert kept == ([(0, 1, 2, 3)] if physical_center == "Nad" else [])
    if kept:
        assert params.tolist() == [[80.0, 0.0]]


@pytest.mark.parametrize("reference", ["missing_type", "HN", "C*", ""])
def test_invalid_reference_is_rejected(default_database, reference):
    term = GenBondedEnergyTerm(default_database, torch.device("cpu"))
    bt = _lysine(default_database, {"CE": reference})
    for _ in range(2):
        with pytest.raises(ValueError, match="LYS atom CE: invalid genbonded_type"):
            term.setup_block_type(bt)
        assert not hasattr(bt, "genbonded_intra_subgraphs")


def _attached_amide(default_database, torch_device, generic_center, references):
    from tmol.io import atom_array_from_cif, pose_stack_from_biotite
    from tmol.ligand import prepare_ligands
    from tmol.tests.data import data_path
    from tmol.tests.pack.test_conjugated_group_packing import FIXTURES

    array = atom_array_from_cif(
        data_path("covalent_fixtures", FIXTURES["biotin"] + ".cif")
    )
    database, _ = prepare_ligands(array, param_db=default_database, seed=20250828)
    changed = []
    for rt in database.chemical.residues:
        if rt.base_name != "LYS" or not any(c.atom == "NZ" for c in rt.connections):
            changed.append(rt)
            continue
        atoms = tuple(
            attr.evolve(
                a,
                atom_type=(
                    ("Nad" if generic_center else "Nbb")
                    if a.name == "NZ"
                    else a.atom_type
                ),
                genbonded_type=(
                    {"NZ": "Nad", "CE": "CS2", "HZ1": "HN"}.get(a.name)
                    if references
                    else None
                ),
            )
            for a in rt.atoms
        )
        changed.append(attr.evolve(rt, atoms=atoms))
    gen = database.scoring.genbonded
    database = attr.evolve(
        database,
        chemical=attr.evolve(database.chemical, residues=tuple(changed)),
        scoring=attr.evolve(
            database.scoring,
            genbonded=attr.evolve(
                gen,
                torsions=(),
                impropers=tuple(
                    p for p in gen.impropers if p.atoms == ("Nad", "CDp", "CS2", "HN")
                ),
            ),
        ),
    )
    pose = pose_stack_from_biotite(array, torch_device, param_db=database, no_optH=True)
    return pose, database


@pytest.mark.parametrize("generic_center", [False, True])
@pytest.mark.parametrize("references", [False, True])
@pytest.mark.parametrize("block_pairs", [False, True])
def test_attached_amide_reference_energy_and_gradient(
    default_database, torch_device, generic_center, references, block_pairs
):
    pose, database = _attached_amide(
        default_database, torch_device, generic_center, references
    )
    term = GenBondedEnergyTerm(database, torch_device)
    pbt = pose.packed_block_types
    for bt in pbt.active_block_types:
        term.setup_block_type(bt)
    term.setup_packed_block_types(pbt)
    term.setup_poses(pose)
    render = (
        term.render_block_pair_scoring_module
        if block_pairs
        else term.render_whole_pose_scoring_module
    )
    module = render(pose)
    coords = pose.coords.double().clone()
    attachment = next(
        (bi, pbt.active_block_types[ti])
        for bi, ti in enumerate(pose.block_type_ind[0].tolist())
        if pbt.active_block_types[ti].base_name == "LYS"
        and any(c.atom == "NZ" for c in pbt.active_block_types[ti].connections)
    )
    bi, bt = attachment
    hydrogen = int(pose.block_coord_offset[0, bi]) + bt.atom_to_idx["HZ1"]
    coords[0, hydrogen] += coords.new_tensor([0.4, -0.3, 0.2])
    coords.requires_grad_(True)
    values = module(coords)
    # Asymmetric weights exercise the block-pair backward path.
    weights = (
        (0.4 + torch.arange(values.numel(), device=torch_device).reshape_as(values))
        if block_pairs
        else torch.ones_like(values)
    )
    total = (values * weights).sum()
    gradient = torch.autograd.grad(total, coords)[0]
    if not (generic_center and references):
        assert total.item() == 0.0
        assert torch.count_nonzero(gradient) == 0
        return
    assert total.item() > 0.1
    assert torch.isfinite(gradient).all()
    ci = next(i for i, c in enumerate(bt.connections) if c.atom == "NZ")
    partner = int(pose.inter_residue_connections[0, bi, ci, 0])
    other = pbt.active_block_types[int(pose.block_type_ind[0, partner])]
    # The fixture orders these neighbors C11, CE, HZ1 in the scoring contract.
    assert other.atom_to_idx["C11"] < bt.atom_to_idx["CE"] < bt.atom_to_idx["HZ1"]
    local = int(pose.block_coord_offset[0, bi])
    remote = int(pose.block_coord_offset[0, partner])
    a, b, c, d = (
        coords[0, local + bt.atom_to_idx["NZ"]],
        coords[0, remote + other.atom_to_idx["C11"]],
        coords[0, local + bt.atom_to_idx["CE"]],
        coords[0, hydrogen],
    )
    first = torch.linalg.cross(b - a, c - b)
    second = torch.linalg.cross(c - b, d - c)
    theta = torch.acos(first.dot(second) / (first.norm() * second.norm()))
    expected = 80 * theta.square()
    if block_pairs:
        expected *= weights[0, 0, min(bi, partner), max(bi, partner)]
    torch.testing.assert_close(total, expected, rtol=1e-7, atol=1e-7)
    expected_gradient = torch.autograd.grad(expected, coords)[0]
    torch.testing.assert_close(gradient, expected_gradient, rtol=2e-6, atol=2e-5)
    for axis in range(3):
        plus, minus = coords.detach().clone(), coords.detach().clone()
        plus[0, hydrogen, axis] += 1e-5
        minus[0, hydrogen, axis] -= 1e-5
        numeric = ((module(plus) - module(minus)) * weights).sum() / 2e-5
        torch.testing.assert_close(
            numeric, gradient[0, hydrogen, axis], rtol=2e-6, atol=2e-5
        )


@pytest.mark.parametrize("generic_center", [False, True])
def test_rotamer_attachment_improper_reference(
    default_database, torch_device, generic_center
):
    from tmol.pack import SetPackerTask
    from tmol.pack.rotamer import build_rotamers
    from tmol.tests.pack.test_conjugated_group_packing import _task

    pose, database = _attached_amide(
        default_database, torch_device, generic_center, True
    )
    task, _ = _task(pose, database, torch_device)
    task.set_chi_sample_budget(1024, 512)
    pose, rotamers = build_rotamers(
        pose, SetPackerTask.from_packer_task(task), database.chemical
    )
    term = GenBondedEnergyTerm(database, torch_device)
    for bt in pose.packed_block_types.active_block_types:
        term.setup_block_type(bt)
    term.setup_packed_block_types(pose.packed_block_types)
    term.setup_poses(pose)
    module = term.render_rotamer_scoring_module(pose, rotamers)
    coords = rotamers.coords.double().clone()
    coords += 0.13 * torch.sin(
        torch.arange(coords.numel(), device=torch_device)
    ).reshape_as(coords)
    coords.requires_grad_(True)
    scores, indices = module(coords)
    pbt = pose.packed_block_types
    bts = [pbt.active_block_types[t] for t in pose.block_type_ind[0].tolist()]
    first = next(
        i
        for i, bt in enumerate(bts)
        if bt.base_name == "LYS" and any(c.atom == "NZ" for c in bt.connections)
    )
    bt = bts[first]
    ci = next(i for i, c in enumerate(bt.connections) if c.atom == "NZ")
    second = int(pose.inter_residue_connections[0, first, ci, 0])
    r1, r2 = indices[1].long(), indices[2].long()
    cross = (rotamers.block_ind_for_rot[r1] == first) & (
        rotamers.block_ind_for_rot[r2] == second
    )
    assert int(cross.sum()) > 1

    def xyz(rots, block, name):
        return coords[
            rotamers.coord_offset_for_rot[rots].long() + bts[block].atom_to_idx[name]
        ]

    a = xyz(r1[cross], first, "NZ")
    b = xyz(r2[cross], second, "C11")
    c = xyz(r1[cross], first, "CE")
    d = xyz(r1[cross], first, "HZ1")
    n1 = torch.linalg.cross(b - a, c - b)
    n2 = torch.linalg.cross(c - b, d - c)
    theta = torch.acos((n1 * n2).sum(-1) / (n1.norm(dim=-1) * n2.norm(dim=-1)))
    expected = (80 if generic_center else 0) * theta.square()
    actual = scores[0, cross]
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-5)
    weights = 0.3 + torch.arange(actual.numel(), device=torch_device) / actual.numel()
    actual_grad = torch.autograd.grad(
        (weights * actual).sum(), coords, retain_graph=True
    )[0]
    expected_grad = torch.autograd.grad((weights * expected).sum(), coords)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-6, atol=2e-5)


@pytest.mark.parametrize("in_patch", [False, True])
def test_reference_bundle_version_and_roundtrip(tmp_path, default_database, in_patch):
    from dataclasses import replace
    from tmol.ligand import load_params_file, write_params_file
    from tmol.tests.ligand.test_ligand_entry_paths import _single_prep

    prep = _single_prep()
    if in_patch:
        patch = next(v for v in default_database.chemical.variants if v.modify_atoms)
        atoms = tuple(
            attr.evolve(a, genbonded_type=a.atom_type) for a in patch.modify_atoms
        )
        prep = replace(prep, adds_patches=(attr.evolve(patch, modify_atoms=atoms),))
    else:
        rt = prep.residue_type
        prep = replace(
            prep,
            residue_type=attr.evolve(
                rt,
                atoms=tuple(
                    attr.evolve(a, genbonded_type=a.atom_type) for a in rt.atoms
                ),
            ),
        )
    path = tmp_path / "refs.tmol"
    write_params_file(prep, path, format="tmol")
    assert yaml.safe_load(path.read_text())["version"] == "3.0"
    loaded = load_params_file(path)[0]
    assert loaded.residue_type.atoms == prep.residue_type.atoms
    assert loaded.adds_patches == prep.adds_patches
    with pytest.raises(ValueError, match="cannot preserve genbonded_type"):
        write_params_file(prep, tmp_path / "refs.params", format="rosetta")
    assert not (tmp_path / "refs.params").exists()
