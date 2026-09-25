"""Renamed backbones and terminal caps retain curated peptide harmonic forces."""

import attr
import pytest
import torch

from tmol.io import pose_stack_from_cif
from tmol.tests.data import data_path
from tmol.tests.score.cartbonded.test_explicit_connection_parameters import scoring_term


@pytest.mark.parametrize("junction", ["fixture", "renamed", "supplied"])
def test_gamma_junction_length_angle_energy_and_gradient(
    torch_device, tmp_path, junction
):
    from tmol.database import ParameterDatabase, inject_residue_params
    from tmol.database.scoring import LengthGroup
    from tmol.io import create_pose_stack_from_sequences
    from tmol.ligand import prepare_polymer_residue, write_params_file, load_params_file
    from tmol.ligand._registry import (
        inject_ligand_preparations,
        rebuild_canonical_ordering,
    )
    from tmol.tests.ligand.test_nonstandard_backbones import _residue

    if junction == "fixture":
        pose, context = pose_stack_from_cif(
            data_path("ncaa_fixtures") / "gamma_peptide_1gac.cif",
            torch_device,
            prepare_ligands=True,
            ligand_seed=20260909,
            no_optH=True,
            return_context=True,
        )
        database = context.parameter_database
    else:
        array = _residue("FGA")
        array.atom_name[array.atom_name == "N"] = "NX"
        database = ParameterDatabase.get_default()
        prep = prepare_polymer_residue(
            array,
            rebuild_canonical_ordering(database),
            database,
            connection_atoms={"NX", "CD"},
            seed=20260909,
        )
        # The new reference metadata must survive the supported public format.
        path = tmp_path / "junction.tmol"
        write_params_file(prep, path)
        loaded = load_params_file(path)
        assert loaded[0].residue_type.atoms == prep.residue_type.atoms
        database = inject_ligand_preparations(database, loaded)
        if junction == "supplied":
            rows = database.scoring.cartbonded.residue_params["FGA"]
            rows = attr.evolve(
                rows,
                length_parameters=(
                    *rows.length_parameters,
                    LengthGroup("CD", "+NX", x0=1.41, K=401.0),
                ),
            )
            database = inject_residue_params(
                database, residue_types=[], cartbonded_params={"FGA": rows}
            )
        pose = create_pose_stack_from_sequences(
            "X[FGA]X[FGA]",
            param_db=database,
            device=torch_device,
            termini=False,
        )
        # Distorted, nondegenerate coordinates exercise forces independently of
        # any coordinate or conformer generator's numerical output.
        pose.coords.copy_(
            torch.randn(
                pose.coords.shape, generator=torch.Generator().manual_seed(19)
            ).to(torch_device)
        )
    pbt = pose.packed_block_types
    types = [pbt.active_block_types[i] for i in pose.block_type_ind[0].tolist()]
    left = next(i for i, bt in enumerate(types) if bt.base_name == "FGA")
    right = int(
        pose.inter_residue_connections[0, left, types[left].up_connection_ind, 0]
    )
    assert right > left
    assert types[right].base_name != "PRO"
    coords = pose.coords.double().clone()
    coords += 0.07 * torch.sin(
        torch.arange(coords.numel(), device=torch_device).reshape_as(coords)
    )
    coords.requires_grad_(True)
    module = scoring_term(pose, database).render_block_pair_scoring_module(pose)
    all_terms = module(coords)[:, 0, left, right]
    actual = all_terms[:2]

    def xyz(block, name):
        return coords[
            0, int(pose.block_coord_offset[0, block]) + types[block].atom_to_idx[name]
        ]

    def constant(value):
        # Native parameter storage is float32 even when coordinates are double.
        return float(torch.tensor(value, dtype=torch.float32))

    nitrogen = "N" if junction == "fixture" else "NX"
    elements = {a.name: a.element for a in database.chemical.atom_types}
    hydrogens = {a.name for a in types[right].atoms if elements[a.atom_type] == "H"}
    hydrogen = next(
        b if a == nitrogen else a
        for a, b, *_ in types[right].bonds
        if (a == nitrogen and b in hydrogens) or (b == nitrogen and a in hydrogens)
    )
    c, n = xyz(left, "CD"), xyz(right, nitrogen)
    if junction == "renamed":
        # Full Cartesian term parity with the same chemistry under its original
        # names also covers proper/improper torsions and their multiplicities.
        original = ParameterDatabase.get_default()
        prep = prepare_polymer_residue(
            _residue("FGA"),
            rebuild_canonical_ordering(original),
            original,
            connection_atoms={"N", "CD"},
            seed=20260909,
        )
        original = inject_ligand_preparations(original, [prep])
        reference = create_pose_stack_from_sequences(
            "X[FGA]X[FGA]",
            param_db=original,
            device=torch_device,
            termini=False,
        )

        def hydrogens_by_parent(bt):
            """Hydrogen names keyed by (parent name, rank among its hydrogens)."""
            names = {a.name for a in bt.atoms if elements[a.atom_type] == "H"}
            parents = {}
            for a, b, *_ in bt.bonds:
                for h, heavy in ((a, b), (b, a)):
                    if h in names:
                        parents.setdefault(heavy, []).append(h)
            return {
                (heavy, rank): h
                for heavy, hs in parents.items()
                for rank, h in enumerate(sorted(hs))
            }

        heavy_name = {"N": nitrogen}
        indices = []
        for block, type_index in enumerate(reference.block_type_ind[0].tolist()):
            bt = reference.packed_block_types.active_block_types[type_index]
            # hydrogens are named after their parent, so match them through it
            renamed_h = hydrogens_by_parent(types[block])
            name_for = {
                h: renamed_h[(heavy_name.get(heavy, heavy), rank)]
                for (heavy, rank), h in hydrogens_by_parent(bt).items()
            }
            for atom in bt.atoms:
                name = name_for.get(atom.name, heavy_name.get(atom.name, atom.name))
                indices.append(
                    int(pose.block_coord_offset[0, block])
                    + types[block].atom_to_idx[name]
                )
        reference_terms = scoring_term(
            reference, original
        ).render_block_pair_scoring_module(reference)(coords[:, indices])[:, 0, 0, 1]
        torch.testing.assert_close(all_terms, reference_terms, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(
            torch.autograd.grad(all_terms.sum(), coords, retain_graph=True)[0],
            torch.autograd.grad(reference_terms.sum(), coords, retain_graph=True)[0],
            atol=2e-5,
            rtol=2e-5,
        )
    # Independent Cartesian formulas with the database's peptide constants;
    # no generated row traversal or tmol geometry/derivative helper is used.
    x0, stiffness = (1.41, 401.0) if junction == "supplied" else (1.32868, 369.445)
    length = 0.5 * constant(stiffness) * ((c - n).norm() - constant(x0)).square()
    angle = coords.new_zeros(())
    for a, b, d, optimum, stiffness in (
        (xyz(left, "CG"), c, n, 2.02807, 160.0),
        (xyz(left, "OE1"), c, n, 2.14676, 170.864),
        (c, n, xyz(right, "CA"), 2.12407, 96.53),
        (c, n, xyz(right, hydrogen), 2.07956, 76.432),
    ):
        first, second = a - b, d - b
        theta = torch.acos(first.dot(second) / (first.norm() * second.norm()))
        angle += 0.5 * constant(stiffness) * (theta - constant(optimum)).square()
    expected = torch.stack((length, angle))
    assert float(expected.detach().sum()) > 0.01
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    weights = coords.new_tensor((0.7, 1.9))
    actual_grad = torch.autograd.grad(
        (actual * weights).sum(), coords, retain_graph=True
    )[0]
    expected_grad = torch.autograd.grad((expected * weights).sum(), coords)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(
    "cap,connection,sequence",
    [
        ("MYR", "C1", "X[MYR]G"),
        ("DIY", "N1", "EX[DIY]"),
        ("4SO", "CAH", "X[4SO]G"),
        ("NH2", "N", "GX[NH2]"),
        ("ACE", "C", "X[ACE]G"),
        ("NME", "N", "GX[NME]"),
    ],
)
def test_cap_junction_forces(torch_device, tmp_path, cap, connection, sequence):
    """Common caps and caps in 3EPC/1A1E/6Q9T retain peptide restoring forces."""
    from tmol.database import ParameterDatabase
    from tmol.database.scoring import AngleGroup, ConnectionCartRes, LengthGroup
    from tmol.io import create_pose_stack_from_sequences
    from tmol.ligand import prepare_polymer_residue, write_params_file, load_params_file
    from tmol.ligand._registry import (
        inject_ligand_preparations,
        rebuild_canonical_ordering,
    )
    from tmol.tests.ligand.test_nonstandard_backbones import _residue_bonded_only_at
    from tmol.ligand._polymer_profile import cap_polymer_profile
    from tmol.tests.score.cartbonded.test_explicit_connection_parameters import (
        harmonic_reference,
    )

    database = ParameterDatabase.get_default()
    source = _residue_bonded_only_at(cap, connection)
    prep = prepare_polymer_residue(
        source,
        rebuild_canonical_ordering(database),
        database,
        connection_atoms={connection},
        profile=cap_polymer_profile(source, connection, database.chemical),
        seed=20260909,
    )
    assert set(source.atom_name[source.element != "H"]) <= {
        atom.name for atom in prep.residue_type.atoms
    }
    path = tmp_path / "cap.tmol"
    write_params_file(prep, path)
    loaded = load_params_file(path)
    assert loaded[0].residue_type.atoms == prep.residue_type.atoms
    database = inject_ligand_preparations(database, loaded)
    pose = create_pose_stack_from_sequences(
        sequence, param_db=database, device=torch_device, termini=False
    )
    if cap == "NH2":
        c, n, before, oxygen, after, last = "C", "+N", "CA", "O", "+H1", "+H2"
        after_x0, after_k = 2.07956, 76.432
        last_x0, last_k = after_x0, after_k
    elif cap in ("MYR", "4SO", "ACE"):
        c, before, oxygen = {
            "MYR": ("C1", "C2", "O1"),
            "4SO": ("CAH", "CAI", "OAB"),
            "ACE": ("C", "CH3", "O"),
        }[cap]
        n, after, last = "+N", "+CA", "+H"
        last_x0, last_k = 2.07956, 76.432
    elif cap == "NME":
        c, n, before, oxygen, after, last = "C", "+N", "CA", "O", "+C", "+H"
        last_x0, last_k = 2.07956, 76.432
    else:
        c, n, before, oxygen, after, last = "C", "+N1", "CA", "O", "+C2", "+C6"
        last_x0, last_k = 1.9548, 125.184
    if cap != "NH2":
        after_x0, after_k = 2.12407, 96.53
    record = ConnectionCartRes(
        block_type1="",
        connection1="up",
        block_type2="",
        connection2="down",
        length_parameters=(LengthGroup(c, n, x0=1.32868, K=369.445),),
        angle_parameters=tuple(
            AngleGroup(a, b, d, x0=x0, K=k)
            for a, b, d, x0, k in (
                (before, c, n, 2.02807, 160.0),
                (oxygen, c, n, 2.14676, 170.864),
                (c, n, after, after_x0, after_k),
                (c, n, last, last_x0, last_k),
            )
        ),
    )
    coords = torch.randn(pose.coords.shape, generator=torch.Generator().manual_seed(19))
    coords = coords.to(device=torch_device, dtype=torch.float64).requires_grad_(True)
    module = scoring_term(pose, database).render_block_pair_scoring_module(pose)
    terms = module(coords)[:, 0, 0, 1]
    assert float(terms[3].detach()) > 1e-6  # Cross-junction planarity stays active.
    actual = terms[:2]
    expected = harmonic_reference(pose, coords, record)
    assert bool((expected > 0.01).all())
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), coords, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), coords)[0],
        atol=2e-5,
        rtol=2e-5,
    )
