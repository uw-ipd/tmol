"""Small actual-protein scoring smoke test without skip-on-error fixtures."""

from pathlib import Path

import jax
import numpy as np

from tmol_jax import pose_stack_from_pdb, beta2016_score_function


def _tripeptide_pdb(tmp_path):
    source = Path(__file__).parents[2] / "tmol/tests/data/pdb/1ubq.pdb"
    lines = [
        line
        for line in source.read_text().splitlines()
        if line.startswith("ATOM") and 1 <= int(line[22:26]) <= 3
    ]
    path = tmp_path / "tripeptide.pdb"
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


def test_protein_score_and_gradient_are_finite(tmp_path):
    path = _tripeptide_pdb(tmp_path)
    pose = pose_stack_from_pdb(path)
    from biotite.structure.io.pdb import PDBFile
    from biotite.structure import stack
    from tmol_jax.io import canonical_form_from_pdb
    from tmol_jax.io.pose_stack_from_biotite import canonical_form_from_biotite

    atoms = PDBFile.read(path).get_structure(model=1)
    canonical = canonical_form_from_pdb(path)
    batched = canonical_form_from_biotite(stack([atoms, atoms]))
    for field in ("coords", "chain_id", "res_types", "atom_is_present"):
        np.testing.assert_array_equal(
            batched[field], np.repeat(canonical[field], 2, axis=0)
        )
    score_function = beta2016_score_function()
    energy, gradient = score_function.jitted_value_and_grad(pose)(pose.coords)
    assert len(pose.packed_block_types.active_block_types) == 3
    # Cartesian setup must depend on topology, even when a real atom is at zero.
    cart = next(
        term
        for term in score_function.terms
        if type(term).__name__ == "CartBondedEnergyTerm"
    )
    shifted = pose.replace(coords=pose.coords - pose.coords[:, :1])
    first = cart.setup_for_pose_stack(pose)
    second = cart.setup_for_pose_stack(shifted)
    for left, right in zip(jax.tree.leaves(first), jax.tree.leaves(second)):
        np.testing.assert_array_equal(left, right)
    moved = pose.to(jax.devices()[0])
    np.testing.assert_array_equal(moved.coords, pose.coords)
    assert moved.device == pose.device
    assert energy.shape == (1,)
    assert gradient.shape == pose.coords.shape
    assert np.isfinite(energy).all()
    assert np.isfinite(gradient).all()
    assert np.linalg.norm(gradient) > 0


def test_mixed_batch_matches_individual_scoring_and_masks_padding(tmp_path):
    from tmol_jax.io import canonical_form_from_pdb, default_packed_block_types
    from tmol_jax.io import pose_stack_from_canonical_form

    canonical = canonical_form_from_pdb(_tripeptide_pdb(tmp_path))
    batch = {name: np.repeat(value, 2, axis=0) for name, value in canonical.items()}
    batch["res_name"][1, -1] = ""
    batch["res_types"][1, -1] = -1
    batch["chain_id"][1, -1] = -1
    batch["coords"][1, -1] = np.nan
    batch["atom_is_present"][1, -1] = False
    pose = pose_stack_from_canonical_form(batch, default_packed_block_types())
    sf = beta2016_score_function()
    energies, gradient = sf.jitted_value_and_grad(pose)(pose.coords)
    assert np.isfinite(energies).all() and np.isfinite(gradient).all()
    np.testing.assert_array_equal(
        np.asarray(gradient)[~np.asarray(pose.atom_is_real)], 0
    )
    for index in range(2):
        individual = pose[index]
        expected = sf.jitted(individual)(individual.coords)
        np.testing.assert_allclose(energies[index], expected[0], atol=1e-4)


def test_pdb_parser_uses_first_model():
    from tmol_jax.io.pdb_parsing import parse_pdb

    def atom(serial, x):
        return (
            f"ATOM  {serial:5d}  CA  ALA A   1    "
            f"{x:8.3f}{2.0:8.3f}{3.0:8.3f}  1.00 10.00           C  "
        )

    text = "\n".join(
        [
            "MODEL        1",
            atom(1, 1.0),
            "ENDMDL",
            "MODEL        2",
            atom(1, 9.0),
            "ENDMDL",
            "END",
        ]
    )
    records = parse_pdb(text)
    np.testing.assert_array_equal(records["coords"], [[1.0, 2.0, 3.0]])
