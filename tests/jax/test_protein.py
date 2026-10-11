"""Small actual-protein scoring smoke test without skip-on-error fixtures."""

from pathlib import Path

import jax
import numpy as np

from tmol_jax import pose_stack_from_pdb, beta2016_score_function


def test_protein_score_and_gradient_are_finite(tmp_path):
    source = Path(__file__).parents[2] / "tmol/tests/data/pdb/1ubq.pdb"
    lines = [
        line
        for line in source.read_text().splitlines()
        if line.startswith("ATOM") and 1 <= int(line[22:26]) <= 3
    ]
    path = tmp_path / "tripeptide.pdb"
    path.write_text("\n".join(lines) + "\nEND\n")
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
