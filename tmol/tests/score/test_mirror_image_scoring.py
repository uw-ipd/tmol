"""A structure and its mirror image must score alike.

The mirror of an L peptide is the D peptide with every coordinate inverted.
Every term in the score function is either invariant under that operation
(distances) or has a mirrored counterpart supplied for the D residue types
(the backbone and rotamer torsion potentials), so the totals must agree.

This is the gate on D-amino-acid support: a term that silently returns zero for
a D residue, or reads its L counterpart's tables without negating torsions,
shows up here as a per-term mismatch without anyone having to guess which
lookups key on residue name.

Glycine is achiral but its tables are not, so the comparison uses the database
that selects the symmetrized glycine tables.
"""

import numpy
import pytest


from tmol.database import ParameterDatabase
from tmol.io import atom_array_from_cif, pose_stack_from_cif
from tmol.score import beta2016_score_function
from tmol.tests.data import data_path

FIXTURE_DIR = data_path("ncaa_fixtures")
MIRROR_PAIR = "6dmz_mod"


def _pose(stem, param_db, torch_device):
    # hydrogen placement must not be optimized: optH would break the mirror
    # symmetry it is being used to measure
    return pose_stack_from_cif(
        FIXTURE_DIR / f"{stem}.cif", torch_device, param_db=param_db, no_optH=True
    )


@pytest.fixture(scope="module")
def symmetric_gly_db():
    return ParameterDatabase.get_default().with_symmetric_gly()


def _scores_by_term(pose_stack, sfxn):
    module = sfxn.render_whole_pose_scoring_module(pose_stack)
    unweighted = module.unweighted_scores(pose_stack.coords)
    return {
        score_type: float(unweighted[i, 0])
        for i, score_type in enumerate(sfxn.all_score_types())
    }


def test_mirror_image_coordinates_are_exact_negations() -> None:
    """The fixtures are a mirror pair, so the comparison means what it says."""
    left = atom_array_from_cif(FIXTURE_DIR / f"{MIRROR_PAIR}_l.cif")
    right = atom_array_from_cif(FIXTURE_DIR / f"{MIRROR_PAIR}_d.cif")
    assert left.array_length() == right.array_length()
    numpy.testing.assert_allclose(left.coord, -right.coord, atol=1e-4)
    assert list(left.atom_name) == list(right.atom_name)


def test_mirror_image_scores_identically(symmetric_gly_db, torch_device) -> None:
    left = _pose(f"{MIRROR_PAIR}_l", symmetric_gly_db, torch_device)
    right = _pose(f"{MIRROR_PAIR}_d", symmetric_gly_db, torch_device)
    sfxn = beta2016_score_function(torch_device, param_db=symmetric_gly_db)

    left_scores = _scores_by_term(left, sfxn)
    right_scores = _scores_by_term(right, sfxn)

    def differs(left_value, right_value):
        tolerance = max(1e-4, 5e-5 * max(abs(left_value), abs(right_value)))
        return abs(left_value - right_value) > tolerance

    differing = {
        str(score_type): (left_scores[score_type], right_scores[score_type])
        for score_type in left_scores
        if differs(left_scores[score_type], right_scores[score_type])
    }
    assert (
        not differing
    ), f"terms differ between a structure and its mirror: {differing}"


def test_repacking_a_d_structure_keeps_it_d(symmetric_gly_db, torch_device) -> None:
    """Repacking must not quietly turn a D residue into its L form.

    The rotamers a D residue is offered come from its own mirrored library, so
    a lookup that fell back to the L tables would either build no rotamers at
    all or place the sidechain as its mirror image. Only the handedness is
    checked: a tautomer or protonation state is a packing degree of freedom.
    """
    from tmol.pack import PackerPalette, PackerTask, pack_rotamers
    from tmol.pack.rotamer import FixedAAChiSampler, IncludeCurrentSampler
    from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database

    pose_stack = _pose(f"{MIRROR_PAIR}_d", symmetric_gly_db, torch_device)
    before = _chirality(pose_stack)
    assert "d" in before, "fixture is not D"

    task = PackerTask(pose_stack, PackerPalette())
    task.restrict_to_repacking()
    task.add_conformer_sampler(IncludeCurrentSampler())
    task.add_conformer_sampler(FixedAAChiSampler())
    task.add_conformer_sampler(
        create_dunbrack_sampler_from_database(symmetric_gly_db, torch_device)
    )

    sfxn = beta2016_score_function(torch_device, param_db=symmetric_gly_db)
    repacked = pack_rotamers(pose_stack, sfxn, task)

    assert _chirality(repacked) == before

    total = sum(_scores_by_term(repacked, sfxn).values())
    assert numpy.isfinite(total)


def _chirality(pose_stack):
    """Handedness of every block, in order."""
    block_types = pose_stack.packed_block_types.active_block_types
    return [
        block_types[int(ind)].properties.polymer.sidechain_chirality
        for ind in pose_stack.block_type_ind[0]
        if int(ind) >= 0
    ]
