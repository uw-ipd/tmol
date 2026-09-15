"""Changed charge databases must take effect on reused poses and terms."""

import attr
import gc
import weakref
import numpy as np
import pytest
import torch

from tmol.database import inject_residue_params
from tmol.io import extended_pose_stack_from_sequences
from tmol.score.elec import ElecEnergyTerm, ElecParamResolver
from tmol.database.scoring import PartialCharges, CountPairReps


def isolated_charge_database(database, first_charge, residue="ALA", atom="CB"):
    elec = attr.evolve(
        database.scoring.elec,
        atom_charge_parameters=tuple(
            attr.evolve(row, charge=0.0)
            for row in database.scoring.elec.atom_charge_parameters
        ),
        atom_cp_reps_parameters=(),
    )
    zeroed = attr.evolve(database, scoring=attr.evolve(database.scoring, elec=elec))
    return inject_residue_params(
        zeroed,
        [],
        partial_charges={
            residue + ":nterm": {atom: first_charge},
            residue + ":cterm": {atom: -0.4},
        },
    )


def setup_term(pose, database):
    term = ElecEnergyTerm(database, pose.device)
    for bt in pose.packed_block_types.active_block_types:
        term.setup_block_type(bt)
    term.setup_packed_block_types(pose.packed_block_types)
    term.setup_poses(pose)
    return term


def coulomb_reference(distance, charge_product, global_params):
    # Interior (unsmoothed) region of the distance-dependent dielectric model.
    def fp32(value):
        return float(np.float32(value))

    D, D0, S = [
        fp32(getattr(global_params, "elec_sigmoidal_die_" + name))
        for name in ("D", "D0", "S")
    ]
    maximum = fp32(global_params.elec_max_dis)
    # Native scoring promotes stored parameters to the coordinate dtype first.
    contrast = D - D0

    def dielectric(r):
        x = r * S
        return D - 0.5 * contrast * (x.square() + 2 * x + 2) * torch.exp(-x)

    cutoff = distance.new_tensor(maximum)
    return (
        322.0637
        * charge_product
        * (1 / (distance * dielectric(distance)) - 1 / (cutoff * dielectric(cutoff)))
    )


@pytest.mark.parametrize("changed_first", [False, True])
@pytest.mark.parametrize("block_pairs", [False, True])
@pytest.mark.parametrize(
    "change", ["charge", "representative", "ligand_mask", "globals"]
)
def test_charge_database_reuse_energy_and_gradient(
    default_database, torch_device, changed_first, block_pairs, change
):
    pose = extended_pose_stack_from_sequences(["AA", "AAA"], device=torch_device)
    databases = {
        "before": isolated_charge_database(default_database, 0.5),
        "after": isolated_charge_database(default_database, 0.75),
    }
    base = databases["before"]
    if change in ("representative", "ligand_mask"):
        with_reps = attr.evolve(
            base,
            scoring=attr.evolve(
                base.scoring,
                elec=attr.evolve(
                    base.scoring.elec,
                    atom_cp_reps_parameters=(CountPairReps("ALA:nterm", "CB", "CA"),),
                ),
            ),
        )
        databases["after"] = with_reps
        if change == "ligand_mask":
            databases["before"] = with_reps
            databases["after"] = attr.evolve(
                with_reps,
                scoring=attr.evolve(
                    with_reps.scoring,
                    genbonded=attr.evolve(
                        with_reps.scoring.genbonded, rosetta_typed=frozenset()
                    ),
                ),
            )
    elif change == "globals":
        databases["after"] = attr.evolve(
            base,
            scoring=attr.evolve(
                base.scoring,
                elec=attr.evolve(
                    base.scoring.elec,
                    global_parameters=attr.evolve(
                        base.scoring.elec.global_parameters, elec_sigmoidal_die_S=0.63
                    ),
                ),
            ),
        )
    terms, modules = {}, {}
    order = ("after", "before") if changed_first else ("before", "after")
    for name in order:
        term = terms[name] = setup_term(pose, databases[name])
        modules[name] = (
            term.render_block_pair_scoring_module
            if block_pairs
            else term.render_whole_pose_scoring_module
        )(pose)
    # Re-render the first term after another database overwrote shared tables.
    first = terms[order[0]]
    modules["again"] = (
        first.render_block_pair_scoring_module
        if block_pairs
        else first.render_whole_pose_scoring_module
    )(pose)
    coords = pose.coords.double().clone()
    pairs = []
    for pi, row in enumerate(pose.block_type_ind.tolist()):
        blocks = [i for i, t in enumerate(row) if t >= 0]
        indices = []
        for bi in (blocks[0], blocks[-1]):
            bt = pose.packed_block_types.active_block_types[row[bi]]
            indices.append(int(pose.block_coord_offset[pi, bi]) + bt.atom_to_idx["CB"])
        coords[pi, indices[1]] = coords[pi, indices[0]] + coords.new_tensor(
            [3.1, 0.7, 0.2]
        )
        pairs.append((pi, blocks[-1], *indices))
    coords.requires_grad_(True)
    weights = (
        1 + 0.1 * torch.arange(pose.n_poses * pose.max_n_blocks**2, device=torch_device)
    ).reshape(pose.n_poses, pose.max_n_blocks, pose.max_n_blocks)
    for name, module in modules.items():
        state = order[0] if name == "again" else name
        q = 0.75 if change == "charge" and state == "after" else 0.5
        expected = coords.new_zeros(())
        for pi, last, a, b in pairs:
            distance = (coords[pi, a] - coords[pi, b]).norm()
            assert 1.85 < float(distance.detach()) < 4.5
            connectivity = (
                0.2
                if last == 1
                and (
                    (change == "representative" and state == "after")
                    or (change == "ligand_mask" and state == "before")
                )
                else 1.0
            )
            expected += (
                connectivity
                * coulomb_reference(
                    distance,
                    q * float(np.float32(-0.4)),
                    databases[state].scoring.elec.global_parameters,
                )
                * (weights[pi, 0, last] if block_pairs else 1)
            )
        values = module(coords)
        actual = (values * weights).sum() if block_pairs else values.sum()
        # Native cutoff constants involve float32 intermediates even when
        # coordinates use double; the independent formula evaluates in double.
        torch.testing.assert_close(actual, expected, rtol=3e-7, atol=1e-7)
        actual_grad = torch.autograd.grad(actual, coords, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, coords, retain_graph=True)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-7, atol=1e-7)


def test_exact_combined_variant_charge_precedes_single_patch(
    default_database, fresh_default_restype_set, torch_device
):
    base = next(r for r in fresh_default_restype_set.residue_types if r.name == "ALA")
    combined = next(
        r
        for r in fresh_default_restype_set.residue_types
        if r.base_name == "ALA" and set(r.name.split(":")[1:]) == {"nterm", "cterm"}
    )
    # An extra patch must not inherit a less-specific combined charge row.
    extended = attr.evolve(combined, name=combined.name + ":other")
    rows = (
        PartialCharges("ALA", "CB", 0.1),
        PartialCharges("ALA:nterm", "CB", 0.2),
        PartialCharges(combined.name, "CB", 0.7),
    )
    elec = attr.evolve(
        default_database.scoring.elec,
        atom_charge_parameters=tuple(
            r
            for r in default_database.scoring.elec.atom_charge_parameters
            if not (r.res.startswith("ALA") and r.atom == "CB")
        )
        + rows,
        atom_cp_reps_parameters=default_database.scoring.elec.atom_cp_reps_parameters
        + (
            CountPairReps("ALA:nterm", "CB", "CA"),
            CountPairReps(combined.name, "CB", "N"),
        ),
    )
    resolver = ElecParamResolver.from_database(elec, torch_device)
    for rt, expected in ((combined, 0.7), (extended, 0.2), (base, 0.1)):
        charges = resolver.get_partial_charges_for_block(rt)
        assert charges[rt.atom_to_idx["CB"]] == np.float32(expected)
    for rt, representative in ((combined, "N"), (extended, "CA"), (base, "CB")):
        mapping = resolver.get_bonded_path_length_mapping_for_block(rt)
        assert mapping[rt.atom_to_idx["CB"]] == rt.atom_to_idx[representative]


def test_charge_switch_reuses_geometry_with_bounded_database_lifetime(
    default_database, torch_device
):
    pose = extended_pose_stack_from_sequences(["AA", "AAA"], device=torch_device)
    pbt = pose.packed_block_types
    owners, charges = [], []
    geometry = None
    for i in range(40):
        database = isolated_charge_database(default_database, 0.5 + 0.01 * i)
        term = setup_term(pose, database)
        params = term.setup_packed_block_types(pbt)
        owners.append(weakref.ref(database.scoring.elec))
        charges.append(weakref.ref(params.charges))
        if geometry is None:
            geometry = (params.inter, params.intra, params.block_geometry)
        else:
            assert params.inter is geometry[0] and params.intra is geometry[1]
            assert all(a is b for a, b in zip(params.block_geometry, geometry[2]))
        del term, database, params
    gc.collect()
    assert all(owner() is None for owner in owners)
    assert sum(charge() is not None for charge in charges) == 2
    assert len(pbt._elec_annotation) == 2


def test_missing_applicable_charge_raises_instead_of_nan(
    default_database, fresh_default_restype_set, torch_device
):
    base = next(r for r in fresh_default_restype_set.residue_types if r.name == "ALA")
    elec = attr.evolve(
        default_database.scoring.elec,
        atom_charge_parameters=tuple(
            row
            for row in default_database.scoring.elec.atom_charge_parameters
            if not (row.res == "ALA" and row.atom == "CB")
        )
        + (PartialCharges("ALA:nterm", "CB", 0.2),),
    )
    resolver = ElecParamResolver.from_database(elec, torch_device)
    with pytest.raises(KeyError, match="Elec charge for atom ALA,CB not found"):
        resolver.get_partial_charges_for_block(base)
