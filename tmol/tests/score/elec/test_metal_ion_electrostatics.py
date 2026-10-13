"""Nominal ion charges obey Coulomb energies and bonded-pair exclusions."""

import attr
import pytest
import torch

from tmol.io import (
    atom_array_from_cif,
    default_canonical_ordering,
    pose_stack_from_biotite,
    pose_stack_from_pdb,
    remove_metal_coordination,
)
from tmol.tests.data import data_path
from tmol.tests.score.elec.test_parameter_identity import coulomb_reference, setup_term


def ion_pose(residue_names, device):
    lines = [
        f"HETATM{i:5d} {name:>4s} {name:>3s} {chr(65 + i)}   1    "
        f"{3.1 * (i - 1):8.3f}{0.:8.3f}{0.:8.3f}  1.00  0.00          {name:>2s}"
        for i, name in enumerate(residue_names, 1)
    ]
    return pose_stack_from_pdb("\n".join(lines + ["END"]) + "\n", device)


def assert_energy_and_gradient(pose, database, expected_energy):
    module = setup_term(pose, database).render_whole_pose_scoring_module(pose)
    coords = pose.coords.double().clone().requires_grad_(True)
    actual = module(coords).sum()
    expected = expected_energy(coords)
    torch.testing.assert_close(actual, expected, rtol=3e-7, atol=1e-7)
    actual_grad = torch.autograd.grad(actual, coords)[0]
    expected_grad = torch.autograd.grad(expected, coords)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("ion", ["CA", "MG"])
def test_lone_ion_has_no_electrostatic_self_energy(default_database, torch_device, ion):
    # Magnesium includes virtual coordination sites; calcium is a single atom.
    pose = ion_pose([ion], torch_device)
    assert_energy_and_gradient(pose, default_database, lambda coords: coords.sum() * 0)


@pytest.mark.parametrize("ion, charge", [("NA", 1.0), ("CA", 2.0), ("MG", 2.0)])
def test_free_ions_coulomb_energy_and_gradient(
    default_database, torch_device, ion, charge
):
    pose = ion_pose([ion, "CA"], torch_device)
    second = int(pose.block_coord_offset[0, 1])
    assert_energy_and_gradient(
        pose,
        default_database,
        lambda coords: coulomb_reference(
            (coords[0, 0] - coords[0, second]).norm(),
            charge * 2.0,
            default_database.scoring.elec.global_parameters,
        ),
    )


def test_coordinated_donor_exclusion_changes_when_link_is_removed(
    default_database, torch_device
):
    structure = atom_array_from_cif(
        data_path("metal_fixtures", "ca_irregular_2fvy.cif.zst")
    )
    structure = structure[
        ((structure.res_name == "ASP") & (structure.res_id == 134))
        | ((structure.res_name == "CA") & (structure.res_id == 311))
    ]
    bound = pose_stack_from_biotite(structure, torch_device)
    assert bound.max_n_blocks == 2
    unbound = remove_metal_coordination(default_canonical_ordering(), bound, 0, 1, 0)
    # Isolate the calcium/donor pair without changing either charge. All other
    # atoms remain present; their electrostatic contributions are set to zero.
    selected = {("CA_irregular", "CA"), ("ASP", "OD1")}
    rows = default_database.scoring.elec.atom_charge_parameters
    donor_charge = next(
        row.charge for row in rows if (row.res, row.atom) == ("ASP", "OD1")
    )
    elec = attr.evolve(
        default_database.scoring.elec,
        atom_charge_parameters=tuple(
            row if (row.res, row.atom) in selected else attr.evolve(row, charge=0.0)
            for row in rows
        ),
        atom_cp_reps_parameters=(),
    )
    database = attr.evolve(
        default_database, scoring=attr.evolve(default_database.scoring, elec=elec)
    )
    # Coordination is represented as a direct connection, so fa_elec excludes
    # its donor pair; the separate metal term supplies that interaction.
    assert_energy_and_gradient(bound, database, lambda coords: coords.sum() * 0)
    donor_type = unbound.packed_block_types.active_block_types[
        unbound.block_type_ind[0, 0]
    ]
    donor = donor_type.atom_to_idx["OD1"]
    metal = int(unbound.block_coord_offset[0, 1])
    donor_charge = float(torch.tensor(donor_charge, dtype=torch.float32))
    assert_energy_and_gradient(
        unbound,
        database,
        lambda coords: coulomb_reference(
            (coords[0, donor] - coords[0, metal]).norm(),
            2.0 * donor_charge,
            database.scoring.elec.global_parameters,
        ),
    )
