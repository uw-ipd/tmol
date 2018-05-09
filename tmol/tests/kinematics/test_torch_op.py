import pytest

import pandas
import torch
import numpy

from tmol.kinematics.builder import KinematicBuilder
from tmol.kinematics.metadata import DOFMetadata, DOFTypes
from tmol.kinematics.torch_op import KinematicOp

from tmol.system.residue.packed import PackedResidueSystem


def test_kinematic_torch_op_refold(ubq_system):
    tsys = ubq_system

    torsion_pairs = numpy.block([
        [tsys.torsion_metadata["atom_index_b"]],
        [tsys.torsion_metadata["atom_index_c"]],
    ]).T
    torsion_bonds = torsion_pairs[numpy.all(torsion_pairs > 0, axis=-1)]

    kintree = KinematicBuilder().append_connected_component(
        *KinematicBuilder.component_for_prioritized_bonds(
            root=0,
            mandatory_bonds=torsion_bonds,
            all_bonds=tsys.bonds,
        )
    ).kintree

    kinematic_metadata = DOFMetadata.for_kintree(kintree)

    torsion_dofs = kinematic_metadata[
        (kinematic_metadata.dof_type == DOFTypes.bond_torsion)
    ]

    coords = torch.from_numpy(tsys.coords)

    kincoords = coords[kintree.id]
    kincoords[torch.isnan(kincoords)] = 0.0

    kop = KinematicOp.from_coords(
        kintree,
        torsion_dofs,
        kincoords,
    )

    refold_kincoords = kop.apply(kop.src_mobile_dofs)

    numpy.testing.assert_allclose(kincoords, refold_kincoords)


@pytest.fixture
def gradcheck_test_system(ubq_res):
    tsys = PackedResidueSystem.from_residues(ubq_res[:4])

    kintree = KinematicBuilder().append_connected_component(
        *KinematicBuilder.bonds_to_connected_component(
            root=0,
            bonds=tsys.bonds,
        )
    ).kintree

    dofs = DOFMetadata.for_kintree(kintree)

    coords = torch.from_numpy(tsys.coords)
    kincoords = coords[kintree.id]
    kincoords[torch.isnan(kincoords)] = 0.0

    return (kintree, dofs, kincoords)


# Update workaround logic in test_kinematic_torch_op_gradcheck when passing
@pytest.mark.xfail
def test_kinematic_torch_op_gradcheck_report(gradcheck_test_system):
    from torch.autograd.gradcheck import get_numerical_jacobian, get_analytical_jacobian
    kintree, dofs, kincoords = gradcheck_test_system

    kop = KinematicOp.from_coords(
        kintree,
        dofs,
        kincoords,
    )
    start_dofs = torch.tensor(kop.src_mobile_dofs, requires_grad=True)

    result = kop(start_dofs)

    eps = 1e-6
    atol = 1e-5
    rtol = 1e-3

    # Extract results from torch/autograd/gradcheck.py
    (analytical, ), reentrant, correct_grad_sizes = get_analytical_jacobian(
        (start_dofs, ), result
    )
    numerical = get_numerical_jacobian(kop, start_dofs, start_dofs, eps)

    a = analytical
    n = numerical

    grad_match = ((a - n).abs() <= (atol + rtol * n.abs()))
    if grad_match.all():
        return

    failures = torch.nonzero(~grad_match)
    nfailures = len(failures)
    failures = pandas.DataFrame({
        "dof_input": failures[:, 0],
        "node_idx": failures[:, 1] / 3,
        "node_coord": numpy.array(list("xyz"))[failures[:, 1] % 3],
        "analytical": a[~grad_match],
        "numerical": n[~grad_match],
    }).drop_duplicates()[[
        "dof_input", "node_idx", "node_coord", "analytical", "numerical"
    ]]

    dof_summary = dofs.to_frame()
    failing_dofs = failures["dof_input"].drop_duplicates().values

    assert nfailures == 0, (
        f"DOFs failed grad check:\n{failures}\n\n"
        f"Failed DOF metadata:\n{dof_summary.iloc[failing_dofs]}\n\n"
    )


def test_kinematic_torch_op_gradcheck(gradcheck_test_system):
    kintree, dofs, kincoords = gradcheck_test_system

    # Temporary workaround for #45, disable theta for post-jump siblings
    post_root_siblings = ((dofs.parent_id == 0) &
                          (dofs.dof_type == DOFTypes.bond_angle))
    dofs = dofs[~post_root_siblings]

    kop = KinematicOp.from_coords(
        kintree,
        dofs,
        kincoords,
    )

    start_dofs = torch.tensor(kop.src_mobile_dofs, requires_grad=True)

    assert torch.autograd.gradcheck(
        kop.apply, (start_dofs, ), raise_exception=True
    )


def test_kinematic_torch_op_smoke(
        gradcheck_test_system, pytorch_backward_coverage
):
    kintree, dofs, kincoords = gradcheck_test_system

    kop = KinematicOp.from_coords(
        kintree,
        dofs,
        kincoords,
    )

    start_dofs = torch.tensor(kop.src_mobile_dofs, requires_grad=True)

    coords = kop(start_dofs)
    coords.register_hook(pytorch_backward_coverage)

    total = coords.sum()
    total.backward()
