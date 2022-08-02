import torch

from tmol.system.kinematics import KinematicDescription
from tmol.system.packed import PackedResidueSystem


from tmol.score.modules.coords import coords_for

from tmol.system.score_support import kincoords_to_coords, get_full_score_system_for

from tmol.tests.autograd import gradcheck


def test_torsion_space_by_real_space_total_score(ubq_system):

    score_system = get_full_score_system_for(ubq_system, torch.device("cpu"))
    xyz_coords = coords_for(ubq_system, score_system)

    sys_kin = KinematicDescription.for_system(
        1, ubq_system.bonds, (ubq_system.torsion_metadata,)
    )
    kincoords = sys_kin.extract_kincoords(ubq_system.coords)
    kinforest = sys_kin.kinforest
    coords_converted_to_torsion_and_back = kincoords_to_coords(
        kincoords, kinforest, ubq_system.system_size
    )

    real_total = score_system.intra_total(xyz_coords)
    torsion_total = score_system.intra_total(coords_converted_to_torsion_and_back)

    assert (real_total == torsion_total).all()


def test_torsion_space_coord_smoke(ubq_system):
    score_system = get_full_score_system_for(ubq_system, torch.device("cpu"))

    start_coords = coords_for(ubq_system, score_system)

    sys_kin = KinematicDescription.for_system(
        1, ubq_system.bonds, (ubq_system.torsion_metadata,)
    )
    start_dofs = sys_kin.extract_kincoords(ubq_system.coords)
    kinforest = sys_kin.kinforest

    cmask = torch.isnan(start_coords).sum(dim=-1) == 0

    def coord_residuals(dofs):
        torsion_space_coords = kincoords_to_coords(
            dofs, kinforest, ubq_system.system_size
        )
        return (torsion_space_coords[cmask] - start_coords[cmask]).norm(dim=-1).sum()

    torch.random.manual_seed(1663)
    pdofs = torch.tensor((torch.rand_like(start_dofs) - 0.5) * 1e-2, requires_grad=True)

    assert pdofs.requires_grad

    res = coord_residuals(pdofs)
    assert res.requires_grad

    res.backward(retain_graph=True)
    assert pdofs.grad is not None


def test_torsion_space_to_cart_space_gradcheck(ubq_res):
    tsys = PackedResidueSystem.from_residues(ubq_res[:6])

    sys_kin = KinematicDescription.for_system(1, tsys.bonds, (tsys.torsion_metadata,))

    start_dofs = (
        sys_kin.extract_kincoords(tsys.coords).detach().clone().requires_grad_()
    )

    dofs_copy = sys_kin.extract_kincoords(tsys.coords)

    def coords(minimizable_dofs):
        dofs_copy[:, :6] = minimizable_dofs
        full_coords = kincoords_to_coords(
            dofs_copy, sys_kin.kinforest, tsys.system_size
        )
        return full_coords[~torch.isnan(full_coords)]

    gradcheck(coords, (start_dofs[:, :6],), eps=1e-1, atol=1e-6, rtol=2e-3)
