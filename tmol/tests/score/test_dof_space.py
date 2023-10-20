import torch

from tmol.system.kinematics import KinematicDescription
from tmol.system.packed import PackedResidueSystem


from tmol.score.modules.coords import coords_for

from tmol.system.score_support import kincoords_to_coords

from tmol.tests.autograd import gradcheck


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
