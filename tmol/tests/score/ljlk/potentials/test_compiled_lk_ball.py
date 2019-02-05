import torch
import torch.autograd
from tmol.utility.units import parse_angle


def test_build_acc_waters():
    from tmol.tests.score.ljlk.potentials.lk_ball import BuildAcceptorWater

    tensor = torch.DoubleTensor

    ## test 1: acceptor water generation + derivatives
    A = tensor((-6.071, -0.619, -3.193))
    B = tensor((-5.250, -1.595, -2.543))
    B0 = tensor((-5.489, 0.060, -3.542))
    dist = 2.65
    angle = parse_angle("109.0 deg")

    torsions = tensor([parse_angle(f"{a} deg") for a in (120.0, 240.0)])
    waters_ref = tensor(
        [[-7.42086525, -1.79165583, -5.14882262], [-7.75428876, 0.40906314, -1.4232189]]
    )

    for torsion, water_ref in zip(torsions, waters_ref):
        water = BuildAcceptorWater.apply(A, B, B0, dist, angle, torsion)
        torch.testing.assert_allclose(water, water_ref)

        torch.autograd.gradcheck(
            lambda A, B, B0: BuildAcceptorWater.apply(A, B, B0, dist, angle, torsion),
            (A.requires_grad_(True), B.requires_grad_(True), B0.requires_grad_(True)),
        )


def test_build_don_water():
    from tmol.tests.score.ljlk.potentials.lk_ball import BuildDonorWater

    tensor = torch.DoubleTensor

    ## test 2: donor water generation + derivatives
    D = tensor((-6.007, 4.706, -0.074))
    H = tensor((-6.747, 4.361, 0.549))
    dist = 2.65

    assert not any(t.requires_grad for t in (D, H))

    waters = BuildDonorWater.apply(D, H, dist)
    waters_ref = tensor([-7.91642236, 3.81579633, 1.5335272])
    torch.testing.assert_allclose(waters, waters_ref)

    torch.autograd.gradcheck(
        lambda D, H: BuildDonorWater.apply(D, H, dist),
        (D.requires_grad_(True), H.requires_grad_(True)),
    )
