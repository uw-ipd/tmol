import torch
import torch.autograd
from tmol.tests.autograd import gradcheck, VectorizedOp


def test_polynomial_gradcheck():

    from tmol.tests.score.common.polynomial.polynomial import poly_v_d

    ds = torch.linspace(0, 3.5, 100).to(dtype=torch.double)
    coeff = torch.tensor(
        [
            0.0,
            -0.5307601,
            6.47949946,
            -22.39522814,
            -55.14303544,
            708.30945242,
            -2619.49318162,
            5227.8805795,
            -6043.31211632,
            3806.04676175,
            -1007.66024144,
        ]
    ).to(dtype=torch.double)

    gradcheck(VectorizedOp(poly_v_d), (ds.requires_grad_(True), coeff))
