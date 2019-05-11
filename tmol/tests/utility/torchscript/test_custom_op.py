import torch
from tmol.utility.cpp_extension import load, relpaths, modulename


def test_load():
    load(
        modulename(f"{__name__}"),
        relpaths(__file__, "custom_op.cpp"),
        is_python_module=False,
    )

    cpow = torch.ops.tmol.cpow
    assert cpow is not None

    def check_form(pow3_f):
        # Checkout without autograd trace
        i: torch.Tensor = torch.arange(10).to(torch.float)
        result = pow3_f(i)
        expected = i.pow(3.0)

        torch.testing.assert_allclose(result, expected)

        assert not result.requires_grad
        assert not expected.requires_grad

        # Check with autograd
        i = torch.arange(10).to(torch.float).requires_grad_(True)
        result = pow3_f(i)
        expected = i.pow(3.0)

        torch.testing.assert_allclose(result, expected)

        assert result.requires_grad
        assert expected.requires_grad

        # Check matching grads
        i = torch.arange(10).to(torch.float).requires_grad_(True)
        i.pow(3.0).sum().backward()
        expected_grad = i.grad

        i = torch.arange(10).to(torch.float).requires_grad_(True)
        pow3_f(i).sum().backward()
        result_grad = i.grad

        torch.testing.assert_allclose(result_grad, expected_grad)

    def pow3(t):
        return cpow(t, 3.0)

    t_pow3 = torch.jit.trace(pow3, torch.rand(10))
    s_pow3 = torch.jit.script(pow3)

    check_form(pow3)
    check_form(t_pow3)
    check_form(s_pow3)
