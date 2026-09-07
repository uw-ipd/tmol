import pytest
import torch

from tmol.optimization._armijo_compiled import (
    armijo_classify,
    armijo_finalize,
    armijo_start,
    armijo_trial,
    armijo_update,
)

_LS_DONE = 0
_LS_INCREASE = 1
_LS_BACKTRACK = 2
_LS_FAILED = 3


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_armijo_start_and_classify_match_tensor_reference(torch_device, dtype):
    searching = torch.tensor([False, True, True, True], device=torch_device)
    alpha0 = torch.ones(4, dtype=dtype, device=torch_device)
    phi0 = torch.full_like(alpha0, 10.0)
    derphi0 = torch.full_like(alpha0, -1.0)
    phi = torch.tensor([20.0, 9.0, 9.5, 10.0], dtype=dtype, device=torch_device)
    sigma_increase = 0.8
    sigma_decrease = 0.1

    expected_alpha = torch.where(searching, alpha0, torch.zeros_like(alpha0))
    linear = phi <= phi0 + expected_alpha * sigma_increase * derphi0
    sufficient = phi <= phi0 + expected_alpha * sigma_decrease * derphi0
    expected_status = torch.full(
        alpha0.shape, _LS_DONE, dtype=torch.int64, device=torch_device
    )
    expected_status = torch.where(searching & linear, _LS_INCREASE, expected_status)
    expected_status = torch.where(
        searching & ~linear & ~sufficient, _LS_BACKTRACK, expected_status
    )
    took = searching & (linear | sufficient)
    expected_accepted = torch.where(took, expected_alpha, torch.zeros_like(alpha0))
    expected_phi_accepted = torch.where(took, phi, phi0)

    alpha = armijo_start(searching, alpha0)
    accepted, phi_accepted, status, active = armijo_classify(
        searching,
        alpha,
        phi,
        phi0,
        derphi0,
        sigma_increase,
        sigma_decrease,
    )

    torch.testing.assert_close(alpha, expected_alpha, rtol=0, atol=0)
    torch.testing.assert_close(accepted, expected_accepted, rtol=0, atol=0)
    torch.testing.assert_close(phi_accepted, expected_phi_accepted, rtol=0, atol=0)
    torch.testing.assert_close(status, expected_status, rtol=0, atol=0)
    expected_active = (expected_status == _LS_INCREASE) | (
        expected_status == _LS_BACKTRACK
    )
    torch.testing.assert_close(active, expected_active, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_armijo_trial_and_update_match_tensor_reference(torch_device, dtype):
    status = torch.tensor(
        [
            _LS_INCREASE,
            _LS_INCREASE,
            _LS_BACKTRACK,
            _LS_BACKTRACK,
            _LS_BACKTRACK,
            _LS_DONE,
        ],
        dtype=torch.int64,
        device=torch_device,
    )
    alpha = torch.tensor(
        [1.0, 1.0, 1.0, 0.1, 0.1, 0.0], dtype=dtype, device=torch_device
    )
    accepted = torch.tensor(
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=dtype, device=torch_device
    )
    phi_accepted = torch.tensor(
        [9.0, 9.0, 10.0, 10.0, 10.0, 10.0], dtype=dtype, device=torch_device
    )
    phi = torch.tensor(
        [9.0, 9.0, 10.0, 10.0, 10.0, 10.0], dtype=dtype, device=torch_device
    )
    phi_trial = torch.tensor(
        [8.0, 9.5, 9.8, 9.999, 10.1, 123.0], dtype=dtype, device=torch_device
    )
    phi0 = torch.full_like(alpha, 10.0)
    derphi0 = torch.full_like(alpha, -1.0)
    factor = 0.5
    sigma_decrease = 0.1
    minstep = 0.03

    expected_trial = torch.where(status == _LS_INCREASE, alpha / factor, accepted)
    expected_trial = torch.where(
        status == _LS_BACKTRACK, alpha * factor * factor, expected_trial
    )

    increasing = status == _LS_INCREASE
    better = increasing & (phi_trial < phi)
    expected_accepted = torch.where(better, expected_trial, accepted)
    expected_phi_accepted = torch.where(better, phi_trial, phi_accepted)
    expected_status = torch.where(increasing, _LS_DONE, status)

    backtracking = expected_status == _LS_BACKTRACK
    armijo = backtracking & (
        phi_trial <= phi0 + expected_trial * sigma_decrease * derphi0
    )
    expected_accepted = torch.where(armijo, expected_trial, expected_accepted)
    expected_phi_accepted = torch.where(armijo, phi_trial, expected_phi_accepted)
    expected_status = torch.where(armijo, _LS_DONE, expected_status)

    floored = backtracking & ~armijo & (expected_trial < minstep)
    downhill = floored & (phi_trial < phi0)
    expected_accepted = torch.where(downhill, expected_trial, expected_accepted)
    expected_phi_accepted = torch.where(downhill, phi_trial, expected_phi_accepted)
    expected_status = torch.where(downhill, _LS_DONE, expected_status)
    expected_failed = floored & ~downhill
    expected_accepted = torch.where(
        expected_failed, torch.zeros_like(alpha), expected_accepted
    )
    expected_phi_accepted = torch.where(expected_failed, phi0, expected_phi_accepted)
    expected_status = torch.where(expected_failed, _LS_FAILED, expected_status)

    trial = armijo_trial(status, alpha, accepted, factor)
    next_accepted, next_phi_accepted, next_status, failed, active = armijo_update(
        status,
        trial,
        accepted,
        phi_accepted,
        phi,
        phi_trial,
        phi0,
        derphi0,
        sigma_decrease,
        minstep,
    )

    torch.testing.assert_close(trial, expected_trial, rtol=0, atol=0)
    torch.testing.assert_close(next_accepted, expected_accepted, rtol=0, atol=0)
    torch.testing.assert_close(next_phi_accepted, expected_phi_accepted, rtol=0, atol=0)
    torch.testing.assert_close(next_status, expected_status, rtol=0, atol=0)
    torch.testing.assert_close(failed, expected_failed, rtol=0, atol=0)
    expected_active = (expected_status == _LS_INCREASE) | (
        expected_status == _LS_BACKTRACK
    )
    torch.testing.assert_close(active, expected_active, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_armijo_finalize_matches_tensor_reference(torch_device, dtype):
    status = torch.tensor(
        [_LS_DONE, _LS_FAILED, _LS_FAILED, _LS_DONE],
        dtype=torch.int64,
        device=torch_device,
    )
    derphi0 = torch.tensor([-4.0, -4.0, -0.25, -1.0], dtype=dtype, device=torch_device)
    accepted = torch.tensor([0.5, 0.0, 0.0, 0.75], dtype=dtype, device=torch_device)
    start = torch.tensor([1.0, 0.8, 0.6, 0.4], dtype=dtype, device=torch_device)
    searching = torch.tensor([True, True, True, False], device=torch_device)
    minstep = 1e-12

    expected_failed = status == _LS_FAILED
    retry = torch.clamp((-derphi0).clamp(min=minstep).rsqrt(), max=1.0)
    expected_step = torch.where(expected_failed, retry, accepted)
    expected_step = torch.where(searching, expected_step, start)

    step, failed = armijo_finalize(status, derphi0, accepted, start, searching, minstep)

    torch.testing.assert_close(step, expected_step)
    torch.testing.assert_close(failed, expected_failed, rtol=0, atol=0)
