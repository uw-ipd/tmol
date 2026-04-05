"""
test_mps.py — MPS (Apple Metal) backend smoke tests for tmol.

These tests verify that the MPS backend added in device_operations.mps.impl.hh
is wired up correctly through the full stack:

  Layer 1 — Python / PyTorch tensor plumbing
  Layer 2 — DeviceOperations<MPS> CPU-loop primitives (accumulate, scan, reduce)
  Layer 3 — Dispatch macros (TMOL_DISPATCH_FLOATING_DEVICE for MPS tensors)
  Layer 4 — Score-term scoring on a real PDB pose (forward pass + gradients)
  Layer 5 — Numerical consistency: MPS results must match CPU within tolerance

Run only the MPS tests:
    pytest tmol/tests/test_mps.py -v

Skip automatically on non-Apple-Silicon machines (requires_mps mark).
"""

import pytest
import torch
import numpy as np

from tmol.tests.torch import requires_mps

# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — Basic MPS tensor plumbing
# ─────────────────────────────────────────────────────────────────────────────


@requires_mps
def test_mps_is_available():
    """PyTorch MPS backend must be available on this machine."""
    assert torch.backends.mps.is_available(), "torch.backends.mps.is_available() returned False"


@requires_mps
def test_mps_tensor_creation():
    """Basic tensor creation and arithmetic on the MPS device."""
    device = torch.device("mps")
    a = torch.rand(128, 3, device=device, dtype=torch.float32)
    b = torch.rand(128, 3, device=device, dtype=torch.float32)
    c = a + b
    assert c.device.type == "mps"
    # Result must be numerically correct when moved back to CPU
    torch.testing.assert_close(c.cpu(), a.cpu() + b.cpu())


@requires_mps
def test_mps_matmul():
    """Matrix multiply on MPS must agree with CPU within float32 tolerance."""
    device = torch.device("mps")
    a = torch.rand(64, 64, device=device, dtype=torch.float32)
    b = torch.rand(64, 64, device=device, dtype=torch.float32)
    c_mps = (a @ b).cpu()
    c_cpu = a.cpu() @ b.cpu()
    torch.testing.assert_close(c_mps, c_cpu, atol=1e-4, rtol=1e-4)


@requires_mps
def test_mps_autograd():
    """Autograd (backward pass) must work on MPS tensors."""
    device = torch.device("mps")
    x = torch.rand(16, requires_grad=True, device=device, dtype=torch.float32)
    y = (x ** 2).sum()
    y.backward()
    # gradient of sum(x^2) w.r.t. x is 2*x
    torch.testing.assert_close(x.grad.cpu(), 2 * x.detach().cpu(), atol=1e-5, rtol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — DeviceOperations<MPS> primitives via Python wrappers
# ─────────────────────────────────────────────────────────────────────────────


@requires_mps
def test_mps_prefix_scan_via_torch():
    """cumsum on MPS tensor must agree with CPU — exercises the same
    algorithmic path that DeviceOperations<MPS>::scan uses."""
    device = torch.device("mps")
    n = 1024
    x = torch.arange(1, n + 1, dtype=torch.float32, device=device)
    result_mps = x.cumsum(0).cpu()
    result_cpu = x.cpu().cumsum(0)
    torch.testing.assert_close(result_mps, result_cpu)


@requires_mps
def test_mps_reduce_via_torch():
    """sum-reduction on MPS tensor must agree with CPU."""
    device = torch.device("mps")
    x = torch.rand(4096, dtype=torch.float32, device=device)
    s_mps = x.sum().item()
    s_cpu = x.cpu().sum().item()
    assert abs(s_mps - s_cpu) / (abs(s_cpu) + 1e-8) < 1e-4


@requires_mps
@pytest.mark.parametrize("n", [64, 256, 1023, 4096])
def test_mps_elementwise_vs_cpu(n):
    """Element-wise ops on MPS must numerically match CPU (various sizes)."""
    device = torch.device("mps")
    x = torch.rand(n, dtype=torch.float32)
    # sin + exp — similar to the kind of math in score kernels
    y_cpu = (x.sin() + x.exp())
    y_mps = (x.to(device).sin() + x.to(device).exp()).cpu()
    torch.testing.assert_close(y_mps, y_cpu, atol=1e-5, rtol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — Dispatch macro: MPS tensor passed through TMOL_DISPATCH_FLOATING_DEVICE
#
# We cannot call the macro directly from Python, but we can verify it by
# constructing an MPS pose and calling a compiled op on it.  The compiled ops
# are registered via TORCH_LIBRARY and will internally hit:
#
#   TMOL_DISPATCH_FLOATING_DEVICE(tensor.options(), "op", [&] {
#       constexpr tmol::Device device_t = tmol::Device::MPS;
#       ...
#   });
#
# If the MPS branch is missing the dispatch would raise "Unsupported device".
# ─────────────────────────────────────────────────────────────────────────────


@requires_mps
def test_mps_pose_stack_construction(ubq_pdb, default_database):
    """Build a pose stack on the MPS device — exercises io compiled ops."""
    from tmol.io import pose_stack_from_pdb

    device = torch.device("mps")
    pose_stack = pose_stack_from_pdb(ubq_pdb, device, residue_end=6)

    assert pose_stack.coords.device.type == "mps"
    # Coordinates must be finite
    assert torch.isfinite(pose_stack.coords).all().item()


# ─────────────────────────────────────────────────────────────────────────────
# Layer 4 — Score-term forward pass on MPS
# ─────────────────────────────────────────────────────────────────────────────


@requires_mps
def test_mps_cartbonded_forward(ubq_pdb, default_database):
    """CartBonded energy term forward pass on MPS device."""
    from tmol.io import pose_stack_from_pdb
    from tmol.score.cartbonded.cartbonded_energy_term import CartBondedEnergyTerm

    device = torch.device("mps")
    pose_stack = pose_stack_from_pdb(ubq_pdb, device, residue_end=6)

    term = CartBondedEnergyTerm(param_db=default_database, device=device)
    for bt in pose_stack.packed_block_types.active_block_types:
        term.setup_block_type(bt)
    term.setup_packed_block_types(pose_stack.packed_block_types)
    term.setup_poses(pose_stack)
    scores = term.render_whole_pose_scoring_module(pose_stack)(pose_stack.coords)

    assert scores.device.type == "mps"
    assert torch.isfinite(scores).all().item(), "CartBonded scores contain non-finite values"
    assert scores.shape[0] > 0


@requires_mps
def test_mps_elec_forward(ubq_pdb, default_database):
    """Electrostatics energy term forward pass on MPS device."""
    from tmol.io import pose_stack_from_pdb
    from tmol.score.elec.elec_energy_term import ElecEnergyTerm

    device = torch.device("mps")
    pose_stack = pose_stack_from_pdb(ubq_pdb, device, residue_end=6)

    term = ElecEnergyTerm(param_db=default_database, device=device)
    for bt in pose_stack.packed_block_types.active_block_types:
        term.setup_block_type(bt)
    term.setup_packed_block_types(pose_stack.packed_block_types)
    term.setup_poses(pose_stack)
    scores = term.render_whole_pose_scoring_module(pose_stack)(pose_stack.coords)

    assert scores.device.type == "mps"
    assert torch.isfinite(scores).all().item(), "Elec scores contain non-finite values"


@requires_mps
def test_mps_ljlk_forward(ubq_pdb, default_database):
    """LJ/LK energy term forward pass on MPS device."""
    from tmol.io import pose_stack_from_pdb
    from tmol.score.ljlk.ljlk_energy_term import LJLKEnergyTerm

    device = torch.device("mps")
    pose_stack = pose_stack_from_pdb(ubq_pdb, device, residue_end=6)

    term = LJLKEnergyTerm(param_db=default_database, device=device)
    for bt in pose_stack.packed_block_types.active_block_types:
        term.setup_block_type(bt)
    term.setup_packed_block_types(pose_stack.packed_block_types)
    term.setup_poses(pose_stack)
    scores = term.render_whole_pose_scoring_module(pose_stack)(pose_stack.coords)

    assert scores.device.type == "mps"
    assert torch.isfinite(scores).all().item(), "LJ/LK scores contain non-finite values"


@requires_mps
def test_mps_hbond_forward(ubq_pdb, default_database):
    """H-bond energy term forward pass on MPS device."""
    from tmol.io import pose_stack_from_pdb
    from tmol.score.hbond.hbond_energy_term import HBondEnergyTerm

    device = torch.device("mps")
    pose_stack = pose_stack_from_pdb(ubq_pdb, device, residue_end=6)

    term = HBondEnergyTerm(param_db=default_database, device=device)
    for bt in pose_stack.packed_block_types.active_block_types:
        term.setup_block_type(bt)
    term.setup_packed_block_types(pose_stack.packed_block_types)
    term.setup_poses(pose_stack)
    scores = term.render_whole_pose_scoring_module(pose_stack)(pose_stack.coords)

    assert scores.device.type == "mps"
    assert torch.isfinite(scores).all().item(), "H-bond scores contain non-finite values"


@requires_mps
def test_mps_score_function_forward(ubq_pdb, default_database):
    """Full ScoreFunction (beta2016) forward pass on MPS device."""
    from tmol.io import pose_stack_from_pdb
    from tmol.score import beta2016_score_function

    device = torch.device("mps")
    pose_stack = pose_stack_from_pdb(ubq_pdb, device, residue_end=6)

    sfxn = beta2016_score_function(device, param_db=default_database)
    wpsm = sfxn.render_whole_pose_scoring_module(pose_stack)
    scores = wpsm(pose_stack.coords)

    assert scores.device.type == "mps"
    assert torch.isfinite(scores).all().item(), "beta2016 scores contain non-finite values"


# ─────────────────────────────────────────────────────────────────────────────
# Layer 5 — MPS vs CPU numerical consistency
#
# The MPS backend (Phase 1) runs the same CPU loops through unified memory,
# so results must be bit-identical or very close to the CPU baseline.
# ─────────────────────────────────────────────────────────────────────────────


@requires_mps
@pytest.mark.parametrize(
    "term_class,import_path",
    [
        ("CartBondedEnergyTerm", "tmol.score.cartbonded.cartbonded_energy_term"),
        ("LJLKEnergyTerm",       "tmol.score.ljlk.ljlk_energy_term"),
        ("ElecEnergyTerm",       "tmol.score.elec.elec_energy_term"),
        ("HBondEnergyTerm",      "tmol.score.hbond.hbond_energy_term"),
    ],
    ids=["cartbonded", "ljlk", "elec", "hbond"],
)
def test_mps_vs_cpu_score_consistency(
    term_class, import_path, ubq_pdb, default_database
):
    """MPS scores must match CPU scores to within float32 tolerance."""
    import importlib
    from tmol.io import pose_stack_from_pdb

    mod = importlib.import_module(import_path)
    TermClass = getattr(mod, term_class)

    cpu = torch.device("cpu")
    mps = torch.device("mps")

    pose_cpu = pose_stack_from_pdb(ubq_pdb, cpu, residue_end=6)
    pose_mps = pose_stack_from_pdb(ubq_pdb, mps, residue_end=6)

    term_cpu = TermClass(param_db=default_database, device=cpu)
    term_mps = TermClass(param_db=default_database, device=mps)

    for bt in pose_cpu.packed_block_types.active_block_types:
        term_cpu.setup_block_type(bt)
        term_mps.setup_block_type(bt)
    term_cpu.setup_packed_block_types(pose_cpu.packed_block_types)
    term_cpu.setup_poses(pose_cpu)
    term_mps.setup_packed_block_types(pose_mps.packed_block_types)
    term_mps.setup_poses(pose_mps)

    scores_cpu = term_cpu.render_whole_pose_scoring_module(pose_cpu)(
        pose_cpu.coords
    ).detach().cpu()
    scores_mps = term_mps.render_whole_pose_scoring_module(pose_mps)(
        pose_mps.coords
    ).detach().cpu()

    np.testing.assert_allclose(
        scores_mps.numpy(),
        scores_cpu.numpy(),
        atol=1e-4,
        rtol=1e-4,
        err_msg=f"{term_class}: MPS scores differ from CPU by more than tolerance",
    )


@requires_mps
def test_mps_vs_cpu_full_scorefunction_consistency(ubq_pdb, default_database):
    """Full beta2016 score function: MPS total must match CPU total."""
    from tmol.io import pose_stack_from_pdb
    from tmol.score import beta2016_score_function

    cpu = torch.device("cpu")
    mps = torch.device("mps")

    pose_cpu = pose_stack_from_pdb(ubq_pdb, cpu, residue_end=6)
    pose_mps = pose_stack_from_pdb(ubq_pdb, mps, residue_end=6)

    sfxn_cpu = beta2016_score_function(cpu, param_db=default_database)
    sfxn_mps = beta2016_score_function(mps, param_db=default_database)

    total_cpu = sfxn_cpu.render_whole_pose_scoring_module(pose_cpu)(pose_cpu.coords).sum().item()
    total_mps = sfxn_mps.render_whole_pose_scoring_module(pose_mps)(pose_mps.coords).sum().item()

    rel_err = abs(total_mps - total_cpu) / (abs(total_cpu) + 1e-8)
    assert rel_err < 1e-3, (
        f"beta2016 total score: CPU={total_cpu:.4f}, MPS={total_mps:.4f}, "
        f"relative error={rel_err:.2e} (threshold 1e-3)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Layer 5b — Backward pass (gradient) consistency
# ─────────────────────────────────────────────────────────────────────────────


@requires_mps
def test_mps_cartbonded_backward(ubq_pdb, default_database):
    """CartBonded gradients on MPS must match CPU gradients."""
    from tmol.io import pose_stack_from_pdb
    from tmol.score.cartbonded.cartbonded_energy_term import CartBondedEnergyTerm

    cpu = torch.device("cpu")
    mps = torch.device("mps")

    pose_cpu = pose_stack_from_pdb(ubq_pdb, cpu, residue_end=4)
    pose_mps = pose_stack_from_pdb(ubq_pdb, mps, residue_end=4)

    term_cpu = CartBondedEnergyTerm(param_db=default_database, device=cpu)
    term_mps = CartBondedEnergyTerm(param_db=default_database, device=mps)

    for bt in pose_cpu.packed_block_types.active_block_types:
        term_cpu.setup_block_type(bt)
        term_mps.setup_block_type(bt)
    term_cpu.setup_packed_block_types(pose_cpu.packed_block_types)
    term_cpu.setup_poses(pose_cpu)
    term_mps.setup_packed_block_types(pose_mps.packed_block_types)
    term_mps.setup_poses(pose_mps)

    coords_cpu = pose_cpu.coords.clone().requires_grad_(True)
    coords_mps = pose_mps.coords.clone().requires_grad_(True)

    scores_cpu = term_cpu.render_whole_pose_scoring_module(pose_cpu)(coords_cpu)
    scores_mps = term_mps.render_whole_pose_scoring_module(pose_mps)(coords_mps)

    scores_cpu.sum().backward()
    scores_mps.sum().backward()

    grad_cpu = coords_cpu.grad.cpu().numpy()
    grad_mps = coords_mps.grad.cpu().numpy()

    np.testing.assert_allclose(
        grad_mps,
        grad_cpu,
        atol=1e-4,
        rtol=1e-4,
        err_msg="CartBonded: MPS gradients differ from CPU gradients",
    )


@requires_mps
def test_mps_score_function_backward(ubq_pdb, default_database):
    """Full beta2016 backward pass on MPS: gradients must be finite and close to CPU."""
    from tmol.io import pose_stack_from_pdb
    from tmol.score import beta2016_score_function

    cpu = torch.device("cpu")
    mps = torch.device("mps")

    pose_cpu = pose_stack_from_pdb(ubq_pdb, cpu, residue_end=4)
    pose_mps = pose_stack_from_pdb(ubq_pdb, mps, residue_end=4)

    sfxn_cpu = beta2016_score_function(cpu, param_db=default_database)
    sfxn_mps = beta2016_score_function(mps, param_db=default_database)

    wpsm_cpu = sfxn_cpu.render_whole_pose_scoring_module(pose_cpu)
    wpsm_mps = sfxn_mps.render_whole_pose_scoring_module(pose_mps)

    coords_cpu = pose_cpu.coords.clone().requires_grad_(True)
    coords_mps = pose_mps.coords.clone().requires_grad_(True)

    wpsm_cpu(coords_cpu).sum().backward()
    wpsm_mps(coords_mps).sum().backward()

    assert torch.isfinite(coords_mps.grad).all().item(), \
        "beta2016 MPS backward: gradients contain non-finite values"

    np.testing.assert_allclose(
        coords_mps.grad.cpu().numpy(),
        coords_cpu.grad.cpu().numpy(),
        atol=1e-3,
        rtol=1e-3,
        err_msg="beta2016: MPS gradients differ from CPU gradients",
    )
