"""Tests for precompiled extension load error messages."""

from tmol._cpp_lib import extension_load_error_details


def test_glibcxx_error_mentions_libstdc():
    exc = OSError(
        "/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.32' not found"
    )
    details = extension_load_error_details(exc)
    assert "libstdc++" in details
    assert "TMOL_DISABLE_WHEEL_FETCH" in details
    assert "TMOL_JIT_FALLBACK" in details
    assert "PyTorch/CUDA" not in details


def test_glibc_error_mentions_build_from_source():
    exc = OSError("/lib64/libc.so.6: version `GLIBC_2.32' not found")
    details = extension_load_error_details(exc)
    assert "glibc" in details.lower()
    assert "TMOL_DISABLE_WHEEL_FETCH" in details


def test_generic_error_mentions_wheel_tags():
    exc = OSError("undefined symbol: some_missing_op")
    details = extension_load_error_details(exc)
    assert "PyTorch >=2.11, and the CUDA major" in details


def test_native_boundary_gradients(torch_device):
    """Exercise strided tensors and derivative graphs through the native ABI."""
    import torch

    coords = torch.randn(4, 4, 3, dtype=torch.double, device=torch_device)
    coords = coords[::2].requires_grad_(True)
    torsion = torch.ops.tmol_constraint.get_torsion_angle
    assert torch.autograd.gradcheck(torsion, (coords,))
    assert torch.autograd.gradcheck(torsion, (torch._neg_view(coords),))
    expected = torch.autograd.grad(torsion(-coords).sum(), coords)[0]
    actual = torch.autograd.grad(torsion(torch._neg_view(coords)).sum(), coords)[0]
    torch.testing.assert_close(actual, expected)
    upstream = torch.ones(
        2, dtype=torch.double, device=torch_device, requires_grad=True
    )
    gradient = torch.autograd.grad(
        torsion(coords), coords, upstream, create_graph=True
    )[0]
    assert torch.autograd.grad(gradient.square().sum(), upstream)[0].isfinite().all()
    # Hooks may replace saved tensors; the native bridge must use the unpacked
    # tensor instead of retaining a private copy that bypasses the hooks.
    with torch.autograd.graph.saved_tensors_hooks(
        lambda tensor: tensor.detach().clone(), lambda tensor: tensor
    ):
        hooked = torch.autograd.grad(torsion(coords).sum(), coords)[0]
    torch.testing.assert_close(hooked, gradient.detach())
    assert not torsion(coords.detach()).requires_grad


def test_native_boundary_saved_input_version(ubq_pdb, torch_device):
    import pytest
    import torch
    from tmol.io import pose_stack_from_file
    from tmol.score import beta2016_score_function

    pose = pose_stack_from_file(
        ubq_pdb,
        torch_device,
        residue_end=4,
        no_optH=True,
        missing_density_distance_threshold=0.0,
    )
    scorer = beta2016_score_function(torch_device).render_whole_pose_scoring_module(
        pose
    )
    coords = pose.coords.clone().requires_grad_(True)
    scores = scorer(coords.view_as(coords))
    with torch.no_grad():
        coords.add_(0)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        scores.sum().backward()

    # A resolved negative view owns a copy; mutation of its source must preserve
    # the stored gradient and its connection to the original leaf.
    scores = scorer(torch._neg_view(coords)).sum()
    expected = torch.autograd.grad(scores, coords, retain_graph=True)[0]
    with torch.no_grad():
        coords.add_(0.3)
    actual = torch.autograd.grad(scores, coords)[0]
    torch.testing.assert_close(actual, expected)


def test_backend_selection(monkeypatch):
    import tmol._cpp_lib as loader

    monkeypatch.setattr(loader.torch.version, "cuda", "13.0")
    monkeypatch.setattr(loader, "version", lambda name: "0.1.63")
    monkeypatch.setattr(loader.importlib.util, "find_spec", lambda name: object())
    assert loader._native_module_name() == "tmol_ops_cu13._C"
    monkeypatch.setattr(loader.torch.version, "cuda", "12.8")
    assert loader._native_module_name() == "tmol_ops_cu12._C"
    monkeypatch.setattr(loader.torch.version, "cuda", None)
    assert loader._native_module_name() == "tmol._C"

    import pytest

    monkeypatch.setattr(loader.torch.version, "cuda", "13.0")
    monkeypatch.setattr(
        loader, "version", lambda name: "0.1.64" if name == "tmol" else "0.1.63"
    )
    with pytest.raises(loader.TmolExtensionIncompatibleError, match="same version"):
        loader._native_module_name()
