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
    assert "Python, PyTorch, and CUDA tags" in details
