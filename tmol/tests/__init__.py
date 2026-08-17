_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "gradcheck": ("autograd", "gradcheck"),
    "VectorizedOp": ("autograd", "VectorizedOp"),
    "make_fixture": ("benchmark", "make_fixture"),
    "make_subfixture": ("benchmark", "make_subfixture"),
    "subfixture": ("benchmark", "subfixture"),
    "stat_frame": ("benchmark", "stat_frame"),
    "stat_frame_from_metadata": ("benchmark", "stat_frame_from_metadata"),
    "stat_frame_from_result_list": ("benchmark", "stat_frame_from_result_list"),
    "BenchmarkPlot": ("benchmark_plot", "BenchmarkPlot"),
    "pytest_collection_modifyitems": ("conftest", "pytest_collection_modifyitems"),
    "pytest_benchmark_update_machine_info": (
        "conftest",
        "pytest_benchmark_update_machine_info",
    ),
    "pytest_addoption": ("conftest", "pytest_addoption"),
    "is_jit_available": ("numba", "is_jit_available"),
    "jit_available": ("numba", "jit_available"),
    "requires_numba_jit": ("numba", "requires_numba_jit"),
    "with_cudasim": ("numba", "with_cudasim"),
    "numba_cudasim": ("numba", "numba_cudasim"),
    "numba_cuda_or_cudasim": ("numba", "numba_cuda_or_cudasim"),
    "test_str_join_method": ("test_benchmark", "test_str_join_method"),
    "test_str_join_invalid": ("test_benchmark", "test_str_join_invalid"),
    "test_candidate_wheels_include_torch_213_cu130_x86_64_fallback": (
        "test_build_backend",
        "test_candidate_wheels_include_torch_213_cu130_x86_64_fallback",
    ),
    "test_candidate_wheels_include_manylinux_aarch64_then_native_fallback": (
        "test_build_backend",
        "test_candidate_wheels_include_manylinux_aarch64_then_native_fallback",
    ),
    "test_candidate_wheels_include_stable_torch_210_variants": (
        "test_build_backend",
        "test_candidate_wheels_include_stable_torch_210_variants",
    ),
    "test_build_wheel_uses_downloaded_wheel_when_available": (
        "test_build_backend",
        "test_build_wheel_uses_downloaded_wheel_when_available",
    ),
    "test_build_wheel_falls_back_when_no_prebuilt_match": (
        "test_build_backend",
        "test_build_wheel_falls_back_when_no_prebuilt_match",
    ),
    "test_build_wheel_skips_fetch_in_repo_checkout": (
        "test_build_backend",
        "test_build_wheel_skips_fetch_in_repo_checkout",
    ),
    "test_build_wheel_force_build_env_skips_fetch": (
        "test_build_backend",
        "test_build_wheel_force_build_env_skips_fetch",
    ),
    "test_build_wheel_attempts_autodetect_in_isolated_build_by_default": (
        "test_build_backend",
        "test_build_wheel_attempts_autodetect_in_isolated_build_by_default",
    ),
    "test_download_retries_then_succeeds": (
        "test_build_backend",
        "test_download_retries_then_succeeds",
    ),
    "test_glibcxx_error_mentions_libstdc": (
        "test_cpp_lib",
        "test_glibcxx_error_mentions_libstdc",
    ),
    "test_glibc_error_mentions_build_from_source": (
        "test_cpp_lib",
        "test_glibc_error_mentions_build_from_source",
    ),
    "test_generic_error_mentions_wheel_tags": (
        "test_cpp_lib",
        "test_generic_error_mentions_wheel_tags",
    ),
    "test_torch_cuda_is_available": ("test_cuda", "test_torch_cuda_is_available"),
    "test_torch_cuda_smoke": ("test_cuda", "test_torch_cuda_smoke"),
    "test_float_perf": ("test_cuda", "test_float_perf"),
    "requires_cuda": ("torch", "requires_cuda"),
    "zero_padded_counts": ("torch", "zero_padded_counts"),
    "torch_device": ("torch", "torch_device"),
    "cuda_not_implemented": ("torch", "cuda_not_implemented"),
    "torch_backward_coverage": ("torch", "torch_backward_coverage"),
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        import importlib

        mod_leaf, attr = _LAZY_ATTRS[name]
        mod = importlib.import_module(f".{mod_leaf}", package=__name__)
        # Re-cache every name from this module so that Python's import
        # machinery (which sets globals()[mod_leaf] = MODULE as a side-effect)
        # does not overwrite previously resolved function/class references.
        for _n, (_m, _a) in _LAZY_ATTRS.items():
            if _m == mod_leaf:
                try:
                    globals()[_n] = getattr(mod, _a)
                except AttributeError:
                    pass
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
