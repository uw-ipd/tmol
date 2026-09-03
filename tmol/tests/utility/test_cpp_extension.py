from tmol.utility import _cpp_extension


def test_required_cxx_standard():
    assert _cpp_extension._required_cxx_standard("2", "12") == 17
    assert _cpp_extension._required_cxx_standard("2", "13") == 20
    assert _cpp_extension._required_cxx_standard("3", "0") == 20


def test_active_torch_cxx_standard_flags_match():
    expected = _cpp_extension._required_cxx_standard(
        *_cpp_extension.get_torch_version()
    )

    assert f"--std=c++{expected}" in _cpp_extension._required_flags
    assert f"-std=c++{expected}" in _cpp_extension._required_cuda_flags


def test_select_cuda_architecture_prefers_active_device():
    select = _cpp_extension._select_cuda_architecture
    assert select(None, (9, 0)) == "9.0"
    assert select("   ", (9, 0)) == "9.0"
    assert select("8.0;9.0;12.0+PTX", (12, 0)) == "12.0"
    assert select("7.5;8.0;8.6;9.0", (8, 9)) == "8.9"
    assert select("8.6 9.0", (9, 0)) == "9.0"
    assert select("8.6", (9, 0)) == "8.6"


def test_custom_extension_flags_preserve_release_optimization():
    kwargs = _cpp_extension._augment_kwargs(
        "test_extension",
        ["test.cpp"],
        extra_cflags=["-DCUSTOM_CXX"],
        extra_cuda_cflags=["-DCUSTOM_CUDA"],
        with_cuda=False,
    )

    assert kwargs["extra_cflags"][:2] == ["-O3", "-DCUSTOM_CXX"]
    assert kwargs["extra_cuda_cflags"][:2] == ["-O3", "-DCUSTOM_CUDA"]
