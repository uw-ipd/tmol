#!/usr/bin/env python3
"""Build each native backend once per Python/platform; test multiple Torch versions."""

import argparse
import json

PYTHON_VERSIONS = ("3.11", "3.12", "3.13", "3.14")
LINUX_ARCHES = ("x86_64", "aarch64")
RELEASE_PLATFORMS = {
    "manylinux_2_28_x86_64",
    "manylinux_2_28_aarch64",
    "macosx_14_0_arm64",
}


def rows(backends, arches):
    return [
        {
            "python-version": python,
            "python-tag": python.replace(".", ""),
            "local-tag": backend,
            "package": "tmol" if backend == "cpu" else f"tmol-ops-{backend}",
            "torch-version": "2.14",
            "torch-package-version": "2.14.1",
            "torch-index-url": (
                "https://pypi.org/simple"
                if arch == "arm64"
                else f"https://download.pytorch.org/whl/{'cpu' if backend == 'cpu' else 'cu132'}"
            ),
            "enable-cuda": backend != "cpu",
            "cuda-version": {"cpu": "", "cu12": "12.8", "cu13": "13.0"}[backend],
            "container-image": "unused-for-manylinux",
            "runs-on": {
                "x86_64": "ubuntu-22.04",
                "aarch64": "ubuntu-24.04-arm",
                "arm64": "macos-15",
            }[arch],
            "arch": arch,
            "label": f"py{python} {backend} {arch}",
            "cuda-archs": "all" if backend != "cpu" else "",
            "portability": python == "3.12" and arch != "arm64",
        }
        for backend in backends
        for python in PYTHON_VERSIONS
        for arch in arches
    ]


def gpu_wheel_rows():
    return rows(("cu12", "cu13"), LINUX_ARCHES)


def cpu_wheel_rows():
    return rows(("cpu",), LINUX_ARCHES)


def macos_wheel_rows():
    return rows(("cpu",), ("arm64",))


def linux_wheel_rows():
    return gpu_wheel_rows() + cpu_wheel_rows()


def test_wheel_rows():
    result = []
    for row in linux_wheel_rows() + macos_wheel_rows():
        # CUDA 12 is supported by Torch 2.11; newer Torch releases use CUDA 13.
        versions = ("2.11.0",) if row["local-tag"] == "cu12" else ("2.11.0", "2.14.1")
        for version in versions:
            test = dict(row, **{"torch-package-version": version})
            cuda = {
                "cpu": "cpu",
                "cu12": "cu128",
                "cu13": "cu130" if version == "2.11.0" else "cu132",
            }[row["local-tag"]]
            if row["arch"] != "arm64":
                test["torch-index-url"] = f"https://download.pytorch.org/whl/{cuda}"
            result.append(test)
    return result


def expected_wheel_keys():
    return {
        f"{row['local-tag']}:cp{row['python-tag']}:{row['arch']}"
        for row in linux_wheel_rows() + macos_wheel_rows()
    }


if __name__ == "__main__":
    matrices = {
        "gpu": gpu_wheel_rows,
        "linux-cpu": cpu_wheel_rows,
        "macos": macos_wheel_rows,
        "linux": linux_wheel_rows,
        "test": test_wheel_rows,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", choices=matrices)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    matrix = matrices[args.matrix]()
    if args.smoke:
        matrix = [row for row in matrix if row["python-version"] == "3.12"]
    print(json.dumps({"include": matrix}))
