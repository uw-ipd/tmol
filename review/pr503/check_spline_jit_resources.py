"""Inspect H200 driver-compiled copies of the actual scoring PTX."""

import argparse
import ctypes
import hashlib
import json
import re
import subprocess
from pathlib import Path
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--before-extensions", type=Path, required=True)
parser.add_argument("--after-extensions", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
p = args.output.parent
torch.cuda.set_device(0)
torch.empty(1, device="cuda")
cuda = ctypes.CDLL("libcuda.so.1")
cuda.cuModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
cuda.cuModuleGetFunction.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_void_p,
    ctypes.c_char_p,
]
cuda.cuFuncGetAttribute.argtypes = [
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_void_p,
]
cuda.cuModuleUnload.argtypes = [ctypes.c_void_p]


def check(code):
    if code:
        raise RuntimeError(f"CUDA driver error {code}")


records = []
for side, extensions in [
    ("before", args.before_extensions.resolve()),
    ("after", args.after_extensions.resolve()),
]:
    for term in ("dunbrack", "backbone_torsion"):
        directory = p / f"{args.output.stem}-{side}-{term}"
        directory.mkdir(parents=True, exist_ok=True)
        library = (
            extensions
            / f"tmol_score_{term}_potentials__compiled"
            / f"tmol_score_{term}_potentials__compiled.so"
        )
        subprocess.run(
            ["cuobjdump", "--extract-ptx", "all", str(library)],
            cwd=directory,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        for path in directory.glob("*.ptx"):
            source = path.read_text()
            module = ctypes.c_void_p()
            check(cuda.cuModuleLoad(ctypes.byref(module), str(path).encode()))
            functions = []
            for name in re.findall(r"\.entry\s+([^\s(]+)", source):
                if "Dunbrack" not in name and "BackboneTorsion" not in name:
                    continue
                function = ctypes.c_void_p()
                check(
                    cuda.cuModuleGetFunction(
                        ctypes.byref(function), module, name.encode()
                    )
                )
                attributes = {}
                for label, code in [
                    ("registers", 4),
                    ("local_bytes_per_thread", 3),
                    ("shared_bytes", 1),
                    ("binary_version", 6),
                    ("ptx_version", 5),
                ]:
                    value = ctypes.c_int()
                    check(cuda.cuFuncGetAttribute(ctypes.byref(value), code, function))
                    attributes[label] = value.value
                functions.append({"name": name, "attributes": attributes})
            check(cuda.cuModuleUnload(module))
            records.append(
                {
                    "side": side,
                    "term": term,
                    "library": str(library),
                    "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                    "ptx_path": str(path),
                    "ptx_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "functions": functions,
                }
            )
result = {
    "gpu": torch.cuda.get_device_name(),
    "torch": torch.__version__,
    "records": records,
    "limits": "CUDA driver loads extracted score PTX with default JIT options on the allocated GPU and queries each function. These are separately loaded copies, not handles from the running Torch scorer. Host allocator peaks do not include these resources. Static cuobjdump resource reports target sm_75 and must not be described as H200 runtime attributes.",
}
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {
            "gpu": result["gpu"],
            "modules": len(records),
            "functions": sum(len(r["functions"]) for r in records),
        },
        indent=2,
    ),
    flush=True,
)
