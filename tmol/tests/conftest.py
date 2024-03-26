import subprocess

from .database import (  # noqa: F401
    default_database,
    default_unpatched_chemical_database,
)

# Import support fixtures
from .support.rosetta import pyrosetta, rosetta_database  # noqa: F401

# Import basic data fixtures
from .data import (  # noqa: F401
    min_pdb,
    min_res,
    min_system,
    big_pdb,
    big_res,
    big_system,
    ubq_pdb,
    ubq_res,
    ubq_system,
    disulfide_pdb,
    disulfide_res,
    cst_system,
    cst_csts,
    ubq_rosetta_baseline,
    water_box_system,
    water_box_res,
    systems_bysize,
    pertuzumab_pdb,
    pertuzumab_and_nearby_erbb2_pdb_and_segments,
    openfold_ubq_and_sumo_pred,
    rosettafold2_ubq_pred,
    rosettafold2_sumo_pred,
)

from .chemical import (  # noqa: F401
    default_restype_set,
    fresh_default_restype_set,
    rts_ubq_res,
    rts_disulfide_res,
)

from .torch import torch_device, torch_backward_coverage  # noqa: F401

from .numba import numba_cudasim, numba_cuda_or_cudasim  # noqa: F401

from .pack.rotamer.dunbrack import dun_sampler  # noqa: F401

from .pose import (  # noqa: F401
    ubq_40_60_pose_stack,
    fresh_default_packed_block_types,
    two_ubq_poses,
)


def pytest_collection_modifyitems(session, config, items):
    # Run all linting-tests *after* the functional tests
    items[:] = sorted(items, key=lambda i: i.nodeid.startswith("tmol/tests/linting"))


def pytest_benchmark_update_machine_info(config, machine_info):
    import torch
    import json
    import cpuinfo
    import psutil

    def device_info_dict(i):
        dp = torch.cuda.get_device_properties(i)
        return {k: getattr(dp, k) for k in dir(dp) if not k.startswith("_")}

    machine_info["cuda"] = {
        "device": {n: device_info_dict(n) for n in range(torch.cuda.device_count())},
        "current_device": (
            torch.cuda.current_device() if torch.cuda.device_count() else None
        ),
    }

    machine_info["cpuinfo"] = cpuinfo.get_cpu_info()

    machine_info["cpu"]["logical"] = psutil.cpu_count(logical=True)
    machine_info["cpu"]["physical"] = psutil.cpu_count(logical=False)

    machine_info["conda"] = {
        "list": json.loads(subprocess.getoutput("conda list --json"))
    }


def pytest_addoption(parser):
    group = parser.getgroup("benchmark")

    group.addoption(
        "--benchmark-cuda-profile",
        action="store_true",
        default=False,
        help="Enable nvrt profile run.",
    )
