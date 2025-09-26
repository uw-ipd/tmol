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
    big_pdb,
    water_box_pdb,
    ubq_pdb,
    disulfide_pdb,
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
    rts_disulfide_res,
)

from .kinematics import (  # noqa: F401
    ff_2ubq_6res_H,
    ff_3_jagged_ubq_465res_H,
    ff_3_jagged_ubq_465res_star,
    ff_2ubq_6res_U,
    ff_2ubq_6res_K,
)

from .torch import torch_device, torch_backward_coverage  # noqa: F401

from .numba import numba_cudasim, numba_cuda_or_cudasim  # noqa: F401

from .pack.rotamer.dunbrack import dun_sampler  # noqa: F401
from .pack import ubq_repacking_rotamers  # noqa: F401

from .pose import (  # noqa: F401
    ubq_40_60_pose_stack,
    fresh_default_packed_block_types,
    stack_of_two_six_res_ubqs,
    stack_of_two_six_res_ubqs_no_term,
    jagged_stack_of_465_res_ubqs,
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
