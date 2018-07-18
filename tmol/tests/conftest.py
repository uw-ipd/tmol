import subprocess

from .database import (  # noqa: F401
    default_database,
)

# Import support fixtures
from .support.rosetta import (  # noqa: F401
    pyrosetta, rosetta_database,
)

# Import basic data fixtures
from .data import ( # noqa: F401
    min_pdb,
    min_res,
    min_system,
    ubq_pdb,
    ubq_res,
    ubq_system,
    ubq_rosetta_baseline,
    water_box_system,
    water_box_res,
)

from .torch import (  # noqa: F401
    torch_device, torch_backward_coverage,
)

from .numba import (  # noqa: F401
    numba_cudasim, numba_cuda_or_cudasim,
)


def pytest_collection_modifyitems(session, config, items):
    # Run all linting-tests *after* the functional tests
    items[:] = sorted(
        items, key=lambda i: i.nodeid.startswith("tmol/tests/linting")
    )


def pytest_benchmark_update_machine_info(config, machine_info):
    import torch
    import json
    import cpuinfo
    import psutil

    def device_info_dict(i):
        dp = torch.cuda.get_device_properties(i)
        return {k: getattr(dp, k) for k in dir(dp) if not k.startswith("_")}

    machine_info["cuda"] = {
        'device':
            {n: device_info_dict(n)
             for n in range(torch.cuda.device_count())},
        "current_device":
            torch.cuda.current_device() if torch.cuda.device_count() else None
    }

    machine_info['cpuinfo'] = cpuinfo.get_cpu_info()

    machine_info['cpu']["logical"] = psutil.cpu_count(logical=True)
    machine_info['cpu']["physical"] = psutil.cpu_count(logical=False)

    machine_info['conda'] = {
        "list": json.loads(subprocess.getoutput("conda list --json"))
    }
