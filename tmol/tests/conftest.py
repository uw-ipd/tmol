import subprocess

# Import support fixtures
from .support.rosetta import (
    pyrosetta,
    rosetta_database,
)

# Import basic data fixtures
from .data import (
    ubq_pdb,
    ubq_res,
    ubq_system,
    ubq_rosetta_baseline,
    water_box_system,
    water_box_res,
)

from .torch import (
    torch_device,
    torch_backward_coverage,
)


def pytest_collection_modifyitems(session, config, items):
    # Run all linting-tests *after* the functional tests
    items[:] = sorted(
        items, key=lambda i: i.nodeid.startswith("tmol/tests/linting")
    )


def pytest_benchmark_update_machine_info(config, machine_info):
    import torch
    import json

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

    machine_info['conda'] = {
        "list": json.loads(subprocess.getoutput("conda list --json"))
    }


__all__ = (
    pytest_benchmark_update_machine_info,
    pyrosetta,
    rosetta_database,
    ubq_pdb,
    ubq_res,
    ubq_system,
    ubq_rosetta_baseline,
    water_box_system,
    water_box_res,
    torch_device,
    torch_backward_coverage,
)
