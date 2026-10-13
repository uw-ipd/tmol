"""Framework-free host preparation contracts.

Modules in this package may depend on NumPy and the Python standard library,
but not on Torch or JAX. Backend adapters consume the immutable plans and own
all device transfer, stream, output, and workspace state.
"""

from .disulfide import (  # noqa: F401
    DISULFIDE_PARAMETER_FIELDS,
    DISULFIDE_TOPOLOGY_SCHEMA_VERSION,
    DisulfideTopologyPlan,
    prepare_disulfide_topology,
)

__all__ = [
    "DISULFIDE_PARAMETER_FIELDS",
    "DISULFIDE_TOPOLOGY_SCHEMA_VERSION",
    "DisulfideTopologyPlan",
    "prepare_disulfide_topology",
]
