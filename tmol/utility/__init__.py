from .args import _signature, bind_to_args, ignore_unused_kwargs  # noqa: F401
from .attr import AttrMapping, AttrMutableMapping  # noqa: F401
from .auto_number import AutoNumber  # noqa: F401
from .biotite_util import (  # noqa: F401
    get_all_residue_positions,
    get_all_segment_positions,
)  # noqa: F401
from .categorical import (  # noqa: F401
    enum_name_catdtype,
    enum_val_catdtype,
    names_to_name_cat,
    names_to_val_cat,
    vals_to_name_cat,
    vals_to_val_cat,
)  # noqa: F401
from .cpp_extension import (  # noqa: F401
    cuda_if_available,
    get_torch_version,
    load,
    load_inline,
    modulename,
    relpaths,
)  # noqa: F401
from .cumsum import (  # noqa: F401
    exclusive_cumsum,
    exclusive_cumsum1d,
    exclusive_cumsum2d,
    exclusive_cumsum2d_w_totals,
)  # noqa: F401
from .device import resolve_device  # noqa: F401
from .dicttoolz import (  # noqa: F401
    flat_items,
    items,
    keys,
    unflatten,
    update_inplace,
    vals,
)  # noqa: F401
from .log import (  # noqa: F401
    ClassLogger,
    LoggerMixin,
    classlogger_for,
    logger_for_class,
)  # noqa: F401
from .mixins import (  # noqa: F401
    QualifiedName,
    cooperative_superclass_factory,
    gather_superclass_properies,
    qualified_name,
)  # noqa: F401
from .numba import torch_cuda_array_interface  # noqa: F401
from .nvtx import nvtx_range  # noqa: F401
from .units import (  # noqa: F401
    Angle,
    BondAngle,
    DihedralAngle,
    parse_angle,
    parse_bond_angle,
    parse_dihedral_angle,
    u,
    ureg,
)  # noqa: F401

from toolz import first


def unique_val(vals):  # noqa: F811
    """Extract a single, unique value from a collection of values."""
    return just_one(set(vals))


def just_one(vals):  # noqa: F811
    """Extract a single value from a length one collection of values."""
    assert len(vals) == 1
    return first(vals)
