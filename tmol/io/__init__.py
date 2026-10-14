"""Structure conversion between external formats and TMol poses."""

import torch
from pathlib import Path
from typing import Optional, Union

from tmol.types._functional import validate_args
from tmol.types._torch import Tensor
from tmol.pose._pose_stack import PoseStack

from ._build_context import PoseBuildContext
from ._canonical_ordering import HisSpecialCaseIndices
from ._cif import pose_stack_from_file
from ._write_pose_stack_pdb import (
    atom_records_from_pose_stack,
    write_pose_stack_pdb,
)
from ._canonical_ordering import default_packed_block_types, default_canonical_ordering
from .details._his_taut_resolution import (
    his_taut_variant_NE2_protonated,
    his_taut_variant_ND1_protonated,
    his_taut_variant_both_ND1_and_NE2_protonated,
)

__all__ = [
    "PoseBuildContext",
    "atom_records_from_pose_stack",
    "pose_stack_from_file",
    "pose_stack_from_pdb",
    "write_pose_stack_pdb",
    "HisSpecialCaseIndices",
    "default_packed_block_types",
    "default_canonical_ordering",
    "his_taut_variant_NE2_protonated",
    "his_taut_variant_ND1_protonated",
    "his_taut_variant_both_ND1_and_NE2_protonated",
]


@validate_args
def pose_stack_from_pdb(
    pdb_lines_or_fname: Union[str, list, Path],
    device: torch.device,
    *,
    residue_start: Optional[int] = None,
    residue_end: Optional[int] = None,
    res_not_connected: Optional[Tensor[torch.bool][:, :, 2]] = None,
    **kwargs,
) -> PoseStack | tuple[PoseStack, dict] | tuple[PoseStack, PoseBuildContext]:
    """Read PDB through AtomWorks and the shared annotated-array constructor.

    Accept a path, PDB text or a list of lines. Residue slicing uses a half-open
    index range. Additional keywords follow :func:`pose_stack_from_file`, including
    ``prepare_ligands`` and ``return_context``. Supplied hydrogens are retained
    and hydrogen optimization is disabled unless explicitly requested.
    """
    import io

    source = pdb_lines_or_fname
    if isinstance(source, list):
        source = io.StringIO("\n".join(line.rstrip("\n") for line in source))
    elif isinstance(source, str) and (
        "\n" in source or source.startswith(("ATOM  ", "HETATM", "MODEL ", "HEADER"))
    ):
        source = io.StringIO(source)
    kwargs.setdefault("no_optH", True)
    kwargs.setdefault("trust_hydrogen_names", True)
    kwargs.setdefault("missing_density_distance_threshold", 0.0)
    if res_not_connected is not None:
        kwargs["res_not_connected"] = res_not_connected
    return pose_stack_from_file(
        source,
        device,
        residue_start=residue_start,
        residue_end=residue_end,
        **kwargs,
    )
