import torch
from typing import Optional, Union
from tmol.types.functional import validate_args
from tmol.pose.pose_stack import PoseStack


@validate_args
def pose_stack_from_pdb(
    pdb_lines_or_fname: Union[str, list],
    device: torch.device,
    *,
    residue_start: Optional[int] = None,
    residue_end: Optional[int] = None,
    **kwargs,
) -> PoseStack:
    """Construct a PoseStack given the contents of a PDB file or the name of a PDB file,
    using the full set of residue types contained in tmol's chemical.yaml file.

    Optionally, a subset of the residues in the range from residue_start to residue_end-1
    can be requested.
    Any additional keyword arguments will be passed to pose_stack_from_canonical_form
    """
    from tmol.io.canonical_ordering import (
        default_canonical_ordering,
        default_packed_block_types,
        canonical_form_from_pdb,
    )
    from tmol.io.pose_stack_construction import pose_stack_from_canonical_form

    co = default_canonical_ordering()
    pbt = default_packed_block_types(device)
    cf = canonical_form_from_pdb(
        co,
        pdb_lines_or_fname,
        device,
        residue_start=residue_start,
        residue_end=residue_end,
    )
    return pose_stack_from_canonical_form(co, pbt, **cf, **kwargs)
