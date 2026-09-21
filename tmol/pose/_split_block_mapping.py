"""Generic mapping for split (fragmented) blocks in a PoseStack.

A split block is a block whose atoms represent a contiguous sub-region of an
original (larger) block type.  The SplitBlockMapping stored on a PoseStack
records, for every split block:

  * Which pose and block slot it occupies.
  * Which other split blocks came from the same original block
    (identified by the shared group_ind within a pose).
  * Which block type the original (unsplit) block had.
  * For each atom of the split block, the corresponding atom index in the
    original block type.

This data structure makes no assumption about why blocks were split;
the ligand-fragmentation logic in tmol.ligand creates instances of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitBlockEntry:
    """Mapping for one split (fragment) block back to its original block type.

    All entries with the same ``(pose_ind, group_ind)`` were split from the
    same original block and should be merged back together when unsplitting.

    ``split_to_orig_atom_inds[i]`` gives the index of the i-th split-block
    atom within the original block type's atom list.

    ``orig_residue_label``, ``orig_chain_label``, and ``orig_ins_code``
    record the PDB residue identity of the original (pre-split) block so
    that the original numbering can be restored without reference to any
    separate ligand-specific mapping object.
    """

    pose_ind: int
    block_ind: int
    group_ind: int  # same-pose fragments of one original block share this
    orig_block_type_ind: int  # index into pose_stack.packed_block_types
    split_to_orig_atom_inds: np.ndarray  # shape (n_atoms_in_split_block,), int32
    orig_residue_label: int  # PDB res_id of the original unsplit block
    orig_chain_label: str  # PDB chain label of the original unsplit block
    orig_ins_code: str  # PDB insertion code of the original unsplit block


@dataclass(frozen=True)
class SplitBlockMapping:
    """All split-block records across every pose in a PoseStack.

    One ``SplitBlockEntry`` per (pose, fragment-block).  Entries are not
    required to be sorted; callers should group by ``(pose_ind, group_ind)``
    to find all fragments of a given original block.
    """

    entries: tuple[SplitBlockEntry, ...]

    def split(self, pose_index: int) -> "SplitBlockMapping":
        """Return the mapping for a single pose, with pose_ind reset to 0."""
        import dataclasses

        return SplitBlockMapping(
            entries=tuple(
                dataclasses.replace(e, pose_ind=0)
                for e in self.entries
                if e.pose_ind == pose_index
            )
        )
