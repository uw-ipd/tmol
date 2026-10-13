"""Per-atom / per-residue PDB metadata for a PoseStack.

This is the JAX-port counterpart of ``tmol.pose.pdb_info.PDBInfo``. The
upstream PyTorch implementation stored everything in numpy arrays of
``object`` dtype, alongside integer residue labels and float occupancy /
B-factor arrays. In the JAX port we trim this down to the information
actually consumed by the PDB writer and keep it as plain Python lists of
strings (for the per-atom / per-chain / per-residue labels) and a single
numpy integer array for residue numbers.

Crucially this class is NOT a pytree. The data it holds has no impact on
any structural computation (energies, gradients, etc.) — it only exists to
preserve author-supplied labels across a round-trip to/from a PDB file. It
therefore lives outside the jitted/traced world as plain Python metadata
attached to a ``PoseStack`` object.

Layout
------
- ``chain_id``:           ``list[list[str]]``  length ``n_poses``; each
  inner list has length ``n_blocks_this_pose`` and holds the chain
  label (e.g. ``"A"``, ``"B"``) for each residue.
- ``res_num``:            ``numpy.ndarray`` of shape
  ``(n_poses, max_n_blocks)`` and dtype ``int32`` — the integer residue
  number for each residue (``-1`` for padding slots beyond a pose's
  real residue count).
- ``res_name``:           ``list[list[str]]``  length ``n_poses``; each
  inner list has length ``n_blocks_this_pose`` and holds the 3-letter
  residue name (e.g. ``"ALA"``, ``"HIS"``) for each residue.
- ``atom_record_type``:   ``list[list[list[str]]]``  length ``n_poses``;
  each pose holds a list per block, each block holds a list per atom,
  and each entry is one of ``"ATOM"`` / ``"HETATM"``.

Shape invariants
----------------
- ``len(chain_id) == len(res_name) == len(atom_record_type) == n_poses``.
- ``res_num.shape == (n_poses, max_n_blocks)``.
- For each pose ``p``:
    - ``len(chain_id[p]) == len(res_name[p]) == len(atom_record_type[p])``
      which equals the number of real residues in pose ``p``.
    - ``len(atom_record_type[p][b]) == n_atoms(block_type[p, b])``.

The constructor is permissive: it does not enforce those invariants so
that callers can build up a ``PDBInfo`` incrementally.
"""

from __future__ import annotations

from typing import List

import attrs
import numpy as np

DEFAULT_ATOM_OCCUPANCY = 1.0
DEFAULT_ATOM_B_FACTOR = 0.0


@attrs.define
class PDBInfo:
    """Plain-Python PDB metadata for a ``PoseStack``.

    See the module docstring for the exact layout; this is NOT a pytree.
    """

    chain_id: List[List[str]]
    res_num: np.ndarray  # shape (n_poses, max_n_blocks), dtype int32
    res_name: List[List[str]]
    atom_record_type: List[List[List[str]]]

    # ---------------------------------------------------------------
    # Shape helpers
    # ---------------------------------------------------------------
    @property
    def n_poses(self) -> int:
        return int(self.res_num.shape[0]) if self.res_num.ndim == 2 else 0

    @property
    def max_n_blocks(self) -> int:
        return int(self.res_num.shape[1]) if self.res_num.ndim == 2 else 0

    # ---------------------------------------------------------------
    # Slicing
    # ---------------------------------------------------------------
    def __getitem__(self, i: int) -> "PDBInfo":
        """Return a single-pose ``PDBInfo`` view.

        The result is always a ``PDBInfo`` with ``n_poses == 1``. Nested
        lists are shallow-copied; ``res_num`` is sliced with ``i:i+1`` so
        the resulting array is still 2D.
        """
        if not isinstance(i, int):
            raise TypeError(
                f"PDBInfo only supports integer single-pose indexing, got {type(i).__name__}"
            )
        n = self.n_poses
        if n > 0 and not (-n <= i < n):
            raise IndexError(
                f"pose index {i} out of range for PDBInfo with n_poses={n}"
            )
        j = i % n if n > 0 else 0
        return PDBInfo(
            chain_id=[list(self.chain_id[j])] if self.chain_id else [[]],
            res_num=np.asarray(self.res_num[j : j + 1]).copy(),
            res_name=[list(self.res_name[j])] if self.res_name else [[]],
            atom_record_type=(
                [[list(a) for a in self.atom_record_type[j]]]
                if self.atom_record_type
                else [[]]
            ),
        )

    # ---------------------------------------------------------------
    # Factories
    # ---------------------------------------------------------------
    @classmethod
    def empty(cls, n_poses: int, max_n_blocks: int) -> "PDBInfo":
        """An all-zero/empty ``PDBInfo`` with no per-atom records yet.

        ``res_num`` is filled with ``-1`` (the "padding" sentinel). The
        nested string lists are created as empty inner lists ready to be
        filled in by higher-level construction code.
        """
        return cls(
            chain_id=[[] for _ in range(n_poses)],
            res_num=np.full((n_poses, max_n_blocks), -1, dtype=np.int32),
            res_name=[[] for _ in range(n_poses)],
            atom_record_type=[[] for _ in range(n_poses)],
        )

    @classmethod
    def for_pose_stack(cls, pose_stack) -> "PDBInfo":
        """Build an empty ``PDBInfo`` matching the shape of ``pose_stack``.

        ``pose_stack.block_type_ind`` tells us where real residues live
        (entries ``>= 0``). For real residues we populate:
        - ``chain_id`` with ``"A"``,
        - ``res_num`` with a 1-based residue number per pose,
        - ``res_name`` with the residue type name from the
          ``PackedBlockTypes``, if available (else ``"UNK"``),
        - ``atom_record_type`` with ``"ATOM"`` for each atom of the block.

        The factory is intentionally robust: if any of these pieces of
        information are missing from the ``PackedBlockTypes`` (e.g. in
        unit tests with toy data) we fall back to safe defaults.
        """
        import numpy as _np  # local alias; keeps the type signature tidy

        bti = _np.asarray(pose_stack.block_type_ind)
        n_poses, max_n_blocks = bti.shape

        # Build res_num: 1-based counter of real residues per pose.
        res_num = _np.full((n_poses, max_n_blocks), -1, dtype=_np.int32)
        for p in range(n_poses):
            real_mask = bti[p] >= 0
            n_real = int(real_mask.sum())
            # fill the first n_real entries (conventional layout — real
            # residues are contiguous at the start of each row).
            res_num[p, :n_real] = _np.arange(1, n_real + 1, dtype=_np.int32)

        # Look up residue-type names from the PackedBlockTypes, if the
        # common "active_block_types" attribute is present and contains
        # objects with a ``.name`` (3-letter or longer) attribute.
        pbt = getattr(pose_stack, "packed_block_types", None)
        abt = getattr(pbt, "active_block_types", None) if pbt is not None else None

        def _bt_name(bt_ind: int) -> str:
            if bt_ind < 0 or abt is None or bt_ind >= len(abt):
                return "UNK"
            return str(getattr(abt[bt_ind], "name", "UNK"))

        # Per-block atom counts — used to size the atom_record_type lists.
        n_atoms_per_bt = getattr(pbt, "n_atoms", None) if pbt is not None else None
        if n_atoms_per_bt is not None:
            n_atoms_per_bt = _np.asarray(n_atoms_per_bt)

        chain_id: List[List[str]] = []
        res_name: List[List[str]] = []
        atom_record_type: List[List[List[str]]] = []
        for p in range(n_poses):
            chain_p: List[str] = []
            res_name_p: List[str] = []
            atom_rec_p: List[List[str]] = []
            for b in range(max_n_blocks):
                bt_ind = int(bti[p, b])
                if bt_ind < 0:
                    break
                chain_p.append("A")
                res_name_p.append(_bt_name(bt_ind))
                if n_atoms_per_bt is not None:
                    natoms = int(n_atoms_per_bt[bt_ind])
                else:
                    natoms = 0
                atom_rec_p.append(["ATOM"] * natoms)
            chain_id.append(chain_p)
            res_name.append(res_name_p)
            atom_record_type.append(atom_rec_p)

        return cls(
            chain_id=chain_id,
            res_num=res_num,
            res_name=res_name,
            atom_record_type=atom_record_type,
        )


__all__ = [
    "PDBInfo",
    "DEFAULT_ATOM_OCCUPANCY",
    "DEFAULT_ATOM_B_FACTOR",
]
