"""Helpers shared by the live (ligand-vs-fixed-receptor) scorers."""

from __future__ import annotations

from dataclasses import fields
from typing import ClassVar

import torch


def cell_grid(
    points: torch.Tensor, cell_width: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the cell-grid origin and dimensions covering ``points``."""

    if points.shape[0] == 0:
        return torch.zeros(3, dtype=torch.float64), torch.ones(3, dtype=torch.int64)
    origin = torch.floor(points.min(dim=0).values / cell_width) * cell_width
    cells = torch.floor((points - origin) / cell_width).to(torch.int64)
    return origin, cells.max(dim=0).values + 1


def build_cell_list(
    anchors: torch.Tensor,
    origin: torch.Tensor,
    dims: torch.Tensor,
    cell_width: float,
    source_indices: torch.Tensor | None = None,
    label: str = "receptor site",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CSR ``(offsets, indices)`` binning ``anchors`` into a cell grid.

    ``indices`` are positions in ``anchors`` (or the matching entries of
    ``source_indices`` when given), ordered cell by cell.
    """

    n_cells = int(torch.prod(dims))
    if anchors.shape[0] == 0:
        return (
            torch.zeros(n_cells + 1, dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
        )
    cells = torch.floor((anchors - origin) / cell_width).to(torch.int64)
    if bool(torch.any(cells < 0) or torch.any(cells >= dims)):
        raise RuntimeError(f"{label} fell outside its cell structure")
    linear = (cells[:, 0] * dims[1] + cells[:, 1]) * dims[2] + cells[:, 2]
    order = torch.argsort(linear, stable=True)
    counts = torch.bincount(linear, minlength=n_cells)
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
    if source_indices is not None:
        order = source_indices[order]
    return offsets.contiguous(), order.to(torch.int64).contiguous()


def validate_ligand_coords(
    ligand_coords: torch.Tensor,
    n_atoms: int,
    reference: torch.Tensor,
    label: str,
) -> None:
    """Check ``[n_poses, n_atoms, 3]`` coordinates against a live topology.

    ``reference`` is any tensor of the topology whose device and floating dtype
    the coordinates must match.
    """
    if ligand_coords.shape[0] == 0 or ligand_coords.shape[1] == 0:
        raise ValueError(f"live {label} scoring requires poses and ligand atoms")
    if ligand_coords.shape[1] != n_atoms:
        raise ValueError(f"ligand atom count does not match live {label} topology")


class LiveData:
    """Mixin for frozen dataclasses of tensors moved between device and dtype."""

    # Floating fields that stay float64 whatever the scoring dtype.
    _float64_fields: ClassVar[frozenset] = frozenset()

    def to(self, device: torch.device | str, dtype: torch.dtype):
        """Copy on ``device`` with coordinates and parameters in ``dtype``."""
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                if value.dtype.is_floating_point:
                    value = value.to(
                        dtype=(
                            torch.float64
                            if field.name in self._float64_fields
                            else dtype
                        )
                    )
                value = value.to(device=device).contiguous()
            values[field.name] = value
        return type(self)(**values)


def derived_atom_indices(pose_stack) -> torch.Tensor:
    """Hbond base atoms ``[atom, (donor heavy, acceptor base, acceptor base0)]``."""
    from tmol.score.hbond.potentials._compiled import _gen_hbond_bases

    pbt = pose_stack.packed_block_types
    params = pbt.hbpbt_params
    with torch.no_grad():
        _, derived = _gen_hbond_bases(
            pose_stack.coords.reshape(-1, 3).contiguous(),
            pose_stack.rot_coord_offset,
            pose_stack.first_rot_for_block,
            pose_stack.first_rot_block_type,
            pose_stack.block_ind_for_rot,
            pose_stack.pose_ind_for_rot,
            pose_stack.block_type_ind_for_rot,
            pose_stack.inter_residue_connections,
            pbt.n_atoms,
            pbt.n_conn,
            pbt.conn_atom,
            pbt.n_all_bonds,
            pbt.all_bonds,
            pbt.atom_all_bond_ranges,
            params.tile_n_donH,
            params.tile_n_acc,
            params.tile_donH_inds,
            params.tile_acc_inds,
            params.tile_acceptor_hybridization,
            params.is_hydrogen,
        )
    return derived.to(torch.int64)
