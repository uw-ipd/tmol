"""Energy-ranked RMS-diverse selection shared by the aligner and the GA."""

import math

import torch


def ligand_heavy_atom_rms(
    coords1: torch.Tensor, coords2: torch.Tensor
) -> torch.Tensor:
    """Unsymmetrized heavy-atom coordinate RMS with broadcasting."""

    if coords1.shape[-2:] != coords2.shape[-2:]:
        raise ValueError("ligand coordinate tensors must have matching atom shapes")
    delta = coords1 - coords2
    return torch.sqrt(torch.mean(torch.sum(delta * delta, dim=-1), dim=-1))


def select_diverse_low_energy(
    energies: torch.Tensor,
    coords: torch.Tensor,
    *,
    npool: int,
    rmsdthreshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select energy-ranked RMS niches, then fill from rejected candidates.

    The bounded host loop walks candidate energy ranks; RMS storage is limited
    to the selected pool instead of an atom-expanded all-pairs candidate tensor.
    """

    if energies.ndim != 1:
        raise ValueError("energies must have shape [candidate]")
    if coords.ndim != 3 or coords.shape[0] != energies.shape[0]:
        raise ValueError("coords must have shape [candidate, heavy_atom, 3]")
    if coords.shape[-1] != 3 or coords.shape[1] < 1:
        raise ValueError("coords must contain at least one 3D heavy atom")
    if not isinstance(npool, int) or isinstance(npool, bool) or npool < 1:
        raise ValueError("npool must be a positive integer")
    n_candidates = energies.shape[0]
    if npool > n_candidates:
        raise ValueError("npool cannot exceed the number of candidates")
    if not math.isfinite(rmsdthreshold) or rmsdthreshold < 0:
        raise ValueError("rmsdthreshold must be finite and nonnegative")

    order = torch.argsort(energies, stable=True)
    selected_order = torch.full((npool,), -1, dtype=torch.int64, device=energies.device)
    selected_mask = torch.zeros(n_candidates, dtype=torch.bool, device=energies.device)
    selected_count = torch.zeros((), dtype=torch.int64, device=energies.device)
    selected_coords = coords[:1].expand(npool, -1, -1).clone()
    pool_columns = torch.arange(npool, device=energies.device)

    for rank in range(n_candidates):
        candidate = order[rank : rank + 1]
        candidate_coords = coords.index_select(0, candidate)
        candidate_distances = ligand_heavy_atom_rms(candidate_coords, selected_coords)
        active_selected = pool_columns < selected_count
        sufficiently_distant = torch.all(
            ~active_selected | (candidate_distances >= rmsdthreshold)
        )
        accept = sufficiently_distant & (selected_count < npool)
        write_column = selected_count.clamp_max(npool - 1).reshape(1)
        previous = selected_order.gather(0, write_column)
        selected_order.scatter_(
            0, write_column, torch.where(accept, candidate, previous)
        )
        selected_mask.scatter_(0, candidate, accept.reshape(1))
        write_coords = write_column[:, None, None].expand(-1, coords.shape[1], 3)
        previous_coords = selected_coords.gather(0, write_coords)
        selected_coords.scatter_(
            0, write_coords, torch.where(accept, candidate_coords, previous_coords)
        )
        selected_count += accept.to(torch.int64)

    selected_in_rank_order = selected_mask.gather(0, order)
    rejected_rank = torch.where(
        ~selected_in_rank_order,
        torch.arange(n_candidates, device=energies.device),
        n_candidates,
    )
    rejected_order = order.gather(0, torch.argsort(rejected_rank))
    fill_columns = (pool_columns - selected_count).clamp_min(0)
    filled = rejected_order.gather(0, fill_columns)
    selected = torch.where(pool_columns < selected_count, selected_order, filled)
    return selected, selected_count
