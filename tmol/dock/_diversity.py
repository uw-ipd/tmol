"""Energy-ranked RMS-diverse selection shared by the aligner and the GA."""

import math

import torch


def ligand_heavy_atom_rms(coords1: torch.Tensor, coords2: torch.Tensor) -> torch.Tensor:
    """Unsymmetrized heavy-atom coordinate RMS with broadcasting."""
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

    The pairwise heavy-atom RMS matrix is computed once on the device; the
    greedy walk down the energy ranking then runs on the host, with one transfer.
    """
    n_candidates = energies.shape[0]
    if npool > n_candidates:
        raise ValueError("npool cannot exceed the number of candidates")

    order = torch.argsort(energies, stable=True)
    flat = coords[order].reshape(n_candidates, -1)
    distance = torch.cdist(flat, flat, compute_mode="donot_use_mm_for_euclid_dist")
    far_enough = (distance / math.sqrt(coords.shape[1]) >= rmsdthreshold).cpu().numpy()
    kept: list[int] = []
    for rank in range(n_candidates):
        if len(kept) == npool:
            break
        if far_enough[rank, kept].all():
            kept.append(rank)
    skipped = set(kept)
    fill = [r for r in range(n_candidates) if r not in skipped][: npool - len(kept)]
    selected = order[torch.tensor(kept + fill, device=energies.device)]
    return selected, torch.tensor(len(kept), device=energies.device)
