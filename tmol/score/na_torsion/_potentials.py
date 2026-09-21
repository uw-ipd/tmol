"""Differentiable nucleic acid torsion potentials.

Each torsion sits in a harmonic well around a mean angle, summed over sugar
pucker states. Rosetta picks the alpha/gamma and syn-chi bins with hard
cutoffs; here they are blended, so the energy is continuous.
"""

from functools import lru_cache

import torch

RAD = torch.pi / 180.0


def wrap_degrees(delta):
    """Wrap an angle difference to [-180, 180)."""
    return (delta + 180.0) % 360.0 - 180.0


def dihedral(p0, p1, p2, p3):
    """Signed dihedral in degrees, batched over leading dims."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / b1.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    v = b0 - (b0 * b1).sum(-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
    y = (torch.cross(b1, v, dim=-1) * w).sum(-1)
    x = (v * w).sum(-1)
    return torch.atan2(y, x) / RAD


def _unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def pucker_weights(ring, temperature):
    """Soft distribution over the 10 sugar pucker states.

    ring: (..., 5, 3) ring coordinates in cyclic order, heteroatom last.

    Walks the 5 cyclic rotations of the ring. Within each, `dot` measures how
    planar the leading four atoms are and `exxo` which face the fifth sits on.
    Softmax over -dot picks the apex; a sigmoid on exxo splits endo from exo.
    """
    rot, unshuffle = _pucker_indices(ring.device)

    a = ring[..., rot, :]  # (..., rotation, atom, xyz)
    n12 = _unit(
        torch.cross(a[..., 1, :] - a[..., 0, :], a[..., 2, :] - a[..., 1, :], -1)
    )
    dot = ((n12 * _unit(a[..., 3, :] - a[..., 2, :])).sum(-1)).abs()
    exxo = (n12 * _unit(a[..., 4, :] - 0.5 * (a[..., 3, :] + a[..., 0, :]))).sum(-1)

    # subtract the min before exponentiating; exp(-dot/T) alone underflows
    w_rot = torch.softmax(-(dot - dot.min(-1, keepdim=True).values) / temperature, -1)
    p_endo = torch.sigmoid(-2.0 * exxo / temperature)

    endo_exo = torch.cat([w_rot * p_endo, w_rot * (1.0 - p_endo)], dim=-1)
    return endo_exo[..., unshuffle]


@lru_cache(maxsize=None)
def _pucker_indices(device):
    """Cyclic-rotation gather, and the permutation onto Rosetta's pucker slots."""
    rot = torch.tensor(
        [[(r + k) % 5 for k in range(5)] for r in range(5)], device=device
    )
    # rotation r contributes to endo slot ENDO[r] and exo slot EXO[r]
    slots = (9, 0, 6, 2, 8) + (4, 5, 1, 7, 3)
    unshuffle = torch.tensor([slots.index(j) for j in range(10)], device=device)
    return rot, unshuffle


def blended_devsq(angle, means, weights):
    """Weighted mixture of harmonic wells, not a well about a blended mean.

    angle: (...), means/weights: (..., n_bins). Mixing the squared deviations
    keeps one minimum per bin; mixing the means would invent one between them.
    """
    dev = wrap_degrees(angle.unsqueeze(-1) - means)
    return (weights * dev * dev).sum(-1)


def triple_bin_weights(angle, means, sdev):
    """Soft g+/t/g- assignment, replacing Rosetta's hard 120-degree bins."""
    dev = wrap_degrees(angle.unsqueeze(-1) - means)
    return torch.softmax(-dev * dev / (2.0 * sdev * sdev), -1)


def bi_bii_weight(epsilon, zeta):
    """Weight on BI, from sin(epsilon - zeta).

    exp(-s*del) / (exp(-s*del) + exp(s.del)) with s = 20
    """
    return torch.sigmoid(-40.0 * torch.sin(wrap_degrees(epsilon - zeta) * RAD))


def syn_weight(chi, width=5.0):
    """Soft version of Rosetta's chi in (20, 100) syn window."""
    chi = chi % 360.0
    return torch.sigmoid((chi - 20.0) / width) * torch.sigmoid((100.0 - chi) / width)
