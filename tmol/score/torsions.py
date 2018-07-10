from typing import Optional
from functools import singledispatch

import torch
import math

from tmol.utility.reactive import reactive_attrs, reactive_property

from tmol.types.functional import validate_args
from tmol.types.torch import Tensor

from .factory import Factory


@reactive_attrs(auto_attribs=True)
class AlphaAABackboneTorsionProvider(Factory):
    @staticmethod
    @singledispatch
    def factory_for(
            other,
            device: torch.device,
            **_,
    ):
        """`clone`-factory, extract coords from other."""
        if requires_grad is None:
            requires_grad = other.coords.requires_grad

        phi_inds = torch.tensor(
            other.phi_inds,
            dtype=torch.long,
            device=device,
        ).requires_grad_(requires_grad)

        psi_inds = torch.tensor(
            other.psi_inds,
            dtype=torch.long,
            device=device,
        ).requires_grad_(requires_grad)

        omega_inds = torch.tensor(
            other.omega_inds,
            dtype=torch.long,
            device=device,
        ).requires_grad_(requires_grad)

        return dict(
            phi_inds=phi_inds, psi_inds=psi_inds, omega_inds=omega_inds
        )

    # global indices used to define the torsions
    # an entry of -1 for any atom means the torsion is undefined
    # and will produce a NaN in the corresponding _tor Tensor
    phi_inds: Tensor(torch.long)[:, 4]
    psi_inds: Tensor(torch.long)[:, 4]
    omega_inds: Tensor(torch.long)[:, 4]

    def reset_total_score(self):
        self.phi_inds = self.phi_inds
        self.psi_inds = self.psi_inds
        self.omega_inds = self.omega_inds

    @reactive_property
    def phi_tor(
            coords: Tensor(torch.float)[:, 3],
            phi_inds: Tensor(torch.long)[:, 4]
    ):
        phi_tor = measure_torsions(coords, phi_inds)
        return phi_tor

    @reactive_property
    def psi_tor(
            coords: Tensor(torch.float)[:, 3],
            psi_inds: Tensor(torch.long)[:, 4]
    ):
        psi_tor = measure_torsions(coords, psi_inds)
        return psi_tor

    @reactive_property
    def omega_tor(
            coords: Tensor(torch.float)[:, 3],
            omega_inds: Tensor(torch.long)[:, 4]
    ):
        omega_tor = measure_torsions(coords, psi_inds)
        return omega_tor


@validate_args
def measure_torsions(
        coords: Tensor(torch.float)[:, 3], inds: Tensor(torch.long)[:, 4]
) -> Tensor(torch.float):
    bad = torch.sum(inds == -1, 1) > 0
    tors = torch.tensor((inds.shape[0]),
                        dtype=torch.float,
                        device=coords.device)
    tors[bad] = torch.nan
    p1 = coords[inds[~bad, 0]]
    p2 = coords[inds[~bad, 1]]
    p3 = coords[inds[~bad, 2]]
    p4 = coords[inds[~bad, 3]]

    v21 = p2 - p1
    v32 = p3 - p2
    v43 = p4 - p3

    norm_123 = torch.cross(v21, v32)
    norm_234 = torch.cross(v32, v43)

    norm_123 /= torch.norm(norm_123, 2, dim=1, keepdim=True)
    norm_234 /= torch.norm(norm_234, 2, dim=1, keepdim=True)

    v32 /= torch.norm(v32, 2, dim=1, keepdim=True)
    m1 = torch.cross(norm_123, v32)

    x = torch.einsum("ij,ij->i", (norm_123, norm_234))
    y = torch.einsum("ij,ij->i", (m1, norm_234))
    tors[~bad] = torch.atan2(y, x)

    return tors
