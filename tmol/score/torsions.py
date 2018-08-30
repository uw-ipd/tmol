from functools import singledispatch

import torch
import numpy
import attr

from tmol.utility.reactive import reactive_attrs, reactive_property

from tmol.numeric.dihedrals import coord_dihedrals

from tmol.types.functional import validate_args
from tmol.types.torch import Tensor

from .factory import Factory


@reactive_attrs(auto_attribs=True)
class AlphaAABackboneTorsionProvider(Factory):
    @staticmethod
    @singledispatch
    def factory_for(other, device: torch.device, **_):
        """``clone``-factory, extract coords from other."""
        # if requires_grad is None:
        #    requires_grad = other.coords.requires_grad

        phi_inds = torch.tensor(other.phi_inds, dtype=torch.long, device=device)
        psi_inds = torch.tensor(other.psi_inds, dtype=torch.long, device=device)
        omega_inds = torch.tensor(other.omega_inds, dtype=torch.long, device=device)
        res_aas = torch.tensor(other.res_aas, dtype=torch.long, device=device)

        return dict(
            phi_inds=phi_inds, psi_inds=psi_inds, omega_inds=omega_inds, res_aas=res_aas
        )

    # global indices used to define the torsions
    # an entry of -1 for any atom means the torsion is undefined
    # and will produce a NaN in the corresponding _tor Tensor
    phi_inds: Tensor(torch.long)[:, :, 4] = attr.ib()
    psi_inds: Tensor(torch.long)[:, :, 4] = attr.ib()
    omega_inds: Tensor(torch.long)[:, :, 4] = attr.ib()

    # Integer representation of the canonical l-aa for each residue; -1 if the residue
    # is not a canonical l-aa. This integer index should come from the pandas index
    # that lives in tmol/chemical/aa.py for consistent use across multiple classes
    res_aas: Tensor(torch.long)[:, :] = attr.ib()

    @reactive_property
    def phi_tor(
        coords64: Tensor(torch.double)[:, :, 3], phi_inds: Tensor(torch.long)[:, :, 4]
    ) -> Tensor(torch.float)[:, :]:
        assert coords64.shape[0] == 1
        assert phi_inds.shape[0] == 1
        phi_tor = measure_torsions(coords64[0, :], phi_inds[0, :])
        return phi_tor[None, :]

    @reactive_property
    def psi_tor(
        coords64: Tensor(torch.double)[:, :, 3], psi_inds: Tensor(torch.long)[:, :, 4]
    ) -> Tensor(torch.float)[:, :]:
        assert coords64.shape[0] == 1
        assert psi_inds.shape[0] == 1
        psi_tor = measure_torsions(coords64[0, :], psi_inds[0, :])
        return psi_tor[None, :]

    @reactive_property
    def omega_tor(
        coords64: Tensor(torch.double)[:, :, 3], omega_inds: Tensor(torch.long)[:, :, 4]
    ) -> Tensor(torch.float)[:, :]:
        assert coords64.shape[0] == 1
        assert omega_inds.shape[0] == 1
        omega_tor = measure_torsions(coords64[0, :], omega_inds[0, :])
        return omega_tor[None, :]


@validate_args
def measure_torsions(
    coords: Tensor(torch.double)[:, 3], inds: Tensor(torch.long)[:, 4]
) -> Tensor(torch.float):
    bad = torch.sum(inds == -1, 1) > 0
    tors = torch.full(
        (inds.shape[0],), numpy.nan, dtype=torch.float, device=coords.device
    )
    tors[bad] = numpy.nan
    p1 = coords[inds[~bad, 0]]
    p2 = coords[inds[~bad, 1]]
    p3 = coords[inds[~bad, 2]]
    p4 = coords[inds[~bad, 3]]

    tors[~bad] = coord_dihedrals(p1, p2, p3, p4)

    return tors
