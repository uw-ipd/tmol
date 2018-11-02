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
    """Provide the named torsions "phi," "psi," and "omega"
    to terms that require them.

    The indices for the atoms that define these torsions for each residue
    must be provided. These incides are then used to retrieve the
    cartesian coordinates of the atoms defining these torsions.
    Torsions are then computed from the coordinates. Torsions are computed
    at most once as part of the reactive system.

    A residue which does not define any of these torsions, for
    whatever reason (e.g. the residue type does not name these torsions,
    or the residue is at a chain terminus and the torsion is not defined)
    should give an atom index of -1 at at least one position in order
    to signify that the torsion should not be computed.
    """

    @staticmethod
    @singledispatch
    def factory_for(other, device: torch.device, **_):
        """``clone``-factory, extract coords from other."""

        phi_inds = torch.tensor(other.phi_inds, dtype=torch.long, device=device)
        psi_inds = torch.tensor(other.psi_inds, dtype=torch.long, device=device)
        omega_inds = torch.tensor(other.omega_inds, dtype=torch.long, device=device)

        return dict(phi_inds=phi_inds, psi_inds=psi_inds, omega_inds=omega_inds)

    phi_inds: Tensor(torch.long)[:, :, 4] = attr.ib()
    psi_inds: Tensor(torch.long)[:, :, 4] = attr.ib()
    omega_inds: Tensor(torch.long)[:, :, 4] = attr.ib()

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
    """Compute a 1D tensor of torsions from a 2D tensor of coordinates and
    a 2D tensor of index quadrouples.

    Torsions are not computed for positions in the index list that contain one
    or more entries of -1; the corresponding position in the returned torsion
    tensor will be NaN.
    """

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
