from functools import singledispatch

import torch
import attr

from tmol.utility.reactive import reactive_attrs

from tmol.types.torch import Tensor

from .factory import Factory


@reactive_attrs(auto_attribs=True)
class PolymericBonds(Factory):
    """Scoring mixin for representing polymeric connections.

    Upper and lower residue incides are given for each residue,
    with the sentinel value of -1 used to designate "no such
    connection."  Most polymeric connections are to i-1 and i+1,
    but not all (e.g. chain ends, cyclic peptides), and therefore
    some class must provide to the scoring terms which need to
    understand polymeric connections that actual indices of the
    upper and lower neighbors.
    """

    @staticmethod
    @singledispatch
    def factory_for(other, device: torch.device, **_):
        """`clone`-factory, extract upper- and lower- residue indices from other."""
        upper = torch.tensor(other.upper, dtype=torch.long, device=device)
        lower = torch.tensor(other.lower, dtype=torch.long, device=device)
        return dict(upper=upper, lower=lower)

    upper: Tensor(torch.long)[:, :] = attr.ib()
    lower: Tensor(torch.long)[:, :] = attr.ib()
