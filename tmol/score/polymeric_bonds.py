from functools import singledispatch

import torch
import attr

from tmol.utility.reactive import reactive_attrs

from tmol.types.torch import Tensor

from .factory import Factory


@reactive_attrs(auto_attribs=True)
class PolymericBonds(Factory):
    @staticmethod
    @singledispatch
    def factory_for(other, device: torch.device, **_):
        """`clone`-factory, extract upper- and lower- residue indices from other."""
        upper = torch.tensor(other.upper, dtype=torch.long, device=device)
        lower = torch.tensor(other.lower, dtype=torch.long, device=device)
        return dict(upper=upper, lower=lower)

    # The index of the residue that residue i has as its upper-connection neighbor
    # (often i+1), with -1 signifying that no upper connection is present
    upper: Tensor(torch.long)[:, :] = attr.ib()

    # The index of the residue that residue i has as its lower-connection neighbor
    # (often i-1), with -1 signifying that no lower connection is present
    lower: Tensor(torch.long)[:, :] = attr.ib()
