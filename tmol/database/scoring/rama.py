import attr
import cattr
import yaml
import json

from typing import Tuple, Optional

import torch
from ..chemical import AAType, aatype_to_three_letter
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args
from tmol.numeric.bspline import BSplineInterpolation


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RamaEntry:
    phi: float
    psi: float
    prob: float
    energy: float


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RamaTable:
    aa_class: str
    phi_step: float
    psi_step: float
    prepro: bool
    entries: Tuple[RamaEntry, ...]


@attr.s(auto_attribs=True, frozen=True, slots=True, hash=False, repr=False)
class RamaDatabase:
    sourcefilepath: str
    tables: Tuple[RamaTable, ...]

    def __hash__(self):
        return hash(self.sourcefilepath)

    def __repr__(self):
        return "RamaDatabase(%s)" % self.sourcefilepath

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as infile:
            raw = json.load(infile)
        raw["sourcefilepath"] = path
        return cattr.structure(raw, cls)

    @validate_args
    def find(self, aaname: str, prepro: bool) -> Optional[RamaTable]:
        for table in self.tables:
            if table.aa_class == aaname and table.prepro == prepro:
                return table
        return None


@attr.s(auto_attribs=True, frozen=True, slots=True)
class CompactedRamaDatabase:
    table: Tensor(torch.float)[20, 2, 36, 36]

    bspline: BSplineInterpolation

    @classmethod
    def from_ramadb(cls, ramadb: RamaDatabase, device: torch.device):

        table = torch.full((20, 2, 36, 36),
                           -1234,
                           dtype=torch.float,
                           device=device)
        for aa in range(len(AAType)):
            aa3name = aatype_to_three_letter[aa]
            for prepro in range(2):
                aatab = ramadb.find(aa3name, bool(prepro))
                assert aatab
                aatab = aatab.entries
                for entry in aatab:
                    phi_i = int(entry.phi) // 10 + 18
                    psi_i = int(entry.psi) // 10 + 18
                    assert phi_i < 36 and psi_i < 36
                    assert phi_i >= 0 and psi_i >= 0
                    table[aa, prepro, phi_i, psi_i] = entry.prob

        # exp of the -energies should get back to the original probabilities
        # so we can calculate the table entropies
        entropy = ((table * torch.log(table)).sum(dim=3)).sum(dim=2).reshape(
            20, 2, 1, 1
        )
        table = -1 * torch.log(table) + entropy

        bspline = BSplineInterpolation.from_coordinates(
            table, degree=3, n_index_dims=2
        )

        return cls(table=table, bspline=bspline)
