import attr
import cattr
import yaml
import json

from typing import Tuple, Optional

import torch
from ..chemical import AAType, aatype_to_three_letter
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args
from tmol.numeric.bspline import BSplineDegree3, compute_coeffs


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


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RamaDatabase:
    tables: Tuple[RamaTable, ...]

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as infile:
            raw = json.load(infile)
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

    bspdeg: BSplineDegree3
    coefficients: Tensor(torch.float)[20, 2, 36, 36]

    @classmethod
    def from_ramadb(cls, ramadb: RamaDatabase, device: torch.device):

        table = torch.full((20, 2, 36, 36),
                           -1234,
                           dtype=torch.float,
                           device=device)
        coefficients = torch.full((20, 2, 36, 36),
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
                    table[aa, prepro, phi_i, psi_i] = entry.energy
        bspline_deg = BSplineDegree3.construct()
        for aa in range(len(AAType)):
            for prepro in range(2):
                coefficients[aa, prepro, :, :] = compute_coeffs(
                    table[aa, prepro, :, :], bspline_deg
                )

        return cls(table=table, bspdeg=bspline_deg, coefficients=coefficients)
