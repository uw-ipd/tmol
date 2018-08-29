import attr
import cattr
import json
import toolz.functoolz

from typing import Tuple, Optional

import torch
from tmol.chemical.aa import AAIndex
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
    _from_rama_db_cache = {}

    table: Tensor(torch.float)[20, 2, 36, 36]
    bspline: BSplineInterpolation
    pro_ind: int

    @classmethod
    @toolz.functoolz.memoize(
        cache=_from_rama_db_cache,
        key=lambda args, kwargs: (args[1], args[2].type, args[2].index),
    )
    def from_ramadb(cls, ramadb: RamaDatabase, device: torch.device):
        """
        Construct a CompactedRamaDatabase from a RamaDatabase.

        Ensure only one compacted copy of the database is created for either
        the CPU or the GPU by using a memoization of the device and the RamaDatabase;
        The RamaDatabase is hashed based on the name of the file that was used
        to create it.
        """

        table = torch.full((20, 2, 36, 36), -1234, dtype=torch.float, device=device)
        ind3 = AAIndex.canonical_laa_ind3()
        for aa in range(len(ind3)):
            aa3name = ind3[aa]
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
        entropy = (
            ((table * torch.log(table)).sum(dim=3)).sum(dim=2).reshape(20, 2, 1, 1)
        )
        table = -1 * torch.log(table) + entropy

        bspline = BSplineInterpolation.from_coordinates(table, degree=3, n_index_dims=2)

        pro_ind = ind3.get_loc("PRO")

        return cls(table=table, bspline=bspline, pro_ind=pro_ind)
