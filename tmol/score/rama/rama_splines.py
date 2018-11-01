import attr
import toolz.functoolz
import torch

from tmol.types.torch import Tensor
from tmol.types.functional import validate_args
from tmol.numeric.bspline import BSplineInterpolation
from tmol.database.scoring.rama import RamaDatabase, RamaMapper


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RamaSplines:
    _from_rama_db_cache = {}

    table: Tensor(torch.float)[:, 36, 36]
    bspline: BSplineInterpolation
    mapper: RamaMapper

    @classmethod
    @toolz.functoolz.memoize(
        cache=_from_rama_db_cache,
        key=lambda args, kwargs: (args[1], args[2].type, args[2].index),
    )
    def from_ramadb(cls, ramadb: RamaDatabase, device: torch.device):
        """
        Construct a RamaSplines from a RamaDatabase.

        Ensure only one compacted copy of the database is created for either
        the CPU or the GPU by using a memoization of the device and the RamaDatabase;
        The RamaDatabase is hashed based on the name of the file that was used
        to create it.
        """

        table = torch.full(
            (len(ramadb.tables), 36, 36), -1234, dtype=torch.float, device=device
        )
        for i, tab in enumerate(ramadb.tables):
            for entry in tab.entries:
                phi_i = int(entry.phi) // 10 + 18
                psi_i = int(entry.psi) // 10 + 18
                assert phi_i < 36 and psi_i < 36
                assert phi_i >= 0 and psi_i >= 0
                table[i, phi_i, psi_i] = entry.prob

        # exp of the -energies should get back to the original probabilities
        # so we can calculate the table entropies
        entropy = (
            ((table * torch.log(table)).sum(dim=2))
            .sum(dim=1)
            .reshape(len(ramadb.tables), 1, 1)
        )
        table = -1 * torch.log(table) + entropy

        bspline = BSplineInterpolation.from_coordinates(table, degree=3, n_index_dims=1)

        return cls(table=table, bspline=bspline, mapper=ramadb.mapper)
