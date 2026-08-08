"""Every donor/acceptor combination must have polynomial parameters.

HBondParamResolver fills its pair table with NaN and overwrites only the pairs
listed in pair_parameters, so an unlisted pair scores NaN rather than zero.
"""

import torch

from tmol.database import ParameterDatabase
from tmol.score.hbond.params import HBondParamResolver


def test_every_donor_acceptor_pair_is_parameterized():
    db = ParameterDatabase.get_default()
    hbdb = db.scoring.hbond

    listed = {(p.donor_type, p.acceptor_type) for p in hbdb.pair_parameters}
    donors = [g.name for g in hbdb.donor_type_params]
    acceptors = [g.name for g in hbdb.acceptor_type_params]

    missing = [(d, a) for d in donors for a in acceptors if (d, a) not in listed]
    assert not missing, f"pairs with no polynomials: {missing}"


def test_resolved_pair_params_are_finite():
    db = ParameterDatabase.get_default()
    resolver = HBondParamResolver.from_database(
        db.chemical, db.scoring.hbond, torch.device("cpu")
    )
    for poly in (
        resolver.pair_params.AHdist,
        resolver.pair_params.cosBAH,
        resolver.pair_params.cosAHD,
    ):
        for field in (poly.range, poly.bound, poly.coeffs):
            assert torch.isfinite(field).all()
