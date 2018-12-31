import torch
import pytest
import numpy
import attr

from tmol.database import ParameterDatabase

from tmol.score.hbond.identification import HBondElementAnalysis
from tmol.score.hbond.params import HBondParamResolver

from tmol.utility.dicttoolz import flat_items, keymap, merge

from tmol.utility.args import ignore_unused_kwargs


@pytest.fixture(scope="session")
def compiled():
    import tmol.score.hbond.potentials.compiled

    return tmol.score.hbond.potentials.compiled


def test_score_op(compiled, ubq_system):
    """Scores generated via compiled hbond dispatch match values from explicit
    evaluation."""
    system = ubq_system
    hbond_database = ParameterDatabase.get_default().scoring.hbond

    # Load coordinates and types from standard test system

    atom_types = system.atom_metadata["atom_type"].copy()
    bonds = system.bonds.copy()
    coords = system.coords.copy()

    # Run donor/acceptor identification over system

    hbond_elements = HBondElementAnalysis.setup(
        hbond_database=hbond_database, atom_types=atom_types, bonds=bonds
    )

    donors = hbond_elements.donors
    acceptors = numpy.concatenate(
        (
            hbond_elements.sp2_acceptors,
            hbond_elements.sp3_acceptors,
            hbond_elements.ring_acceptors,
        )
    )

    # Load type pair parameter arrays and covert to tensor types
    hbond_param_resolver = HBondParamResolver.from_database(hbond_database)

    def _t(v):
        t = torch.tensor(v)
        if t.dtype == torch.float64:
            t = t.to(torch.float32)
        return t

    def _t_coords(donors, acceptors):
        """fetch coordinates as tensors"""
        return dict(
            D=_t(coords[donors["d"]]),
            H=_t(coords[donors["h"]]),
            donor_type_index=_t(
                hbond_param_resolver.donor_type_index.get_indexer(donors["donor_type"])
            ).to(torch.int32),
            A=_t(coords[acceptors["a"]]),
            B=_t(coords[acceptors["b"]]),
            B0=_t(coords[acceptors["b0"]]),
            acceptor_type_index=_t(
                hbond_param_resolver.acceptor_type_index.get_indexer(
                    acceptors["acceptor_type"]
                )
            ).to(torch.int32),
        )

    def _t_params(param_set):
        """fetch type pair parameters as flattened args"""
        return {"_".join(k): _t(v) for k, v in flat_items(attr.asdict(param_set))}

    # Run score dispatch over acceptors/donors
    inputs = _t_coords(donors, acceptors)
    pparams = _t_params(hbond_param_resolver.pair_params)
    gparams = attr.asdict(hbond_database.global_parameters)

    inds, scores, *derivs = ignore_unused_kwargs(compiled.hbond_pair_score)(
        **merge(inputs, pparams, gparams)
    )

    # assert that some healthy number of bonds were scored
    assert (scores < 0).sum() > 10

    # Verify scores via back comparison to explicit evaluation
    dind, aind = inds.transpose(0, 1)

    batch_coords = _t_coords(donors[dind], acceptors[aind])
    batch_params = _t_params(
        hbond_param_resolver.pair_params[
            batch_coords["donor_type_index"].to(torch.int64),
            batch_coords["acceptor_type_index"].to(torch.int64),
        ]
    )

    batch_scores, *batch_derivs = ignore_unused_kwargs(compiled.hbond_score_V_dV)(
        **merge(batch_coords, batch_params, gparams)
    )

    torch.testing.assert_allclose(scores, batch_scores)
    for d, bd in zip(derivs, batch_derivs):
        torch.testing.assert_allclose(d, bd)
