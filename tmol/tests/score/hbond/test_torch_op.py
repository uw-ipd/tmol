import torch
import pytest
import attr

from tmol.database import ParameterDatabase

from tmol.score.hbond.identification import HBondElementAnalysis
from tmol.score.hbond.params import HBondParamResolver

from tmol.utility.dicttoolz import flat_items, merge

from tmol.utility.args import ignore_unused_kwargs, bind_to_args

from tmol.tests.autograd import gradcheck


@pytest.fixture(scope="session")
def compiled():
    import tmol.score.hbond.potentials.compiled

    return tmol.score.hbond.potentials.compiled


def _t(v):
    t = torch.tensor(v)
    if t.dtype == torch.float64:
        t = t.to(torch.float32)
    return t


def _setup_inputs(coords, params, donors, acceptors):
    """fetch coordinates as tensors"""
    return dict(
        D=_t(coords[donors["d"]]),
        H=_t(coords[donors["h"]]),
        donor_type=_t(params.donor_type_index.get_indexer(donors["donor_type"])).to(
            torch.int32
        ),
        A=_t(coords[acceptors["a"]]),
        B=_t(coords[acceptors["b"]]),
        B0=_t(coords[acceptors["b0"]]),
        acceptor_type=_t(
            params.acceptor_type_index.get_indexer(acceptors["acceptor_type"])
        ).to(torch.int32),
    )


def test_score_op(compiled, ubq_system):
    """Scores generated via compiled hbond dispatch match values from explicit
    evaluation."""

    from tmol.score.hbond.torch_op import HBondOp

    system = ubq_system
    hbond_database = ParameterDatabase.get_default().scoring.hbond
    hbond_param_resolver = HBondParamResolver.from_database(hbond_database)

    # Load coordinates and types from standard test system

    atom_types = system.atom_metadata["atom_type"].copy()
    bonds = system.bonds.copy()
    coords = system.coords.copy()

    # Run donor/acceptor identification over system

    hbond_elements = HBondElementAnalysis.setup(
        hbond_database=hbond_database, atom_types=atom_types, bonds=bonds
    )

    donors = hbond_elements.donors
    acceptors = hbond_elements.acceptors

    # setup input coordinate tensors

    inputs = _setup_inputs(coords, hbond_param_resolver, donors, acceptors)

    ### score via op

    op = HBondOp.from_database(hbond_database, hbond_param_resolver)
    inds, op_scores = op.score(**inputs)
    assert (op_scores < 0).sum() > 10

    ### Score via direct invoke of dispatch loop
    def _t_params(param_set):
        """fetch type pair parameters as flattened args"""
        return {"_".join(k): _t(v) for k, v in flat_items(attr.asdict(param_set))}

    # Load type pair parameter arrays and covert to tensor types
    # Run score dispatch over acceptors/donors
    pparams = _t_params(hbond_param_resolver.pair_params)
    gparams = attr.asdict(hbond_database.global_parameters)

    inds, dispatch_scores, *derivs = ignore_unused_kwargs(compiled.hbond_pair_score)(
        **merge(inputs, pparams, gparams)
    )

    # assert that some healthy number of bonds were scored
    assert (dispatch_scores < 0).sum() > 10

    # Verify scores via back comparison to explicit evaluation
    dind, aind = inds.transpose(0, 1)

    batch_coords = _setup_inputs(
        coords, hbond_param_resolver, donors[dind], acceptors[aind]
    )
    batch_params = _t_params(
        hbond_param_resolver.pair_params[
            batch_coords["donor_type"].to(torch.int64),
            batch_coords["acceptor_type"].to(torch.int64),
        ]
    )

    batch_scores, *batch_derivs = ignore_unused_kwargs(compiled.hbond_score_V_dV)(
        **merge(batch_coords, batch_params, gparams)
    )

    torch.testing.assert_allclose(batch_scores, batch_scores)
    for d, bd in zip(derivs, batch_derivs):
        torch.testing.assert_allclose(d, bd)


def test_score_op_gradcheck(compiled, ubq_system):
    """Scores generated via compiled hbond dispatch match values from explicit
    evaluation."""

    from tmol.score.hbond.torch_op import HBondOp

    system = ubq_system
    hbond_database = ParameterDatabase.get_default().scoring.hbond
    hbond_param_resolver = HBondParamResolver.from_database(hbond_database)

    # Load coordinates and types from standard test system

    atom_types = system.atom_metadata["atom_type"].copy()
    bonds = system.bonds.copy()
    coords = system.coords.copy()

    # Run donor/acceptor identification over system

    hbond_elements = HBondElementAnalysis.setup(
        hbond_database=hbond_database, atom_types=atom_types, bonds=bonds
    )

    donors = hbond_elements.donors[:10]
    acceptors = hbond_elements.acceptors[:10]

    op = HBondOp.from_database(
        hbond_database, hbond_param_resolver, dtype=torch.float64
    )

    # setup input coordinate tensors
    inputs = {
        n: t.to(torch.float64).requires_grad_(True) if t.is_floating_point() else t
        for n, t in _setup_inputs(
            coords, hbond_param_resolver, donors, acceptors
        ).items()
    }
    input_args = bind_to_args(op.score, **inputs)

    gradcheck(
        lambda *i: op.score(*i)[1].sum(), input_args, eps=1e-2, rtol=2.5e-3, nfail=1
    )
