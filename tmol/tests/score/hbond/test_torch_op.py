import torch
import attr

from tmol.database import ParameterDatabase

from tmol.score.hbond.identification import HBondElementAnalysis
from tmol.score.hbond.params import HBondParamResolver

from tmol.utility.dicttoolz import merge, valmap

from tmol.utility.args import ignore_unused_kwargs, bind_to_args

from tmol.tests.autograd import gradcheck


def _setup_inputs(coords, params, donors, acceptors, torch_device):
    """fetch coordinates as tensors"""

    def _t(v):
        t = torch.tensor(v).to(device=torch_device)
        if t.dtype == torch.float64:
            t = t.to(torch.float32)
        return t

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


def test_score_op(ubq_system, torch_device):
    """Scores computed via op-based dispatch match values from explicit
    evaluation."""

    from tmol.score.hbond.torch_op import HBondOp
    import tmol.tests.score.hbond.potentials.compiled as compiled

    system = ubq_system
    hbond_database = ParameterDatabase.get_default().scoring.hbond
    hbond_param_resolver = HBondParamResolver.from_database(
        hbond_database, torch_device
    )

    # Load coordinates and types from standard test system

    atom_types = system.atom_metadata["atom_type"].copy()
    atom_elements = system.atom_metadata["atom_element"].copy()
    bonds = system.bonds.copy()
    coords = system.coords.copy()

    # Run donor/acceptor identification over system

    hbond_elements = HBondElementAnalysis.setup(
        hbond_database=hbond_database,
        atom_types=atom_types,
        atom_is_hydrogen=atom_elements == "H",
        bonds=bonds,
    )

    donors = hbond_elements.donors
    acceptors = hbond_elements.acceptors

    # setup input coordinate tensors

    inputs = _setup_inputs(
        coords, hbond_param_resolver, donors, acceptors, torch_device
    )

    ### score via op

    op = HBondOp.from_database(hbond_database, hbond_param_resolver)
    (dind, aind), op_scores = op.score(**inputs)
    assert (op_scores < 0).sum() > 10

    # Verify scores via back comparison to explicit evaluation
    hbond_param_resolver = HBondParamResolver.from_database(
        hbond_database, torch.device("cpu")
    )
    batch_coords = _setup_inputs(
        coords, hbond_param_resolver, donors[dind], acceptors[aind], torch.device("cpu")
    )

    gparams = attr.asdict(hbond_database.global_parameters)
    batch_params = valmap(
        lambda t: t[
            batch_coords["donor_type"].to(torch.int64),
            batch_coords["acceptor_type"].to(torch.int64),
        ],
        HBondOp._setup_pair_params(hbond_param_resolver, torch.float32),
    )

    batch_scores, *batch_derivs = ignore_unused_kwargs(compiled.hbond_score_V_dV)(
        **merge(batch_coords, batch_params, gparams)
    )

    torch.testing.assert_allclose(
        op_scores,
        torch.from_numpy(batch_scores).to(device=torch_device, dtype=torch.float),
    )
    # Derivative values validated via gradcheck


def test_score_op_gradcheck(ubq_system, torch_device):

    from tmol.score.hbond.torch_op import HBondOp

    system = ubq_system
    hbond_database = ParameterDatabase.get_default().scoring.hbond
    hbond_param_resolver = HBondParamResolver.from_database(
        hbond_database, torch_device
    )

    # Load coordinates and types from standard test system

    atom_types = system.atom_metadata["atom_type"].copy()
    atom_elements = system.atom_metadata["atom_element"].copy()
    bonds = system.bonds.copy()
    coords = system.coords.copy()

    # Run donor/acceptor identification over system

    hbond_elements = HBondElementAnalysis.setup(
        hbond_database=hbond_database,
        atom_types=atom_types,
        atom_is_hydrogen=atom_elements == "H",
        bonds=bonds,
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
            coords, hbond_param_resolver, donors, acceptors, torch_device
        ).items()
    }
    input_args = bind_to_args(op.score, **inputs)

    gradcheck(
        lambda *i: op.score(*i)[1].sum(), input_args, eps=1e-2, rtol=2.5e-3, nfail=1
    )
