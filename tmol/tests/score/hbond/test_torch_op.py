import torch
import attr

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
        donor_coords=_t(coords),
        acceptor_coords=_t(coords),
        D=_t(donors["d"]),
        H=_t(donors["h"]),
        donor_type=_t(params.donor_type_index.get_indexer(donors["donor_type"])).to(
            torch.int32
        ),
        A=_t(acceptors["a"]),
        B=_t(acceptors["b"]),
        B0=_t(acceptors["b0"]),
        acceptor_type=_t(
            params.acceptor_type_index.get_indexer(acceptors["acceptor_type"])
        ).to(torch.int32),
    )


def test_score_op(default_database, ubq_system, torch_device):
    """Scores computed via op-based dispatch match values from explicit
    evaluation."""

    from tmol.score.hbond.torch_op import HBondOp
    import tmol.tests.score.hbond.potentials.compiled as compiled

    system = ubq_system
    hbond_param_resolver = HBondParamResolver.from_database(
        default_database.chemical, default_database.scoring.hbond, torch_device
    )

    # Load coordinates and types from standard test system

    atom_types = system.atom_metadata["atom_type"].copy()
    bonds = system.bonds.copy()
    coords = system.coords.copy()

    # Run donor/acceptor identification over system

    hbond_elements = HBondElementAnalysis.setup_from_database(
        chemical_database=default_database.chemical,
        hbond_database=default_database.scoring.hbond,
        atom_types=atom_types,
        bonds=bonds,
    )

    donors = hbond_elements.donors
    acceptors = hbond_elements.acceptors

    # setup input coordinate tensors

    inputs = _setup_inputs(
        coords, hbond_param_resolver, donors, acceptors, torch_device
    )

    ### score via op

    op = HBondOp.from_database(default_database.scoring.hbond, hbond_param_resolver)
    op_score = op.score(**inputs)

    # Verify scores via back comparison to explicit evaluation
    dind, aind = map(
        torch.flatten,
        torch.meshgrid((torch.arange(len(donors)), torch.arange(len(acceptors)))),
    )
    hbond_param_resolver = HBondParamResolver.from_database(
        default_database.chemical, default_database.scoring.hbond, torch.device("cpu")
    )
    batch_coords = _setup_inputs(
        coords, hbond_param_resolver, donors[dind], acceptors[aind], torch.device("cpu")
    )
    donor_coords = batch_coords["donor_coords"]
    D = batch_coords["D"]
    D = donor_coords[D]

    H = batch_coords["H"]
    H = donor_coords[H]

    acceptor_coords = batch_coords["acceptor_coords"]
    A = batch_coords["A"]
    A = acceptor_coords[A]

    B = batch_coords["B"]
    B = acceptor_coords[B]

    B0 = batch_coords["B0"]
    B0 = acceptor_coords[B0]

    gparams = attr.asdict(default_database.scoring.hbond.global_parameters)
    batch_params = valmap(
        lambda t: t[
            batch_coords["donor_type"].to(torch.int64),
            batch_coords["acceptor_type"].to(torch.int64),
        ],
        HBondOp._setup_pair_params(hbond_param_resolver, torch.float32),
    )

    batch_scores, *batch_derivs = ignore_unused_kwargs(compiled.hbond_score_V_dV)(
        **merge(dict(D=D, H=H, A=A, B=B, B0=B0), batch_params, gparams)
    )

    torch.testing.assert_allclose(
        op_score,
        torch.from_numpy(batch_scores).to(device=torch_device, dtype=torch.float).sum(),
    )
    # Derivative values validated via gradcheck


def test_score_op_gradcheck(default_database, ubq_system, torch_device):

    from tmol.score.hbond.torch_op import HBondOp

    system = ubq_system
    hbond_param_resolver = HBondParamResolver.from_database(
        default_database.chemical, default_database.scoring.hbond, torch_device
    )

    # Load coordinates and types from standard test system

    atom_types = system.atom_metadata["atom_type"].copy()
    bonds = system.bonds.copy()
    coords = system.coords.copy()

    # Run donor/acceptor identification over system

    hbond_elements = HBondElementAnalysis.setup_from_database(
        chemical_database=default_database.chemical,
        hbond_database=default_database.scoring.hbond,
        atom_types=atom_types,
        bonds=bonds,
    )

    donors = hbond_elements.donors[:10]
    acceptors = hbond_elements.acceptors[:10]

    op = HBondOp.from_database(
        default_database.scoring.hbond, hbond_param_resolver, dtype=torch.float64
    )

    # setup input coordinate tensors
    inputs = {
        n: t.to(torch.float64).requires_grad_(True) if t.is_floating_point() else t
        for n, t in _setup_inputs(
            coords, hbond_param_resolver, donors, acceptors, torch_device
        ).items()
    }
    input_args = bind_to_args(op.score, **inputs)

    gradcheck(lambda *i: op.score(*i), input_args, eps=1e-3, rtol=2.5e-3, nfail=1)
