import torch

from tmol.score.hbond.identification import HBondElementAnalysis
from tmol.score.hbond.params import HBondParamResolver, CompactedHBondDatabase


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


def test_script_module_scores(default_database, ubq_system, torch_device):
    """Scores computed via torch module match values from explicit
    evaluation."""

    from tmol.score.hbond.script_modules import HBondIntraModule

    system = ubq_system
    hbond_param_resolver = HBondParamResolver.from_database(
        default_database.chemical, default_database.scoring.hbond, torch_device
    )

    compact_db = CompactedHBondDatabase.from_database(
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
    intra_module = HBondIntraModule(compact_db)
    module_score = intra_module.forward(**inputs)

    # Verify scores via back comparison to explicit evaluation
    dind, aind = map(
        torch.flatten,
        torch.meshgrid((torch.arange(len(donors)), torch.arange(len(acceptors)))),
    )
    cpu_hbond_param_resolver = HBondParamResolver.from_database(
        default_database.chemical, default_database.scoring.hbond, torch.device("cpu")
    )
    cpu_compact_db = CompactedHBondDatabase.from_database(
        default_database.chemical, default_database.scoring.hbond, torch.device("cpu")
    )
    batch_coords = _setup_inputs(
        coords, cpu_hbond_param_resolver, donors, acceptors, torch.device("cpu")
    )

    batch_score = torch.ops.tmol.score_hbond(
        batch_coords["donor_coords"],
        batch_coords["acceptor_coords"],
        batch_coords["D"],
        batch_coords["H"],
        batch_coords["donor_type"],
        batch_coords["A"],
        batch_coords["B"],
        batch_coords["B0"],
        batch_coords["acceptor_type"],
        cpu_compact_db.pair_param_table,
        cpu_compact_db.pair_poly_table,
        cpu_compact_db.global_param_table,
    )

    torch.testing.assert_allclose(module_score.cpu(), batch_score)
    # Derivative values validated via gradcheck
