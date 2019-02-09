import torch
from tmol.database.chemical import ChemicalDatabase
from tmol.score.chemical_database import AtomTypeParamResolver, ChemicalDB


def test_database_parameter_resolution(default_database, torch_device):
    """Chemical database parameters are packed into indexed torch tensors.
    """
    resolver: AtomTypeParamResolver = AtomTypeParamResolver.from_database(
        chemical_database=default_database.chemical, device=torch_device
    )

    validate_param_resolver(default_database, resolver, torch_device)


def test_score_graph(default_database, torch_device):
    """Chemical database is loaded from default db via score graph."""

    graph: ChemicalDB = ChemicalDB.build_for(None, device=torch_device)

    validate_param_resolver(default_database, graph.atom_type_params, torch_device)


def validate_param_resolver(
    database: ChemicalDatabase,
    resolver: AtomTypeParamResolver,
    torch_device: torch.device,
):
    """Assert over valid AtomTypeParamResolver
    Verify that atom type parameters from database layer are packed into tensor
    data on target device, with proper mapping from boolean/symbolic data types
    into tensor primitive datatypes.
    """

    atom_types = {t.name: t for t in database.chemical.atom_types}

    assert len(resolver.index) == len(atom_types) + 1

    for a in atom_types:
        aidx, = resolver.index.get_indexer_for([a])

        assert resolver.params.is_acceptor[aidx] == atom_types[a].is_acceptor
        assert (
            resolver.params.acceptor_hybridization[aidx]
            == {None: 0, "sp2": 1, "sp3": 2, "ring": 3}[
                atom_types[a].acceptor_hybridization
            ]
        )

        assert resolver.params.is_donor[aidx] == atom_types[a].is_donor

        assert resolver.params.is_hydrogen[aidx] == (atom_types[a].element == "H")
        assert resolver.params.is_hydroxyl[aidx] == atom_types[a].is_hydroxyl
        assert resolver.params.is_polarh[aidx] == atom_types[a].is_polarh

    assert resolver.params.is_acceptor.device == torch_device
    assert resolver.params.acceptor_hybridization.device == torch_device
    assert resolver.params.is_donor.device == torch_device
    assert resolver.params.is_hydroxyl.device == torch_device
    assert resolver.params.is_polarh.device == torch_device

    assert resolver.type_idx(list(atom_types)).device == torch_device
