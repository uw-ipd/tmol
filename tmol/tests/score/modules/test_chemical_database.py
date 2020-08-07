from tmol.score.modules.chemical_database import ChemicalDB
from tmol.score.modules.bases import ScoreSystem

from tmol.tests.score.test_chemical_database import validate_param_resolver


def test_score_component(default_database, torch_device):
    """Chemical database is loaded from default db via score component."""
    system = ScoreSystem._build_with_modules(None, [ChemicalDB], device=torch_device)

    validate_param_resolver(
        default_database, ChemicalDB.get(system).atom_type_params, torch_device
    )
