import pytest


@pytest.fixture()
def dun_sampler(default_database, torch_device):
    from tmol.score.dunbrack.params import DunbrackParamResolver
    from tmol.pack.rotamer.dunbrack.dunbrack_chi_sampler import DunbrackChiSampler

    param_resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    return DunbrackChiSampler.from_database(param_resolver)
