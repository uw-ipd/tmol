import pytest


@pytest.fixture()
def dun_sampler(default_database, torch_device):
    from tmol.pack.rotamer.dunbrack.dunbrack_chi_sampler import DunbrackChiSampler
    from tmol.score.dunbrack.params import DunbrackParamResolver

    param_resolver = DunbrackParamResolver.from_database(default_database.scoring.dun, torch_device)
    return DunbrackChiSampler.from_database(param_resolver)
