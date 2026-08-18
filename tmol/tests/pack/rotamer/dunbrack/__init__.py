import pytest


@pytest.fixture()
def dun_sampler(default_database, torch_device):
    from tmol.pack.rotamer.dunbrack import (
        create_dunbrack_sampler_from_database,
    )

    return create_dunbrack_sampler_from_database(default_database, torch_device)
