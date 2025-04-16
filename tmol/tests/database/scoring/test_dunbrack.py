import pytest
import os
import torch


@pytest.mark.benchmark(group="dun_load", min_rounds=1)
def test_load_dunbrack_from_binary(benchmark):
    dirname = os.path.dirname(__file__)

    @benchmark
    def db():
        return torch.load(
            os.path.join(dirname, "../../../database/default/scoring/dunbrack.bin")
        )

    assert db is not None
