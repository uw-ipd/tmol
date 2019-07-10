import torch

from .params import CompactedHBondDatabase

import tmol.score.hbond.potentials.compiled  # noqa
from tmol.utility.cuda.synchronize import synchronize_if_cuda_available


class _HBondScoreModule(torch.jit.ScriptModule):
    """torch.autograd hbond baseline operator."""

    def __init__(self, compact_db: CompactedHBondDatabase):
        super().__init__()
        self.pair_param_table = compact_db.pair_param_table
        self.pair_poly_table = compact_db.pair_poly_table
        self.global_param_table = compact_db.global_param_table


class HBondIntraModule(_HBondScoreModule):
    def __init__(self, compact_db: CompactedHBondDatabase):
        super().__init__(compact_db)

    @torch.jit.script_method
    def forward(
        self, donor_coords, acceptor_coords, D, H, donor_type, A, B, B0, acceptor_type
    ):
        return torch.ops.tmol.score_hbond(
            donor_coords,
            acceptor_coords,
            D,
            H,
            donor_type,
            A,
            B,
            B0,
            acceptor_type,
            self.pair_param_table,
            self.pair_poly_table,
            self.global_param_table,
        )

    def final(
        self, donor_coords, acceptor_coords, D, H, donor_type, A, B, B0, acceptor_type
    ):
        res = self(
            donor_coords, acceptor_coords, D, H, donor_type, A, B, B0, acceptor_type
        )
        synchronize_if_cuda_available()
        return res
