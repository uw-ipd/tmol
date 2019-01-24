import attr

from typing import Mapping, Union, Callable

import torch

from tmol.utility.dicttoolz import flat_items, merge

from tmol.database.scoring import HBondDatabase
from .params import HBondParamResolver


@attr.s(auto_attribs=True, frozen=True)
class HBondOp:
    """torch.autograd hbond baseline operator."""

    params: Mapping[str, Union[float, torch.Tensor]]
    hbond_pair_score: Callable = attr.ib()

    @hbond_pair_score.default
    def _load_hbond_pair_score(self):
        from .potentials.compiled import hbond_pair_score

        return hbond_pair_score

    @staticmethod
    def _setup_pair_params(param_resolver, dtype):
        def _t(n, v):
            t = torch.tensor(v)
            if t.is_floating_point():
                if any(dkey in n for dkey in ("range", "bound", "coeffs")):
                    # High degree polynomial parameters stored as double precision
                    # to allow accurate double evaluation.
                    t = t.to(torch.float64)
                else:
                    t = t.to(dtype)
            return t

        return {
            "_".join(k): _t(k, v)
            for k, v in flat_items(attr.asdict(param_resolver.pair_params))
        }

    @classmethod
    def from_database(
        cls,
        database: HBondDatabase,
        param_resolver: HBondParamResolver,
        dtype=torch.float32,
    ):
        pair_params = cls._setup_pair_params(param_resolver, dtype)
        global_params = attr.asdict(database.global_parameters)

        return cls(params=merge(pair_params, global_params))

    def score(self, D, H, donor_type, A, B, B0, acceptor_type):
        inds, scores = HBondFun(self)(D, H, donor_type, A, B, B0, acceptor_type)
        return inds.detach(), scores


class HBondFun(torch.autograd.Function):
    def __init__(self, op):
        self.op = op
        super().__init__()

    def forward(ctx, D, H, donor_type, A, B, B0, acceptor_type):
        assert D.dim() == 2
        assert D.shape[1] == 3
        assert D.shape == H.shape
        assert D.shape[:1] == donor_type.shape
        assert not donor_type.requires_grad

        assert A.dim() == 2
        assert A.shape[1] == 3
        assert A.shape == B.shape
        assert A.shape == B0.shape
        assert A.shape[:1] == acceptor_type.shape
        assert not acceptor_type.requires_grad

        assert all(
            t.device.type == "cpu" for t in (D, H, donor_type, A, B, B0, acceptor_type)
        )

        inds, E, *dE_dC = ctx.op.hbond_pair_score(
            D, H, donor_type, A, B, B0, acceptor_type, **ctx.op.params
        )

        inds = inds.transpose(0, 1)

        ctx.donor_shape = donor_type.shape
        ctx.acceptor_shape = acceptor_type.shape

        ctx.save_for_backward(*([inds] + dE_dC))

        return (inds, E)

    def backward(ctx, _ind_grads, dV_dE):
        ind, dE_dD, dE_dH, dE_dA, dE_dB, dE_dB0 = ctx.saved_tensors
        donor_ind, acceptor_ind = ind

        def _chain_donor(dE_dDonor):
            return torch.sparse_coo_tensor(
                donor_ind[None, :], dV_dE[..., None] * dE_dDonor, ctx.donor_shape + (3,)
            ).to_dense()

        def _chain_acceptor(dE_dAcceptor):
            return torch.sparse_coo_tensor(
                acceptor_ind[None, :],
                dV_dE[..., None] * dE_dAcceptor,
                ctx.acceptor_shape + (3,),
            ).to_dense()

        return (
            _chain_donor(dE_dD),
            _chain_donor(dE_dH),
            None,
            _chain_acceptor(dE_dA),
            _chain_acceptor(dE_dB),
            _chain_acceptor(dE_dB0),
            None,
        )
