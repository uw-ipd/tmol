import attr
from attr import asdict
from typing import Mapping, Callable

from .params import PackedDunbrackDatabase, DunbrackParams, DunbrackScratch

import torch


@attr.s(auto_attribs=True, frozen=True)
class DunbrackOp:
    device: torch.device
    params: Mapping[str, torch.Tensor]
    packed_db: PackedDunbrackDatabase
    dun_params: DunbrackParams
    scratch: DunbrackScratch

    f: Callable = attr.ib()
    df: Callable = attr.ib()

    @f.default
    def _load_f(self):
        from .potentials import compiled

        return compiled.dunbrack_energy

    @df.default
    def _load_df(self):
        from .potentials import compiled

        return compiled.dunbrack_deriv

    @classmethod
    def from_params(
        cls,
        packed_db: PackedDunbrackDatabase,
        dun_params: DunbrackParams,
        dun_scratch: DunbrackScratch,
    ):
        res = cls(
            device=packed_db.rotameric_bb_start.device,
            params={**asdict(packed_db), **asdict(dun_params), **asdict(dun_scratch)},
            packed_db=packed_db,
            dun_params=dun_params,
            scratch=dun_scratch,
        )
        assert all(
            res.device == t.device
            for t in res.params.values()
            if not isinstance(t, list)
        )
        assert all(
            all(res.device == t.device for t in l)
            for l in res.params.values()
            if isinstance(l, list)
        )
        return res

    def intra(self, coords):
        return DunbrackScoreFun(self)(coords)


class DunbrackScoreFun(torch.autograd.Function):
    def __init__(self, op):
        self.op = op
        super().__init__()

    def forward(ctx, coords):

        # dE_dphi/psi are returned as ntors x 12 arrays
        rot_nlpE, drot_nlp_dbb, devpen, ddevpen_dtor, nonrot_nlpE, dnonrot_nlpE_dtor = ctx.op.f(
            coords, **ctx.op.params
        )

        ctx.save_for_backward(coords, drot_nlp_dbb, ddevpen_dtor, dnonrot_nlpE_dtor)

        return rot_nlpE, devpen, nonrot_nlpE

    def backward(ctx, dE_drotnlp, dE_ddevpen, dE_dnonrotnlp):
        coords, drot_nlp_dbb, ddevpen_dtor, dnonrot_nlpE_dtor = ctx.saved_tensors

        dE_drotnlp = dE_drotnlp.contiguous()
        dE_ddevpen = dE_ddevpen.contiguous()
        dE_dnonrotnlp = dE_dnonrotnlp.contiguous()

        # print("dE_drotnlp", dE_drotnlp.shape, dE_drotnlp.dtype)

        # dE_dxyz = torch.zeros(ctx.coords_shape, dtype=torch.float, device=drot_nlp_dphi.device)
        dE_dxyz = ctx.op.df(
            coords,
            dE_drotnlp=dE_drotnlp,
            drot_nlp_dbb_xyz=drot_nlp_dbb,
            dE_ddevpen=dE_ddevpen,
            ddevpen_dtor_xyz=ddevpen_dtor,
            dE_dnonrotnlp=dE_dnonrotnlp,
            dnonrot_nlp_dtor_xyz=dnonrot_nlpE_dtor,
            dihedral_offset_for_res=ctx.op.dun_params.dihedral_offset_for_res,
            dihedral_atom_inds=ctx.op.dun_params.dihedral_atom_inds,
            rotres2resid=ctx.op.dun_params.rotres2resid,
            rotameric_chi_desc=ctx.op.dun_params.rotameric_chi_desc,
            semirotameric_chi_desc=ctx.op.dun_params.semirotameric_chi_desc,
        )
        return dE_dxyz

        # return dE_dxyz
        # return (dE_dxyz, None, None, None)
