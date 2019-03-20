import attr
from attr import asdict
from typing import Mapping, Callable

from .params import PackedDunbrackDatabase, DunbrackParams

import torch


@attr.s(auto_attribs=True, frozen=True)
class DunbrackOp:
    device: torch.device
    params: Mapping[str, torch.Tensor]
    packed_db: PackedDunbrackDatabase
    dun_params: DunbrackParams

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
    def from_params(cls, packed_db: PackedDunbrackDatabase, dun_params: DunbrackParams):
        res = cls(
            device=packed_db.rotameric_bb_start.device,
            params={**asdict(packed_db), **asdict(dun_params)},
            packed_db=packed_db,
            dun_params=dun_params,
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

        # allocate the temporary tensors to hold information needed
        ndihe = ctx.op.dun_params.dihedral_atom_inds.shape[0]
        nrotchi = ctx.op.dun_params.rotameric_chi_desc.shape[0]
        nres = ctx.op.dun_params.ndihe_for_res.shape[0]

        dihedrals = torch.zeros((ndihe,), dtype=torch.float, device=ctx.op.device)
        ddihe_dxyz = torch.zeros((ndihe, 4, 3), dtype=torch.float, device=ctx.op.device)
        dihedral_dE_ddihe = torch.zeros(
            (ndihe,), dtype=torch.float, device=ctx.op.device
        )
        #  rotchi_devpen = torch.zeros((nrotchi,), dtype=torch.float, device=ctx.op.device)
        #  ddevpen_dbb = torch.zeros((nrotchi, 2), dtype=torch.float, device=ctx.op.device)
        rotameric_rottable_assignment = torch.zeros(
            (nres,), dtype=torch.int32, device=ctx.op.device
        )
        semirotameric_rottable_assignment = torch.zeros(
            (nres,), dtype=torch.int32, device=ctx.op.device
        )

        # for key, val in ctx.op.params.items():
        #    print("key in ctx.op.params:", key, print(type(val)))

        # dE_dphi/psi are returned as ntors x 12 arrays
        rot_nlpE, drot_nlp_dphi, drot_nlp_dpsi, devpen, ddevpen_dphi, ddevpen_dpsi, ddevpen_dchi, nonrot_nlpE, dnonrot_nlpE_dphi, dnonrot_nlpE_dpsi, dnonrot_nlpE_dchi = ctx.op.f(
            coords,
            dihedrals=dihedrals,
            ddihe_dxyz=ddihe_dxyz,
            dihedral_dE_ddihe=dihedral_dE_ddihe,
            # rotchi_devpen=rotchi_devpen,
            # ddevpen_dbb=ddevpen_dbb,
            rotameric_rottable_assignment=rotameric_rottable_assignment,
            semirotameric_rottable_assignment=semirotameric_rottable_assignment,
            **ctx.op.params,
        )

        ctx.save_for_backward(
            coords,
            drot_nlp_dphi,
            drot_nlp_dpsi,
            ddevpen_dphi,
            ddevpen_dpsi,
            ddevpen_dchi,
            dnonrot_nlpE_dphi,
            dnonrot_nlpE_dpsi,
            dnonrot_nlpE_dchi,
        )

        return rot_nlpE, devpen, nonrot_nlpE

    def backward(ctx, dE_drotnlp, dE_ddevpen, dE_dnonrotnlp):
        coords, drot_nlp_dphi, drot_nlp_dpsi, ddevpen_dphi, ddevpen_dpsi, ddevpen_dchi, dnonrot_nlpE_dphi, dnonrot_nlpE_dpsi, dnonrot_nlpE_dchi = (
            ctx.saved_tensors
        )

        dE_drotnlp = dE_drotnlp.contiguous()
        dE_ddevpen = dE_ddevpen.contiguous()
        dE_dnonrotnlp = dE_dnonrotnlp.contiguous()

        # print("dE_drotnlp", dE_drotnlp.shape, dE_drotnlp.dtype)

        # dE_dxyz = torch.zeros(ctx.coords_shape, dtype=torch.float, device=drot_nlp_dphi.device)
        dE_dxyz = ctx.op.df(
            coords,
            dE_drotnlp=dE_drotnlp,
            drot_nlp_dphi_xyz=drot_nlp_dphi,
            drot_nlp_dpsi_xyz=drot_nlp_dpsi,
            dE_ddevpen=dE_ddevpen,
            ddevpen_dphi_xyz=ddevpen_dphi,
            ddevpen_dpsi_xyz=ddevpen_dpsi,
            ddevpen_dchi_xyz=ddevpen_dchi,
            dE_dnonrotnlp=dE_dnonrotnlp,
            dnonrot_nlp_dphi_xyz=dnonrot_nlpE_dphi,
            dnonrot_nlp_dpsi_xyz=dnonrot_nlpE_dpsi,
            dnonrot_nlp_dchi_xyz=dnonrot_nlpE_dchi,
            dihedral_offset_for_res=ctx.op.dun_params.dihedral_offset_for_res,
            dihedral_atom_inds=ctx.op.dun_params.dihedral_atom_inds,
            rotres2resid=ctx.op.dun_params.rotres2resid,
            rotameric_chi_desc=ctx.op.dun_params.rotameric_chi_desc,
            semirotameric_chi_desc=ctx.op.dun_params.semirotameric_chi_desc,
        )
        return dE_dxyz

        # return dE_dxyz
        # return (dE_dxyz, None, None, None)
