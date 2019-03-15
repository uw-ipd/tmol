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

    @f.default
    def _load_f(self):
        from .potentials import compiled

        return compiled.dunbrack_energy

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
        rotchi_devpen = torch.zeros((nrotchi,), dtype=torch.float, device=ctx.op.device)
        ddevpen_dbb = torch.zeros((nrotchi, 2), dtype=torch.float, device=ctx.op.device)
        rottable_assignment = torch.zeros(
            (nres,), dtype=torch.int32, device=ctx.op.device
        )

        for key, val in ctx.op.params.items():
            print("key in ctx.op.params:", key, print(type(val)))

        # dE_dphi/psi are returned as ntors x 12 arrays
        E = ctx.op.f(
            coords,
            dihedrals=dihedrals,
            ddihe_dxyz=ddihe_dxyz,
            dihedral_dE_ddihe=dihedral_dE_ddihe,
            rotchi_devpen=rotchi_devpen,
            ddevpen_dbb=ddevpen_dbb,
            rottable_assignment=rottable_assignment,
            **ctx.op.params,
        )

        # ctx.save_for_backward(ddihe_dxyz, dihedral_dE_ddihe)

        return E

    def backward(ctx, dV_dE):
        # ddihe_dxyz, dihedral_dE_ddihe = ctx.saved_tensors

        return (None, None, None, None)
