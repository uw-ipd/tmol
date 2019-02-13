import attr

import torch

from tmol.score.chemical_database import AtomTypeParamResolver
from tmol.score.ljlk.params import LJLKParamResolver, LJLKGlobalParams

from tmol.utility.cpp_extension import load, modulename, relpaths

_compiled = load(modulename(__name__), relpaths(__file__, "compiled.cc"))


@attr.s(auto_attribs=True, frozen=True, slots=True)
class LKBall:
    atom_resolver: AtomTypeParamResolver
    param_resolver: LJLKParamResolver

    @classmethod
    def from_database(cls, database, torch_device):
        atom_resolver = AtomTypeParamResolver.from_database(
            database.chemical, torch.device("cpu")
        )
        param_resolver = LJLKParamResolver.from_param_resolver(
            atom_resolver, database.scoring.ljlk
        )

        return cls(atom_resolver, param_resolver)

    @property
    def apply(self,):
        return LKBallFun(self)


class LKBallFun(torch.autograd.Function):
    def __init__(self, op):
        self.op = op

    def forward(
        ctx,
        coords_i,
        coords_j,
        waters_i,
        waters_j,
        atom_types_i,
        atom_types_j,
        bonded_path_lengths,
    ):
        i_heavy = ~ctx.op.atom_resolver.params.is_hydrogen[atom_types_i]
        j_heavy = ~ctx.op.atom_resolver.params.is_hydrogen[atom_types_j]

        i_idx = torch.nonzero(i_heavy)[:, 0]
        j_idx = torch.nonzero(j_heavy)[:, 0]

        ind, v = _compiled.lk_ball(
            coords_i[i_idx],
            coords_j[j_idx],
            waters_i[i_idx],
            waters_j[j_idx],
            atom_types_i[i_idx],
            atom_types_j[j_idx],
            bonded_path_lengths[i_idx, :][:, j_idx],
            ctx.op.param_resolver.type_params,
            ctx.op.param_resolver.global_params,
            ctx.op.param_resolver.global_params,
        )

        return torch.stack((i_idx[ind[:, 0]], j_idx[ind[:, 1]]), dim=1), v


@attr.s(auto_attribs=True, frozen=True, slots=True)
class AttachedWaters:

    atom_resolver: AtomTypeParamResolver
    global_params: LJLKGlobalParams

    @classmethod
    def from_database(cls, database, torch_device):
        return cls(
            AtomTypeParamResolver.from_database(database.chemical, torch.device("cpu")),
            global_params=LJLKParamResolver.from_database(
                database.chemical, database.scoring.ljlk, torch.device("cpu")
            ).global_params,
        )

    def apply(self, coords, atom_types, indexed_bonds):
        atom_type_params = self.atom_resolver.params[atom_types]

        return AttachedWatersFun(self, indexed_bonds, atom_type_params)(coords)


class AttachedWatersFun(torch.autograd.Function):
    def __init__(self, op, indexed_bonds, atom_type_params):
        self.op = op
        self.atom_type_params = atom_type_params
        self.indexed_bonds = indexed_bonds

        super().__init__()

    def forward(ctx, coords):

        water_locations = _compiled.attached_waters_forward(
            coords, ctx.indexed_bonds, ctx.atom_type_params, ctx.op.global_params
        )

        ctx.inputs = (
            coords,
            ctx.indexed_bonds,
            ctx.atom_type_params,
            ctx.op.global_params,
        )

        return water_locations

    def backward(ctx, dE_dW):
        water_locations = _compiled.attached_waters_backward(dE_dW, *ctx.inputs)

        return water_locations
