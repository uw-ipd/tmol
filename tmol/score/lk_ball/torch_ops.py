import attr

import torch

from tmol.score.chemical_database import AtomTypeParamResolver
from tmol.score.ljlk.params import LJLKParamResolver, LJLKGlobalParams


@attr.s(auto_attribs=True, frozen=True, slots=True)
class LKBall:
    atom_resolver: AtomTypeParamResolver
    param_resolver: LJLKParamResolver

    @classmethod
    def from_database(cls, database, torch_device):
        atom_resolver = AtomTypeParamResolver.from_database(
            database.chemical, torch_device
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

        from .potentials import compiled

        i_heavy = ~ctx.op.atom_resolver.params.is_hydrogen[atom_types_i]
        j_heavy = ~ctx.op.atom_resolver.params.is_hydrogen[atom_types_j]

        i_idx = torch.nonzero(i_heavy)[:, 0]
        j_idx = torch.nonzero(j_heavy)[:, 0]

        inputs = (
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

        ind, v = compiled.lk_ball_V(*inputs)

        ctx.inputs = inputs
        ctx.i_idx = i_idx
        ctx.j_idx = j_idx

        ctx.shape_I = atom_types_i.shape
        ctx.shape_J = atom_types_j.shape

        return torch.stack((i_idx[ind[:, 0]], j_idx[ind[:, 1]])), v

    def backward(ctx, _, dT_dV):
        from .potentials import compiled

        i_idx = ctx.i_idx
        j_idx = ctx.j_idx

        ind, dV_dCI, dV_dCJ, dV_dWI, dV_dWJ = compiled.lk_ball_dV(*ctx.inputs)

        ind_I = i_idx[ind[:, 0]].detach()
        ind_J = j_idx[ind[:, 1]].detach()

        dT_dI = (
            torch.sparse_coo_tensor(
                # Insert leading dimension: [1, n]
                ind_I[None, :],
                # [n, 4] -> [n, 4, 1] * [n, 4, 3]
                dT_dV[..., None] * dV_dCI,
                ctx.shape_J + (4, 3),
            )
            .to_dense()  # [i, 4, 3]
            .sum(dim=1)  # [i, 3]
        )
        dT_dJ = (
            torch.sparse_coo_tensor(
                # Jnsert leading dimension: [1, n]
                ind_J[None, :],
                # [n, 4] -> [n, 4, 1] * [n, 4, 3]
                dT_dV[..., None] * dV_dCJ,
                ctx.shape_J + (4, 3),
            )
            .to_dense()  # [i, 4, 3]
            .sum(dim=1)  # [i,    3]
        )
        dT_dWI = (
            torch.sparse_coo_tensor(
                # Insert leading dimension: [1, n]
                ind_I[None, :],
                # [n, 4] -> [n, 4, 1, 1] * [n, 4, 4, 3]
                dT_dV[..., None, None] * dV_dWI,
                ctx.shape_I + (4, 4, 3),
            )
            .to_dense()  # [i, 4, 4, 3]
            .sum(dim=1)  # [i,    4, 3]
        )

        dT_dWJ = (
            torch.sparse_coo_tensor(
                # Insert leading dimension: [1, n]
                ind_J[None, :],
                # [n, 4] -> [n, 4, 1, 1] * [n, 4, 4, 3]
                dT_dV[..., None, None] * dV_dWJ,
                ctx.shape_J + (4, 4, 3),
            )
            .to_dense()  # [i, 4, 4, 3]
            .sum(dim=1)  # [i,    4, 3]
        )

        return (dT_dI, dT_dJ, dT_dWI, dT_dWJ, None, None, None)


@attr.s(auto_attribs=True, frozen=True, slots=True)
class AttachedWaters:

    atom_resolver: AtomTypeParamResolver
    global_params: LJLKGlobalParams

    @classmethod
    def from_database(cls, database, torch_device):
        return cls(
            AtomTypeParamResolver.from_database(database.chemical, torch_device),
            global_params=LJLKParamResolver.from_database(
                database.chemical, database.scoring.ljlk, torch_device
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
        from .potentials import compiled

        water_locations = compiled.attached_waters_forward(
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
        from .potentials import compiled

        water_locations = compiled.attached_waters_backward(dE_dW, *ctx.inputs)

        return water_locations
