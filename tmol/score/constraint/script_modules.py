import torch
import numpy
from tmol.score.constraint.params import ConstraintResolver

from tmol.database.chemical import ChemicalDatabase
from tmol.database.scoring.ljlk import LJLKDatabase

# Import compiled components to load torch_ops
import tmol.score.ljlk.potentials.compiled  # noqa

# Workaround for https://github.com/pytorch/pytorch/pull/15340
# on torch<1.0.1
if "to" in torch.jit.ScriptModule.__dict__:
    delattr(torch.jit.ScriptModule, "to")


class ConstraintIntraModule(torch.jit.ScriptModule):
    def __init__(self, param_resolver: ConstraintResolver):
        super().__init__()

        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        def _tfloat(ts):
            return tuple(map(lambda t: t.to(torch.float), ts))

        self.cb_frames = _p(param_resolver.cb_frames)
        self.cb_stacks = _p(param_resolver.cb_stacks)
        self.cb_res_indices = _p(param_resolver.cb_res_indices)
        self.geometry_params = _p(
            torch.stack(
                _tfloat(
                    [
                        torch.tensor(1.521736),  # dist
                        torch.tensor(1.2151),  # angle
                        torch.tensor(-2.14326),  # torsion
                    ]
                ),
                dim=0,
            )
        )
        self.spline_xs = _p(param_resolver.spline_xs)
        self.spline_ys = _p(param_resolver.spline_ys)
        self.nres = param_resolver.nres

    # @torch.jit.script_method
    def axis_angle(self, Vs, angle):
        W = torch.sign(angle) * torch.cos(angle / 2)
        S = torch.sqrt((1 - W * W))
        V = S * Vs
        Rs = torch.zeros((*Vs.shape[:-1], 3, 3), device=Vs.device)
        Rs[..., 0, 0] = 1 - 2 * (V[..., 1] * V[..., 1] + V[..., 2] * V[..., 2])
        Rs[..., 0, 1] = 2 * (V[..., 0] * V[..., 1] - V[..., 2] * W)
        Rs[..., 0, 2] = 2 * (V[..., 0] * V[..., 2] + V[..., 1] * W)
        Rs[..., 1, 0] = 2 * (V[..., 0] * V[..., 1] + V[..., 2] * W)
        Rs[..., 1, 1] = 1 - 2 * (V[..., 0] * V[..., 0] + V[..., 2] * V[..., 2])
        Rs[..., 1, 2] = 2 * (V[..., 1] * V[..., 2] - V[..., 0] * W)
        Rs[..., 2, 0] = 2 * (V[..., 0] * V[..., 2] - V[..., 1] * W)
        Rs[..., 2, 1] = 2 * (V[..., 1] * V[..., 2] + V[..., 0] * W)
        Rs[..., 2, 2] = 1 - 2 * (V[..., 0] * V[..., 0] + V[..., 1] * V[..., 1])
        return Rs

    # @torch.jit.script_method
    def cbs_from_frames(self, Ns, Cs, CAs, geom):
        dist, angle, tor = geom[0], geom[1], geom[2]

        CANs = CAs - Ns
        NCs = Ns - Cs
        Xp = torch.cross(NCs, CANs, dim=-1)
        Xp = Xp / torch.sqrt(torch.sum(Xp * Xp, dim=-1))[..., None]

        # rotate angle
        Rs = self.axis_angle(Xp, angle)
        RXs = torch.matmul(Rs, CANs[..., None]).squeeze(dim=-1)

        # rotate torsion
        CANs = CANs / torch.sqrt(torch.sum(CANs * CANs, dim=-1))[..., None]
        Rs = self.axis_angle(CANs, tor)
        RXs = torch.matmul(Rs, RXs[..., None]).squeeze(dim=-1)
        return CAs + dist * RXs / torch.sqrt(torch.sum(RXs * RXs, dim=-1))[..., None]

    # interpolate a dense array of splines
    # x = spline interpolant xs (N)
    # ys = spline interpolant ys (nres x nres x N)
    # xs = parwise distance stacks (nstack x nres x nres)
    def interp(self, x, ys, xs):
        def h_poly(t):
            tt = torch.stack((torch.ones_like(t), t, t * t, t * t * t))
            A = torch.tensor(
                [[1, 0, -3, 2], [0, 1, -2, 1], [0, 0, 3, -2], [0, 0, -1, 1]],
                dtype=tt.dtype,
                device=tt.device,
            )
            return torch.matmul(A, tt.reshape(4, -1)).reshape(4, *t.shape)

        nspl = x.shape[0]
        nres = ys.shape[0]
        nstk = xs.shape[0]

        ms = (ys[..., 1:] - ys[..., :-1]) / (x[1:] - x[:-1])[None, None, :]
        ms = torch.cat(
            [ms[..., 0:1], (ms[..., 1:] + ms[..., :-1]) / 2, ms[..., -1:]], dim=-1
        )
        # torch 1.6
        # I = torch.searchsorted(x[1:], xs.detach()) # fd

        # quick and dirty pre-1.6
        I = torch.zeros(xs.shape, dtype=torch.long, device=xs.device)
        for i, xi in enumerate(x[1:].flip(0)):
            I[xs <= xi] = x.shape[0] - i - 2

        dx = x[I + 1] - x[I]
        hh = h_poly((xs - x[I]) / dx)

        I = I.reshape(nstk, -1).transpose(0, 1)
        ys = ys.reshape(-1, nspl)
        ms = ms.reshape(-1, nspl)

        yI = torch.gather(ys, 1, I).reshape((nres, nres, nstk)).permute([2, 0, 1])
        yIp1 = torch.gather(ys, 1, I + 1).reshape((nres, nres, nstk)).permute([2, 0, 1])
        mI = torch.gather(ms, 1, I).reshape((nres, nres, nstk)).permute([2, 0, 1])
        mIp1 = torch.gather(ms, 1, I + 1).reshape((nres, nres, nstk)).permute([2, 0, 1])

        return hh[0] * yI + hh[1] * mI * dx + hh[2] * yIp1 + hh[3] * mIp1 * dx

    # @torch.jit.script_method
    def forward(self, coords):
        # 1 - build 'ideal' CBs
        Ns = torch.full(
            (coords.shape[0], self.nres, 3), numpy.nan, device=coords.device
        )
        Cs = torch.full(
            (coords.shape[0], self.nres, 3), numpy.nan, device=coords.device
        )
        CAs = torch.full(
            (coords.shape[0], self.nres, 3), numpy.nan, device=coords.device
        )
        Ns[self.cb_stacks, self.cb_res_indices, :] = coords[
            self.cb_stacks, self.cb_frames[:, 0], :
        ]
        Cs[self.cb_stacks, self.cb_res_indices, :] = coords[
            self.cb_stacks, self.cb_frames[:, 1], :
        ]
        CAs[self.cb_stacks, self.cb_res_indices, :] = coords[
            self.cb_stacks, self.cb_frames[:, 2], :
        ]
        CBs = self.cbs_from_frames(Ns, Cs, CAs, self.geometry_params)

        # 2 - compute pairwise distances
        cbdel = CBs[:, None, :, :] - CBs[:, :, None, :]
        ds = torch.sqrt(torch.sum(cbdel * cbdel, dim=-1))

        # 3 - spline interpolation
        E = self.interp(self.spline_xs, self.spline_ys, ds)

        Etot = torch.sum(torch.triu(E, 3), dim=(1, 2))

        return Etot
