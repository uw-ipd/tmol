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
        self.dense_cbcb_dist_xs = _p(param_resolver.dense_cbcb_dist_xs)
        self.dense_cbcb_dist_ys = _p(param_resolver.dense_cbcb_dist_ys)
        self.dense_cacbcbca_tors_xs = _p(param_resolver.dense_cacbcbca_tors_xs)
        self.dense_cacbcbca_tors_ys = _p(param_resolver.dense_cacbcbca_tors_ys)
        self.dense_ncacacb_tors_xs = _p(param_resolver.dense_ncacacb_tors_xs)
        self.dense_ncacacb_tors_ys = _p(param_resolver.dense_ncacacb_tors_ys)
        self.dense_cacacb_angle_xs = _p(param_resolver.dense_cacacb_angle_xs)
        self.dense_cacacb_angle_ys = _p(param_resolver.dense_cacacb_angle_ys)

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
    def interp(self, x, ys, xs, verbose=False):
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
        I = torch.searchsorted(x[1:], xs.detach())  # fd
        dx = x[I + 1] - x[I]
        hh = h_poly((xs - x[I]) / dx)

        I = I.reshape(nstk, -1).transpose(0, 1)
        ys = ys.reshape(-1, nspl)
        ms = ms.reshape(-1, nspl)

        yI = torch.gather(ys, 1, I).reshape((nres, nres, nstk)).permute([2, 0, 1])
        yIp1 = torch.gather(ys, 1, I + 1).reshape((nres, nres, nstk)).permute([2, 0, 1])
        mI = torch.gather(ms, 1, I).reshape((nres, nres, nstk)).permute([2, 0, 1])
        mIp1 = torch.gather(ms, 1, I + 1).reshape((nres, nres, nstk)).permute([2, 0, 1])

        interp = hh[0] * yI + hh[1] * mI * dx + hh[2] * yIp1 + hh[3] * mIp1 * dx

        return interp

    def get_angles(self, I, J, K):
        F = I - J
        G = K - J
        angle = torch.acos(
            torch.clamp(
                torch.sum(F * G, dim=-1)
                / torch.sqrt(
                    (torch.sum(F * F, dim=-1) * torch.sum(G * G, dim=-1)) + 1e-12
                ),
                -1,
                1,
            )
        )
        return angle

    def get_dihedrals(self, I, J, K, L):
        def cross_broadcast(x, y):
            z0 = x[..., 1] * y[..., 2] - x[..., 2] * y[..., 1]
            z1 = x[..., 2] * y[..., 0] - x[..., 0] * y[..., 2]
            z2 = x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
            return torch.stack((z0, z1, z2), dim=-1)

        F = I - J
        G = J - K
        H = L - K
        A = cross_broadcast(F, G)
        B = cross_broadcast(H, G)
        sign = torch.sign(torch.sum(G * cross_broadcast(A, B), dim=-1))
        dih = -sign * torch.acos(
            torch.clamp(
                torch.sum(A * B, dim=-1)
                / torch.sqrt(
                    (torch.sum(A * A, dim=-1) * torch.sum(B * B, dim=-1)) + 1e-12
                ),
                -1,
                1,
            )
        )
        return dih

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

        # 2 - compute pairwise distances and angles
        cbdel = CBs[:, None, :, :] - CBs[:, :, None, :]
        ds = torch.sqrt(torch.sum(cbdel * cbdel, dim=-1))
        omegas = self.get_dihedrals(
            CAs[:, :, None, :],
            CBs[:, :, None, :],
            CBs[:, None, :, :],
            CAs[:, None, :, :],
        )
        thetas = self.get_dihedrals(
            Ns[:, :, None, :],
            CAs[:, :, None, :],
            CBs[:, :, None, :],
            CBs[:, None, :, :],
        )
        phis = self.get_angles(
            CAs[:, :, None, :], CBs[:, :, None, :], CBs[:, None, :, :]
        )

        # 3 - spline interpolation
        Ed = self.interp(self.dense_cbcb_dist_xs, self.dense_cbcb_dist_ys, ds)
        Eomega = self.interp(
            self.dense_cacbcbca_tors_xs, self.dense_cacbcbca_tors_ys, omegas, True
        )
        Etheta = self.interp(
            self.dense_ncacacb_tors_xs, self.dense_ncacacb_tors_ys, thetas
        )
        Ephi = self.interp(self.dense_cacacb_angle_xs, self.dense_cacacb_angle_ys, phis)

        return torch.sum(Ed), torch.sum(Eomega) + torch.sum(Etheta), torch.sum(Ephi)
