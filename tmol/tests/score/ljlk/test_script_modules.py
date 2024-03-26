import toolz
import attr

import numpy
import torch

from tmol.score.bonded_atom import bonded_path_length

import tmol.database

from tmol.utility.args import ignore_unused_kwargs
from tmol.tests.autograd import gradcheck

from tmol.tests.benchmark import subfixture
from tmol.tests.numba import requires_numba_jit

from tmol.score.ljlk.script_modules import (
    LJIntraModule,
    LJInterModule,
    LKIsotropicIntraModule,
    LKIsotropicInterModule,
)

from tmol.score.common.stack_condense import condense_torch_inds
from tmol.score.chemical_database import AtomTypeParamResolver


@attr.s(auto_attribs=True)
class ScoreSetup:
    param_resolver: tmol.score.ljlk.params.LJLKParamResolver
    coords: torch.tensor
    atom_type_idx: torch.tensor
    atom_pair_bpl: numpy.ndarray

    tcoords: torch.Tensor
    ttype: torch.Tensor
    thvy_at_inds: torch.Tensor
    tbpl: torch.Tensor

    @classmethod
    def from_fixture(cls, database, system, torch_device) -> "ScoreSetup":
        param_resolver = tmol.score.ljlk.params.LJLKParamResolver.from_database(
            database.chemical, database.scoring.ljlk, torch_device
        )

        coords = system.coords[None, :]
        atom_type_idx = param_resolver.type_idx(system.atom_metadata["atom_type"])[
            None, :
        ]
        atom_pair_bpl = bonded_path_length(system.bonds, system.coords.shape[0], 6)[
            None, :
        ]

        tcoords = (
            torch.from_numpy(system.coords[None, :])
            .to(device=torch_device)
            .requires_grad_(True)
        )
        ttype = atom_type_idx[:]

        atype_params = AtomTypeParamResolver.from_database(
            database.chemical, torch_device
        )
        thvy_at_inds = condense_torch_inds(
            ~atype_params.params.is_hydrogen[ttype], torch_device
        )

        tbpl = torch.from_numpy(atom_pair_bpl).to(device=torch_device)[:, :]

        return cls(
            param_resolver=param_resolver,
            coords=coords,
            atom_type_idx=atom_type_idx,
            atom_pair_bpl=atom_pair_bpl,
            tcoords=tcoords,
            ttype=ttype,
            thvy_at_inds=thvy_at_inds,
            tbpl=tbpl,
        )


def _dense_lj(coords, atom_type_idx, atom_pair_bpl, param_resolver):
    from tmol.score.ljlk.numba.vectorized import lj

    return _dense_potential(lj, coords, atom_type_idx, atom_pair_bpl, param_resolver)


def _dense_lk(coords, atom_type_idx, atom_pair_bpl, param_resolver):
    from tmol.score.ljlk.numba.vectorized import lk_isotropic

    return _dense_potential(
        lk_isotropic, coords, atom_type_idx, atom_pair_bpl, param_resolver
    )


def _dense_potential(potential, coords, atom_type_idx, atom_pair_bpl, param_resolver):
    """Compute dense atom-atom lj score table via vectorized op."""
    atom_pair_d = numpy.linalg.norm(
        coords[:, None, :, :] - coords[:, :, None, :], axis=-1
    )
    atom_type_params = toolz.valmap(
        lambda t: t.cpu().numpy(),
        attr.asdict(param_resolver.type_params[atom_type_idx]),
    )

    return ignore_unused_kwargs(potential)(
        atom_pair_d,
        atom_pair_bpl,
        **toolz.merge(
            {k + "_i": t[:, :, None] for k, t in atom_type_params.items()},
            {k + "_j": t[:, None, :] for k, t in atom_type_params.items()},
            toolz.valmap(
                lambda t: t.cpu().numpy(), attr.asdict(param_resolver.global_params)
            ),
        ),
    )


@requires_numba_jit
def test_lj_intra_op(benchmark, default_database, ubq_system, torch_device):
    """LJIntraModule returns sum of triu entries of the dense lj score matrix."""
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)

    expected_dense = numpy.triu(
        numpy.nan_to_num(
            _dense_lj(s.coords, s.atom_type_idx, s.atom_pair_bpl, s.param_resolver)
        )
    ).sum()

    op = LJIntraModule(s.param_resolver)
    op.to(s.tcoords)

    @subfixture(benchmark)
    def op_val():
        return op(s.tcoords, s.ttype, s.tbpl)

    torch.testing.assert_close(op_val, torch.tensor((expected_dense,)).to(torch_device))

    @subfixture(benchmark)
    def op_full():
        res = op(s.tcoords, s.ttype, s.tbpl)
        res.backward()

        return res

    torch.testing.assert_close(
        op_full, torch.tensor((expected_dense,)).to(torch_device)
    )

    subind = torch.arange(0, s.tcoords.shape[1], 100)

    def op_subset(c):
        fcoords = s.tcoords.clone()
        fcoords[:, subind] = c

        return op(fcoords, s.ttype, s.tbpl)

    gradcheck(op_subset, (s.tcoords[:, subind].requires_grad_(True),), eps=1e-3)


@requires_numba_jit
def test_lj_intra_op_stacked(benchmark, default_database, torch_device, ubq_system):
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)

    expected_dense = numpy.triu(
        numpy.nan_to_num(
            _dense_lj(s.coords, s.atom_type_idx, s.atom_pair_bpl, s.param_resolver)
        )
    )

    op = LJIntraModule(s.param_resolver)
    op.to(s.tcoords)

    coords2 = torch.cat((s.tcoords, s.tcoords), dim=0)
    atype2 = torch.cat((s.ttype, s.ttype), dim=0)
    atbpl = torch.cat((s.tbpl, s.tbpl), dim=0)

    @subfixture(benchmark)
    def op_val():
        return op(coords2, atype2, atbpl)

    torch.testing.assert_close(
        op_val,
        torch.tensor(expected_dense).to(torch_device).sum().unsqueeze(0).repeat(2),
    )


@requires_numba_jit
def test_lj_inter_op(default_database, torch_device, ubq_system):
    """LJInterModule returns sum of the dense lj score matrix."""

    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)

    part = ubq_system.system_size // 2

    expected_dense = numpy.nan_to_num(
        _dense_lj(s.coords, s.atom_type_idx, s.atom_pair_bpl, s.param_resolver)
    )[:, :part, part:].sum()

    op = LJInterModule(s.param_resolver)
    op.to(s.tcoords)

    val = op(
        s.tcoords[:, :part],
        s.ttype[:, :part],
        s.tcoords[:, part:],
        s.ttype[:, part:],
        s.tbpl[:, :part, part:],
    )

    torch.testing.assert_close(val, torch.tensor((expected_dense,)).to(torch_device))

    subind = torch.arange(0, s.tcoords.shape[1], 100)

    def op_subset(c):
        fcoords = s.tcoords.clone()
        fcoords[:, subind] = c

        return op(
            fcoords[:, :part],
            s.ttype[:, :part],
            fcoords[:, part:],
            s.ttype[:, part:],
            s.tbpl[:, :part, part:],
        )

    gradcheck(op_subset, (s.tcoords[:, subind].requires_grad_(True),), eps=1e-3)


@requires_numba_jit
def test_lk_intra_op(benchmark, default_database, ubq_system, torch_device):
    """LKIsotropicIntraModule returns sum of triu entries of the dense
    lk_isotropic score matrix."""

    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)

    expected_dense = numpy.triu(
        numpy.nan_to_num(
            _dense_lk(s.coords, s.atom_type_idx, s.atom_pair_bpl, s.param_resolver)
        )
    ).sum()

    op = LKIsotropicIntraModule(s.param_resolver)
    op.to(s.tcoords)

    @subfixture(benchmark)
    def op_val():
        return op(s.tcoords, s.ttype, s.thvy_at_inds, s.tbpl)

    torch.testing.assert_close(op_val, torch.tensor((expected_dense,)).to(torch_device))

    @subfixture(benchmark)
    def op_full():
        res = op(s.tcoords, s.ttype, s.thvy_at_inds, s.tbpl)
        res.backward()

        return res

    torch.testing.assert_close(
        op_full, torch.tensor((expected_dense,)).to(torch_device)
    )

    subind = torch.arange(0, s.tcoords.shape[1], 100)

    def op_subset(c):
        fcoords = s.tcoords.clone()
        fcoords[:, subind] = c

        return op(fcoords, s.ttype, s.thvy_at_inds, s.tbpl)

    gradcheck(op_subset, (s.tcoords[:, subind].requires_grad_(True),), eps=1e-3)


@requires_numba_jit
def test_lk_inter_op(default_database, torch_device, ubq_system):
    """LKIsotropicInterModule returns sum of the dense lj score matrix."""

    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)

    part = ubq_system.system_size // 2

    expected_dense = numpy.nan_to_num(
        _dense_lk(s.coords, s.atom_type_idx, s.atom_pair_bpl, s.param_resolver)
    )[:, :part, part:].sum()

    op = LKIsotropicInterModule(s.param_resolver)
    op.to(s.tcoords)

    inds_part_lo = s.thvy_at_inds[:, s.thvy_at_inds[0, :] < part]
    inds_part_hi = s.thvy_at_inds[:, s.thvy_at_inds[0, :] >= part] - part

    val = op(
        s.tcoords[:, :part],
        s.ttype[:, :part],
        inds_part_lo,
        s.tcoords[:, part:],
        s.ttype[:, part:],
        inds_part_hi,
        s.tbpl[:, :part, part:],
    )

    torch.testing.assert_close(val, torch.tensor((expected_dense,)).to(torch_device))

    subind = torch.arange(0, s.tcoords.shape[1], 100)

    def op_subset(c):
        fcoords = s.tcoords.clone()
        fcoords[:, subind] = c

        return op(
            fcoords[:, :part],
            s.ttype[:, :part],
            inds_part_lo,
            fcoords[:, part:],
            s.ttype[:, part:],
            inds_part_hi,
            s.tbpl[:, :part, part:],
        )

    gradcheck(op_subset, (s.tcoords[:, subind].requires_grad_(True),), eps=1e-3)
