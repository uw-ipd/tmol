import toolz
import attr

import numpy
import torch

from tmol.score.ljlk.numba.vectorized import lj, lk_isotropic
from tmol.score.bonded_atom import bonded_path_length

import tmol.database

from tmol.utility.args import ignore_unused_kwargs
from tmol.tests.autograd import gradcheck

from tmol.tests.benchmark import subfixture

from tmol.score.ljlk.script_modules import (
    LJIntraModule,
    LJInterModule,
    LKIsotropicIntraModule,
    LKIsotropicInterModule,
)


@attr.s(auto_attribs=True)
class ScoreSetup:
    param_resolver: tmol.score.ljlk.params.LJLKParamResolver
    coords: torch.tensor
    atom_type_idx: torch.tensor
    atom_pair_bpl: numpy.ndarray

    tcoords: torch.Tensor
    ttype: torch.Tensor
    tbpl: torch.Tensor

    @classmethod
    def from_fixture(cls, database, system, torch_device) -> "ScoreSetup":
        param_resolver = tmol.score.ljlk.params.LJLKParamResolver.from_database(
            database.chemical, database.scoring.ljlk, torch_device
        )

        coords = system.coords
        atom_type_idx = param_resolver.type_idx(system.atom_metadata["atom_type"])
        atom_pair_bpl = bonded_path_length(system.bonds, system.coords.shape[0], 6)

        tcoords = (
            torch.from_numpy(system.coords).to(device=torch_device).requires_grad_(True)
        )
        ttype = atom_type_idx
        tbpl = torch.from_numpy(atom_pair_bpl).to(device=torch_device)

        return cls(
            param_resolver=param_resolver,
            coords=coords,
            atom_type_idx=atom_type_idx,
            atom_pair_bpl=atom_pair_bpl,
            tcoords=tcoords,
            ttype=ttype,
            tbpl=tbpl,
        )


def _dense_lj(coords, atom_type_idx, atom_pair_bpl, param_resolver):
    return _dense_potential(lj, coords, atom_type_idx, atom_pair_bpl, param_resolver)


def _dense_lk(coords, atom_type_idx, atom_pair_bpl, param_resolver):
    return _dense_potential(
        lk_isotropic, coords, atom_type_idx, atom_pair_bpl, param_resolver
    )


def _dense_potential(potential, coords, atom_type_idx, atom_pair_bpl, param_resolver):
    """Compute dense atom-atom lj score table via vectorized op."""
    atom_pair_d = numpy.linalg.norm(coords[None, :, :] - coords[:, None, :], axis=-1)
    atom_type_params = toolz.valmap(
        lambda t: t.cpu().numpy(),
        attr.asdict(param_resolver.type_params[atom_type_idx]),
    )

    return ignore_unused_kwargs(potential)(
        atom_pair_d,
        atom_pair_bpl,
        **toolz.merge(
            {k + "_i": t[:, None] for k, t in atom_type_params.items()},
            {k + "_j": t[None, :] for k, t in atom_type_params.items()},
            toolz.valmap(
                lambda t: t.cpu().numpy(), attr.asdict(param_resolver.global_params)
            ),
        ),
    )


def test_lj_intra_op(benchmark, default_database, ubq_system, torch_device):
    """LJIntraModule returns sum of triu entries of the dense lj score matrix."""
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)

    expected_dense = numpy.triu(
        numpy.nan_to_num(
            _dense_lj(s.coords, s.atom_type_idx, s.atom_pair_bpl, s.param_resolver)
        )
    )

    op = LJIntraModule(s.param_resolver)
    op.to(s.tcoords)

    @subfixture(benchmark)
    def op_val():
        retval = op(s.tcoords, s.ttype, s.tbpl)
        torch.cuda.synchronize()
        return retval

    torch.testing.assert_allclose(
        op_val, torch.tensor(expected_dense).to(torch_device).sum()
    )

    @subfixture(benchmark)
    def op_full():
        res = op(s.tcoords, s.ttype, s.tbpl)
        torch.cuda.synchronize()
        res.backward()

        return res

    torch.testing.assert_allclose(
        op_full, torch.tensor(expected_dense).to(torch_device).sum()
    )

    subind = torch.arange(0, s.tcoords.shape[0], 100)

    def op_subset(c):
        fcoords = s.tcoords.clone()
        fcoords[subind] = c

        retval = op(fcoords, s.ttype, s.tbpl)
        torch.cuda.synchronize()
        return retval

    gradcheck(op_subset, (s.tcoords[subind].requires_grad_(True),), eps=1e-3)


def test_lj_inter_op(default_database, torch_device, ubq_system):
    """LJInterModule returns sum of the dense lj score matrix."""

    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)

    part = ubq_system.system_size // 2

    expected_dense = numpy.nan_to_num(
        _dense_lj(s.coords, s.atom_type_idx, s.atom_pair_bpl, s.param_resolver)
    )[:part, part:]

    op = LJInterModule(s.param_resolver)
    op.to(s.tcoords)

    val = op(
        s.tcoords[:part],
        s.ttype[:part],
        s.tcoords[part:],
        s.ttype[part:],
        s.tbpl[:part, part:],
    )

    torch.testing.assert_allclose(
        val, torch.tensor(expected_dense).to(torch_device).sum()
    )

    subind = torch.arange(0, s.tcoords.shape[0], 100)

    def op_subset(c):
        fcoords = s.tcoords.clone()
        fcoords[subind] = c

        retval = op(
            fcoords[:part],
            s.ttype[:part],
            fcoords[part:],
            s.ttype[part:],
            s.tbpl[:part, part:],
        )
        torch.cuda.synchronize()
        return retval

    gradcheck(op_subset, (s.tcoords[subind].requires_grad_(True),), eps=1e-3)


def test_lk_intra_op(benchmark, default_database, ubq_system, torch_device):
    """LKIsotropicIntraModule returns sum of triu entries of the dense
    lk_isotropic score matrix."""

    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)

    expected_dense = numpy.triu(
        numpy.nan_to_num(
            _dense_lk(s.coords, s.atom_type_idx, s.atom_pair_bpl, s.param_resolver)
        )
    )

    op = LKIsotropicIntraModule(s.param_resolver)
    op.to(s.tcoords)

    @subfixture(benchmark)
    def op_val():
        retval = op(s.tcoords, s.ttype, s.tbpl)
        torch.cuda.synchronize()
        return retval

    torch.testing.assert_allclose(
        op_val, torch.tensor(expected_dense).to(torch_device).sum()
    )

    @subfixture(benchmark)
    def op_full():
        res = op(s.tcoords, s.ttype, s.tbpl)
        torch.cuda.synchronize()
        res.backward()

        return res

    torch.testing.assert_allclose(
        op_full, torch.tensor(expected_dense).to(torch_device).sum()
    )

    subind = torch.arange(0, s.tcoords.shape[0], 100)

    def op_subset(c):
        fcoords = s.tcoords.clone()
        fcoords[subind] = c

        retval = op(fcoords, s.ttype, s.tbpl)
        torch.cuda.synchronize()
        return retval

    gradcheck(op_subset, (s.tcoords[subind].requires_grad_(True),), eps=1e-3)


def test_lk_inter_op(default_database, torch_device, ubq_system):
    """LKIsotropicInterModule returns sum of the dense lj score matrix."""

    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)

    part = ubq_system.system_size // 2

    expected_dense = numpy.nan_to_num(
        _dense_lk(s.coords, s.atom_type_idx, s.atom_pair_bpl, s.param_resolver)
    )[:part, part:]

    op = LKIsotropicInterModule(s.param_resolver)
    op.to(s.tcoords)

    val = op(
        s.tcoords[:part],
        s.ttype[:part],
        s.tcoords[part:],
        s.ttype[part:],
        s.tbpl[:part, part:],
    )

    torch.testing.assert_allclose(
        val, torch.tensor(expected_dense).to(torch_device).sum()
    )

    subind = torch.arange(0, s.tcoords.shape[0], 100)

    def op_subset(c):
        fcoords = s.tcoords.clone()
        fcoords[subind] = c

        retval = op(
            fcoords[:part],
            s.ttype[:part],
            fcoords[part:],
            s.ttype[part:],
            s.tbpl[:part, part:],
        )
        torch.cuda.synchronize()
        return retval

    gradcheck(op_subset, (s.tcoords[subind].requires_grad_(True),), eps=1e-3)
