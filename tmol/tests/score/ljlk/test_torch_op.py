import toolz
import attr

import numpy
import torch
import sparse

from tmol.score.ljlk.numba.vectorized import lj, lk_isotropic
from tmol.score.ljlk.torch_op import LJOp, LKOp
from tmol.score.bonded_atom import bonded_path_length

import tmol.database

from tmol.utility.args import ignore_unused_kwargs


@attr.s(auto_attribs=True)
class ScoreSetup:
    param_resolver: tmol.score.ljlk.params.LJLKParamResolver
    coords: numpy.ndarray
    atom_type_idx: numpy.ndarray
    atom_pair_bpl: numpy.ndarray

    tcoords: torch.Tensor
    ttype: torch.Tensor
    tbpl: torch.Tensor

    @classmethod
    def from_fixture(cls, database, system, torch_device) -> "ScoreSetup":
        param_resolver = tmol.score.ljlk.params.LJLKParamResolver.from_database(
            database.scoring.ljlk, torch_device
        )

        coords = system.coords
        atom_type_idx = param_resolver.type_idx(system.atom_metadata["atom_type"])
        atom_pair_bpl = bonded_path_length(system.bonds, system.coords.shape[0], 6)

        tcoords = (
            torch.from_numpy(system.coords).to(device=torch_device).requires_grad_(True)
        )
        ttype = torch.from_numpy(atom_type_idx).to(device=torch_device)
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


def _todense(inds, val, shape=None):
    return sparse.COO(
        inds.cpu().numpy(), val.detach().cpu().numpy(), shape=shape
    ).todense()


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
            toolz.valmap(float, attr.asdict(param_resolver.global_params)),
        ),
    )


def test_lj_intra_op(default_database, ubq_system, torch_device):
    """LJOp.intra returns triu entries of the dense lj score matrix."""
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)

    expected_dense = numpy.triu(
        numpy.nan_to_num(
            _dense_lj(s.coords, s.atom_type_idx, s.atom_pair_bpl, s.param_resolver)
        )
    )

    op = LJOp.from_param_resolver(s.param_resolver)

    v_inds, v_val = op.intra(s.tcoords, s.ttype, s.tbpl)

    assert not v_inds.requires_grad
    assert v_val.requires_grad

    op_dense = _todense(v_inds, v_val, (s.coords.shape[0],) * 2)
    numpy.testing.assert_allclose(op_dense, expected_dense, rtol=1e-6, atol=1e-7)


def test_lj_inter_op(default_database, torch_device, ubq_system):
    """LJOp.intra returns entries of the dense lj score matrix."""
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)

    part = ubq_system.system_size // 2

    expected_dense = numpy.triu(
        numpy.nan_to_num(
            _dense_lj(s.coords, s.atom_type_idx, s.atom_pair_bpl, s.param_resolver)
        )
    )[:part, part:]

    op = LJOp.from_param_resolver(s.param_resolver)

    v_inds, v_val = op.inter(
        s.tcoords[:part],
        s.ttype[:part],
        s.tcoords[part:],
        s.ttype[part:],
        s.tbpl[:part, part:],
    )

    assert not v_inds.requires_grad
    assert v_val.requires_grad

    op_dense = _todense(v_inds, v_val, shape=(part, ubq_system.system_size - part))

    numpy.testing.assert_allclose(op_dense, expected_dense, rtol=1e-6, atol=1e-7)


def test_lj_inter_op_gradcheck(default_database, ubq_system, torch_device):
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)

    natoms = 16
    op = LJOp.from_param_resolver(s.param_resolver)

    coords_a = s.tcoords[0:natoms]
    coords_b = s.tcoords[natoms : natoms * 2]

    def eval_inter(coords_a, coords_b):

        i, v = op.inter(
            coords_a,
            s.ttype[0:natoms],
            coords_b,
            s.ttype[natoms : natoms * 2],
            s.tbpl[0:natoms, natoms : natoms * 2],
        )

        return v

    torch.autograd.gradcheck(
        eval_inter,
        (coords_a.requires_grad_(True), coords_b.requires_grad_(True)),
        eps=1e-3,
    )


def test_lj_intra_op_gradcheck(default_database, ubq_system, torch_device):
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)

    natoms = 32
    op = LJOp.from_param_resolver(s.param_resolver)

    coords = s.tcoords[:natoms]

    def eval_intra(coords):

        i, v = op.intra(coords, s.ttype[:natoms], s.tbpl[:natoms, :natoms])

        return v

    torch.autograd.gradcheck(eval_intra, (coords.requires_grad_(True),), eps=1e-3)


def test_lk_intra_op(default_database, ubq_system, torch_device):
    """LKOp.intra returns triu entries of the dense lk score matrix."""

    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)

    expected_dense = numpy.triu(
        numpy.nan_to_num(
            _dense_lk(s.coords, s.atom_type_idx, s.atom_pair_bpl, s.param_resolver)
        )
    )

    op = LKOp.from_param_resolver(s.param_resolver)

    v_inds, v_val = op.intra(s.tcoords, s.ttype, s.tbpl)

    assert not v_inds.requires_grad
    assert v_val.requires_grad

    op_dense = _todense(v_inds, v_val, (s.coords.shape[0],) * 2)
    numpy.testing.assert_allclose(op_dense, expected_dense, rtol=1e-6, atol=1e-7)


def test_lk_inter_op(default_database, ubq_system, torch_device):
    """LKOp.intra returns entries of the dense lk score matrix."""

    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)

    part = ubq_system.system_size // 2

    expected_dense = numpy.triu(
        numpy.nan_to_num(
            _dense_lk(s.coords, s.atom_type_idx, s.atom_pair_bpl, s.param_resolver)
        )
    )[:part, part:]

    op = LKOp.from_param_resolver(s.param_resolver)

    v_inds, v_val = op.inter(
        s.tcoords[:part],
        s.ttype[:part],
        s.tcoords[part:],
        s.ttype[part:],
        s.tbpl[:part, part:],
    )

    assert not v_inds.requires_grad
    assert v_val.requires_grad

    op_dense = _todense(v_inds, v_val, shape=(part, ubq_system.system_size - part))

    numpy.testing.assert_allclose(op_dense, expected_dense, rtol=1e-6, atol=1e-7)


def test_lk_inter_op_gradcheck(default_database, ubq_system, torch_device):
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)

    natoms = 16
    op = LKOp.from_param_resolver(s.param_resolver)

    coords_a = s.tcoords[0:natoms]
    coords_b = s.tcoords[natoms : natoms * 2]

    def eval_inter(coords_a, coords_b):

        i, v = op.inter(
            coords_a,
            s.ttype[0:natoms],
            coords_b,
            s.ttype[natoms : natoms * 2],
            s.tbpl[0:natoms, natoms : natoms * 2],
        )

        return v

    torch.autograd.gradcheck(
        eval_inter,
        (coords_a.requires_grad_(True), coords_b.requires_grad_(True)),
        eps=1e-3,
    )


def test_lk_intra_op_gradcheck(default_database, ubq_system, torch_device):
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)

    natoms = 32
    op = LKOp.from_param_resolver(s.param_resolver)

    coords = s.tcoords[:natoms]

    def eval_intra(coords):

        i, v = op.intra(coords, s.ttype[:natoms], s.tbpl[:natoms, :natoms])

        return v

    torch.autograd.gradcheck(eval_intra, (coords.requires_grad_(True),), eps=1e-3)
