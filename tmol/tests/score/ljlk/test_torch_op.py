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


def test_lj_intra_op(default_database, ubq_system):
    """LJOp.intra returns triu entries of the dense lj score matrix."""

    param_resolver = tmol.score.ljlk.params.LJLKParamResolver.from_database(
        default_database.scoring.ljlk, torch.device("cpu")
    )

    op = LJOp.from_param_resolver(param_resolver)

    atom_type_idx = param_resolver.type_idx(ubq_system.atom_metadata["atom_type"])
    atom_pair_bpl = bonded_path_length(ubq_system.bonds, ubq_system.coords.shape[0], 6)

    expected_dense = numpy.triu(
        numpy.nan_to_num(
            _dense_lj(ubq_system.coords, atom_type_idx, atom_pair_bpl, param_resolver)
        )
    )

    v_inds, v_val = op.intra(
        torch.from_numpy(ubq_system.coords).requires_grad_(True),
        torch.from_numpy(atom_type_idx),
        torch.from_numpy(atom_pair_bpl),
    )

    assert not v_inds.requires_grad
    assert v_val.requires_grad

    op_dense = sparse.COO(
        v_inds.numpy(), v_val.detach().numpy(), shape=(ubq_system.system_size,) * 2
    ).todense()

    numpy.testing.assert_allclose(op_dense, expected_dense, rtol=1e-6, atol=1e-7)


def test_lj_inter_op(default_database, ubq_system):
    """LJOp.intra returns entries of the dense lj score matrix."""

    param_resolver = tmol.score.ljlk.params.LJLKParamResolver.from_database(
        default_database.scoring.ljlk, torch.device("cpu")
    )

    op = LJOp.from_param_resolver(param_resolver)

    atom_type_idx = param_resolver.type_idx(ubq_system.atom_metadata["atom_type"])
    atom_pair_bpl = bonded_path_length(ubq_system.bonds, ubq_system.coords.shape[0], 6)

    part = ubq_system.system_size // 2

    expected_dense = numpy.nan_to_num(
        _dense_lj(ubq_system.coords, atom_type_idx, atom_pair_bpl, param_resolver)
    )[:part, part:]

    v_inds, v_val = op.inter(
        torch.from_numpy(ubq_system.coords[:part]).requires_grad_(True),
        torch.from_numpy(atom_type_idx[:part]),
        torch.from_numpy(ubq_system.coords[part:]).requires_grad_(True),
        torch.from_numpy(atom_type_idx[part:]),
        torch.from_numpy(atom_pair_bpl[:part, part:]),
    )

    assert not v_inds.requires_grad
    assert v_val.requires_grad

    op_dense = sparse.COO(
        v_inds.numpy(),
        v_val.detach().numpy(),
        shape=(part, ubq_system.system_size - part),
    ).todense()

    numpy.testing.assert_allclose(op_dense, expected_dense, rtol=1e-6, atol=1e-7)


def test_lj_inter_op_gradcheck(default_database, ubq_system):
    natoms = 16

    param_resolver = tmol.score.ljlk.params.LJLKParamResolver.from_database(
        default_database.scoring.ljlk, torch.device("cpu")
    )

    op = LJOp.from_param_resolver(param_resolver)

    coords_a = ubq_system.coords[0:natoms]
    atom_type_idx_a = param_resolver.type_idx(ubq_system.atom_metadata["atom_type"])[
        0:natoms
    ]
    coords_b = ubq_system.coords[natoms : natoms * 2]
    atom_type_idx_b = param_resolver.type_idx(ubq_system.atom_metadata["atom_type"])[
        natoms : natoms * 2
    ]
    atom_pair_bpl = bonded_path_length(ubq_system.bonds, ubq_system.coords.shape[0], 6)[
        0:natoms, natoms : natoms * 2
    ]

    def eval_inter(coords_a, coords_b):

        i, v = op.inter(
            coords_a,
            torch.from_numpy(atom_type_idx_a),
            coords_b,
            torch.from_numpy(atom_type_idx_b),
            torch.from_numpy(atom_pair_bpl),
        )

        return v

    torch.autograd.gradcheck(
        eval_inter,
        (
            torch.from_numpy(coords_a).requires_grad_(True),
            torch.from_numpy(coords_b).requires_grad_(True),
        ),
        eps=1e-3,
    )


def test_lj_intra_op_gradcheck(default_database, ubq_system):
    natoms = 32

    param_resolver = tmol.score.ljlk.params.LJLKParamResolver.from_database(
        default_database.scoring.ljlk, torch.device("cpu")
    )

    op = LJOp.from_param_resolver(param_resolver)

    coords = ubq_system.coords[:natoms]
    atom_type_idx = param_resolver.type_idx(ubq_system.atom_metadata["atom_type"])[
        :natoms
    ]
    atom_pair_bpl = bonded_path_length(ubq_system.bonds, ubq_system.coords.shape[0], 6)[
        :natoms, :natoms
    ]

    def eval_intra(coords):

        i, v = op.intra(
            coords, torch.from_numpy(atom_type_idx), torch.from_numpy(atom_pair_bpl)
        )

        return v

    torch.autograd.gradcheck(
        eval_intra, (torch.from_numpy(coords).requires_grad_(True),), eps=1e-3
    )


def test_lk_intra_op(default_database, ubq_system):
    """LKOp.intra returns triu entries of the dense lk score matrix."""

    param_resolver = tmol.score.ljlk.params.LJLKParamResolver.from_database(
        default_database.scoring.ljlk, torch.device("cpu")
    )

    op = LKOp.from_param_resolver(param_resolver)

    atom_type_idx = param_resolver.type_idx(ubq_system.atom_metadata["atom_type"])
    atom_pair_bpl = bonded_path_length(ubq_system.bonds, ubq_system.coords.shape[0], 6)

    expected_dense = numpy.triu(
        numpy.nan_to_num(
            _dense_lk(ubq_system.coords, atom_type_idx, atom_pair_bpl, param_resolver)
        )
    )

    v_inds, v_val = op.intra(
        torch.from_numpy(ubq_system.coords).requires_grad_(True),
        torch.from_numpy(atom_type_idx),
        torch.from_numpy(atom_pair_bpl),
    )

    assert not v_inds.requires_grad
    assert v_val.requires_grad

    op_dense = sparse.COO(
        v_inds.numpy(), v_val.detach().numpy(), shape=(ubq_system.system_size,) * 2
    ).todense()

    numpy.testing.assert_allclose(op_dense, expected_dense, atol=1e-8)


def test_lk_inter_op(default_database, ubq_system):
    """LKOp.intra returns entries of the dense lk score matrix."""

    param_resolver = tmol.score.ljlk.params.LJLKParamResolver.from_database(
        default_database.scoring.ljlk, torch.device("cpu")
    )

    op = LKOp.from_param_resolver(param_resolver)

    atom_type_idx = param_resolver.type_idx(ubq_system.atom_metadata["atom_type"])
    atom_pair_bpl = bonded_path_length(ubq_system.bonds, ubq_system.coords.shape[0], 6)

    part = ubq_system.system_size // 2

    expected_dense = numpy.nan_to_num(
        _dense_lk(ubq_system.coords, atom_type_idx, atom_pair_bpl, param_resolver)
    )[:part, part:]

    v_inds, v_vals = op.inter(
        torch.from_numpy(ubq_system.coords[:part]).requires_grad_(True),
        torch.from_numpy(atom_type_idx[:part]),
        torch.from_numpy(ubq_system.coords[part:]).requires_grad_(True),
        torch.from_numpy(atom_type_idx[part:]),
        torch.from_numpy(atom_pair_bpl[:part, part:]),
    )

    assert not v_inds.requires_grad
    assert v_vals.requires_grad

    op_dense = sparse.COO(
        v_inds.numpy(),
        v_vals.detach().numpy(),
        shape=(part, ubq_system.system_size - part),
    ).todense()

    numpy.testing.assert_allclose(op_dense, expected_dense, atol=1e-8)


def test_lk_inter_op_gradcheck(default_database, ubq_system):
    natoms = 16

    param_resolver = tmol.score.ljlk.params.LJLKParamResolver.from_database(
        default_database.scoring.ljlk, torch.device("cpu")
    )

    op = LKOp.from_param_resolver(param_resolver)

    coords_a = ubq_system.coords[0:natoms]
    atom_type_idx_a = param_resolver.type_idx(ubq_system.atom_metadata["atom_type"])[
        0:natoms
    ]
    coords_b = ubq_system.coords[natoms : natoms * 2]
    atom_type_idx_b = param_resolver.type_idx(ubq_system.atom_metadata["atom_type"])[
        natoms : natoms * 2
    ]
    atom_pair_bpl = bonded_path_length(ubq_system.bonds, ubq_system.coords.shape[0], 6)[
        0:natoms, natoms : natoms * 2
    ]

    def eval_inter(coords_a, coords_b):

        i, v = op.inter(
            coords_a,
            torch.from_numpy(atom_type_idx_a),
            coords_b,
            torch.from_numpy(atom_type_idx_b),
            torch.from_numpy(atom_pair_bpl),
        )

        return v

    torch.autograd.gradcheck(
        eval_inter,
        (
            torch.from_numpy(coords_a).requires_grad_(True),
            torch.from_numpy(coords_b).requires_grad_(True),
        ),
        eps=1e-3,
    )


def test_lk_intra_op_gradcheck(default_database, ubq_system):
    natoms = 32

    param_resolver = tmol.score.ljlk.params.LJLKParamResolver.from_database(
        default_database.scoring.ljlk, torch.device("cpu")
    )

    op = LJOp.from_param_resolver(param_resolver)

    coords = ubq_system.coords[:natoms]
    atom_type_idx = param_resolver.type_idx(ubq_system.atom_metadata["atom_type"])[
        :natoms
    ]
    atom_pair_bpl = bonded_path_length(ubq_system.bonds, ubq_system.coords.shape[0], 6)[
        :natoms, :natoms
    ]

    def eval_intra(coords):

        i, v = op.intra(
            coords, torch.from_numpy(atom_type_idx), torch.from_numpy(atom_pair_bpl)
        )

        return v

    torch.autograd.gradcheck(
        eval_intra, (torch.from_numpy(coords).requires_grad_(True),), eps=1e-3
    )


def _dense_lj(coords, atom_type_idx, atom_pair_bpl, param_resolver):
    """Compute dense atom-atom lj score table via vectorized op."""
    atom_pair_d = numpy.linalg.norm(coords[None, :, :] - coords[:, None, :], axis=-1)
    atom_type_params = toolz.valmap(
        torch.Tensor.numpy, attr.asdict(param_resolver.type_params[atom_type_idx])
    )

    return ignore_unused_kwargs(lj)(
        atom_pair_d,
        atom_pair_bpl,
        **toolz.merge(
            {k + "_i": t[:, None] for k, t in atom_type_params.items()},
            {k + "_j": t[None, :] for k, t in atom_type_params.items()},
            toolz.valmap(float, attr.asdict(param_resolver.global_params)),
        ),
    )


def _dense_lk(coords, atom_type_idx, atom_pair_bpl, param_resolver):
    """Compute dense atom-atom lj score table via vectorized op."""
    atom_pair_d = numpy.linalg.norm(coords[None, :, :] - coords[:, None, :], axis=-1)
    atom_type_params = toolz.valmap(
        torch.Tensor.numpy, attr.asdict(param_resolver.type_params[atom_type_idx])
    )

    return ignore_unused_kwargs(lk_isotropic)(
        atom_pair_d,
        atom_pair_bpl,
        **toolz.merge(
            {k + "_i": t[:, None] for k, t in atom_type_params.items()},
            {k + "_j": t[None, :] for k, t in atom_type_params.items()},
            toolz.valmap(float, attr.asdict(param_resolver.global_params)),
        ),
    )
