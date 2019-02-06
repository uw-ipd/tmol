import attr

import numpy
import torch

from tmol.score.elec.torch_op import ElecOp
from tmol.score.elec.params import ElecParamResolver
from tmol.score.bonded_atom import bonded_path_length


@attr.s(auto_attribs=True)
class ScoreSetup:
    param_resolver: ElecParamResolver
    tcoords: torch.Tensor
    trbpl: torch.Tensor
    tpcs: torch.Tensor

    @classmethod
    def from_fixture(cls, database, system, torch_device) -> "ScoreSetup":
        param_resolver = ElecParamResolver.from_database(
            database.scoring.elec, torch_device
        )

        atom_names = system.atom_metadata["atom_name"].copy()
        res_names = system.atom_metadata["residue_name"].copy()
        res_indices = system.atom_metadata["residue_index"].copy()
        atom_pair_bpl = bonded_path_length(system.bonds, system.coords.shape[0], 6)
        rbpl = param_resolver.remap_bonded_path_lengths(
            atom_pair_bpl[None, :],
            res_names[None, :],
            res_indices[None, :],
            atom_names[None, :],
        )
        pcs = param_resolver.resolve_partial_charge(
            res_names[None, :], atom_names[None, :]
        )

        tcoords = (
            torch.from_numpy(system.coords).to(torch_device).requires_grad_(True)
        )[None, :]
        trbpl = torch.from_numpy(rbpl).to(torch_device, tcoords.dtype)
        tpcs = torch.from_numpy(pcs).to(torch_device, tcoords.dtype)

        return cls(
            param_resolver=param_resolver, tcoords=tcoords, tpcs=tpcs, trbpl=trbpl
        )


# sweep fa_elec in 0.1A intervals from 0 to 6A
def test_elec_sweep(default_database, torch_device):
    coords = numpy.zeros((2, 3))
    bpl = numpy.array([[0.0, 6.0], [6.0, 0.0]])
    pcs = numpy.array([1.0, 1.0])

    tcoords = torch.from_numpy(coords).to(torch_device).requires_grad_(True)
    tbpl = torch.from_numpy(bpl).to(torch_device, tcoords.dtype)
    tpcs = torch.from_numpy(pcs).to(torch_device, tcoords.dtype)

    import tmol.score.elec.potentials.compiled as compiled

    scores_expected = numpy.array(
        [
            20.34806378,
            20.34806378,
            20.34806378,
            20.34806378,
            20.34806378,
            20.34806378,
            20.34806378,
            20.34806378,
            20.34806378,
            20.34806378,
            20.34806378,
            20.34806378,
            20.34806378,
            20.34806378,
            20.18762061,
            19.12568959,
            17.56817737,
            16.10605807,
            15.33030576,
            14.56187166,
            13.05769881,
            11.72195127,
            10.53389978,
            9.475596469,
            8.531431983,
            7.68779116,
            6.932776108,
            6.255977294,
            5.648280232,
            5.101699794,
            4.6092368,
            4.164753198,
            3.762863074,
            3.398837406,
            3.068520792,
            2.768258715,
            2.494834048,
            2.245411682,
            2.017490276,
            1.808860241,
            1.617567174,
            1.441880041,
            1.280263497,
            1.131353806,
            0.993937877,
            0.866935017,
            0.743738205,
            0.620451592,
            0.500150796,
            0.385911435,
            0.280809126,
            0.187919488,
            0.110318139,
            0.051080696,
            0.013282777,
            0,
            0,
            0,
            0,
            0,
        ]
    )

    min_dis, max_dis, D, D0, S = 1.6, 5.5, 79.931, 6.648, 0.441546
    scores = numpy.zeros(60)
    for i in range(60):
        tcoords[1, 2] = i / 10.0
        pairs, batch_scores, *batch_derivs = compiled.elec_triu(
            tcoords, tpcs, tcoords, tpcs, tbpl, D, D0, S, min_dis, max_dis
        )
        scores[i] = batch_scores[1]
    numpy.testing.assert_allclose(scores, scores_expected, atol=1e-4)


# sweep fa_elec in 0.1A intervals from 0 to 6A
# check numeric v analytic gradients
def test_elec_sweep_gradcheck(default_database, torch_device):
    coords = numpy.zeros((2, 3))
    bpl = numpy.array([[0.0, 6.0], [6.0, 0.0]])
    pcs = numpy.array([1.0, 1.0])

    tcoords = torch.from_numpy(coords).to(torch_device).requires_grad_(True)
    tbpl = torch.from_numpy(bpl).to(torch_device, tcoords.dtype)
    tpcs = torch.from_numpy(pcs).to(torch_device, tcoords.dtype)

    min_dis, max_dis, D, D0, S = 1.6, 5.5, 79.931, 6.648, 0.441546

    def eval_intra(coords):
        import tmol.score.elec.potentials.compiled as compiled

        pairs, scores, derivs_i, derivs_j = compiled.elec_triu(
            tcoords, tpcs, tcoords, tpcs, tbpl, D, D0, S, min_dis, max_dis
        )
        return (scores[1], derivs_i[1])

    dscores_A = numpy.zeros((60, 3))
    dscores_N = numpy.zeros((60, 3))
    for i in range(60):
        eps = 1e-5
        tcoords[1, 2] = i / 10.0
        _, dscores_A[i, :] = eval_intra(tcoords)
        for j in range(3):
            tcoords[0, j] = eps
            score_p, _ = eval_intra(tcoords)
            tcoords[0, j] = -eps
            score_m, _ = eval_intra(tcoords)
            tcoords[0, j] = 0
            dscores_N[i, j] = (score_p - score_m) / (2 * eps)

        print(dscores_A[i], dscores_N[i])

    numpy.testing.assert_allclose(dscores_A, dscores_N, atol=1e-5)


# torch forward op
def test_elec_intra(default_database, ubq_system, torch_device):
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)
    func = ElecOp.from_param_resolver(s.param_resolver)

    pairs, batch_scores, *batch_derivs = func.intra(s.tcoords[0], s.tpcs[0], s.trbpl[0])

    numpy.testing.assert_allclose(batch_scores.detach().sum(), -131.9225, atol=1e-4)


# torch gradcheck
def test_elec_intra_gradcheck(default_database, ubq_system, torch_device):
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)
    func = ElecOp.from_param_resolver(s.param_resolver)

    natoms = 32

    def eval_intra(coords):
        i, v = func.intra(coords, s.tpcs[0, :natoms], s.trbpl[0, :natoms, :natoms])
        return v

    coords = s.tcoords[0, :natoms]
    torch.autograd.gradcheck(eval_intra, (coords.requires_grad_(True),), eps=1e-4)
