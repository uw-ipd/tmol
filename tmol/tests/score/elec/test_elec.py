import toolz
import attr

import numpy
import torch
import sparse

from tmol.score.elec.torch_op import ElecScoreFun
from tmol.score.elec.params import ElecParamResolver
from tmol.score.bonded_atom import bonded_path_length

import tmol.database

from tmol.utility.args import ignore_unused_kwargs


def test_elec_sweep(default_database, torch_device):
    coords = numpy.zeros((2, 3))
    bpl = numpy.array([[0.0, 6.0], [6.0, 0.0]])
    pcs = numpy.array([1.0, 1.0])
    epr = ElecParamResolver.from_database(default_database.scoring.elec, torch_device)

    tcoords = torch.from_numpy(coords).to(device=torch_device).requires_grad_(True)
    tbpl = torch.from_numpy(bpl).to(device=torch_device)
    tpcs = torch.from_numpy(pcs).to(device=torch_device)

    print(pcs)

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

    scores = numpy.zeros(60)
    for i in range(60):
        tcoords[1, 2] = i / 10.0
        pairs, batch_scores, *batch_derivs = compiled.elec_triu(
            tcoords, tpcs, tcoords, tpcs, tbpl, **attr.asdict(epr.global_params)
        )
        scores[i] = batch_scores[1]
        print(scores[i])
    numpy.testing.assert_allclose(scores, scores_expected, atol=1e-6)


def test_elec_torch(default_database, ubq_system, torch_device):
    atom_names = ubq_system.atom_metadata["atom_name"].copy()
    res_names = ubq_system.atom_metadata["residue_name"].copy()
    res_indices = ubq_system.atom_metadata["residue_index"].copy()

    coords = ubq_system.coords.copy()
    bpl = bonded_path_length(ubq_system.bonds, ubq_system.coords.shape[0], 6)

    epr = ElecParamResolver.from_database(default_database.scoring.elec, torch_device)
    rbpl = epr.remap_bonded_path_lengths(bpl, res_names, res_indices, atom_names)
    pcs = epr.resolve_partial_charge(res_names, atom_names)

    tcoords = torch.from_numpy(coords).to(device=torch_device).requires_grad_(True)

    import tmol.score.elec.potentials.compiled as compiled

    pairs, batch_scores, *batch_derivs = compiled.elec_triu(
        tcoords, pcs, tcoords, pcs, rbpl, **attr.asdict(epr.global_params)
    )

    numpy.testing.assert_allclose(batch_scores.sum(), -131.9225, atol=1e-4)


def test_elec_deriv(default_database, ubq_system, torch_device):
    pass
