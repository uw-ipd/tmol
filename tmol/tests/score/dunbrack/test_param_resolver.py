import numpy
import torch
import attr

import itertools

from tmol.score.dunbrack.params import DunbrackParamResolver


def test_dun_param_resolver_tables(default_database, torch_device):
    resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    dun_params = resolver.packed_db

    # rot tables 2d, semirot tables 3d
    assert dun_params.rotameric_bbsteps.shape[1] == 2
    assert dun_params.rotameric_bbstarts.shape[1] == 2
    assert dun_params.semirotameric_bbsteps.shape[1] == 3
    assert dun_params.semirotameric_bbstarts.shape[1] == 3

    nrottables = 0
    nsemirottables = 0
    for f in default_database.scoring.dun.rotameric_libraries:
        nchi = f.rotameric_data.nchi()
        nrot = f.rotameric_data.nrotamers()
        nrottables += nrot * (2 + 2 * nchi)  # prob/logprob/nchi*(mean/stdev)
    for f in default_database.scoring.dun.semi_rotameric_libraries:
        nrotchi = f.rotameric_data.nchi()
        nrot = f.rotameric_data.nrotamers()
        nrottables += 2 * nrot * nrotchi  # nchi*(mean/stdev)
        nsemirottables += 2 * nrot * nrotchi  # prob/logprob

    assert dun_params.rotameric_prob_tables.shape[0] == nrottables
    assert dun_params.semirotameric_prob_tables.shape[0] == nsemirottables

    ndunaas = len(default_database.scoring.dun.rotameric_libraries) + len(
        default_database.scoring.dun.semi_rotameric_libraries
    )

    # check device is correct
    for _, t in attr.asdict(dun_params).items():
        assert t.device == torch_device


def test_dun_param_resolver_construction(default_database, torch_device):

    resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )

    example_names = numpy.array(["ALA", "PHE", "ARG", "LEU", "GLY", "GLU", "MET"])

    phis = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7],
            [4, 5, 6, 7, 8],
            [5, 6, 7, 8, 9],
            [6, 7, 8, 9, 10],
        ],
        dtype=torch.int32,
        device=torch_device,
    )
    psis = torch.tensor(
        [
            [0, 2, 2, 3, 4],
            [1, 3, 3, 4, 5],
            [2, 4, 4, 5, 6],
            [3, 5, 5, 6, 7],
            [4, 6, 6, 7, 8],
            [5, 7, 7, 8, 9],
            [6, 8, 8, 9, 10],
        ],
        dtype=torch.int32,
        device=torch_device,
    )
    chi = torch.tensor(
        [
            [1, 0, 3, 5, 7, 9],
            [1, 1, 5, 7, 9, 11],
            [2, 0, 9, 11, 13, 15],
            [2, 1, 11, 13, 15, 17],
            [2, 2, 13, 15, 17, 19],
            [2, 3, 15, 17, 19, 21],
            [3, 0, 17, 19, 21, 23],
            [3, 1, 19, 21, 23, 25],
            [5, 0, 31, 33, 35, 37],
            [5, 1, 33, 35, 37, 39],
            [5, 2, 35, 36, 37, 39],
            [6, 0, 41, 42, 43, 44],
            [6, 1, 42, 43, 44, 45],
            [6, 2, 43, 44, 45, 46],
        ],
        dtype=torch.int32,
        device=torch_device,
    )

    dun_params = resolver.resolve_dunbrack_parameters(
        example_names, phis, psis, chi, torch_device
    )

    bb_indices_gold = torch.tensor(
        [
            [[2, 3, 4, 5], [3, 3, 4, 5]],
            [[3, 4, 5, 6], [4, 4, 5, 6]],
            [[4, 5, 6, 7], [5, 5, 6, 7]],
            [[6, 7, 8, 9], [7, 7, 8, 9]],
            [[7, 8, 9, 10], [8, 8, 9, 10]],
        ],
        dtype=torch.int32,
    )

    chi_indices_gold = torch.tensor(
        [
            [[3, 5, 7, 9], [5, 7, 9, 11], [-1, -1, -1, -1], [-1, -1, -1, -1]],
            [[9, 11, 13, 15], [11, 13, 15, 17], [13, 15, 17, 19], [15, 17, 19, 21]],
            [[17, 19, 21, 23], [19, 21, 23, 25], [-1, -1, -1, -1], [-1, -1, -1, -1]],
            [[31, 33, 35, 37], [33, 35, 37, 39], [35, 36, 37, 39], [-1, -1, -1, -1]],
            [[41, 42, 43, 44], [42, 43, 44, 45], [43, 44, 45, 46], [-1, -1, -1, -1]],
        ],
        dtype=torch.int32,
    )

    aa_indices_gold = torch.tensor([12, 6, 3, 11, 4], dtype=torch.int32)

    assert (bb_indices_gold == dun_params.bb_indices.cpu()).all()
    assert (chi_indices_gold == dun_params.chi_indices.cpu()).all()
    assert (aa_indices_gold == dun_params.aa_indices.cpu()).all()
