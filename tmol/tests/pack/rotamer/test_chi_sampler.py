import torch
import numpy

import tmol.pack.rotamer.compiled

from tmol.system.packed import PackedResidueSystem
from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.device import TorchDevice
from tmol.score.dunbrack.score_graph import DunbrackScoreGraph
from tmol.score.score_graph import score_graph
from tmol.score.dunbrack.params import DunbrackParamResolver

from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available


def get_compiled():
    compiled = load(
        modulename(__name__),
        cuda_if_available(
            relpaths(__file__, ["compiled.pybind.cpp", "test.cpp"])  #   "test.cu"??
        ),
    )
    return compiled


def test_sample_chi_for_rotamers_smoke(ubq_system, default_database, torch_device):
    # print("starting test sample chi for rotamers smoke")

    resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    dun_params = resolver.sampling_db
    dun_params_aux = resolver.scoring_db_aux

    @score_graph
    class CartDunbrackGraph(
        CartesianAtomicCoordinateProvider, DunbrackScoreGraph, TorchDevice
    ):
        pass

    dun_graph = CartDunbrackGraph.build_for(
        ubq_system, device=torch_device, parameter_database=default_database
    )

    # resolver = dun_graph.dun_resolve_indices
    # dun_params = resolver.sampling_db #

    def _ti32(l):
        return torch.tensor(l, dtype=torch.int32, device=torch_device)

    def _tf32(l):
        return torch.tensor(l, dtype=torch.float32, device=torch_device)

    # print(dun_graph.dun_phi)
    # print(dun_graph.dun_psi)

    ndihe_for_res = _ti32([2, 2, 2])
    dihedral_offset_for_res = _ti32([0, 2, 4])
    dihedral_atom_inds = _ti32(
        [
            [50, 72, 73, 74],  # res 3
            [72, 73, 74, 96],
            [322, 336, 337, 338],  # res 17
            [336, 337, 338, 352],
            [1010, 1016, 1017, 1018],  # res 53
            [1016, 1017, 1018, 1040],
        ]
    )

    rottable_set_for_buildable_restype = _ti32(
        [
            [0, 2],  # lys
            [0, 7],  # ser
            [1, 3],  # leu
            [1, 16],  # trp
            [1, 0],  # cys
            [2, 17],
        ]  # tyr
    )
    chi_expansion_for_buildable_restype = _ti32(
        [
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
        ]
    )
    nans_for_expansion = numpy.empty((6, 4, 18), dtype=float)
    nans_for_expansion[:] = numpy.nan
    nans_for_expansion[1, 1, :] = 20 * numpy.pi / 180 * numpy.arange(18, dtype=float)
    nans_for_expansion[4, 1, 0:3] = 120 * numpy.pi / 180 * numpy.arange(3, dtype=float)
    nans_for_expansion[5, 2, 0:2] = 180 * numpy.pi / 180 * numpy.arange(2, dtype=float)
    non_dunbrack_expansion_for_buildable_restype = _tf32(nans_for_expansion)
    non_dunbrack_expansion_counts_for_buildable_restype = _ti32(numpy.zeros((6, 4)))
    non_dunbrack_expansion_counts_for_buildable_restype[1, 1] = 18
    non_dunbrack_expansion_counts_for_buildable_restype[4, 1] = 3
    non_dunbrack_expansion_counts_for_buildable_restype[5, 2] = 2
    prob_cumsum_limit_for_buildable_restype = _tf32(numpy.full((6,), 0.95, dtype=float))
    nchi_for_buildable_restype = _ti32([4, 2, 2, 2, 2, 3])

    retval = torch.ops.tmol.dun_sample_chi(
        dun_graph.coords[0, :],
        dun_params.rotameric_prob_tables,
        dun_params.rotprob_table_sizes,
        dun_params.rotprob_table_strides,
        dun_params.rotameric_mean_tables,
        dun_params.rotameric_sdev_tables,
        dun_params.rotmean_table_sizes,
        dun_params.rotmean_table_strides,
        dun_params.rotameric_meansdev_tableset_offsets,
        dun_params.rotameric_bb_start,
        dun_params.rotameric_bb_step,
        dun_params.rotameric_bb_periodicity,
        dun_params.semirotameric_tables,
        dun_params.semirot_table_sizes,
        dun_params.semirot_table_strides,
        dun_params.semirot_start,
        dun_params.semirot_step,
        dun_params.semirot_periodicity,
        dun_params.rotameric_rotind2tableind,
        dun_params.semirotameric_rotind2tableind,
        dun_params.all_chi_rotind2tableind,
        dun_params.all_chi_rotind2tableind_offsets,
        dun_params.n_rotamers_for_tableset,
        dun_params.n_rotamers_for_tableset_offsets,
        dun_params.sorted_rotamer_2_rotamer,
        dun_params.nchi_for_table_set,
        dun_params.rotwells,
        ndihe_for_res,
        dihedral_offset_for_res,
        dihedral_atom_inds,
        rottable_set_for_buildable_restype,
        chi_expansion_for_buildable_restype,
        non_dunbrack_expansion_for_buildable_restype,
        non_dunbrack_expansion_counts_for_buildable_restype,
        prob_cumsum_limit_for_buildable_restype,
        nchi_for_buildable_restype,
    )

    # print(retval)
    n_rots_for_brt, n_rots_for_brt_offsets, brt_for_rotamer, chi_for_rotamers = retval
    assert n_rots_for_brt.shape == (6,)
    assert n_rots_for_brt_offsets.shape == (6,)
    assert brt_for_rotamer.shape == (826,)
    assert chi_for_rotamers.shape == (826, 4)


def test_determine_n_possible_rots(default_database, torch_device):
    compiled = get_compiled()
    resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    dun_params = resolver.sampling_db

    rottable_set_for_buildable_restype = torch.tensor(
        [
            [0, 2],  # lys
            [0, 7],  # ser
            [1, 3],  # leu
            [1, 16],  # trp
            [1, 0],  # cys
            [2, 17],
        ],  # tyr
        dtype=torch.int32,
        device=torch_device,
    )

    n_possible_rotamers_per_brt = torch.zeros(
        (6,), dtype=torch.int32, device=torch_device
    )

    compiled.determine_n_possible_rots(
        rottable_set_for_buildable_restype,
        dun_params.n_rotamers_for_tableset,
        n_possible_rotamers_per_brt,
    )

    n_poss_gold = dun_params.n_rotamers_for_tableset[
        rottable_set_for_buildable_restype[:, 1].to(torch.int64)
    ].to(torch.int32)
    # print(n_possible_rotamers_per_brt)
    # print(n_poss_gold)
    numpy.testing.assert_equal(
        n_poss_gold.cpu().numpy(), n_possible_rotamers_per_brt.cpu().numpy()
    )


def test_fill_in_brt_for_possrots(torch_device):
    compiled = get_compiled()

    def _ti32(l):
        return torch.tensor(l, dtype=torch.int32, device=torch_device)

    offsets = numpy.array([0, 5, 10, 15], dtype=numpy.int32)
    possible_rotamer_offset_for_brt = _ti32(offsets)
    brt_for_possrot = torch.zeros((20,), dtype=torch.int32, device=torch_device)
    brt_for_possrot_bndry = torch.zeros((20,), dtype=torch.int32, device=torch_device)
    compiled.fill_in_brt_for_possrots(
        possible_rotamer_offset_for_brt, brt_for_possrot, brt_for_possrot_bndry
    )

    brt_for_possrot_gold = numpy.array(
        [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5, dtype=numpy.int32
    )
    numpy.testing.assert_equal(brt_for_possrot_gold, brt_for_possrot)
    brt_for_possrot_bndry_gold = numpy.zeros((20,), dtype=numpy.int32)
    brt_for_possrot_bndry_gold[offsets] = 1
    numpy.testing.assert_equal(brt_for_possrot_bndry_gold, brt_for_possrot_bndry)


def test_interpolate_probabilities_for_possible_rotamers(
    default_database, torch_device
):
    # let's just test phe at a grid point

    compiled = get_compiled()
    resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    dun_params = resolver.sampling_db

    rottable_set_for_buildable_restype = torch.tensor(
        [[0, 12]], dtype=torch.int32, device=torch_device
    )
    brt_for_possible_rotamer = torch.zeros(
        (18,), dtype=torch.int32, device=torch_device
    )
    possible_rotamer_offset_for_brt = torch.zeros(
        (1,), dtype=torch.int32, device=torch_device
    )
    backbone_dihedrals = torch.tensor(
        numpy.pi / 180 * numpy.array([-110, 140]),
        dtype=torch.float32,
        device=torch_device,
    )
    rotamer_probability = torch.full(
        (18,), -1.0, dtype=torch.float32, device=torch_device
    )

    compiled.interpolate_probabilities_for_possible_rotamers(
        dun_params.rotameric_prob_tables,
        dun_params.rotprob_table_sizes,
        dun_params.rotprob_table_strides,
        dun_params.rotameric_bb_start,
        dun_params.rotameric_bb_step,
        dun_params.rotameric_bb_periodicity,
        dun_params.n_rotamers_for_tableset_offsets,
        dun_params.sorted_rotamer_2_rotamer,
        rottable_set_for_buildable_restype,
        brt_for_possible_rotamer,
        possible_rotamer_offset_for_brt,
        backbone_dihedrals,
        rotamer_probability,
    )

    # PHE -110 140 0 3 1 0 0 0.18027    -66.8 93 0 0 8.4 8.1 0 0
    # PHE -110 140 0 3 6 0 0 0.0823506  -66.8 71 0 0 8.4 6.8 0 0
    # PHE -110 140 0 3 2 0 0 0.116819   -66.8 119.6 0 0 8.4 8.1 0 0
    # PHE -110 140 0 2 1 0 0 0.165068    178.4 78.2 0 0 10.5 8 0 0
    # PHE -110 140 0 3 3 0 0 0.109428    -66.8 -26.6 0 0 8.4 8.7 0 0
    # PHE -110 140 0 2 2 0 0 0.0801794   178.4 101 0 0 10.5 6.7 0 0
    # PHE -110 140 0 3 4 0 0 0.0965739   -66.8 2.1 0 0 8.4 8.4 0 0
    # PHE -110 140 0 2 6 0 0 0.0376458   178.4 54.5 0 0 10.5 7 0 0
    # PHE -110 140 0 3 5 0 0 0.0861078   -66.8 35.2 0 0 8.4 9.3 0 0
    # PHE -110 140 0 1 1 0 0 0.0259214    62.7 90.6 0 0 8.6 7.3 0 0
    # PHE -110 140 0 2 5 0 0 0.00589813   178.4 24.3 0 0 10.5 7.1 0 0
    # PHE -110 140 0 2 3 0 0 0.00565283   178.4 130.3 0 0 10.5 6.8 0 0
    # PHE -110 140 0 1 6 0 0 0.00435724   62.7 69.6 0 0 8.6 5.8 0 0
    # PHE -110 140 0 1 2 0 0 0.00288873   62.7 111.3 0 0 8.6 5.2 0 0
    # PHE -110 140 0 2 4 0 0 0.000622143  178.4 -10.3 0 0 10.5 9.4 0 0
    # PHE -110 140 0 1 3 0 0 9.6065e-05   62.7 147 0 0 8.6 8.3 0 0
    # PHE -110 140 0 1 5 0 0 9.55346e-05 62.7 37.7 0 0 8.6 7.2 0 0
    # PHE -110 140 0 1 4 0 0 2.59811e-05 62.7 -1.2 0 0 8.6 8.8 0 0

    rotprob_gold = numpy.array(
        sorted(
            [
                0.18027,
                0.0823506,
                0.116819,
                0.165068,
                0.109428,
                0.0801794,
                0.0965739,
                0.0376458,
                0.0861078,
                0.0259214,
                0.00589813,
                0.00565283,
                0.00435724,
                0.00288873,
                0.000622143,
                9.6065e-05,
                9.55346e-05,
                2.59811e-05,
            ],
            reverse=True,
        )
    )
    numpy.testing.assert_allclose(
        rotprob_gold, rotamer_probability.cpu().numpy(), rtol=1e-5, atol=1e-5
    )


def test_determine_n_base_rotamers_to_build_1(torch_device):
    compiled = get_compiled()
    prob_cumsum_limit_for_buildable_restype = torch.tensor([0.95], device=torch_device)
    n_possible_rotamers_per_brt = torch.tensor(
        [18], dtype=torch.int32, device=torch_device
    )
    brt_for_possrot = torch.zeros((18,), dtype=torch.int32, device=torch_device)
    brt_for_possrot_bndry = torch.zeros((18,), dtype=torch.int32, device=torch_device)
    brt_for_possrot_bndry[0] = 1
    possrot_offset_for_brt = torch.zeros((1,), dtype=torch.int32, device=torch_device)

    rotprob_gold = numpy.array(
        sorted(
            [
                0.18027,
                0.0823506,
                0.116819,
                0.165068,
                0.109428,
                0.0801794,
                0.0965739,
                0.0376458,
                0.0861078,
                0.0259214,
                0.00589813,
                0.00565283,
                0.00435724,
                0.00288873,
                0.000622143,
                9.6065e-05,
                9.55346e-05,
                2.59811e-05,
            ],
            reverse=True,
        )
    )

    rotamer_probability = torch.tensor(
        rotprob_gold, dtype=torch.float32, device=torch_device
    )
    n_rotamers_to_build_per_brt = torch.zeros(
        (1,), dtype=torch.int32, device=torch_device
    )

    compiled.determine_n_base_rotamers_to_build(
        prob_cumsum_limit_for_buildable_restype,
        n_possible_rotamers_per_brt,
        brt_for_possrot,
        brt_for_possrot_bndry,
        possrot_offset_for_brt,
        rotamer_probability,
        n_rotamers_to_build_per_brt,
    )

    assert 9 == n_rotamers_to_build_per_brt[0]


def test_determine_n_base_rotamers_to_build_2(torch_device):
    compiled = get_compiled()
    prob_cumsum_limit_for_buildable_restype = torch.tensor(
        [0.95, 0.99], device=torch_device
    )
    n_possible_rotamers_per_brt = torch.tensor(
        [18, 18], dtype=torch.int32, device=torch_device
    )
    brt_for_possrot_a = torch.zeros((18,), dtype=torch.int32, device=torch_device)
    brt_for_possrot_b = torch.ones((18,), dtype=torch.int32, device=torch_device)
    brt_for_possrot = torch.cat((brt_for_possrot_a, brt_for_possrot_b))
    brt_for_possrot_bndry = torch.zeros((36,), dtype=torch.int32, device=torch_device)
    brt_for_possrot_bndry[0] = 1
    brt_for_possrot_bndry[18] = 1
    possrot_offset_for_brt = torch.zeros((2,), dtype=torch.int32, device=torch_device)
    possrot_offset_for_brt[1] = 18

    rotprob_gold = numpy.array(
        sorted(
            [
                0.18027,
                0.0823506,
                0.116819,
                0.165068,
                0.109428,
                0.0801794,
                0.0965739,
                0.0376458,
                0.0861078,
                0.0259214,
                0.00589813,
                0.00565283,
                0.00435724,
                0.00288873,
                0.000622143,
                9.6065e-05,
                9.55346e-05,
                2.59811e-05,
            ],
            reverse=True,
        )
    )

    rotamer_probability = torch.tensor(
        numpy.concatenate((rotprob_gold, rotprob_gold)),
        dtype=torch.float32,
        device=torch_device,
    )
    # rotamer probability cumsum:
    # 0 0 1 0.18027 2 0.345338 3 0.462157 4 0.571585
    # 5 0.668159 6 0.754267 7 0.836617 8 0.916797 9 0.954443
    # 10 0.980364 11 0.986262 12 0.991915 13 0.996272 14 0.999161
    # 15 0.999783 16 0.999879 17 0.999975 18 0 19 0.18027
    # 20 0.345338 21 0.462157 22 0.571585 23 0.668159 24 0.754267
    # 25 0.836617 26 0.916797 27 0.954443 28 0.980364 29 0.986262
    # 30 0.991915 31 0.996272 32 0.999161 33 0.999783 34 0.999879
    # 35 0.999975

    n_rotamers_to_build_per_brt = torch.zeros(
        (2,), dtype=torch.int32, device=torch_device
    )

    compiled.determine_n_base_rotamers_to_build(
        prob_cumsum_limit_for_buildable_restype,
        n_possible_rotamers_per_brt,
        brt_for_possrot,
        brt_for_possrot_bndry,
        possrot_offset_for_brt,
        rotamer_probability,
        n_rotamers_to_build_per_brt,
    )

    assert 9 == n_rotamers_to_build_per_brt[0]
    assert 12 == n_rotamers_to_build_per_brt[1]


def test_count_expanded_rotamers(default_database, torch_device):
    print("test count expanded rotamers", torch_device)
    compiled = get_compiled()
    resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    dun_params = resolver.sampling_db

    def _ti32(l):
        return torch.tensor(l, dtype=torch.int32, device=torch_device)

    nchi_for_buildable_restype = _ti32([4, 2, 2, 2, 2, 3])
    rottable_set_for_buildable_restype = _ti32(
        [
            [0, 2],  # lys
            [0, 7],  # ser
            [1, 3],  # leu
            [1, 16],  # trp
            [1, 0],  # cys
            [2, 17],
        ]  # tyr
    )
    print("rottable_set_for_buildable_restype")
    print(rottable_set_for_buildable_restype.shape)
    chi_expansion_for_buildable_restype = _ti32(
        [
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
        ]
    )
    non_dunbrack_expansion_counts_for_buildable_restype = _ti32(numpy.zeros((6, 4)))
    non_dunbrack_expansion_counts_for_buildable_restype[1, 1] = 18
    non_dunbrack_expansion_counts_for_buildable_restype[4, 1] = 3
    non_dunbrack_expansion_counts_for_buildable_restype[5, 2] = 2
    n_expansions_for_brt = _ti32([0] * 6)
    expansion_dim_prods_for_brt = _ti32([0] * 24).reshape((6, 4))
    n_rotamers_to_build_per_brt = _ti32([1] * 6)
    n_rotamers_to_build_per_brt_offsets = _ti32([0] * 6)

    print("nchi_for_buildable_restype.shape")
    print(nchi_for_buildable_restype.shape)
    print("rottable_set_for_buildable_restype.shape")
    print(rottable_set_for_buildable_restype.shape)
    print("dun_params.nchi_for_table_set.shape")
    print(dun_params.nchi_for_table_set.shape)
    print("chi_expansion_for_buildable_restype.shape")
    print(chi_expansion_for_buildable_restype.shape)
    print("non_dunbrack_expansion_counts_for_buildable_restype.shape")
    print(non_dunbrack_expansion_counts_for_buildable_restype.shape)
    print("n_expansions_for_brt.shape")
    print(n_expansions_for_brt.shape)
    print("expansion_dim_prods_for_brt.shape")
    print(expansion_dim_prods_for_brt.shape)
    print("n_rotamers_to_build_per_brt.shape")
    print(n_rotamers_to_build_per_brt.shape)
    print("n_rotamers_to_build_per_brt_offsets.shape")
    print(n_rotamers_to_build_per_brt_offsets.shape)

    nrots = compiled.count_expanded_rotamers(
        nchi_for_buildable_restype,
        rottable_set_for_buildable_restype,
        dun_params.nchi_for_table_set,
        chi_expansion_for_buildable_restype,
        non_dunbrack_expansion_counts_for_buildable_restype,
        n_expansions_for_brt,
        expansion_dim_prods_for_brt,
        n_rotamers_to_build_per_brt,
        n_rotamers_to_build_per_brt_offsets,
    )
    print("n_expansions_for_brt")
    print(n_expansions_for_brt)
    print("expansion_dim_prods_for_brt")
    print(expansion_dim_prods_for_brt)
    print("n_rotamers_to_build_per_brt")
    print(n_rotamers_to_build_per_brt)
    print("n_rotamers_to_build_per_brt_offsets")
    print(n_rotamers_to_build_per_brt_offsets)
    print("nrots", nrots)
