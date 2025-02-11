import torch
import numpy

# import attr
import cattr

import tmol.pack.rotamer.dunbrack.compiled  # noqa F401

from tmol.utility.tensor.common_operations import exclusive_cumsum1d

from tmol.chemical.restypes import RefinedResidueType, ResidueTypeSet
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack_builder import PoseStackBuilder

from tmol.io import pose_stack_from_pdb
from tmol.score.dunbrack.params import DunbrackParamResolver
from tmol.pack.packer_task import PackerTask, PackerPalette
from tmol.pack.rotamer.dunbrack.dunbrack_chi_sampler import DunbrackChiSampler

from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available
from tmol.tests.data import no_termini_pose_stack_from_pdb


def get_compiled():
    compiled = load(
        modulename(__name__),
        cuda_if_available(
            relpaths(__file__, ["compiled.pybind.cpp", "test.cpp", "test.cu"])
        ),
    )
    return compiled


def test_annotate_residue_type(default_database):
    torch_device = torch.device("cpu")
    param_resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )

    tyr_restype = cattr.structure(
        cattr.unstructure(
            next(res for res in default_database.chemical.residues if res.name == "TYR")
        ),
        RefinedResidueType,
    )

    sampler = DunbrackChiSampler.from_database(param_resolver)
    sampler.annotate_residue_type(tyr_restype)

    assert hasattr(tyr_restype, "dun_sampler_cache")

    assert isinstance(tyr_restype.dun_sampler_cache.bbdihe_uaids, numpy.ndarray)
    assert tyr_restype.dun_sampler_cache.bbdihe_uaids.shape == (2, 4, 3)
    assert tyr_restype.dun_sampler_cache.bbdihe_uaids[0, 0, 0] == -1
    assert tyr_restype.dun_sampler_cache.bbdihe_uaids[0, 0, 1] == 0
    assert tyr_restype.dun_sampler_cache.bbdihe_uaids[0, 0, 2] == 0
    assert tyr_restype.dun_sampler_cache.bbdihe_uaids[1, 3, 0] == -1
    assert tyr_restype.dun_sampler_cache.bbdihe_uaids[1, 3, 1] == 1
    assert tyr_restype.dun_sampler_cache.bbdihe_uaids[1, 3, 2] == 0

    assert isinstance(tyr_restype.dun_sampler_cache.chi_defining_atom, numpy.ndarray)
    assert tyr_restype.dun_sampler_cache.chi_defining_atom.shape == (3,)
    tyr_restype.dun_sampler_cache.chi_defining_atom[0] == tyr_restype.atom_to_idx["CA"]
    tyr_restype.dun_sampler_cache.chi_defining_atom[1] == tyr_restype.atom_to_idx["CB"]
    tyr_restype.dun_sampler_cache.chi_defining_atom[2] == tyr_restype.atom_to_idx["CZ"]


def test_annotate_packed_block_types(default_database, torch_device):
    # torch_device = torch.device("cpu")
    param_resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )

    desired = set(["ALA", "CYS", "ASP", "GLU", "PHE", "HIS", "ARG", "SER", "TYR"])

    all_restypes = [
        cattr.structure(cattr.unstructure(res), RefinedResidueType)
        for res in default_database.chemical.residues
        if res.name in desired
    ]
    restype_set = ResidueTypeSet.from_restype_list(
        default_database.chemical, all_restypes
    )

    sampler = DunbrackChiSampler.from_database(param_resolver)
    for restype in all_restypes:
        sampler.annotate_residue_type(restype)

    pbt = PackedBlockTypes.from_restype_list(
        default_database.chemical, restype_set, all_restypes, torch_device
    )
    sampler.annotate_packed_block_types(pbt)

    assert hasattr(pbt, "dun_sampler_cache")

    assert isinstance(pbt.dun_sampler_cache.bbdihe_uaids, torch.Tensor)
    assert pbt.dun_sampler_cache.bbdihe_uaids.shape == (pbt.n_types, 2, 4, 3)
    assert pbt.dun_sampler_cache.bbdihe_uaids.device == torch_device

    assert isinstance(pbt.dun_sampler_cache.chi_defining_atom, torch.Tensor)
    assert pbt.dun_sampler_cache.chi_defining_atom.shape == (pbt.n_types, 4)
    assert pbt.dun_sampler_cache.chi_defining_atom.device == torch_device


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

    def _ti32(the_list):
        return torch.tensor(the_list, dtype=torch.int32, device=torch_device)

    offsets = numpy.array([0, 5, 10, 15], dtype=numpy.int32)
    possible_rotamer_offset_for_brt = _ti32(offsets)
    brt_for_possrot = torch.zeros((20,), dtype=torch.int32, device=torch_device)
    compiled.fill_in_brt_for_possrots(possible_rotamer_offset_for_brt, brt_for_possrot)

    brt_for_possrot_gold = numpy.array(
        [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5, dtype=numpy.int32
    )
    numpy.testing.assert_equal(brt_for_possrot_gold, brt_for_possrot.cpu())


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
        possrot_offset_for_brt,
        rotamer_probability,
        n_rotamers_to_build_per_brt,
    )

    assert 9 == n_rotamers_to_build_per_brt[0]
    assert 12 == n_rotamers_to_build_per_brt[1]


def test_count_expanded_rotamers(default_database, torch_device):
    compiled = get_compiled()
    resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    dun_params = resolver.sampling_db

    def _ti32(the_list):
        return torch.tensor(the_list, dtype=torch.int32, device=torch_device)

    nchi_for_buildable_restype = _ti32([4, 2, 2, 2, 2, 3])
    rottable_set_for_buildable_restype = _ti32(
        [
            [0, 2],  # lys
            [0, 7],  # ser
            [1, 3],  # leu
            [1, 16],  # trp
            [1, 0],  # cys
            [2, 17],  # tyr
        ]
    )
    chi_expansion_for_buildable_restype = _ti32(
        [
            [1, 1, 0, 0],  # 9      9
            [1, 0, 0, 0],  # 3*18  54
            [0, 0, 0, 0],  # 1      1
            [1, 1, 0, 0],  # 9      9
            [1, 0, 0, 0],  # 1*3    9
            [1, 1, 0, 0],  # 9*2   18
        ]
    )
    non_dunbrack_expansion_counts_for_buildable_restype = _ti32(numpy.zeros((6, 4)))
    non_dunbrack_expansion_counts_for_buildable_restype[1, 1] = 18
    non_dunbrack_expansion_counts_for_buildable_restype[4, 1] = 3
    non_dunbrack_expansion_counts_for_buildable_restype[5, 2] = 2
    n_expansions_for_brt = _ti32([0] * 6)
    expansion_dim_prods_for_brt = _ti32([0] * 24).reshape((6, 4))
    n_rotamers_to_build_per_brt = _ti32([2] * 6)
    n_rotamers_to_build_per_brt_offsets = _ti32([0] * 6)

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
    assert 200 == nrots
    n_expansions_for_brt_gold = numpy.array([9, 54, 1, 9, 9, 18], dtype=numpy.int32)
    expansion_dim_prods_for_brt_gold = numpy.array(
        [
            [3, 1, 1, 1],
            [18, 1, 0, 0],
            [1, 1, 0, 0],
            [3, 1, 0, 0],
            [3, 1, 0, 0],
            [6, 2, 1, 0],
        ],
        dtype=numpy.int32,
    )
    n_rotamers_to_build_per_brt_gold = numpy.array(
        [18, 108, 2, 18, 18, 36], dtype=numpy.int32
    )
    n_rotamers_to_build_per_brt_offsets_gold = numpy.array(
        [0, 18, 126, 128, 146, 164], dtype=numpy.int32
    )

    numpy.testing.assert_equal(n_expansions_for_brt_gold, n_expansions_for_brt.cpu())
    numpy.testing.assert_equal(
        expansion_dim_prods_for_brt_gold, expansion_dim_prods_for_brt.cpu()
    )
    numpy.testing.assert_equal(
        n_rotamers_to_build_per_brt_gold, n_rotamers_to_build_per_brt.cpu()
    )
    numpy.testing.assert_equal(
        n_rotamers_to_build_per_brt_offsets_gold,
        n_rotamers_to_build_per_brt_offsets.cpu(),
    )


def test_map_from_rotamer_index_to_brt(torch_device):
    compiled = get_compiled()
    n_rotamers_to_build_per_brt_offsets = torch.tensor(
        [0, 18, 126, 128, 146, 164], dtype=torch.int32, device=torch_device
    )
    brt_for_rotamer = torch.zeros((200,), dtype=torch.int32, device=torch_device)

    compiled.map_from_rotamer_index_to_brt(
        n_rotamers_to_build_per_brt_offsets, brt_for_rotamer
    )

    brt_for_rotamer_gold = numpy.array(
        [0] * 18 + [1] * 108 + [2] * 2 + [3] * 18 + [4] * 18 + [5] * 36,
        dtype=numpy.int32,
    )
    numpy.testing.assert_equal(brt_for_rotamer_gold, brt_for_rotamer.cpu())


def test_sample_chi_for_rotamers(default_database, torch_device):
    def _ti32(the_list):
        return torch.tensor(the_list, dtype=torch.int32, device=torch_device)

    compiled = get_compiled()
    resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    dun_params = resolver.sampling_db

    # look, we're going to use phe but treat it like tyr.
    # why? why not just use tyr? Because I already have all the numbers
    # for phe, and I don't have the numbers for tyr and because the
    # code does not care at all that we are adding an extra chi to phe.
    # The code says "you want phe with an extra chi, you got it" and
    # maybe that's becuse you're modeling a weird phe + phosphate group
    # or something.
    rottable_set_for_buildable_restype = _ti32([[0, 12]])
    chi_expansion_for_buildable_restype = _ti32([[1, 1, 0, 0]])
    # non_dunbrack_expansion_counts_for_buildable_restype = _ti32(numpy.zeros((1, 4)))
    # non_dunbrack_expansion_counts_for_buildable_restype[0, 2] = 2
    non_dunbrack_expansion_for_buildable_restype = torch.tensor(
        numpy.array([[[numpy.nan, numpy.nan], [numpy.nan, numpy.nan], [0.25, 1.25]]]),
        dtype=torch.float32,
        device=torch_device,
    )
    nchi_for_buildable_restype = _ti32([3])

    # sample on a grid point so we can read off the correct answers
    # from the database file itself
    backbone_dihedrals = torch.tensor(
        numpy.pi / 180 * numpy.array([-110, 140]),
        dtype=torch.float32,
        device=torch_device,
    )

    n_rotamers_to_build_per_brt_offsets = _ti32([0])
    brt_for_rotamer = _ti32([0] * 18 * 9)
    n_expansions_for_brt = _ti32([18])
    expansion_dim_prods_for_brt = _ti32([[6, 2, 1]])
    chi_for_rotamers = torch.zeros(
        (18 * 9, 3), dtype=torch.float32, device=torch_device
    )

    compiled.sample_chi_for_rotamers(
        dun_params.rotameric_mean_tables,
        dun_params.rotameric_sdev_tables,
        dun_params.rotmean_table_sizes,
        dun_params.rotmean_table_strides,
        dun_params.rotameric_meansdev_tableset_offsets,
        dun_params.rotameric_bb_start,
        dun_params.rotameric_bb_step,
        dun_params.rotameric_bb_periodicity,
        dun_params.sorted_rotamer_2_rotamer,
        dun_params.nchi_for_table_set,
        rottable_set_for_buildable_restype,
        chi_expansion_for_buildable_restype,
        non_dunbrack_expansion_for_buildable_restype,
        nchi_for_buildable_restype,
        backbone_dihedrals,
        n_rotamers_to_build_per_brt_offsets,
        brt_for_rotamer,
        n_expansions_for_brt,
        dun_params.n_rotamers_for_tableset_offsets,
        expansion_dim_prods_for_brt,
        chi_for_rotamers,
    )

    # PHE -110 140 0 3 1 0 0 0.18027    -66.8   93   0 0 8.4  8.1 0 0
    # PHE -110 140 0 2 1 0 0 0.165068   178.4   78.2 0 0 10.5 8   0 0
    # PHE -110 140 0 3 2 0 0 0.116819   -66.8  119.6 0 0 8.4  8.1 0 0
    # PHE -110 140 0 3 3 0 0 0.109428   -66.8  -26.6 0 0 8.4  8.7 0 0
    # PHE -110 140 0 3 4 0 0 0.0965739  -66.8    2.1 0 0 8.4  8.4 0 0
    # PHE -110 140 0 3 5 0 0 0.0861078  -66.8   35.2 0 0 8.4  9.3 0 0
    # PHE -110 140 0 3 6 0 0 0.0823506  -66.8   71   0 0 8.4  6.8 0 0
    # PHE -110 140 0 2 2 0 0 0.0801794  178.4  101   0 0 10.5 6.7 0 0
    # PHE -110 140 0 2 6 0 0 0.0376458  178.4  54.5  0 0 10.5 7   0 0

    chi1_means = numpy.array(
        [-66.8, 178.4, -66.8, -66.8, -66.8, -66.8, -66.8, 178.4, 178.4]
    )
    chi2_means = numpy.array([93, 78.2, 119.6, -26.6, 2.1, 35.2, 71, 101, 54.5])
    chi1_sdevs = numpy.array([8.4, 10.5, 8.4, 8.4, 8.4, 8.4, 8.4, 10.5, 10.5])
    chi2_sdevs = numpy.array([8.1, 8, 8.1, 8.7, 8.4, 9.3, 6.8, 6.7, 7])
    chi_for_rotamers_gold = numpy.zeros((18 * 9, 3), dtype=numpy.float32)
    ar162 = numpy.arange(162, dtype=int)
    chi_for_rotamers_gold[:, 0] = chi1_means[ar162 // 18]
    chi_for_rotamers_gold[:, 1] = chi2_means[ar162 // 18]
    chi1sd_minus = ar162 % 18 < 6
    chi1sd_plus = ar162 % 18 >= 12
    chi_for_rotamers_gold[chi1sd_minus, 0] -= chi1_sdevs[ar162 // 18][chi1sd_minus]
    chi_for_rotamers_gold[chi1sd_plus, 0] += chi1_sdevs[ar162 // 18][chi1sd_plus]
    chi2sd_minus = ar162 % 6 < 2
    chi2sd_plus = ar162 % 6 >= 4
    chi_for_rotamers_gold[chi2sd_minus, 1] -= chi2_sdevs[ar162 // 18][chi2sd_minus]
    chi_for_rotamers_gold[chi2sd_plus, 1] += chi2_sdevs[ar162 // 18][chi2sd_plus]
    chi_for_rotamers_gold *= numpy.pi / 180

    chi_for_rotamers_gold[ar162 % 2 == 0, 2] = 0.25
    chi_for_rotamers_gold[ar162 % 2 == 1, 2] = 1.25

    numpy.testing.assert_allclose(
        chi_for_rotamers_gold, chi_for_rotamers.cpu(), atol=1e-5, rtol=1e-5
    )


# def test_chi_sampler_smoke(ubq_system, default_database, torch_device):
#     # print("ubq system:", len(ubq_system.residues))
#     # torch_device = torch.device("cpu")
#     palette = PackerPalette(default_database.chemical)
#     task = PackerTask(ubq_system, palette)
#     # print("task size:", len(task.rlts))
#     for rlt in task.rlts:
#         rlt.restrict_to_repacking()
#
#     param_resolver = DunbrackParamResolver.from_database(
#         default_database.scoring.dun, torch_device
#     )
#     sampler = ChiSampler.from_database(param_resolver)
#
#     coords = torch.tensor(ubq_system.coords, dtype=torch.float32, device=torch_device)
#     result = sampler.chi_samples_for_residues(ubq_system, coords, task)
#     n_rots_for_brt, n_rots_for_brt_offsets, brt_for_rotamer, chi_for_rotamers = result
#     # print("n_rots_for_brt")
#     # print(n_rots_for_brt.shape)
#     # print("n_rots_for_brt_offsets")
#     # print(n_rots_for_brt_offsets.shape)
#     # print("brt_for_rotamer")
#     # print(brt_for_rotamer.shape)
#     # print("chi_for_rotamers")
#     # print(chi_for_rotamers.shape)
#
#     assert n_rots_for_brt.shape == (69,)
#     assert n_rots_for_brt_offsets.shape == n_rots_for_brt.shape
#     assert brt_for_rotamer.shape == (1524,)
#     assert brt_for_rotamer.shape[0] == chi_for_rotamers.shape[0]
#     assert chi_for_rotamers.shape[1] == 4


def test_package_samples_for_output(default_database, ubq_pdb, torch_device):
    # torch_device = torch.device("cpu")
    param_resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    dun_sampler = DunbrackChiSampler.from_database(param_resolver)

    rts = ResidueTypeSet.from_database(default_database.chemical)

    # replace them with residues constructed from the residue types
    # that live in our locally constructed set of refined residue types
    # ubq_res = [
    #     attr.evolve(
    #         res,
    #         residue_type=next(
    #             rt for rt in rts.residue_types if rt.name == res.residue_type.name
    #         ),
    #     )
    #     for res in ubq_res
    # ]

    # p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
    #     default_database.chemical, ubq_res[5:11], torch_device
    # )
    # p2 = PoseStackBuilder.one_structure_from_polymeric_residues(
    #     default_database.chemical, ubq_res[:7], torch_device
    # )
    p1 = no_termini_pose_stack_from_pdb(
        ubq_pdb, torch_device, residue_start=5, residue_end=11
    )
    p2 = no_termini_pose_stack_from_pdb(
        ubq_pdb, torch_device, residue_start=1, residue_end=8
    )
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)
    pbt = poses.packed_block_types

    palette = PackerPalette(rts)
    task = PackerTask(poses, palette)
    task.add_chi_sampler(dun_sampler)
    task.restrict_to_repacking()

    all_allowed_restypes = numpy.array(
        [
            rt
            for one_pose_rlts in task.rlts
            for rlt in one_pose_rlts
            for rt in rlt.allowed_restypes
        ],
        dtype=object,
    )
    dun_restype_allowed = numpy.array(
        [
            1 if rt.name != "ALA" and rt.name != "GLY" else 0
            for rt in all_allowed_restypes
        ],
        dtype=int,
    )
    rt_names = numpy.array([rt.name for rt in all_allowed_restypes], dtype=object)
    rt_base_names = numpy.array(
        [rt.name.partition(":")[0] for rt in all_allowed_restypes], dtype=object
    )

    dun_rot_inds_for_rts = param_resolver._indices_from_names(
        param_resolver.all_table_indices, rt_base_names[None, :], torch.device("cpu")
    ).squeeze()
    block_type_ind_for_brt = torch.tensor(
        pbt.restype_index.get_indexer(
            rt_names[dun_rot_inds_for_rts.cpu().numpy() != -1]
        ),
        dtype=torch.int64,
        device=torch_device,
    )

    nonzero_dunrot_inds_for_rts = torch.nonzero(dun_rot_inds_for_rts != -1)

    # let the sampler annotate the residue types and the PBT
    for rt in poses.packed_block_types.active_block_types:
        dun_sampler.annotate_residue_type(rt)
    dun_sampler.annotate_packed_block_types(poses.packed_block_types)

    n_rots_for_brt = torch.arange(12, dtype=torch.int32, device=torch_device) + 1
    offsets_for_brt = exclusive_cumsum1d(n_rots_for_brt)
    n_rots = torch.sum(n_rots_for_brt).item()
    brt_for_rotamer = torch.zeros((n_rots,), dtype=torch.int32, device=torch_device)
    brt_for_rotamer[offsets_for_brt.to(torch.int64)] = 1
    brt_for_rotamer[0] = 0
    brt_for_rotamer = torch.cumsum(brt_for_rotamer, dim=0)
    chi_for_rotamers = torch.zeros(
        (n_rots, 4), dtype=torch.float32, device=torch_device
    )

    sampled_chi = (n_rots_for_brt, offsets_for_brt, brt_for_rotamer, chi_for_rotamers)

    results = dun_sampler.package_samples_for_output(
        pbt, task, block_type_ind_for_brt, 4, nonzero_dunrot_inds_for_rts, sampled_chi
    )

    n_rots_for_rt_gold = numpy.zeros(all_allowed_restypes.shape[0], dtype=numpy.int32)
    n_rots_for_rt_gold[dun_restype_allowed != 0] = n_rots_for_brt.cpu().numpy()

    # rot_offset_for_rt_gold = numpy.zeros_like(n_rots_for_rt_gold)
    # rot_offset_for_rt_gold[dun_restype_allowed != 0] = offsets_for_brt.cpu().numpy()

    rt_for_rot_gold = numpy.zeros((n_rots,), dtype=numpy.int32)
    rt_for_rot_gold[offsets_for_brt.cpu()] = 1
    rt_for_rot_gold[0] = 0
    rt_for_rot_gold[offsets_for_brt[4].cpu()] += 1
    rt_for_rot_gold = numpy.cumsum(rt_for_rot_gold)

    assert results[0].device == torch_device
    assert results[1].device == torch_device
    assert results[2].device == torch_device
    assert results[3].device == torch_device

    numpy.testing.assert_equal(n_rots_for_rt_gold, results[0].cpu().numpy())
    numpy.testing.assert_equal(rt_for_rot_gold, results[1].cpu().numpy())


def test_chi_sampler_smoke(ubq_pdb, default_database, default_restype_set):
    torch_device = torch.device("cpu")
    # p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
    #     default_database.chemical, ubq_res[:5], torch_device
    # )
    # p2 = PoseStackBuilder.one_structure_from_polymeric_residues(
    #     default_database.chemical, ubq_res[:7], torch_device
    # )
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=5)
    p2 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=7)
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)
    palette = PackerPalette(default_restype_set)
    task = PackerTask(poses, palette)
    task.restrict_to_repacking()

    param_resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    sampler = DunbrackChiSampler.from_database(param_resolver)
    task.add_chi_sampler(sampler)

    for rt in poses.packed_block_types.active_block_types:
        sampler.annotate_residue_type(rt)
    sampler.annotate_packed_block_types(poses.packed_block_types)
    sampler.sample_chi_for_poses(poses, task)


def test_chi_sampler_build_lots_of_rotamers(
    ubq_pdb, default_database, default_restype_set, torch_device
):
    # torch_device = torch.device("cpu")
    n_poses = 10
    # print([res.residue_type.name for res in ubq_res[:10]])
    # p = PoseStackBuilder.one_structure_from_polymeric_residues(
    #     default_database.chemical, ubq_res[:10], torch_device
    # )
    p = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=10)
    poses = PoseStackBuilder.from_poses([p] * n_poses, torch_device)
    palette = PackerPalette(default_restype_set)
    task = PackerTask(poses, palette)
    task.restrict_to_repacking()

    param_resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    sampler = DunbrackChiSampler.from_database(param_resolver)
    task.add_chi_sampler(sampler)

    for rt in poses.packed_block_types.active_block_types:
        sampler.annotate_residue_type(rt)
    sampler.annotate_packed_block_types(poses.packed_block_types)
    chi_samples = sampler.sample_chi_for_poses(poses, task)

    n_rots_for_rt, rt_for_rotamer, chi_defining_atom, chi = chi_samples

    # print("rt_for_rotamer")
    # print(rt_for_rotamer)

    n_rots = chi_defining_atom.shape[0]
    n_rots_per_pose = n_rots // n_poses
    assert n_rots_per_pose * n_poses == n_rots
