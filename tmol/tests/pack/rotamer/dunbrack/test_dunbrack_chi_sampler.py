import cattr
import numpy
import torch

import tmol.pack.rotamer.dunbrack.compiled  # noqa F401
from tmol.chemical.restypes import RefinedResidueType, ResidueTypeSet
from tmol.io import pose_stack_from_pdb
from tmol.pack.packer_task import PackerPalette, PackerTask, SetPackerTask
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.score.dunbrack.params import DunbrackParamResolver
from tmol.pack.rotamer.dunbrack.dunbrack_chi_sampler import (
    create_dunbrack_sampler_from_database,
)
from tmol.tests.data import no_termini_pose_stack_from_pdb
from tmol.utility.tensor.common_operations import exclusive_cumsum1d


def get_compiled():
    from tmol._load_ext import load_module

    return load_module(
        __name__,
        __file__,
        ["compiled.pybind.cpp", "test_cpu.cpp", "test_cuda.cu"],
        "tmol.tests.pack.rotamer.dunbrack._ext",
    )


def test_annotate_residue_type(default_database):
    torch_device = torch.device("cpu")

    tyr_restype = cattr.structure(
        cattr.unstructure(
            next(res for res in default_database.chemical.residues if res.name == "TYR")
        ),
        RefinedResidueType,
    )

    sampler = create_dunbrack_sampler_from_database(default_database, torch_device)
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
    # chi_defining_atom stores the 3rd atom of each chi dihedral:
    # chi1: N-CA-CB-CG  →  CB
    # chi2: CA-CB-CG-CD1  →  CG
    # chi2.5: ...-CZ-OH  →  OH
    assert (
        tyr_restype.dun_sampler_cache.chi_defining_atom[0]
        == tyr_restype.atom_to_idx["CB"]
    )
    assert (
        tyr_restype.dun_sampler_cache.chi_defining_atom[1]
        == tyr_restype.atom_to_idx["CG"]
    )
    assert (
        tyr_restype.dun_sampler_cache.chi_defining_atom[2]
        == tyr_restype.atom_to_idx["OH"]
    )


def test_annotate_packed_block_types(default_database, torch_device):
    # torch_device = torch.device("cpu")
    desired = {"ALA", "CYS", "ASP", "GLU", "PHE", "HIS", "ARG", "SER", "TYR"}

    all_restypes = [
        cattr.structure(cattr.unstructure(res), RefinedResidueType)
        for res in default_database.chemical.residues
        if res.name in desired
    ]
    restype_set = ResidueTypeSet.from_restype_list(
        default_database.chemical, all_restypes
    )

    sampler = create_dunbrack_sampler_from_database(default_database, torch_device)
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

    assert n_rotamers_to_build_per_brt[0] == 9


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

    assert n_rotamers_to_build_per_brt[0] == 9
    assert n_rotamers_to_build_per_brt[1] == 12


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
    assert nrots == 200
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


def test_package_samples_for_output(default_database, ubq_pdb, torch_device):
    dun_sampler = create_dunbrack_sampler_from_database(default_database, torch_device)

    p1 = no_termini_pose_stack_from_pdb(
        ubq_pdb, torch_device, residue_start=5, residue_end=11
    )
    p2 = no_termini_pose_stack_from_pdb(
        ubq_pdb, torch_device, residue_start=1, residue_end=8
    )
    pose_stack = PoseStackBuilder.from_poses([p1, p2], torch_device)
    pbt = pose_stack.packed_block_types

    palette = PackerPalette()
    task = PackerTask(pose_stack, palette)
    task.add_conformer_sampler(dun_sampler)
    task.restrict_to_repacking()
    task = SetPackerTask.from_packer_task(task)

    ################ contents of sample_chi_for_poses

    for rt in pose_stack.packed_block_types.active_block_types:
        dun_sampler.annotate_residue_type(rt)
    dun_sampler.annotate_packed_block_types(pose_stack.packed_block_types)

    # pbt = pose_stack.packed_block_types
    self_ind_in_packer_task = task.conformer_sampler_index[id(dun_sampler)]

    # the subset of blocktypes which are allowed at the positions and
    # for which the block-level tasks include this DunbrackChiSampler
    # the "Dunbrack allowed" restypes
    is_dun_allowed_gbt = torch.logical_and(
        task.per_block_conformer_sampler_allowed[
            task.cons_bt_pose, task.cons_bt_block, self_ind_in_packer_task
        ],
        task.is_cons_bt_allowed,
    )
    dun_allowed_bt_to_gbt = torch.nonzero(is_dun_allowed_gbt, as_tuple=True)[0]

    n_gbt_total = task.cons_bt_pose.shape[0]
    # n_dun_allowed_bt = dun_allowed_bt_to_gbt.shape[0]

    # removing: dun_allowed_bt_to_gbt is equivalent to dun_allowed_gbt,
    # dun_allowed_bt_to_gbt = numpy.arange(n_gbt_total, dtype=numpy.int64)[
    #     is_gbt_dun_allowed
    # ]
    # dun_allowed_bt_to_gbt_torch = torch.tensor(
    #     dun_allowed_bt_to_gbt, device=self.device
    # )

    # OLD dun_allowed_bt_names = numpy.array(
    # OLD     [bt.name for bt in dun_allowed_blocktypes], dtype=object
    # OLD )
    # OLD dun_allowed_bt_base_names = numpy.array(
    # OLD     [bt.name.partition(":")[0] for bt in dun_allowed_blocktypes], dtype=object
    # OLD )
    pbt = pose_stack.packed_block_types

    # the source block for each dun-allowed block type
    # OLD dun_allowed_bt_block = torch.tensor(
    # OLD     [
    # OLD         i * max_n_blocks + j
    # OLD         for i, one_pose_blts in enumerate(task.blts)
    # OLD         for j, blt in enumerate(one_pose_blts)
    # OLD         for k, _ in enumerate(blt.considered_block_types)
    # OLD         if blt.block_type_allowed[k] and self in blt.conformer_samplers
    # OLD     ],
    # OLD     dtype=torch.int32,
    # OLD     device=self.device,
    # OLD )
    dun_allowed_bt_block = task.global_block_ind_for_considered_block_types[
        dun_allowed_bt_to_gbt
    ]

    # the dunbrack-assigned table index for each dun-allowed block type;
    # -1 if the block type is not built by any dunbrack table;
    # TO DO: if the BLT holds a boolean vector for considered block types,
    # then we just know what the dun-rot-inds are for each PBT-assigned
    # block type index.
    # OLD rottable_set_for_dun_allowed_bts_cpu = (
    # OLD     self.dun_param_resolver._indices_from_names(
    # OLD         self.dun_param_resolver.all_table_indices,
    # OLD         dun_allowed_bt_base_names[None, :],
    # OLD         device=torch.device("cpu"),
    # OLD     ).squeeze(dim=0)
    # OLD )
    dun_allowed_bt = task.cons_bt_block_type[dun_allowed_bt_to_gbt]
    rottable_set_for_dun_allowed_bts = pbt.dun_sampler_cache.rottable_set_for_bt[
        dun_allowed_bt
    ]

    # rottable_set_for_dun_allowed_bts = rottable_set_for_dun_allowed_bts_cpu.to(
    #     self.device
    # )

    # the pbt-assigned block-type indices for each buildable block type
    # the subset of dun_rot_inds_for_dun_allowed_bts with a non-sentinel
    # value represents the buildable block types
    # block_type_ind_for_bbt = torch.tensor(
    #     pbt.restype_index.get_indexer(
    #         dun_allowed_bt_names[rottable_set_for_dun_allowed_bts_cpu.numpy() != -1]
    #     ),
    #     dtype=torch.int64,
    #     device=self.device,
    # )

    inds_of_phi = dun_sampler.atom_indices_for_backbone_dihedral(pose_stack, 0).reshape(
        -1, 4
    )
    inds_of_psi = dun_sampler.atom_indices_for_backbone_dihedral(pose_stack, 1).reshape(
        -1, 4
    )

    # what is the subset of dun-allowed block types that are buildable by the Dunbrack library?
    is_dun_allowed_bt_bbt = rottable_set_for_dun_allowed_bts != -1

    dun_allowed_bt_that_are_bbt = torch.nonzero(is_dun_allowed_bt_bbt, as_tuple=True)[0]
    bbt_to_gbt_torch = dun_allowed_bt_to_gbt[dun_allowed_bt_that_are_bbt]
    rottable_set_for_bbt = rottable_set_for_dun_allowed_bts[dun_allowed_bt_that_are_bbt]
    block_type_ind_for_bbt = dun_allowed_bt[dun_allowed_bt_that_are_bbt]

    # the "indices" of the blocks that the block types we will be building come
    # from, assuming we are colapsing the n_sys x max_n_blocks into a single
    # numbering. We will need to keep this array as it will be used by the
    # caller to understand what block types we are defining samples for.
    # We will shortly be renumbering the residues to talk about only the ones
    # that we will build rotamers for: BBT = "buildable block type"
    block_for_bbt = dun_allowed_bt_block[dun_allowed_bt_that_are_bbt]

    global_block_ind_for_bubl, bubl_for_bbt = torch.unique(
        block_for_bbt, return_inverse=True
    )
    global_block_ind_for_bubl = global_block_ind_for_bubl.to(torch.int64)

    # There are two things we need to know about each BBT:
    # 1. what BUildable-BLock index did it come from? (not all blocks are buildable,
    #    and we only care about the subset that are. When we call "unique" above,
    #    that reduces our focus to the subset of all blocks to the ones that are
    #    buildable; later, when we measure phi/psi, we will only measure phi/psi
    #    for the subset that are buildable.)
    # 2. what rottable set does the Dunbrack library assign to it?
    # We will put them together into a single tensor.
    bubl_and_rottable_set_for_bbt = (
        torch.cat(
            (
                bubl_for_bbt.reshape(-1, 1),
                rottable_set_for_bbt.reshape(-1, 1),
            ),
            dim=1,
        )
        .to(torch.int32)
        .to(device=dun_sampler.device)
    )

    # phi_psi_res_inds = numpy.arange(n_sys * max_n_blocks, dtype=numpy.int32)

    n_sampling_blocks = global_block_ind_for_bubl.shape[0]

    # map the residue-numbered list of dihedral angles to their positions in
    # the set of residues that the Dunbrack library will provide chi samples for
    dihedral_atom_inds = torch.full(
        (2 * n_sampling_blocks, 4), -1, dtype=torch.int32, device=dun_sampler.device
    )
    dihedral_atom_inds[
        2
        * torch.arange(n_sampling_blocks, dtype=torch.int64, device=dun_sampler.device),
        :,
    ] = inds_of_phi[global_block_ind_for_bubl, :]
    dihedral_atom_inds[
        2
        * torch.arange(n_sampling_blocks, dtype=torch.int64, device=dun_sampler.device)
        + 1,
        :,
    ] = inds_of_psi[global_block_ind_for_bubl, :]

    n_dihe_for_block = torch.full(
        (n_sampling_blocks,), 2, dtype=torch.int32, device=dun_sampler.device
    )
    dihedral_offset_for_block = 2 * torch.arange(
        n_sampling_blocks, dtype=torch.int32, device=dun_sampler.device
    )

    n_bbts = dun_allowed_bt_that_are_bbt.shape[0]

    max_n_chi = pose_stack.packed_block_types.dun_sampler_cache.max_n_chi

    # chi_expansion_for_gbt = torch.cat(
    #     [
    #         torch.tensor(blt.chi_expansion)
    #         for one_pose_blts in task.blts
    #         for blt in one_pose_blts
    #     ],
    # ).to(self.device)
    chi_expansion_for_gbt = task.per_block_chi_expansion[
        task.cons_bt_pose, task.cons_bt_block, task.cons_bt_which_block_type
    ]

    chi_expansion_for_bbt = (chi_expansion_for_gbt[dun_allowed_bt_to_gbt])[
        dun_allowed_bt_that_are_bbt
    ]
    # chi_expansion_for_bbt = chi_expansion_for_bbt

    # ok, we'll go to the block types and look at their protonation
    # state expansions and we'll put that information into the
    # chi_expansions_for_buildable_restype tensor

    # Treat all residues as buried (index 1). Burial classification is not
    # yet implemented; treating everything as buried is the conservative
    # choice (more rotamers).
    sc = pbt.dun_sampler_cache

    # Use total chi count per residue type (Dunbrack chis + proton chis)
    # rather than only the Dunbrack library's nchi. The C++ kernel loops
    # over indices [n_dun_chi .. n_chi) to sample non-Dunbrack (proton)
    # chis; if n_chi == n_dun_chi that loop never runs.
    n_chi_for_bbt = (
        (sc.chi_defining_atom[block_type_ind_for_bbt] >= 0).sum(dim=1).to(torch.int32)
    )

    non_dunbrack_expansion_counts_for_bbt = sc.non_dunbrack_sample_counts[
        block_type_ind_for_bbt, :, 1  # dim2: burial state (0=exposed, 1=buried)
    ]

    # treat all residues as buried (index 1)
    non_dunbrack_expansion_for_bbt = sc.non_dunbrack_samples[
        block_type_ind_for_bbt, :, 1  # dim2: burial state (0=exposed, 1=buried)
    ]

    # Rosetta defaults (buried): rotameric=0.98, semi-rotameric=0.95.
    # Based on testing (alf) semi-rot should also be 0.98
    # Table sets are ordered rotameric-first; semi-rotameric sets start at
    # index n_rotameric_sets.
    n_rotameric_sets = int(
        dun_sampler.dun_param_resolver.rotameric_table_indices["dun_table_name"].max()
        + 1
    )
    is_semi = bubl_and_rottable_set_for_bbt[:, 1].to(torch.int64) >= n_rotameric_sets
    prob_cumsum_limit_for_bbt = torch.where(
        is_semi,
        torch.full((n_bbts,), 0.98, dtype=torch.float32, device=dun_sampler.device),
        torch.full((n_bbts,), 0.98, dtype=torch.float32, device=dun_sampler.device),
    )

    # the sampled chi returned are a tuple containing info for BBTs:
    # these have to be mapped back to info for GBTs, which is handled
    # in the next step
    sampled_chi = dun_sampler.launch_rotamer_building(
        pose_stack.coords.reshape(-1, 3),
        n_dihe_for_block,
        dihedral_offset_for_block,
        dihedral_atom_inds,
        bubl_and_rottable_set_for_bbt,
        chi_expansion_for_bbt,
        non_dunbrack_expansion_for_bbt,
        non_dunbrack_expansion_counts_for_bbt,
        prob_cumsum_limit_for_bbt,
        n_chi_for_bbt,
    )

    n_rots_for_bbt = sampled_chi[0]
    chi_for_rotamers = sampled_chi[3]
    n_rots = chi_for_rotamers.shape[0]

    results = dun_sampler.package_samples_for_output(
        pbt,
        task,
        n_gbt_total,
        bbt_to_gbt_torch,
        block_type_ind_for_bbt,
        max_n_chi,
        sampled_chi,
    )
    (
        n_rots_for_gbt,
        gbt_for_rotamer,
        chi_defining_atom_for_rotamer,
        chi_for_rotamers,
    ) = results

    all_considered_restypes = task.cons_bt_block_type.cpu().numpy()
    offsets_for_gbt = exclusive_cumsum1d(n_rots_for_gbt)

    n_rots_for_gbt_gold = numpy.zeros(
        all_considered_restypes.shape[0], dtype=numpy.int32
    )

    n_rots_for_gbt_gold = numpy.zeros(
        all_considered_restypes.shape[0], dtype=numpy.int32
    )

    dun_allowed_bt_to_gbt_np = dun_allowed_bt_to_gbt.cpu().numpy()
    n_rots_for_gbt_gold[
        dun_allowed_bt_to_gbt_np[is_dun_allowed_bt_bbt.cpu().numpy()]
    ] = n_rots_for_bbt.cpu().numpy()

    rt_for_rot_gold = numpy.zeros((n_rots + 1,), dtype=numpy.int32)
    offsets_for_gbt_np = offsets_for_gbt.cpu().numpy()
    for _i, rotamer in enumerate(offsets_for_gbt_np):
        rt_for_rot_gold[rotamer] += 1
    rt_for_rot_gold = rt_for_rot_gold[:-1]  # lop off the "last" rotamer
    rt_for_rot_gold = numpy.cumsum(rt_for_rot_gold) - 1

    assert results[0].device == torch_device
    assert results[1].device == torch_device
    assert results[2].device == torch_device
    assert results[3].device == torch_device

    numpy.testing.assert_equal(n_rots_for_gbt_gold, results[0].cpu().numpy())
    numpy.testing.assert_equal(rt_for_rot_gold, results[1].cpu().numpy())


def test_chi_sampler_smoke(ubq_pdb, default_database):
    torch_device = torch.device("cpu")
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=5)
    p2 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=7)
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)
    palette = PackerPalette()
    task = PackerTask(poses, palette)
    task.restrict_to_repacking()

    sampler = create_dunbrack_sampler_from_database(default_database, torch_device)
    task.add_conformer_sampler(sampler)
    task = SetPackerTask.from_packer_task(task)

    for rt in poses.packed_block_types.active_block_types:
        sampler.annotate_residue_type(rt)
    sampler.annotate_packed_block_types(poses.packed_block_types)
    sampler.sample_chi_for_poses(poses, task)


def test_chi_sampler_build_lots_of_rotamers(ubq_pdb, default_database, torch_device):
    n_poses = 10
    p = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=10)
    poses = PoseStackBuilder.from_poses([p] * n_poses, torch_device)
    palette = PackerPalette()
    task = PackerTask(poses, palette)
    task.restrict_to_repacking()

    sampler = create_dunbrack_sampler_from_database(default_database, torch_device)
    task.add_conformer_sampler(sampler)
    task = SetPackerTask.from_packer_task(task)

    for rt in poses.packed_block_types.active_block_types:
        sampler.annotate_residue_type(rt)
    sampler.annotate_packed_block_types(poses.packed_block_types)
    chi_samples = sampler.sample_chi_for_poses(poses, task)

    n_rots_for_rt, rt_for_rotamer, chi_defining_atom, chi = chi_samples

    assert n_rots_for_rt.shape[0] == 2100
    n_rots = chi_defining_atom.shape[0]
    n_rots_per_pose = n_rots // n_poses
    assert n_rots_per_pose * n_poses == n_rots
