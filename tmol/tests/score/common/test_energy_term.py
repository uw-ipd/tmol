import numpy
import torch
import pytest
import pickle
import os

from tmol.pose.packed_block_types import residue_types_from_residues, PackedBlockTypes
from tmol.pose.pose_stack_builder import PoseStackBuilder

from tmol.tests.autograd import gradcheck


class EnergyTermTestBase:
    energy_term_class = None

    @classmethod
    def get_test_baseline_data_filename(cls, testname):
        dirname = os.path.join("tmol", "tests", "data", "term_baselines", cls.__name__)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        filename = os.path.join(dirname, testname + ".pickle")
        return filename

    # save the baselines to disk
    @classmethod
    def save_test_baseline_data(cls, testname, data):
        filename = cls.get_test_baseline_data_filename(testname)
        with open(filename, "wb") as outfile:
            pickle.dump(data, outfile)

    # fetch the baselines from disk, or NULL if they dont exist yet
    @classmethod
    def get_test_baseline_data(cls, testname):
        filename = cls.get_test_baseline_data_filename(testname)
        try:
            with open(filename, "rb") as infile:
                return pickle.load(infile)
        except:  # FileNotFoundError or whatever else
            raise Exception(
                "Baselines not found for "
                + cls.__name__
                + ":"
                + testname
                + ". Re-run with update_baselines=True"
            )

    @classmethod
    def test_whole_pose_scoring_10(
        cls,
        rts_res,
        default_database,
        torch_device,
        update_baseline=False,
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
        nondet_tol=0,
    ):
        n_poses = 10
        energy_term = cls.energy_term_class(
            param_db=default_database, device=torch_device
        )

        p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
            default_database.chemical, res=rts_res, device=torch_device
        )
        pn = PoseStackBuilder.from_poses([p1] * n_poses, device=torch_device)

        for bt in pn.packed_block_types.active_block_types:
            energy_term.setup_block_type(bt)
        energy_term.setup_packed_block_types(pn.packed_block_types)
        energy_term.setup_poses(pn)

        pose_scorer = energy_term.render_whole_pose_scoring_module(pn)

        coords = torch.nn.Parameter(pn.coords.clone())
        scores = pose_scorer(coords).cpu().detach().numpy()

        if True:  # update_baseline:
            cls.save_test_baseline_data(cls.test_whole_pose_scoring_10.__name__, scores)
        gold_vals = cls.get_test_baseline_data(cls.test_whole_pose_scoring_10.__name__)

        numpy.testing.assert_allclose(gold_vals, scores, atol=1e-5, rtol=1e-5)

    @classmethod
    def test_whole_pose_scoring_gradcheck(
        cls,
        rts_res,
        default_database,
        torch_device,
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
        nondet_tol=0,
    ):
        energy_term = cls.energy_term_class(
            param_db=default_database, device=torch_device
        )
        p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
            default_database.chemical, res=rts_res, device=torch_device
        )
        for bt in p1.packed_block_types.active_block_types:
            energy_term.setup_block_type(bt)
        energy_term.setup_packed_block_types(p1.packed_block_types)
        energy_term.setup_poses(p1)

        ljlk_pose_scorer = energy_term.render_whole_pose_scoring_module(p1)

        def score(coords):
            scores = ljlk_pose_scorer(coords)
            return torch.sum(scores)

        gradcheck(
            score,
            (p1.coords.requires_grad_(True),),
            eps=eps,
            atol=atol,
            rtol=rtol,
            nondet_tol=nondet_tol,
        )

    @classmethod
    def test_whole_pose_scoring_jagged(
        cls,
        rts_res,
        default_database,
        torch_device: torch.device,
        update_baseline=False,
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
        nondet_tol=0,
    ):
        rts_60 = rts_res[:60]
        rts_33 = rts_res[:33]
        p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
            default_database.chemical, res=rts_res, device=torch_device
        )
        p2 = PoseStackBuilder.one_structure_from_polymeric_residues(
            default_database.chemical, res=rts_60, device=torch_device
        )
        p3 = PoseStackBuilder.one_structure_from_polymeric_residues(
            default_database.chemical, res=rts_33, device=torch_device
        )
        pn = PoseStackBuilder.from_poses([p1, p2, p3], device=torch_device)

        energy_term = cls.energy_term_class(
            param_db=default_database, device=torch_device
        )
        for bt in pn.packed_block_types.active_block_types:
            energy_term.setup_block_type(bt)
        energy_term.setup_packed_block_types(pn.packed_block_types)
        energy_term.setup_poses(pn)

        pose_scorer = energy_term.render_whole_pose_scoring_module(pn)
        scores = pose_scorer(pn.coords).cpu().detach().numpy()

        if True:  # update_baseline:
            cls.save_test_baseline_data(
                cls.test_whole_pose_scoring_jagged.__name__, scores
            )
        gold_vals = cls.get_test_baseline_data(
            cls.test_whole_pose_scoring_jagged.__name__
        )

        numpy.testing.assert_allclose(gold_vals, scores, rtol=1e-5)

    @classmethod
    def test_block_scoring(
        cls,
        rts_res,
        default_database,
        torch_device,
        update_baseline=False,
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
        nondet_tol=0,
    ):
        energy_term = cls.energy_term_class(
            param_db=default_database, device=torch_device
        )
        p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
            default_database.chemical, res=rts_res, device=torch_device
        )
        for bt in p1.packed_block_types.active_block_types:
            energy_term.setup_block_type(bt)
        energy_term.setup_packed_block_types(p1.packed_block_types)
        energy_term.setup_poses(p1)

        pose_scorer = energy_term.render_whole_pose_scoring_module(p1)

        coords = torch.nn.Parameter(p1.coords.clone())
        scores = (
            pose_scorer(coords, output_block_pair_energies=True).cpu().detach().numpy()
        )

        if True:  # update_baseline:
            cls.save_test_baseline_data(cls.test_block_scoring.__name__, scores)
        gold_vals = cls.get_test_baseline_data(cls.test_block_scoring.__name__)
        print(gold_vals)

        numpy.testing.assert_allclose(gold_vals, scores, atol=1e-4)

    @classmethod
    def test_block_scoring_reweighted_gradcheck(
        cls,
        rts_res,
        default_database,
        torch_device,
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
        nondet_tol=0,
    ):
        energy_term = cls.energy_term_class(
            param_db=default_database, device=torch_device
        )
        p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
            default_database.chemical, res=rts_res, device=torch_device
        )
        for bt in p1.packed_block_types.active_block_types:
            energy_term.setup_block_type(bt)
        energy_term.setup_packed_block_types(p1.packed_block_types)
        energy_term.setup_poses(p1)

        pose_scorer = energy_term.render_whole_pose_scoring_module(p1)

        def score(coords):
            scores = pose_scorer(coords, output_block_pair_energies=True)
            scale = 0.01 * torch.arange(
                torch.numel(scores), device=scores.device
            ).reshape(scores.shape)
            return torch.sum(scale * scores)

        gradcheck(
            score,
            (p1.coords.requires_grad_(True),),
            eps=eps,
            atol=atol,
            rtol=rtol,
            nondet_tol=nondet_tol,  # fd this is necessary here...
        )
