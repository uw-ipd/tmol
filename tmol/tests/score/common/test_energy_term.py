import numpy
import torch
import pytest
import pickle
import os
import yaml
import importlib
import functools
from unittest.mock import patch

from tmol.pose.packed_block_types import residue_types_from_residues, PackedBlockTypes
from tmol.pose.pose_stack_builder import PoseStackBuilder

from tmol.tests.autograd import gradcheck
import torch.autograd.gradcheck as torchgrad


# monkeypatch function to give more sane output from torch gradcheck
def _get_notallclose_msg(
    analytical,
    numerical,
    output_idx,
    input_idx,
    complex_indices,
    test_imag=False,
    is_forward_ad=False,
    atol=None,
    rtol=None,
) -> str:
    # original code from torch:
    """
    out_is_complex = (not is_forward_ad) and complex_indices and output_idx in complex_indices
    inp_is_complex = is_forward_ad and complex_indices and input_idx in complex_indices
    part = "imaginary" if test_imag else "real"
    element = "inputs" if is_forward_ad else "outputs"
    prefix = "" if not (out_is_complex or inp_is_complex) else \
        f"While considering the {part} part of complex {element} only, "
    mode = "computed with forward mode " if is_forward_ad else ""
    results['numerical'] = numerical.cpu()
    results['analytical'] = analytical.cpu()
    return prefix + 'Jacobian %smismatch for output %d with respect to input %d,\n' \
        'numerical:%s\nanalytical:%s\n' % (mode, output_idx, input_idx, numerical, analytical)
    """

    # custom code:
    analytical = analytical.cpu().numpy()
    numerical = numerical.cpu().numpy()

    return get_notallclose_msg(analytical, numerical, atol, rtol)


def get_notallclose_msg(analytical, numerical, atol, rtol):
    resstr = "Difference between analytical and numerical tensors exceeds tolerances:\n"
    close = numpy.isclose(analytical, numerical, atol=atol, rtol=rtol)
    badvals = numpy.argwhere(close == False)
    maxval = 0.0
    table = [("index", "analytical", "numerical", "difference")]
    table += [
        (
            bv,
            analytical[tuple(bv)],
            numerical[tuple(bv)],
            abs(analytical[tuple(bv)] - numerical[tuple(bv)]),
        )
        for bv in badvals
    ]
    for bv in badvals:
        ind = tuple(bv)
        diff = abs(analytical[ind] - numerical[ind])
        maxval = max(diff, maxval)
        # resstr += "%s: (analytical):%f (numerical):%f (diff):%f" % (bv, analytical[ind], numerical[ind], diff) + "\n"
    resstr += print_table(table)

    resstr += "Tolerances: atol=%f, rtol=%f\n" % (atol, rtol)
    resstr += "Measured: atol=%f" % (maxval)

    return resstr


def assert_allclose(baseline, measured, atol, rtol):
    try:
        numpy.testing.assert_allclose(baseline, measured, atol=atol, rtol=rtol)
    except AssertionError:
        raise AssertionError(get_notallclose_msg(measured, baseline, atol, rtol))


def print_table(table):
    col_width = [max(len(str(x)) for x in col) for col in zip(*table)]
    retstr = ""
    for line in table:
        retstr += (
            " | ".join("{:{}}".format(str(x), col_width[i]) for i, x in enumerate(line))
            + "\n"
        )
    return retstr


class EnergyTermTestBase:
    energy_term_class = None

    @classmethod
    def get_test_baseline_data_filename(cls, testname):
        dirname = os.path.join(
            "tmol", "tests", "data", "term_baselines", cls.energy_term_class.__name__
        )
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        filename = os.path.join(dirname, testname + ".yaml")
        return filename

    # save the baselines to disk
    @classmethod
    def save_test_baseline_data(cls, testname, data):
        filename = cls.get_test_baseline_data_filename(testname)
        with open(filename, "w") as outfile:
            yaml.safe_dump(data, outfile)  # , default_flow_style=None)

    @classmethod
    def block_pair_to_dict(cls, data):
        indat = data.tolist()
        out = cls.recursive_reformat_to_dicts(
            data, names=["term", "pose", "res", "res"]
        )
        return out

    @classmethod
    def whole_pose_to_dict(cls, data):
        indat = data.tolist()
        out = cls.recursive_reformat_to_dicts(data, names=["term", "pose"])
        return out

    @classmethod
    def recursive_reformat_to_dicts(cls, data, names, dim=0):
        if dim == len(names):
            return float(data)
        else:
            return {
                names[dim]
                + str(ind): cls.recursive_reformat_to_dicts(val, names, dim + 1)
                for ind, val in enumerate(data)
            }

    @classmethod
    def recursive_reformat_from_dicts(cls, data):
        if isinstance(data, dict):
            return [cls.recursive_reformat_from_dicts(val) for key, val in data.items()]
        else:
            return data

    # fetch the baselines from disk, or NULL if they dont exist yet
    @classmethod
    def get_test_baseline_data(cls, testname):
        filename = cls.get_test_baseline_data_filename(testname)
        try:
            with open(filename, "r") as infile:
                # return pickle.load(infile)
                return numpy.array(
                    cls.recursive_reformat_from_dicts(yaml.safe_load(infile))
                )
        except FileNotFoundError as e:  # FileNotFoundError or whatever else
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
            cls.save_test_baseline_data(
                cls.test_whole_pose_scoring_10.__name__, cls.whole_pose_to_dict(scores)
            )
        gold_vals = cls.get_test_baseline_data(cls.test_whole_pose_scoring_10.__name__)

        assert_allclose(gold_vals, scores, atol, rtol)

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

        # monkeypatch more sane error reporting
        torchgrad = importlib.import_module("torch.autograd.gradcheck")
        torchgrad._get_notallclose_msg = functools.partial(
            _get_notallclose_msg, atol=atol, rtol=rtol
        )

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
                cls.test_whole_pose_scoring_jagged.__name__,
                cls.whole_pose_to_dict(scores),
            )
        gold_vals = cls.get_test_baseline_data(
            cls.test_whole_pose_scoring_jagged.__name__
        )

        assert_allclose(gold_vals, scores, atol, rtol)

    @classmethod
    def test_block_scoring(
        cls,
        rts_res,
        default_database,
        torch_device,
        update_baseline=False,
        atol=1e-5,
        rtol=1e-3,
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
            cls.save_test_baseline_data(
                cls.test_block_scoring.__name__, cls.block_pair_to_dict(scores)
            )
        gold_vals = cls.get_test_baseline_data(cls.test_block_scoring.__name__)
        # print(gold_vals)
        """
        close = numpy.isclose(gold_vals, scores, atol=atol, rtol=rtol)
        badvals = numpy.argwhere(close==False)
        for bv in badvals:
            ind = tuple(bv)
            #print("%s: (baseline):%f (computed):%f (diff):%f" % (bv, gold_vals[ind], scores[ind], gold_vals[ind]-scores[ind]))
            """

        # numpy.testing.assert_allclose(gold_vals, scores, rtol=1e-5)
        assert_allclose(gold_vals, scores, atol, rtol)

    @classmethod
    # @patch(importlib.import_module('torch.autograd.gradcheck'), new_callable=_get_notallclose_msg)
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

        # monkeypatch more sane error reporting
        torchgrad = importlib.import_module("torch.autograd.gradcheck")
        torchgrad._get_notallclose_msg = functools.partial(
            _get_notallclose_msg, atol=atol, rtol=rtol
        )

        torchgrad.gradcheck(
            score,
            (p1.coords.requires_grad_(True),),
            eps=eps,
            atol=atol,
            rtol=rtol,
            nondet_tol=nondet_tol,
        )
