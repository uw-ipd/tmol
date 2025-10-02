import numpy
import torch
import os
import yaml
import importlib
import functools
import pandas
import torchshow

from tmol.io import pose_stack_from_pdb
from tmol.io.pdb_parsing import parse_pdb
from tmol.io.canonical_ordering import (
    default_canonical_ordering,
    default_packed_block_types,
    select_atom_records_res_subset,
    canonical_form_from_atom_records,
)
from tmol.io.pose_stack_construction import pose_stack_from_canonical_form
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.score.ref.ref_energy_term import RefEnergyTerm


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
    badvals = numpy.argwhere(close == False)  # noqa: E712
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


def pose_stack_from_pdb_and_resnums(pdb, torch_device, resnums=None):
    if resnums is None:
        return pose_stack_from_pdb(pdb, torch_device)

    atom_records = parse_pdb(pdb)
    atom_subsets = pandas.concat(
        [select_atom_records_res_subset(atom_records, i, j) for i, j in resnums]
    )
    canonical_ordering = default_canonical_ordering()
    packed_block_types = default_packed_block_types(torch_device)
    canonical_form = canonical_form_from_atom_records(
        canonical_ordering, atom_subsets, torch_device
    )

    res_not_connected = []
    for i, j in resnums:
        for k in range(i, j):
            if k == i:
                if k == 0:
                    res_not_connected.append((False, False))
                else:
                    res_not_connected.append((True, False))
            elif k == j - 1:
                res_not_connected.append((False, True))
            else:
                res_not_connected.append((False, False))
    canonical_form["res_not_connected"] = torch.tensor(
        [res_not_connected], dtype=torch.bool, device=torch_device
    )

    pose_stack = pose_stack_from_canonical_form(
        canonical_ordering, packed_block_types, **canonical_form
    )
    return pose_stack


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
            yaml.safe_dump(data, outfile)

    @classmethod
    def block_pair_to_dict(cls, data):
        out = cls.recursive_reformat_to_dicts(
            data, names=["term", "pose", "res", "res"]
        )
        return out

    @classmethod
    def whole_pose_to_dict(cls, data):
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
                return numpy.array(
                    cls.recursive_reformat_from_dicts(yaml.safe_load(infile))
                )
        except FileNotFoundError:  # FileNotFoundError or whatever else
            raise Exception(
                "Baselines not found for "
                + cls.__name__
                + ":"
                + testname
                + ". Re-run with update_baselines=True"
            )

    @classmethod
    def get_pose_scorer(cls, pose_stack, param_db, device, block_pair_scoring=False):
        energy_term = cls.energy_term_class(param_db=param_db, device=device)

        for bt in pose_stack.packed_block_types.active_block_types:
            energy_term.setup_block_type(bt)
        energy_term.setup_packed_block_types(pose_stack.packed_block_types)
        energy_term.setup_poses(pose_stack)

        return (
            energy_term.render_whole_pose_scoring_module(pose_stack)
            if block_pair_scoring is False
            else energy_term.render_block_pair_scoring_module(pose_stack)
        )

    @classmethod
    def test_whole_pose_scoring_10(
        cls,
        pdb,
        default_database,
        torch_device,
        resnums=None,
        edit_pose_stack_fn=None,
        update_baseline=False,
        atol=1e-5,
        rtol=1e-3,
    ):
        n_poses = 10

        p1 = pose_stack_from_pdb_and_resnums(pdb, torch_device, resnums)
        pn = PoseStackBuilder.from_poses([p1] * n_poses, device=torch_device)

        # from tmol.io.write_pose_stack_pdb import write_pose_stack_pdb

        # write_pose_stack_pdb(
        # pn,
        # "test_whole_pose_scoring_10_new.pdb",
        # chain_ind_for_block=torch.zeros(
        # (pn.n_poses, pn.max_n_blocks), dtype=torch.int64
        # ),
        # )

        if edit_pose_stack_fn is not None:
            edit_pose_stack_fn(pn)

        pose_scorer = cls.get_pose_scorer(pn, default_database, torch_device)

        coords = torch.nn.Parameter(pn.coords.clone())
        scores = pose_scorer(coords).cpu().detach().numpy()

        if update_baseline:
            cls.save_test_baseline_data(
                cls.test_whole_pose_scoring_10.__name__, cls.whole_pose_to_dict(scores)
            )
        gold_vals = cls.get_test_baseline_data(cls.test_whole_pose_scoring_10.__name__)

        assert_allclose(gold_vals, scores, atol, rtol)

    @classmethod
    def test_whole_pose_scoring_gradcheck(
        cls,
        pdb,
        default_database,
        torch_device,
        resnums=None,
        edit_pose_stack_fn=None,
        eps=1e-6,  # torch default
        atol=1e-5,  # torch default
        rtol=1e-3,  # torch default
        nondet_tol=0.0,  # torch default
    ):
        p1 = pose_stack_from_pdb_and_resnums(pdb, torch_device, resnums)

        if edit_pose_stack_fn is not None:
            edit_pose_stack_fn(p1)

        pose_scorer = cls.get_pose_scorer(p1, default_database, torch_device)

        # wt = torch.rand((10,), device=torch_device)

        # print(wt)

        def score(coords):
            scores = pose_scorer(coords)

            # wt = torch.full_like(scores, 0.5)
            # score = wt * scores

            return scores

        # monkeypatch more sane error reporting
        torchgrad = importlib.import_module("torch.autograd.gradcheck")
        torchgrad._get_notallclose_msg = functools.partial(
            _get_notallclose_msg, atol=atol, rtol=rtol
        )

        torchgrad.gradcheck(
            score,
            (p1.coords.double().requires_grad_(True),),
            eps=eps,
            atol=atol,
            rtol=rtol,
            nondet_tol=nondet_tol,
        )

    @classmethod
    def test_whole_pose_scoring_jagged(
        cls,
        pdb,
        default_database,
        torch_device: torch.device,
        edit_pose_stack_fn=None,
        update_baseline=False,
        atol=1e-5,
        rtol=1e-3,
    ):
        res_50 = [(0, 50)]
        res_30 = [(20, 50)]
        p1 = pose_stack_from_pdb_and_resnums(pdb, torch_device)
        p2 = pose_stack_from_pdb_and_resnums(pdb, torch_device, res_50)
        p3 = pose_stack_from_pdb_and_resnums(pdb, torch_device, res_30)
        pn = PoseStackBuilder.from_poses([p1, p2, p3], device=torch_device)

        if edit_pose_stack_fn is not None:
            edit_pose_stack_fn(pn)

        pose_scorer = cls.get_pose_scorer(pn, default_database, torch_device)
        scores = pose_scorer(pn.coords).cpu().detach().numpy()

        if update_baseline:
            cls.save_test_baseline_data(
                cls.test_whole_pose_scoring_jagged.__name__,
                cls.whole_pose_to_dict(scores),
            )
        gold_vals = cls.get_test_baseline_data(
            cls.test_whole_pose_scoring_jagged.__name__
        )

        assert_allclose(gold_vals, scores, atol, rtol)

    @classmethod
    def test_block_scoring_matches_whole_pose_scoring(
        cls,
        pdb,
        default_database,
        torch_device,
        resnums=None,
        edit_pose_stack_fn=None,
        atol=1e-5,
        rtol=1e-3,
    ):
        p1 = pose_stack_from_pdb_and_resnums(pdb, torch_device, resnums)

        if edit_pose_stack_fn is not None:
            edit_pose_stack_fn(p1)

        block_pose_scorer = cls.get_pose_scorer(
            p1, default_database, torch_device, True
        )
        whole_pose_scorer = cls.get_pose_scorer(p1, default_database, torch_device)

        coords = torch.nn.Parameter(p1.coords.clone())
        block_pair_scores = block_pose_scorer(coords).to_dense().cpu().detach().numpy()
        full_pose_scores = whole_pose_scorer(coords).cpu().detach().numpy()

        assert_allclose(full_pose_scores, block_pair_scores.sum((2, 3)), atol, rtol)

    @classmethod
    def test_block_scoring(
        cls,
        pdb,
        default_database,
        torch_device,
        resnums=None,
        edit_pose_stack_fn=None,
        update_baseline=False,
        override_baseline_name=None,
        atol=1e-5,
        rtol=1e-3,
    ):
        p1 = pose_stack_from_pdb_and_resnums(pdb, torch_device, resnums)

        if edit_pose_stack_fn is not None:
            edit_pose_stack_fn(p1)

        pose_scorer = cls.get_pose_scorer(p1, default_database, torch_device, True)

        coords = torch.nn.Parameter(p1.coords.clone())
        scores = pose_scorer(coords)
        scores = scores.to_dense().cpu().detach().numpy()

        test_name = (
            cls.test_block_scoring.__name__
            if (override_baseline_name is None)
            else override_baseline_name
        )
        if update_baseline:
            cls.save_test_baseline_data(test_name, cls.block_pair_to_dict(scores))
        gold_vals = cls.get_test_baseline_data(test_name)

        assert_allclose(gold_vals, scores, atol, rtol)

    @classmethod
    def test_block_scoring_reweighted_gradcheck(
        cls,
        pdb,
        default_database,
        torch_device,
        resnums=None,
        edit_pose_stack_fn=None,
        eps=1e-6,  # torch default
        atol=1e-5,  # torch default
        rtol=1e-3,  # torch default
        nondet_tol=0.0,  # torch default
    ):
        p1 = pose_stack_from_pdb_and_resnums(pdb, torch_device, resnums)

        if edit_pose_stack_fn is not None:
            edit_pose_stack_fn(p1)

        pose_scorer = cls.get_pose_scorer(p1, default_database, torch_device)

        def score(coords):
            scores = pose_scorer(coords)
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
            (p1.coords.double().requires_grad_(True),),
            eps=eps,
            atol=atol,
            rtol=rtol,
            nondet_tol=nondet_tol,
        )


# We use Ref as dummy for generic energy term tests. This new class ensures it gets its own test data directory
class DummyEnergyTerm(RefEnergyTerm):
    pass


class EnergyTermBaseTester(EnergyTermTestBase):
    energy_term_class = DummyEnergyTerm


# This test just makes sure the updating_baseline functionality works (only tests the '10' variant currently)
def test_energy_term_base_write_baseline_smoke(ubq_pdb, default_database, torch_device):
    test_class = EnergyTermBaseTester()
    test_class.test_whole_pose_scoring_10(
        ubq_pdb, default_database, torch_device, update_baseline=True
    )


# This test makes sure that the 'jagged' test of the DummyEnergyTerm (Ref) fails. The baselines for the Dummy term's 'jagged' test have been manually modified and should always fail
def test_energy_term_fail(ubq_pdb, default_database, torch_device):
    test_class = EnergyTermBaseTester()
    failed = False

    try:
        test_class.test_whole_pose_scoring_jagged(
            ubq_pdb, default_database, torch_device, update_baseline=False
        )
        failed = True
    except AssertionError:
        pass

    if failed:
        raise AssertionError("Test passed with bad baselines.")
