import attrs
import time
import torch
from tmol.pose.pose_stack import PoseStack

from tmol.kinematics.fold_forest import FoldForest
from tmol.kinematics.move_map import MoveMap, MinimizerMap
from tmol.score.score_function import ScoreFunction

from tmol.optimization.lbfgs_armijo import LBFGS_Armijo


def build_kinforest_network(
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    ff: FoldForest,
    mm: MoveMap,
    verbose=False,
):
    from tmol.kinematics.script_modules import PoseStackKinematicsModule
    from tmol.optimization.sfxn_modules import KinForestSfxnNetwork

    # TEMP!
    # pose_stack.coords = pose_stack.coords.to(torch.float64)

    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    kin_module = PoseStackKinematicsModule(pose_stack, ff)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time1 = time.perf_counter()

    minimizer_map = MinimizerMap(pose_stack, kin_module.kmd, mm)
    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time2 = time.perf_counter()

    kf_network = KinForestSfxnNetwork(
        sfxn, pose_stack, kin_module, minimizer_map.dof_mask
    )
    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time3 = time.perf_counter()

    if verbose:
        print(
            f"build_kinforest_network {end_time3 - start_time: .2f}"
            + f" s1: {end_time1-start_time: .2f} s2: {end_time2 - end_time1: .2f}"
            + f" s3: {end_time3-end_time2: .2f}"
        )

    return kf_network


def run_kin_min(
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    ff: FoldForest,
    mm: MoveMap,
    verbose=False,
):
    """Run LBFGS minimization on a PoseStack in internal DOF space.

    Note that this function constructs a new kinematics module and discards it
    at the conclusion of minimization.
    """
    input_pose_stack_coords_dtype = pose_stack.coords.dtype
    # TEMP!
    pose_stack = attrs.evolve(pose_stack, coords=pose_stack.coords.to(torch.float64))

    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    kf_network = build_kinforest_network(pose_stack, sfxn, ff, mm, verbose)
    optimizer = LBFGS_Armijo(kf_network.parameters())

    class debug_closure:
        def __call__(self):
            optimizer.zero_grad()
            E = kf_network().sum()
            E.backward()
            return E

        def no_debug(self, coords1, coords_center, coords2, step):
            old_wpsm = kf_network.whole_pose_scoring_module

            for st in sfxn.all_score_types():
                print("Grad check only with term:", st)
                new_sfxn = ScoreFunction(sfxn._param_db, sfxn._device)
                new_sfxn.set_weight(st, 1.0)
                new_wpsm = new_sfxn.render_whole_pose_scoring_module(pose_stack)

                # kf_network.whole_pose_scoring_module = new_wpsm

                # get rid of any gradients from the previous iteration
                kf_network.full_dofs = kf_network.full_dofs.detach()
                kf_network.full_coords = kf_network.full_coords.detach()
                kf_network.flat_coords = kf_network.flat_coords.detach()

                kf_network.full_dofs[kf_network.dof_mask] = coords_center
                kin_coords = kf_network.kin_module(kf_network.full_dofs)
                kf_network.flat_coords[kf_network.id[1:]] = kin_coords[1:]
                kf_network.full_coords = kf_network.flat_coords.view(
                    kf_network.orig_coords_shape
                )
                kf_network.full_coords = kf_network.full_coords.to(torch.float64)

                # # now evaluate the score
                # da_score = kf_network.whole_pose_scoring_module(kf_network.full_coords).sum()
                # # print("da score:", da_score.item())
                # return da_score

                def score(coords):
                    return new_wpsm(coords).sum()

                # monkeypatch more sane error reporting
                from tmol.tests.score.common.test_energy_term import (
                    _get_notallclose_msg,
                )
                import importlib
                import functools

                eps = 1e-6  # torch default
                atol = 1e-4  # torch default
                rtol = 1e-5  # torch default
                nondet_tol = 1e-5

                torchgrad = importlib.import_module("torch.autograd.gradcheck")
                torchgrad._get_notallclose_msg = functools.partial(
                    _get_notallclose_msg, atol=atol, rtol=rtol
                )

                torchgrad.gradcheck(
                    score,
                    (kf_network.full_coords.requires_grad_(True),),
                    eps=eps,
                    atol=atol,
                    rtol=rtol,
                    nondet_tol=nondet_tol,
                )

                # term.debug_energy_gradient(coords1, coords2)

    closure = debug_closure()

    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time1 = time.perf_counter()

    # Perform the minimization
    optimizer.step(closure)

    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time2 = time.perf_counter()

    new_pose_stack = kf_network.pose_stack_from_dofs()
    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time3 = time.perf_counter()

    if verbose:
        print(
            f"kin_min {end_time3 - start_time: .2f} setup: {end_time1-start_time: .2f}"
            + f" opt {end_time2 - end_time1: .2f} stack-ctor: {end_time3-end_time2: .2f}"
        )

    if new_pose_stack.coords.dtype != input_pose_stack_coords_dtype:
        new_pose_stack = attrs.evolve(
            new_pose_stack,
            coords=new_pose_stack.coords.to(input_pose_stack_coords_dtype),
        )
    return new_pose_stack
