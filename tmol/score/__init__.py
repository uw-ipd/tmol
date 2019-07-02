"""Composable, scoring components for molecular systems."""

from . import (  # noqa: F401
    device,
    bonded_atom,
    interatomic_distance,
    ljlk,
    lk_ball,
    elec,
    cartbonded,
    dunbrack,
    hbond,
    rama,
    omega,
    coordinates,
    score_graph,
    score_weights,
    viewer,  # import viewer to register io overloads
)


@score_graph.score_graph
class TotalScoreGraph(
    ljlk.LJScoreGraph,
    ljlk.LKScoreGraph,
    lk_ball.LKBallScoreGraph,
    hbond.HBondScoreGraph,
    dunbrack.DunbrackScoreGraph,
    rama.RamaScoreGraph,
    omega.OmegaScoreGraph,
    # elec.ElecScoreGraph,
    cartbonded.CartBondedScoreGraph,
    score_weights.ScoreWeights,  # per-term reweighing
):
    pass


@score_graph.score_graph
class KinematicTotalScoreGraph(
    coordinates.KinematicAtomicCoordinateProvider, TotalScoreGraph
):
    pass


@score_graph.score_graph
class CartesianTotalScoreGraph(
    coordinates.CartesianAtomicCoordinateProvider, TotalScoreGraph
):
    pass


__all__ = ("TotalScoreGraph", "KinematicTotalScoreGraph", "CartesianTotalScoreGraph")
