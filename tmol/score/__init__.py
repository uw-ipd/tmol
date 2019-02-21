"""Composable, scoring components for molecular systems."""

from . import (  # noqa: F401
    device,
    bonded_atom,
    interatomic_distance,
    ljlk,
    elec,
    cartbonded,
    hbond,
    coordinates,
    score_graph,
    viewer,  # import viewer to register io overloads
    rama,
)


@score_graph.score_graph
class TotalScoreGraph(
    ljlk.LJScoreGraph,
    ljlk.LKScoreGraph,
    hbond.HBondScoreGraph,
    elec.ElecScoreGraph,
    cartbonded.CartBondedScoreGraph,
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
