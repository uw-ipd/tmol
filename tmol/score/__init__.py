"""Composable, scoring components for molecular systems."""

from tmol.utility.reactive import reactive_attrs

from . import (  # noqa: F401
    device,
    bonded_atom,
    interatomic_distance,
    ljlk,
    hbond,
    coordinates,
    viewer,  # import viewer to register io overloads
)


@reactive_attrs
class TotalScoreGraph(
    hbond.HBondScoreGraph,
    ljlk.LJLKScoreGraph,
    interatomic_distance.BlockedInteratomicDistanceGraph,
    bonded_atom.BondedAtomScoreGraph,
    device.TorchDevice,
):
    pass


@reactive_attrs
class KinematicTotalScoreGraph(
    coordinates.KinematicAtomicCoordinateProvider, TotalScoreGraph
):
    pass


@reactive_attrs
class CartesianTotalScoreGraph(
    coordinates.CartesianAtomicCoordinateProvider, TotalScoreGraph
):
    pass


__all__ = ("TotalScoreGraph", "KinematicTotalScoreGraph", "CartesianTotalScoreGraph")
