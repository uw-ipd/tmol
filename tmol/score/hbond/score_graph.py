import attr
from typing import Optional

import torch
import numpy

from ..database import ParamDB
from ..device import TorchDevice
from ..bonded_atom import BondedAtomScoreGraph
from ..factory import Factory

from .potentials import (
    hbond_donor_sp2_score,
    hbond_donor_sp3_score,
    hbond_donor_ring_score,
)

from .identification import HBondElementAnalysis, donor_dtype, acceptor_dtype
from .params import HBondParamResolver, HBondPairParams

from tmol.database import ParameterDatabase
from tmol.database.scoring import HBondDatabase

from tmol.utility.reactive import reactive_attrs, reactive_property

from tmol.types.attrs import ValidateAttrs
from tmol.types.functional import validate_args
from tmol.types.array import NDArray
from tmol.types.torch import Tensor

pair_descr_dtype = numpy.dtype(donor_dtype.descr + acceptor_dtype.descr)


@attr.s(frozen=True, auto_attribs=True, slots=True)
class HBondPairs(ValidateAttrs):
    """Atom indices of all donor/acceptor pairs in system.

    The combination of all donors against all acceptors, (these lists having
    been constructed by the HBondElementAnalysis class) where the acceptors are
    broken down by their functional form (sp2, sp3, and ring). For each pair,
    then copy down the set of hbond parameters that describe how to evaluate
    the energy for that pair.

    (Making a copy of the parameters for each perspective donor/acceptor pair
    seems like it would be pretty expensive!)

    This work is performed on the CPU and then copied to the device.
    """

    donor_sp2_pairs: NDArray(pair_descr_dtype)[:]
    donor_sp2_pair_params: Optional[HBondPairParams]

    donor_sp3_pairs: NDArray(pair_descr_dtype)[:]
    donor_sp3_pair_params: Optional[HBondPairParams]

    donor_ring_pairs: NDArray(pair_descr_dtype)[:]
    donor_ring_pair_params: Optional[HBondPairParams]

    @staticmethod
    @validate_args
    def _cross_pairs(
        donors: NDArray(donor_dtype)[:], acceptors: NDArray(acceptor_dtype)[:]
    ):
        pairs = numpy.empty((len(donors), len(acceptors)), dtype=pair_descr_dtype)

        for n in donors.dtype.names:
            pairs[n] = donors[n].reshape((-1, 1))

        for n in acceptors.dtype.names:
            pairs[n] = acceptors[n].reshape((1, -1))

        pairs = pairs.ravel()

        return pairs[pairs["a"] != pairs["d"]]

    @classmethod
    @validate_args
    def setup(
        cls,
        params: HBondParamResolver,
        elems: HBondElementAnalysis,
        device: torch.device,
    ):
        # Get blocks of acceptor/donor pairs for each set of identified acceptors
        donor_sp2_pairs = cls._cross_pairs(elems.donors, elems.sp2_acceptors)
        donor_sp2_pair_params = (
            (
                params[donor_sp2_pairs["donor_type"], donor_sp2_pairs["acceptor_type"]]
            ).to(device)
            if len(donor_sp2_pairs)
            else None
        )

        donor_sp3_pairs = cls._cross_pairs(elems.donors, elems.sp3_acceptors)
        donor_sp3_pair_params = (
            (
                params[donor_sp3_pairs["donor_type"], donor_sp3_pairs["acceptor_type"]]
            ).to(device)
            if len(donor_sp3_pairs)
            else None
        )

        donor_ring_pairs = cls._cross_pairs(elems.donors, elems.ring_acceptors)
        donor_ring_pair_params = (
            (
                params[
                    donor_ring_pairs["donor_type"], donor_ring_pairs["acceptor_type"]
                ]
            ).to(device)
            if len(donor_ring_pairs)
            else None
        )

        return cls(
            donor_sp2_pairs=donor_sp2_pairs,
            donor_sp2_pair_params=donor_sp2_pair_params,
            donor_sp3_pairs=donor_sp3_pairs,
            donor_sp3_pair_params=donor_sp3_pair_params,
            donor_ring_pairs=donor_ring_pairs,
            donor_ring_pair_params=donor_ring_pair_params,
        )


@reactive_attrs(auto_attribs=True)
class HBondScoreGraph(BondedAtomScoreGraph, ParamDB, TorchDevice, Factory):
    """Compute graph for the HBond term.

    It uses the reactive system to compute the list of donors and acceptors
    (via the HBondElementAnalysis class) and then the list of donor/acceptor
    pairs (via the HBondPairs class) once, and then reuses these lists.

    The h-bond functional form differs for the three classes of acceptors:
    sp2-, sp3-, and ring-hybridized acceptors. For this reason, these three are
    handled separately. Different donor types and different acceptor types within
    a group of the same hybridization will have different parameters / polynomials,
    but their functional forms will be the same, and so they can be processed
    together.

    The code (hopefully only for now?) evaluates the hydrogen bond potential
    between all pairs of acceptors and donors, in contrast to the Lennard-Jones
    term which looks only at nearby atom pairs. Perhaps future versions of
    this code will cull distant acceptor/donor pairs.
    """

    @staticmethod
    def factory_for(
        val,
        parameter_database: ParameterDatabase,
        hbond_database: Optional[HBondDatabase] = None,
        **_,
    ):
        """Overridable clone-constructor.

        Initialize from ``val.hbond_database`` if possible, otherwise from
        ``parameter_database.scoring.hbond``.
        """
        if hbond_database is None:
            if getattr(val, "hbond_database", None):
                hbond_database = val.hbond_database
            else:
                hbond_database = parameter_database.scoring.hbond

        return dict(hbond_database=hbond_database)

    hbond_database: HBondDatabase

    @property
    def component_atom_pair_dist_threshold(self):
        """Expose threshold distance for InteratomicDisanceGraph."""
        return self.hbond_database.global_parameters.threshold_distance

    @reactive_property
    @validate_args
    def hbond_param_resolver(hbond_database: HBondDatabase) -> HBondParamResolver:
        "hbond pair parameter resolver"
        return HBondParamResolver.from_database(hbond_database)

    @reactive_property
    @validate_args
    def hbond_elements(
        hbond_database: HBondDatabase,
        atom_types: NDArray(object)[:, :],
        bonds: NDArray(int)[:, 3],
    ) -> HBondElementAnalysis:
        """hbond score elements in target graph"""
        assert atom_types.shape[0] == 1
        assert numpy.all(bonds[:, 0] == 0)

        return HBondElementAnalysis.setup(
            hbond_database=hbond_database, atom_types=atom_types[0], bonds=bonds[:, 1:]
        )

    @reactive_property
    @validate_args
    def hbond_pairs(
        hbond_param_resolver: HBondParamResolver,
        hbond_elements: HBondElementAnalysis,
        device: torch.device,
    ) -> HBondPairs:
        """hbond pair metadata and parameters in target graph"""
        return HBondPairs.setup(hbond_param_resolver, hbond_elements, device)

    @reactive_property
    @validate_args
    def donor_sp2_hbond(
        coords: Tensor(torch.float)[:, :, 3],
        hbond_pairs: HBondPairs,
        hbond_database: HBondDatabase,
    ) -> Tensor(torch.float)[:]:
        """donor-sp2 hbond scores"""

        assert len(coords) == 1, "Only single depth supported"
        coords = coords[0]

        donor_sp2_pairs = hbond_pairs.donor_sp2_pairs
        donor_sp2_pair_params = hbond_pairs.donor_sp2_pair_params

        if len(donor_sp2_pairs) == 0:
            return coords.new(0)

        return hbond_donor_sp2_score(
            d=coords[donor_sp2_pairs["d"]],
            h=coords[donor_sp2_pairs["h"]],
            a=coords[donor_sp2_pairs["a"]],
            b=coords[donor_sp2_pairs["b"]],
            b0=coords[donor_sp2_pairs["b0"]],
            # type pair parameters
            glob_accwt=(donor_sp2_pair_params.acceptor_weight),
            glob_donwt=(donor_sp2_pair_params.donor_weight),
            AHdist_coeffs=(donor_sp2_pair_params.AHdist.coeffs),
            AHdist_ranges=(donor_sp2_pair_params.AHdist.range),
            AHdist_bounds=(donor_sp2_pair_params.AHdist.bound),
            cosBAH_coeffs=(donor_sp2_pair_params.cosBAH.coeffs),
            cosBAH_ranges=(donor_sp2_pair_params.cosBAH.range),
            cosBAH_bounds=(donor_sp2_pair_params.cosBAH.bound),
            cosAHD_coeffs=(donor_sp2_pair_params.cosAHD.coeffs),
            cosAHD_ranges=(donor_sp2_pair_params.cosAHD.range),
            cosAHD_bounds=(donor_sp2_pair_params.cosAHD.bound),
            # global parameters
            hb_sp2_range_span=(hbond_database.global_parameters.hb_sp2_range_span),
            hb_sp2_BAH180_rise=(hbond_database.global_parameters.hb_sp2_BAH180_rise),
            hb_sp2_outer_width=(hbond_database.global_parameters.hb_sp2_outer_width),
        )

    @reactive_property
    @validate_args
    def donor_sp3_hbond(
        coords: Tensor(torch.float)[:, :, 3],
        hbond_pairs: HBondPairs,
        hbond_database: HBondDatabase,
    ) -> Tensor(torch.float)[:]:
        donor_sp3_pairs = hbond_pairs.donor_sp3_pairs
        donor_sp3_pair_params = hbond_pairs.donor_sp3_pair_params

        assert len(coords) == 1, "Only single depth supported"
        coords = coords[0]

        if len(donor_sp3_pairs) == 0:
            return coords.new(0)

        return hbond_donor_sp3_score(
            d=coords[donor_sp3_pairs["d"]],
            h=coords[donor_sp3_pairs["h"]],
            a=coords[donor_sp3_pairs["a"]],
            b=coords[donor_sp3_pairs["b"]],
            b0=coords[donor_sp3_pairs["b0"]],
            # type pair parameters
            glob_accwt=(donor_sp3_pair_params.acceptor_weight),
            glob_donwt=(donor_sp3_pair_params.donor_weight),
            AHdist_coeffs=(donor_sp3_pair_params.AHdist.coeffs),
            AHdist_ranges=(donor_sp3_pair_params.AHdist.range),
            AHdist_bounds=(donor_sp3_pair_params.AHdist.bound),
            cosBAH_coeffs=(donor_sp3_pair_params.cosBAH.coeffs),
            cosBAH_ranges=(donor_sp3_pair_params.cosBAH.range),
            cosBAH_bounds=(donor_sp3_pair_params.cosBAH.bound),
            cosAHD_coeffs=(donor_sp3_pair_params.cosAHD.coeffs),
            cosAHD_ranges=(donor_sp3_pair_params.cosAHD.range),
            cosAHD_bounds=(donor_sp3_pair_params.cosAHD.bound),
            # global parameters
            hb_sp3_softmax_fade=(hbond_database.global_parameters.hb_sp3_softmax_fade),
        )

    @reactive_property
    @validate_args
    def donor_ring_hbond(
        coords: Tensor(torch.float)[:, :, 3],
        hbond_pairs: HBondPairs,
        hbond_database: HBondDatabase,
    ) -> Tensor(torch.float)[:]:
        """donor-ring hbond scores"""

        donor_ring_pairs = hbond_pairs.donor_ring_pairs
        donor_ring_pair_params = hbond_pairs.donor_ring_pair_params
        assert len(coords) == 1, "Only single layer supported."
        coords = coords[0]

        if len(donor_ring_pairs) == 0:
            return coords.new(0)

        return hbond_donor_ring_score(
            d=coords[donor_ring_pairs["d"]],
            h=coords[donor_ring_pairs["h"]],
            a=coords[donor_ring_pairs["a"]],
            b=coords[donor_ring_pairs["b"]],
            b0=coords[donor_ring_pairs["b0"]],
            # type pair parameters
            glob_accwt=(donor_ring_pair_params.acceptor_weight),
            glob_donwt=(donor_ring_pair_params.donor_weight),
            AHdist_coeffs=(donor_ring_pair_params.AHdist.coeffs),
            AHdist_ranges=(donor_ring_pair_params.AHdist.range),
            AHdist_bounds=(donor_ring_pair_params.AHdist.bound),
            cosBAH_coeffs=(donor_ring_pair_params.cosBAH.coeffs),
            cosBAH_ranges=(donor_ring_pair_params.cosBAH.range),
            cosBAH_bounds=(donor_ring_pair_params.cosBAH.bound),
            cosAHD_coeffs=(donor_ring_pair_params.cosAHD.coeffs),
            cosAHD_ranges=(donor_ring_pair_params.cosAHD.range),
            cosAHD_bounds=(donor_ring_pair_params.cosAHD.bound),
        )

    @reactive_property
    @validate_args
    def total_hbond(
        donor_sp2_hbond: Tensor(torch.float)[:],
        donor_sp3_hbond: Tensor(torch.float)[:],
        donor_ring_hbond: Tensor(torch.float)[:],
    ) -> Tensor(torch.float)[:]:
        """total hbond score"""
        return (
            donor_sp2_hbond.sum() + donor_sp3_hbond.sum() + donor_ring_hbond.sum()
        ).reshape((1,))

    @reactive_property
    @validate_args
    def hbond_scores(
        donor_sp2_hbond: Tensor(torch.float)[:],
        donor_sp3_hbond: Tensor(torch.float)[:],
        donor_ring_hbond: Tensor(torch.float)[:],
    ) -> Tensor(torch.float)[:]:
        return torch.cat((donor_sp2_hbond, donor_sp3_hbond, donor_ring_hbond))

    @reactive_property
    @validate_args
    def hbond_pair_metadata(hbond_pairs: HBondPairs) -> NDArray(pair_descr_dtype)[:]:
        """All hbond pairs, in order of "sp2"/"sp3"/"ring"."""
        return numpy.concatenate(
            (
                hbond_pairs.donor_sp2_pairs,
                hbond_pairs.donor_sp3_pairs,
                hbond_pairs.donor_ring_pairs,
            )
        )
