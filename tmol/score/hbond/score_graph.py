import attr
from properties import Instance

import torch
import numpy

from tmol.properties.array import VariableT, Array
from tmol.properties.reactive import derived_from

from ..bonded_atom import ScoreComponentAttributes
from ..interatomic_distance import InteratomicDistanceGraphBase

from .potentials import (
    hbond_donor_sp2_score,
    hbond_donor_sp3_score,
    hbond_donor_ring_score,
)

from .identification import (
    HBondElementAnalysis,
    donor_dtype,
    acceptor_dtype,
)
from .params import HBondParamResolver, HBondPairParams

import tmol.database
from tmol.database.scoring import HBondDatabase

from tmol.types.attrs import ValidateAttrs
from tmol.types.functional import validate_args
from tmol.types.array import NDArray

pair_descr_dtype = numpy.dtype(donor_dtype.descr + acceptor_dtype.descr)


@attr.s(frozen=True, auto_attribs=True, slots=True)
class HBondPairs(ValidateAttrs):
    donor_sp2_pairs: NDArray(pair_descr_dtype)[:]
    donor_sp2_pair_params: HBondPairParams

    donor_sp3_pairs: NDArray(pair_descr_dtype)[:]
    donor_sp3_pair_params: HBondPairParams

    donor_ring_pairs: NDArray(pair_descr_dtype)[:]
    donor_ring_pair_params: HBondPairParams

    @staticmethod
    @validate_args
    def _cross_pairs(
            donors: NDArray(donor_dtype)[:],
            acceptors: NDArray(acceptor_dtype)[:]
    ):
        pairs = numpy.empty(
            (len(donors), len(acceptors)),
            dtype=pair_descr_dtype,
        )

        for n in donors.dtype.names:
            pairs[n] = donors[n].reshape((-1, 1))

        for n in acceptors.dtype.names:
            pairs[n] = acceptors[n].reshape((1, -1))

        pairs = pairs.ravel()

        return pairs[pairs["a"] != pairs["d"]]

    @classmethod
    @validate_args
    def setup(cls, params: HBondParamResolver, elems: HBondElementAnalysis):
        # Get blocks of acceptor/donor pairs for each set of identified acceptors
        donor_sp2_pairs = cls._cross_pairs(elems.donors, elems.sp2_acceptors)
        donor_sp2_pair_params = (
            params[donor_sp2_pairs["donor_type"],
                   donor_sp2_pairs["acceptor_type"]]
        )

        donor_sp3_pairs = cls._cross_pairs(elems.donors, elems.sp3_acceptors)
        donor_sp3_pair_params = (
            params[donor_sp3_pairs["donor_type"],
                   donor_sp3_pairs["acceptor_type"]]
        )

        donor_ring_pairs = cls._cross_pairs(elems.donors, elems.ring_acceptors)
        donor_ring_pair_params = (
            params[donor_ring_pairs["donor_type"],
                   donor_ring_pairs["acceptor_type"]]
        )

        return cls(
            donor_sp2_pairs=donor_sp2_pairs,
            donor_sp2_pair_params=donor_sp2_pair_params,
            donor_sp3_pairs=donor_sp3_pairs,
            donor_sp3_pair_params=donor_sp3_pair_params,
            donor_ring_pairs=donor_ring_pairs,
            donor_ring_pair_params=donor_ring_pair_params,
        )


class HBondScoreGraph(InteratomicDistanceGraphBase):

    hbond_database: HBondDatabase = Instance(
        "hbond parameter database",
        HBondDatabase,
        default=tmol.database.default.scoring.hbond
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.atom_pair_dist_thresholds.add(
            self.hbond_database.global_parameters.threshold_distance
        )
        self.score_components.add(
            ScoreComponentAttributes("hbond", "total_hbond", None)
        )

    @derived_from(
        "hbond_database",
        Instance("hbond pair parameter resolver", HBondParamResolver)
    )
    def hbond_param_resolver(self) -> HBondParamResolver:
        return HBondParamResolver.from_database(self.hbond_database)

    @derived_from(("hbond_database", "atom_types", "bonds"),
                  Instance(
                      "hbond score elements in target =graph",
                      HBondElementAnalysis
                  ))
    def hbond_elements(self) -> HBondElementAnalysis:
        return HBondElementAnalysis.setup(
            hbond_database=self.hbond_database,
            atom_types=self.atom_types,
            bonds=self.bonds
        )

    @derived_from(("hbond_param_resolver", "hbond_elements"),
                  Instance(
                      "hbond pair metadata and parameters in target graph",
                      HBondPairs
                  ))
    def hbond_pairs(self) -> HBondPairs:
        return HBondPairs.setup(self.hbond_param_resolver, self.hbond_elements)

    @derived_from(
        ("coords", "hbond_pairs"),
        VariableT("donor-sp2 hbond scores"),
    )
    def donor_sp2_hbond(self):

        donor_sp2_pairs = self.hbond_pairs.donor_sp2_pairs
        donor_sp2_pair_params = self.hbond_pairs.donor_sp2_pair_params

        if len(donor_sp2_pairs) == 0:
            return self.coords.new(0)

        return hbond_donor_sp2_score(
            d=self.coords[donor_sp2_pairs["d"]],
            h=self.coords[donor_sp2_pairs["h"]],
            a=self.coords[donor_sp2_pairs["a"]],
            b=self.coords[donor_sp2_pairs["b"]],
            b0=self.coords[donor_sp2_pairs["b0"]],

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
            hb_sp2_range_span=(
                self.hbond_database.global_parameters.hb_sp2_range_span
            ),
            hb_sp2_BAH180_rise=(
                self.hbond_database.global_parameters.hb_sp2_BAH180_rise
            ),
            hb_sp2_outer_width=(
                self.hbond_database.global_parameters.hb_sp2_outer_width
            ),
        )

    @derived_from(
        ("coords", "hbond_elements"),
        VariableT("donor-sp3 hbond scores"),
    )
    def donor_sp3_hbond(self):
        donor_sp3_pairs = self.hbond_pairs.donor_sp3_pairs
        donor_sp3_pair_params = self.hbond_pairs.donor_sp3_pair_params

        if len(donor_sp3_pairs) == 0:
            return self.coords.new(0)

        return hbond_donor_sp3_score(
            d=self.coords[donor_sp3_pairs["d"]],
            h=self.coords[donor_sp3_pairs["h"]],
            a=self.coords[donor_sp3_pairs["a"]],
            b=self.coords[donor_sp3_pairs["b"]],
            b0=self.coords[donor_sp3_pairs["b0"]],

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
            hb_sp3_softmax_fade=(
                self.hbond_database.global_parameters.hb_sp3_softmax_fade
            ),
        )

    @derived_from(
        ("coords", "hbond_elements"),
        VariableT("donor-ring hbond scores"),
    )
    def donor_ring_hbond(self):
        donor_ring_pairs = self.hbond_pairs.donor_ring_pairs
        donor_ring_pair_params = self.hbond_pairs.donor_ring_pair_params

        if len(donor_ring_pairs) == 0:
            return self.coords.new(0)

        return hbond_donor_ring_score(
            d=self.coords[donor_ring_pairs["d"]],
            h=self.coords[donor_ring_pairs["h"]],
            a=self.coords[donor_ring_pairs["a"]],
            b=self.coords[donor_ring_pairs["b"]],
            b0=self.coords[donor_ring_pairs["b0"]],

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

    @derived_from(
        ("donor_sp2_hbond", "donor_sp3_hbond", "donor_ring_hbond"),
        VariableT("total hbond score"),
    )
    def total_hbond(self):
        return self.donor_sp2_hbond.sum() + self.donor_sp3_hbond.sum(
        ) + self.donor_ring_hbond.sum()

    @derived_from(
        ("donor_sp2_hbond", "donor_sp3_hbond", "donor_ring_hbond"),
        VariableT("total hbond score"),
    )
    def hbond_scores(self):
        return torch.cat((
            self.donor_sp2_hbond,
            self.donor_sp3_hbond,
            self.donor_ring_hbond,
        ))

    @derived_from(
        "hbond_pairs",
        Array(
            "hbond pair metadata in target graph",
            dtype=pair_descr_dtype,
        )[:]
    )
    def hbond_pair_metadata(self):
        """All hbond pairs, in order of "sp2"/"sp3"/"ring"."""
        return numpy.concatenate((
            self.hbond_pairs.donor_sp2_pairs,
            self.hbond_pairs.donor_sp3_pairs,
            self.hbond_pairs.donor_ring_pairs,
        ))
