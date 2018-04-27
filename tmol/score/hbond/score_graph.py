import properties
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

from .identification import HBondElementAnalysis
from .params import HBondParamResolver, HBondPairParams

import tmol.database
from tmol.database.scoring import HBondDatabase


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
        analysis = HBondElementAnalysis(
            hbond_database=self.hbond_database,
            atom_types=self.atom_types,
            bonds=self.bonds
        ).setup()

        analysis.validate()

        return analysis

    @derived_from(
        "hbond_elements",
        Array(
            "donor to sp2 acceptor pairs",
            dtype=(
                HBondElementAnalysis.donor_dtype.descr +
                HBondElementAnalysis.sp2_acceptor_dtype.descr
            )
        )
    )
    def donor_sp2_pairs(self):
        dons = self.hbond_elements.donors
        accs = self.hbond_elements.sp2_acceptors

        pairs = numpy.empty((len(dons), len(accs)),
                            dtype=dons.dtype.descr + accs.dtype.descr)

        for n in dons.dtype.names:
            pairs[n] = dons[n].reshape((-1, 1))

        for n in accs.dtype.names:
            pairs[n] = accs[n].reshape((1, -1))

        return pairs.ravel()

    @derived_from(("hbond_param_resolver", "donor_sp2_pairs"),
                  properties.Instance(
                      "donor to sp2 acceptor pair parameters",
                      HBondPairParams,
                  ))
    def donor_sp2_params(self):
        return self.hbond_param_resolver[self.donor_sp2_pairs["donor_type"],
                                         self.donor_sp2_pairs["acceptor_type"]]

    @derived_from(
        "hbond_elements",
        Array(
            "donor to sp3 acceptor pairs",
            dtype=(
                HBondElementAnalysis.donor_dtype.descr +
                HBondElementAnalysis.sp3_acceptor_dtype.descr
            )
        )
    )
    def donor_sp3_pairs(self):
        dons = self.hbond_elements.donors
        accs = self.hbond_elements.sp3_acceptors

        pairs = numpy.empty(
            (len(dons), len(accs)),
            dtype=dons.dtype.descr + accs.dtype.descr,
        )

        for n in dons.dtype.names:
            pairs[n] = dons[n].reshape((-1, 1))

        for n in accs.dtype.names:
            pairs[n] = accs[n].reshape((1, -1))

        return pairs.ravel()

    @derived_from(("hbond_param_resolver", "donor_sp3_pairs"),
                  properties.Instance(
                      "donor to sp3 acceptor pair parameters",
                      HBondPairParams,
                  ))
    def donor_sp3_params(self):
        return self.hbond_param_resolver[self.donor_sp3_pairs["donor_type"],
                                         self.donor_sp3_pairs["acceptor_type"]]

    @derived_from(
        "hbond_elements",
        Array(
            "donor to ring acceptor pairs",
            dtype=(
                HBondElementAnalysis.donor_dtype.descr +
                HBondElementAnalysis.ring_acceptor_dtype.descr
            )
        )
    )
    def donor_ring_pairs(self):
        dons = self.hbond_elements.donors
        accs = self.hbond_elements.ring_acceptors

        pairs = numpy.empty((len(dons), len(accs)),
                            dtype=dons.dtype.descr + accs.dtype.descr)

        for n in dons.dtype.names:
            pairs[n] = dons[n].reshape((-1, 1))

        for n in accs.dtype.names:
            pairs[n] = accs[n].reshape((1, -1))

        return pairs.ravel()

    @derived_from(("hbond_param_resolver", "donor_ring_pairs"),
                  properties.Instance(
                      "donor to ring acceptor pair parameters",
                      HBondPairParams,
                  ))
    def donor_ring_params(self):
        return (
            self.hbond_param_resolver[self.donor_ring_pairs["donor_type"],
                                      self.donor_ring_pairs["acceptor_type"]]
        )

    @derived_from(
        ("coords", "hbond_elements"),
        VariableT("donor-sp2 hbond scores"),
    )
    def donor_sp2_hbond(self):
        if len(self.donor_sp2_pairs) == 0:
            return self.coords.new(0)

        return hbond_donor_sp2_score(
            d=self.coords[self.donor_sp2_pairs["d"]],
            h=self.coords[self.donor_sp2_pairs["h"]],
            a=self.coords[self.donor_sp2_pairs["a"]],
            b=self.coords[self.donor_sp2_pairs["b"]],
            b0=self.coords[self.donor_sp2_pairs["b0"]],

            # type pair parameters
            glob_accwt=(self.donor_sp2_params.acceptor_weight),
            glob_donwt=(self.donor_sp2_params.donor_weight),
            AHdist_coeffs=(self.donor_sp2_params.AHdist.coeffs),
            AHdist_ranges=(self.donor_sp2_params.AHdist.range),
            AHdist_bounds=(self.donor_sp2_params.AHdist.bound),
            cosBAH_coeffs=(self.donor_sp2_params.cosBAH.coeffs),
            cosBAH_ranges=(self.donor_sp2_params.cosBAH.range),
            cosBAH_bounds=(self.donor_sp2_params.cosBAH.bound),
            cosAHD_coeffs=(self.donor_sp2_params.cosAHD.coeffs),
            cosAHD_ranges=(self.donor_sp2_params.cosAHD.range),
            cosAHD_bounds=(self.donor_sp2_params.cosAHD.bound),

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
        if len(self.donor_sp3_pairs) == 0:
            return self.coords.new(0)

        return hbond_donor_sp3_score(
            d=self.coords[self.donor_sp3_pairs["d"]],
            h=self.coords[self.donor_sp3_pairs["h"]],
            a=self.coords[self.donor_sp3_pairs["a"]],
            b=self.coords[self.donor_sp3_pairs["b"]],
            b0=self.coords[self.donor_sp3_pairs["b0"]],

            # type pair parameters
            glob_accwt=(self.donor_sp3_params.acceptor_weight),
            glob_donwt=(self.donor_sp3_params.donor_weight),
            AHdist_coeffs=(self.donor_sp3_params.AHdist.coeffs),
            AHdist_ranges=(self.donor_sp3_params.AHdist.range),
            AHdist_bounds=(self.donor_sp3_params.AHdist.bound),
            cosBAH_coeffs=(self.donor_sp3_params.cosBAH.coeffs),
            cosBAH_ranges=(self.donor_sp3_params.cosBAH.range),
            cosBAH_bounds=(self.donor_sp3_params.cosBAH.bound),
            cosAHD_coeffs=(self.donor_sp3_params.cosAHD.coeffs),
            cosAHD_ranges=(self.donor_sp3_params.cosAHD.range),
            cosAHD_bounds=(self.donor_sp3_params.cosAHD.bound),

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
        if len(self.donor_ring_pairs) == 0:
            return self.coords.new(0)

        return hbond_donor_ring_score(
            d=self.coords[self.donor_ring_pairs["d"]],
            h=self.coords[self.donor_ring_pairs["h"]],
            a=self.coords[self.donor_ring_pairs["a"]],
            b=self.coords[self.donor_ring_pairs["b"]],
            bp=self.coords[self.donor_ring_pairs["bp"]],

            # type pair parameters
            glob_accwt=(self.donor_ring_params.acceptor_weight),
            glob_donwt=(self.donor_ring_params.donor_weight),
            AHdist_coeffs=(self.donor_ring_params.AHdist.coeffs),
            AHdist_ranges=(self.donor_ring_params.AHdist.range),
            AHdist_bounds=(self.donor_ring_params.AHdist.bound),
            cosBAH_coeffs=(self.donor_ring_params.cosBAH.coeffs),
            cosBAH_ranges=(self.donor_ring_params.cosBAH.range),
            cosBAH_bounds=(self.donor_ring_params.cosBAH.bound),
            cosAHD_coeffs=(self.donor_ring_params.cosAHD.coeffs),
            cosAHD_ranges=(self.donor_ring_params.cosAHD.range),
            cosAHD_bounds=(self.donor_ring_params.cosAHD.bound),
        )

    @derived_from(
        ("donor_sp2_hbond", "donor_sp3_hbond", "donor_ring_hbond"),
        VariableT("total hbond score"),
    )
    def total_hbond(self):
        return self.donor_sp2_hbond.sum() + self.donor_sp3_hbond.sum(
        ) + self.donor_ring_hbond.sum()

    @derived_from(
        "hbond_elements",
        VariableT("Donor atom indices, all hbond types."),
    )
    def hbond_donor_ind(self):
        return torch.LongTensor(
            numpy.concatenate((
                self.donor_sp2_pairs["d"],
                self.donor_sp3_pairs["d"],
                self.donor_ring_pairs["d"],
            ))
        )

    @derived_from(
        "hbond_elements",
        VariableT("Hydrogen atom indices, all hbond types."),
    )
    def hbond_h_ind(self):
        return torch.LongTensor(
            numpy.concatenate((
                self.donor_sp2_pairs["h"],
                self.donor_sp3_pairs["h"],
                self.donor_ring_pairs["h"],
            ))
        )

    @derived_from(
        "hbond_elements",
        VariableT("Acceptor atom indices, all hbond types."),
    )
    def hbond_acceptor_ind(self):
        return torch.LongTensor(
            numpy.concatenate((
                self.donor_sp2_pairs["a"],
                self.donor_sp3_pairs["a"],
                self.donor_ring_pairs["a"],
            ))
        )

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
