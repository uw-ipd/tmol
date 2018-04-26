import attr
import cattr
import properties
from properties import Instance
import toolz

from typing import Dict

import torch
import numpy
import pandas

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

import tmol.database
from tmol.database.scoring import HBondDatabase


@attr.s(frozen=True, slots=True)
class HBondParamResolver:
    hbdb: HBondDatabase = attr.ib()

    param_lookup: pandas.DataFrame = attr.ib()

    @param_lookup.default
    def _init_param_lookup(self):
        to_frame = toolz.compose(
            pandas.DataFrame.from_records,
            cattr.unstructure,
        )

        # Get polynomial parameters index by polynomial name
        poly_params = (
            to_frame(self.hbdb.polynomial_parameters)
            .rename(columns={"name": "polynomial"})
            .set_index("polynomial")
            .drop(columns=["degree", "dimension"])
        )  # yapf: disable

        # Convert pair parameter table into a multi-index frame specifying
        # *just* the polynomial name used for that term/don/acc set
        pair_params = (
            to_frame(self.hbdb.pair_parameters)
            .set_index(["don_chem_type", "acc_chem_type"])
            .rename_axis("term", 1)
            .stack()  # Convert column names into an index level
            .reorder_levels(("term", "don_chem_type", "acc_chem_type"))
            .to_frame("polynomial")
        )  # yapf: disable

        # Merge into a single frame of poly paramerters, indexed by
        # term/donor/acceptor triple
        return pandas.merge(
            pair_params,
            poly_params,
            how="left",
            left_on="polynomial",
            right_index=True
        )

    weight_lookup: pandas.DataFrame = attr.ib()

    @weight_lookup.default
    def _init_weight_lookup(self):
        to_frame = toolz.compose(
            pandas.DataFrame.from_records,
            cattr.unstructure,
        )

        # acc/don weights
        acc_params = (
            to_frame(self.hbdb.acc_weights)
            .rename(columns={
                "weight": "acc_weight",
                "name": "acc_name"
            })
        )

        don_params = (
            to_frame(self.hbdb.don_weights)
            .rename(columns={
                "weight": "don_weight",
                "name": "don_name"
            })
        )

        # cross join
        result_index = pandas.MultiIndex.from_product(
            [acc_params["acc_name"], don_params["don_name"]],
            names=["acc_name", "don_name"],
        )

        donor_acceptor_pair_frame = pandas.DataFrame(index=result_index
                                                     ).reset_index()

        result_frame = toolz.reduce(
            pandas.merge,
            (donor_acceptor_pair_frame, acc_params, don_params)
        ).set_index(["don_name", "acc_name"])

        return result_frame

    type_pair_index: pandas.Series = attr.ib()

    @type_pair_index.default
    def _init_type_pair_index(self):
        type_index = self.param_lookup.index.droplevel("term"
                                                       ).drop_duplicates()
        return (
            pandas.Series(
                index=type_index, data=numpy.arange(len(type_index))
            ).sort_index()
        )

    param_tensors: Dict[str, torch.Tensor] = attr.ib()

    @param_tensors.default
    def _init_param_tensors(self):
        # {term : {param : tensor}}
        normalized_param_tensors: Dict[str, Dict[str, torch.Tensor]] = {
            t: toolz.dicttoolz.merge({
                "coeffs":
                    torch.Tensor(
                        params[["c_" + i for i in "abcdefghijk"]].values
                    ),
            }, {
                "ranges": torch.Tensor(params[["xmin", "xmax"]].values),
            }, {
                "bounds": torch.Tensor(params[["min_val", "max_val"]].values),
            })
            for t, params in self.param_lookup.groupby(level="term")
        }

        # is weight_lookup to be in the same order as param_lookup?
        # both are indexed on ["don_name","acc_name"]
        normalized_param_tensors["glob"] = {
            "donwt": torch.Tensor(self.weight_lookup["don_weight"].values),
            "accwt": torch.Tensor(self.weight_lookup["acc_weight"].values)
        }

        return {
            f"{term}_{param}": tensor
            for term in normalized_param_tensors
            for param, tensor in normalized_param_tensors[term].items()
        }

    def type_pair_indices(self, donor_types, acceptor_types):
        """Resolve donor/acceptor types into first-dimension index arrays into param tensors."""
        query_index = pandas.MultiIndex.from_arrays(
            (donor_types, acceptor_types),
            names=self.type_pair_index.index.names
        )

        return pandas.merge(
            pandas.Series(
                data=numpy.arange(len(query_index)), index=query_index
            ).to_frame("pair_idx"),
            self.type_pair_index.to_frame("type_pair_idx"),
            left_index=True,
            right_index=True,
            how="left"
        )["type_pair_idx"].values


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
    def hbond_param_resolver(self):
        return HBondParamResolver(self.hbond_database)

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

    @derived_from(
        ("hbond_param_resolver", "donor_sp2_pairs"),
        properties.Dictionary("donor to sp2 acceptor pair parameters")
    )
    def donor_sp2_params(self):
        d_sp2_param_indices = self.hbond_param_resolver.type_pair_indices(
            self.donor_sp2_pairs["donor_type"],
            self.donor_sp2_pairs["acceptor_type"]
        )

        return {
            param: tensor[d_sp2_param_indices]
            for param, tensor in
            self.hbond_param_resolver.param_tensors.items()
        }

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

    @derived_from(
        ("hbond_param_resolver", "donor_sp3_pairs"),
        properties.Dictionary("donor to sp3 acceptor pair parameters")
    )
    def donor_sp3_params(self):
        d_sp3_param_indices = self.hbond_param_resolver.type_pair_indices(
            self.donor_sp3_pairs["donor_type"],
            self.donor_sp3_pairs["acceptor_type"]
        )

        return {
            param: tensor[d_sp3_param_indices]
            for param, tensor in
            self.hbond_param_resolver.param_tensors.items()
        }

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

    @derived_from(
        ("hbond_param_resolver", "donor_ring_pairs"),
        properties.Dictionary("donor to ring acceptor pair parameters")
    )
    def donor_ring_params(self):
        d_ring_param_indices = self.hbond_param_resolver.type_pair_indices(
            self.donor_ring_pairs["donor_type"],
            self.donor_ring_pairs["acceptor_type"]
        )

        return {
            param: tensor[d_ring_param_indices]
            for param, tensor in
            self.hbond_param_resolver.param_tensors.items()
        }

    @derived_from(
        ("coords", "hbond_elements"),
        VariableT("donor-sp2 hbond scores"),
    )
    def donor_sp2_hbond(self):
        if len(self.donor_sp2_pairs) == 0:
            return self.coords.new(0)

        coord_params = dict(
            d=self.coords[self.donor_sp2_pairs["d"]],
            h=self.coords[self.donor_sp2_pairs["h"]],
            a=self.coords[self.donor_sp2_pairs["a"]],
            b=self.coords[self.donor_sp2_pairs["b"]],
            b0=self.coords[self.donor_sp2_pairs["b0"]],
        )

        pair_params = self.donor_sp2_params

        global_params = dict(
            hb_sp2_range_span=self.hbond_database.global_parameters.
            hb_sp2_range_span,
            hb_sp2_BAH180_rise=self.hbond_database.global_parameters.
            hb_sp2_BAH180_rise,
            hb_sp2_outer_width=self.hbond_database.global_parameters.
            hb_sp2_outer_width,
        )

        return hbond_donor_sp2_score(
            **toolz.dicttoolz.merge(coord_params, pair_params, global_params)
        )

    @derived_from(
        ("coords", "hbond_elements"),
        VariableT("donor-sp3 hbond scores"),
    )
    def donor_sp3_hbond(self):
        if len(self.donor_sp3_pairs) == 0:
            return self.coords.new(0)

        coord_params = dict(
            d=self.coords[self.donor_sp3_pairs["d"]],
            h=self.coords[self.donor_sp3_pairs["h"]],
            a=self.coords[self.donor_sp3_pairs["a"]],
            b=self.coords[self.donor_sp3_pairs["b"]],
            b0=self.coords[self.donor_sp3_pairs["b0"]],
        )

        pair_params = self.donor_sp3_params

        global_params = dict(
            hb_sp3_softmax_fade=self.hbond_database.global_parameters.
            hb_sp3_softmax_fade,
        )

        return hbond_donor_sp3_score(
            **toolz.dicttoolz.merge(coord_params, pair_params, global_params)
        )

    @derived_from(
        ("coords", "hbond_elements"),
        VariableT("donor-ring hbond scores"),
    )
    def donor_ring_hbond(self):
        if len(self.donor_ring_pairs) == 0:
            return self.coords.new(0)

        coord_params = dict(
            d=self.coords[self.donor_ring_pairs["d"]],
            h=self.coords[self.donor_ring_pairs["h"]],
            a=self.coords[self.donor_ring_pairs["a"]],
            b=self.coords[self.donor_ring_pairs["b"]],
            bp=self.coords[self.donor_ring_pairs["bp"]],
        )

        pair_params = self.donor_ring_params

        return hbond_donor_ring_score(
            **toolz.dicttoolz.merge(coord_params, pair_params)
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
