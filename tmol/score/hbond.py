import attr
import cattr
import properties
from properties import Instance
import toolz
from toolz.curried import compose

from typing import Dict

import torch
import numpy
import pandas

from tmol.properties.array import VariableT, Array
from tmol.properties.reactive import derived_from

from .bonded_atom import ScoreComponentAttributes
from .interatomic_distance import InteratomicDistanceGraphBase

import tmol.database
from tmol.database.scoring import HBondDatabase


def hbond_donor_sp2_score(
        # Input coordinates
        d,
        h,
        a,
        b,
        b0,

        # type pair parameters
        AHdist_xmin,
        AHdist_xmax,
        AHdist_min_val,
        AHdist_max_val,
        AHdist_root1,
        AHdist_root2,
        AHdist_coeffs,
        cosBAH_short_xmin,
        cosBAH_short_xmax,
        cosBAH_short_min_val,
        cosBAH_short_max_val,
        cosBAH_short_root1,
        cosBAH_short_root2,
        cosBAH_short_coeffs,
        cosBAH_long_xmin,
        cosBAH_long_xmax,
        cosBAH_long_min_val,
        cosBAH_long_max_val,
        cosBAH_long_root1,
        cosBAH_long_root2,
        cosBAH_long_coeffs,
        cosBAH2_long_xmin,
        cosBAH2_long_xmax,
        cosBAH2_long_min_val,
        cosBAH2_long_max_val,
        cosBAH2_long_root1,
        cosBAH2_long_root2,
        cosBAH2_long_coeffs,
        cosAHD_short_xmin,
        cosAHD_short_xmax,
        cosAHD_short_min_val,
        cosAHD_short_max_val,
        cosAHD_short_root1,
        cosAHD_short_root2,
        cosAHD_short_coeffs,
        cosAHD_long_xmin,
        cosAHD_long_xmax,
        cosAHD_long_min_val,
        cosAHD_long_max_val,
        cosAHD_long_root1,
        cosAHD_long_root2,
        cosAHD_long_coeffs,

        # Global score parameters
        max_dis
):
    # h_d_vec = (d - h)
    # h_d_dist = h_d_vec.norm(dim=-1)
    # h_d_unit = h_d_vec / h_d_dist.unsqueeze(dim=-1)

    a_h_vec = (h - a)
    a_h_dist = a_h_vec.norm(dim=-1)
    # a_h_unit = a_h_vec / a_h_dist.unsqueeze(dim=-1)

    # b_a_vec = (a - b)
    # b_a_dist = b_a_vec.norm(dim=-1)
    # b_a_unit = b_a_vec / b_a_dist.unsqueeze(dim=-1)

    # d_a_dist = (d - a).norm(dim=-1)
    return (a_h_dist < max_dis).type(d.dtype)


def hbond_donor_sp3_score(
        # Input coordinates
        d,
        h,
        a,
        b,
        b0,

        # type pair parameters
        AHdist_xmin,
        AHdist_xmax,
        AHdist_min_val,
        AHdist_max_val,
        AHdist_root1,
        AHdist_root2,
        AHdist_coeffs,
        cosBAH_short_xmin,
        cosBAH_short_xmax,
        cosBAH_short_min_val,
        cosBAH_short_max_val,
        cosBAH_short_root1,
        cosBAH_short_root2,
        cosBAH_short_coeffs,
        cosBAH_long_xmin,
        cosBAH_long_xmax,
        cosBAH_long_min_val,
        cosBAH_long_max_val,
        cosBAH_long_root1,
        cosBAH_long_root2,
        cosBAH_long_coeffs,
        cosBAH2_long_xmin,
        cosBAH2_long_xmax,
        cosBAH2_long_min_val,
        cosBAH2_long_max_val,
        cosBAH2_long_root1,
        cosBAH2_long_root2,
        cosBAH2_long_coeffs,
        cosAHD_short_xmin,
        cosAHD_short_xmax,
        cosAHD_short_min_val,
        cosAHD_short_max_val,
        cosAHD_short_root1,
        cosAHD_short_root2,
        cosAHD_short_coeffs,
        cosAHD_long_xmin,
        cosAHD_long_xmax,
        cosAHD_long_min_val,
        cosAHD_long_max_val,
        cosAHD_long_root1,
        cosAHD_long_root2,
        cosAHD_long_coeffs,

        # Global score parameters
        max_dis
):
    a_h_vec = (h - a)
    a_h_dist = a_h_vec.norm(dim=-1)

    return (a_h_dist < max_dis).type(d.dtype)


def hbond_donor_ring_score(
        # Input coordinates
        d,
        h,
        a,
        b,
        bp,

        # type pair parameters
        AHdist_xmin,
        AHdist_xmax,
        AHdist_min_val,
        AHdist_max_val,
        AHdist_root1,
        AHdist_root2,
        AHdist_coeffs,
        cosBAH_short_xmin,
        cosBAH_short_xmax,
        cosBAH_short_min_val,
        cosBAH_short_max_val,
        cosBAH_short_root1,
        cosBAH_short_root2,
        cosBAH_short_coeffs,
        cosBAH_long_xmin,
        cosBAH_long_xmax,
        cosBAH_long_min_val,
        cosBAH_long_max_val,
        cosBAH_long_root1,
        cosBAH_long_root2,
        cosBAH_long_coeffs,
        cosBAH2_long_xmin,
        cosBAH2_long_xmax,
        cosBAH2_long_min_val,
        cosBAH2_long_max_val,
        cosBAH2_long_root1,
        cosBAH2_long_root2,
        cosBAH2_long_coeffs,
        cosAHD_short_xmin,
        cosAHD_short_xmax,
        cosAHD_short_min_val,
        cosAHD_short_max_val,
        cosAHD_short_root1,
        cosAHD_short_root2,
        cosAHD_short_coeffs,
        cosAHD_long_xmin,
        cosAHD_long_xmax,
        cosAHD_long_min_val,
        cosAHD_long_max_val,
        cosAHD_long_root1,
        cosAHD_long_root2,
        cosAHD_long_coeffs,

        # Global score parameters
        max_dis
):
    a_h_vec = (h - a)
    a_h_dist = a_h_vec.norm(dim=-1)

    return (a_h_dist < max_dis).type(d.dtype)


class HBondElementAnalysis(properties.HasProperties):
    hbond_database: HBondDatabase = Instance(
        "hbond parameter database",
        HBondDatabase,
        default=tmol.database.default.scoring.hbond
    )

    atom_types = Array("atomic types", dtype=object)[:]
    bonds = Array("inter-atomic bond graph", dtype=int)[:, 2]

    donor_dtype = numpy.dtype([("d", int), ("h", int), ("donor_type", object)])
    donors = Array("Identified donor atom indices.", dtype=donor_dtype)[:]

    sp2_acceptor_dtype = numpy.dtype([("a", int),
                                      ("b", int),
                                      ("b0", int),
                                      ("acceptor_type", object)])
    sp2_acceptors = Array(
        "Identified sp2 acceptor atom indices.", dtype=sp2_acceptor_dtype
    )[:]

    sp3_acceptor_dtype = numpy.dtype([("a", int),
                                      ("b", int),
                                      ("b0", int),
                                      ("acceptor_type", object)])
    sp3_acceptors = Array(
        "Identified sp3 acceptor atom indices.", dtype=sp3_acceptor_dtype
    )[:]

    ring_acceptor_dtype = numpy.dtype([("a", int),
                                       ("b", int),
                                       ("bp", int),
                                       ("acceptor_type", object)])
    ring_acceptors = Array(
        "Identified ring acceptor atom indices.", dtype=ring_acceptor_dtype
    )[:]

    def setup(self):
        self: HBondElementAnalysis
        bond_types = self.atom_types[self.bonds]

        bond_table = pandas.DataFrame.from_dict({
            "i_i": self.bonds[:, 0],
            "i_t": bond_types[:, 0],
            "j_i": self.bonds[:, 1],
            "j_t": bond_types[:, 1],
        })

        def inc_cols(*args):
            order = {"i": "j", "j": "k"}
            res = []
            for n in args:
                nn = order[n]
                res.append((n + "_i", nn + "_i"))
                res.append((n + "_t", nn + "_t"))
            return dict(res)

        def df_to_struct(df):
            rec = df.to_records(index=False)
            return rec.view(rec.dtype.fields)

        if self.hbond_database.atom_groups.donors:
            donor_types = pandas.DataFrame.from_records(
                cattr.unstructure(self.hbond_database.atom_groups.donors)
            )
            donor_table = pandas.merge(
                donor_types,
                bond_table,
                how="inner",
                left_on=["d", "h"],
                right_on=["i_t", "j_t"]
            )
            donor_pairs = {"i_i": "d", "j_i": "h", "donor_type": "donor_type"}
            self.donors = df_to_struct(
                donor_table[list(donor_pairs)].rename(columns=donor_pairs)
            )
        else:
            self.donors = numpy.empty(0, self.donor_dtype)

        if self.hbond_database.atom_groups.sp2_acceptors:
            sp2_acceptor_types = pandas.DataFrame.from_records(
                cattr.unstructure(
                    self.hbond_database.atom_groups.sp2_acceptors
                )
            )
            sp2_ab_table = pandas.merge(
                sp2_acceptor_types,
                bond_table,
                how="inner",
                left_on=["a", "b"],
                right_on=["i_t", "j_t"]
            )
            sp2_bb0_table = pandas.merge(
                sp2_acceptor_types,
                bond_table.rename(columns=inc_cols("i", "j")),
                how="inner",
                left_on=["b", "b0"],
                right_on=["j_t", "k_t"]
            )
            sp2_acceptor_table = pandas.merge(sp2_ab_table, sp2_bb0_table)
            sp2_pairs = {
                "i_i": "a",
                "j_i": "b",
                "k_i": "b0",
                "acceptor_type": "acceptor_type"
            }
            self.sp2_acceptors = df_to_struct(
                sp2_acceptor_table[list(sp2_pairs)].rename(columns=sp2_pairs)
            )
        else:
            self.sp2_acceptors = numpy.empty(0, self.sp2_acceptor_dtype)

        if self.hbond_database.atom_groups.sp3_acceptors:
            sp3_acceptor_types = pandas.DataFrame.from_records(
                cattr.unstructure(
                    self.hbond_database.atom_groups.sp3_acceptors
                )
            )
            sp3_ab_table = pandas.merge(
                sp3_acceptor_types,
                bond_table,
                how="inner",
                left_on=["a", "b"],
                right_on=["i_t", "j_t"]
            )
            sp3_ab0_table = pandas.merge(
                sp3_acceptor_types,
                bond_table.rename(columns=inc_cols("j")),
                how="inner",
                left_on=["a", "b0"],
                right_on=["i_t", "k_t"]
            )
            sp3_acceptor_table = pandas.merge(sp3_ab_table, sp3_ab0_table)
            sp3_pairs = {
                "i_i": "a",
                "j_i": "b",
                "k_i": "b0",
                "acceptor_type": "acceptor_type"
            }
            self.sp3_acceptors = df_to_struct(
                sp3_acceptor_table[list(sp3_pairs)].rename(columns=sp3_pairs)
            )
        else:
            self.sp3_acceptors = numpy.empty(0, self.sp3_acceptor_dtype)

        if self.hbond_database.atom_groups.ring_acceptors:
            ring_acceptor_types = pandas.DataFrame.from_records(
                cattr.unstructure(
                    self.hbond_database.atom_groups.ring_acceptors
                )
            )
            ring_ab_table = pandas.merge(
                ring_acceptor_types,
                bond_table,
                how="inner",
                left_on=["a", "b"],
                right_on=["i_t", "j_t"]
            )
            ring_abp_table = pandas.merge(
                ring_acceptor_types,
                bond_table.rename(columns=inc_cols("j")),
                how="inner",
                left_on=["a", "bp"],
                right_on=["i_t", "k_t"]
            )
            ring_acceptor_table = pandas.merge(ring_ab_table, ring_abp_table)
            ring_pairs = {
                "i_i": "a",
                "j_i": "b",
                "k_i": "bp",
                "acceptor_type": "acceptor_type"
            }
            self.ring_acceptors = df_to_struct(
                ring_acceptor_table[list(ring_pairs)].rename(
                    columns=ring_pairs
                )
            )
        else:
            self.ring_acceptors = numpy.empty(0, self.ring_acceptor_dtype)

        return self


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
            t: toolz.dicttoolz.merge(
                {
                    "coeffs":
                    compose(torch.Tensor, numpy.nan_to_num)(
                        params[["c_" + i for i in "abcdefghijk"]].values
                    )  # yapf: disable
                },
                {
                    t: torch.Tensor(params[t])
                    for t in
                    ("max_val", "min_val", "root1", "root2", "xmax", "xmin")
                }
            )
            for t, params in self.param_lookup.groupby(level="term")
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
            self.hbond_database.global_parameters.max_dis
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
            max_dis=self.hbond_database.global_parameters.max_dis,
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
            max_dis=self.hbond_database.global_parameters.max_dis,
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

        global_params = dict(
            max_dis=self.hbond_database.global_parameters.max_dis,
        )

        return hbond_donor_ring_score(
            **toolz.dicttoolz.merge(coord_params, pair_params, global_params)
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
