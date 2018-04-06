import cattr
import properties
from properties import Instance
import torch

import numpy
import pandas

from tmol.properties.array import VariableT, Array
from tmol.properties.reactive import derived_from

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

    # Global score parameters
    max_dis
):
    d_a_dist = (d.reshape((-1, 1, 3)) - a.reshape((1, -1, 3))).norm(dim=-1)
    return (d_a_dist < max_dis).type(d.dtype)

def hbond_donor_sp3_score(
    # Input coordinates
    d,
    h,
    a,
    b,
    b0,

    # Global score parameters
    max_dis
):
    d_a_dist = (d.reshape((-1, 1, 3)) - a.reshape((1, -1, 3))).norm(dim=-1)
    return (d_a_dist < max_dis).type(d.dtype)

def hbond_donor_ring_score(
    # Input coordinates
    d,
    h,
    a,
    b,
    bp,

    # Global score parameters
    max_dis
):
    d_a_dist = (d.reshape((-1, 1, 3)) - a.reshape((1, -1, 3))).norm(dim=-1)
    return (d_a_dist < max_dis).type(d.dtype)

class HBondElementAnalysis(properties.HasProperties):
    hbond_database: HBondDatabase = Instance(
        "hbond parameter database", HBondDatabase, default=tmol.database.default.scoring.hbond)

    atom_types = Array("atomic types", dtype=object)[:]
    bonds = Array("inter-atomic bond graph", dtype=int)[:, 2]

    donor_dtype = numpy.dtype([("d", int), ("h", int), ("donor_type", object)])
    donors = Array("Identified donor atom indices.", dtype=donor_dtype)[:]

    sp2_acceptor_dtype = numpy.dtype([("a", int), ("b", int), ("b0", int), ("acceptor_type", object)])
    sp2_acceptors = Array("Identified sp2 acceptor atom indices.", dtype=sp2_acceptor_dtype)[:]

    sp3_acceptor_dtype = numpy.dtype([("a", int), ("b", int), ("b0", int), ("acceptor_type", object)])
    sp3_acceptors = Array("Identified sp3 acceptor atom indices.", dtype=sp3_acceptor_dtype)[:]

    ring_acceptor_dtype = numpy.dtype([("a", int), ("b", int), ("bp", int), ("acceptor_type", object)])
    ring_acceptors = Array("Identified ring acceptor atom indices.", dtype=ring_acceptor_dtype)[:]

    def setup(self):
        self : HBondElementAnalysis
        bond_types = self.atom_types[self.bonds]

        bond_table = pandas.DataFrame.from_dict({
            "i_i" : self.bonds[:,0],
            "i_t" : bond_types[:,0],
            "j_i" : self.bonds[:,1],
            "j_t" : bond_types[:,1],
        })


        def inc_cols(*args):
            order = {"i" : "j", "j" : "k"}
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
            donor_types = pandas.DataFrame.from_records(cattr.unstructure(
                self.hbond_database.atom_groups.donors
            ))
            donor_table = pandas.merge(donor_types, bond_table, how="inner", left_on=["d", "h"], right_on=["i_t", "j_t"])
            donor_pairs = {"i_i" : "d", "j_i" : "h", "donor_type" : "donor_type"}
            self.donors = df_to_struct(donor_table[list(donor_pairs)].rename(columns=donor_pairs))

        if self.hbond_database.atom_groups.sp2_acceptors:
            sp2_acceptor_types = pandas.DataFrame.from_records(cattr.unstructure(
                self.hbond_database.atom_groups.sp2_acceptors
            ))
            sp2_ab_table = pandas.merge(
                sp2_acceptor_types, bond_table,
                how="inner", left_on=["a", "b"], right_on=["i_t", "j_t"]
            )
            sp2_bb0_table = pandas.merge(
                sp2_acceptor_types, bond_table.rename(columns = inc_cols("i", "j")),
                how="inner", left_on=["b", "b0"], right_on=["j_t", "k_t"]
            )
            sp2_acceptor_table = pandas.merge(sp2_ab_table, sp2_bb0_table)
            sp2_pairs = {"i_i" : "a", "j_i" : "b", "k_i" : "b0",  "acceptor_type" : "acceptor_type"}
            self.sp2_acceptors = df_to_struct(
                sp2_acceptor_table[list(sp2_pairs)].rename(columns=sp2_pairs))

        if self.hbond_database.atom_groups.sp3_acceptors:
            sp3_acceptor_types = pandas.DataFrame.from_records(cattr.unstructure(
                self.hbond_database.atom_groups.sp3_acceptors
            ))
            sp3_ab_table = pandas.merge(
                sp3_acceptor_types, bond_table,
                how="inner", left_on=["a", "b"], right_on=["i_t", "j_t"]
            )
            sp3_ab0_table = pandas.merge(
                sp3_acceptor_types, bond_table.rename(columns = inc_cols("j")),
                how="inner", left_on=["a", "b0"], right_on=["i_t", "k_t"]
            )
            sp3_acceptor_table = pandas.merge(sp3_ab_table, sp3_ab0_table)
            sp3_pairs = {"i_i" : "a", "j_i" : "b", "k_i" : "b0",  "acceptor_type" : "acceptor_type"}
            self.sp3_acceptors = df_to_struct(
                sp3_acceptor_table[list(sp3_pairs)].rename(columns=sp3_pairs))

        if self.hbond_database.atom_groups.ring_acceptors:
            ring_acceptor_types = pandas.DataFrame.from_records(cattr.unstructure(
                self.hbond_database.atom_groups.ring_acceptors
            ))
            ring_ab_table = pandas.merge(
                ring_acceptor_types, bond_table,
                how="inner", left_on=["a", "b"], right_on=["i_t", "j_t"]
            )
            ring_abp_table = pandas.merge(
                ring_acceptor_types, bond_table.rename(columns = inc_cols("j")),
                how="inner", left_on=["a", "bp"], right_on=["i_t", "k_t"]
            )
            ring_acceptor_table = pandas.merge(ring_ab_table, ring_abp_table)
            ring_pairs = {"i_i" : "a", "j_i" : "b", "k_i" : "bp",  "acceptor_type" : "acceptor_type"}
            self.ring_acceptors = df_to_struct(
                ring_acceptor_table[list(ring_pairs)].rename(columns=ring_pairs))

        return self

class HBondScoreGraph(InteratomicDistanceGraphBase):

    hbond_database: HBondDatabase = Instance(
        "ljlk parameter database", HBondDatabase, default=tmol.database.default.scoring.hbond)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.atom_pair_dist_thresholds.add(
            self.hbond_database.global_parameters.max_dis)
        self.score_components.add("total_hbond")

    @derived_from(("hbond_database", "atom_types", "bonds"),
        Instance("hbond score elements in target =graph", HBondElementAnalysis))
    def hbond_elements(self) -> HBondElementAnalysis:
        return HBondElementAnalysis(
                hbond_database = self.hbond_database,
                atom_types = self.atom_types,
                bonds = self.bonds
            ).setup()

    @derived_from(("coords", "hbond_elements"), VariableT("donor-sp2 hbond scores"))
    def donor_sp2_hbond(self):
        return hbond_donor_sp2_score(
            d = self.coords[self.hbond_elements.donors["d"]],
            h = self.coords[self.hbond_elements.donors["h"]],
            a = self.coords[self.hbond_elements.sp2_acceptors["a"]],
            b = self.coords[self.hbond_elements.sp2_acceptors["b"]],
            b0 = self.coords[self.hbond_elements.sp2_acceptors["b0"]],
            max_dis = self.hbond_database.global_parameters.max_dis,
        )

    @derived_from(("coords", "hbond_elements"), VariableT("donor-sp3 hbond scores"))
    def donor_sp3_hbond(self):
        return hbond_donor_sp3_score(
            d = self.coords[self.hbond_elements.donors["d"]],
            h = self.coords[self.hbond_elements.donors["h"]],
            a = self.coords[self.hbond_elements.sp3_acceptors["a"]],
            b = self.coords[self.hbond_elements.sp3_acceptors["b"]],
            b0 = self.coords[self.hbond_elements.sp3_acceptors["b0"]],
            max_dis = self.hbond_database.global_parameters.max_dis,
        )

    @derived_from(("coords", "hbond_elements"), VariableT("donor-ring hbond scores"))
    def donor_ring_hbond(self):
        return hbond_donor_ring_score(
            d = self.coords[self.hbond_elements.donors["d"]],
            h = self.coords[self.hbond_elements.donors["h"]],
            a = self.coords[self.hbond_elements.ring_acceptors["a"]],
            b = self.coords[self.hbond_elements.ring_acceptors["b"]],
            bp = self.coords[self.hbond_elements.ring_acceptors["bp"]],
            max_dis = self.hbond_database.global_parameters.max_dis,
        )

    @derived_from(
        ("donor_sp2_hbond", "donor_sp3_hbond", "donor_ring_hbond"),
        VariableT("total hbond score"))
    def total_hbond(self):
        return self.donor_sp2_hbond.sum() + self.donor_sp3_hbond.sum() + self.donor_ring_hbond.sum()
