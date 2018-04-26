import properties
from properties import Instance

import cattr

import numpy
import pandas

from tmol.properties.array import Array

import tmol.database
from tmol.database.scoring import HBondDatabase


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
