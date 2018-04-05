import cattr
import properties
from properties import Instance, Dictionary, StringChoice

import numpy
import pandas

from tmol.properties.array import VariableT, Array
from tmol.properties.reactive import cached

from .interatomic_distance import InteratomicDistanceGraphBase

import tmol.database
from tmol.database.scoring import HBondDatabase


class HBondScoreGraph(InteratomicDistanceGraphBase):

    hbond_database: HBondDatabase = Instance(
        "ljlk parameter database", HBondDatabase, default=tmol.database.default.scoring.hbond)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.atom_pair_dist_thresholds.add(
            self.hond_database.parameters["global_parameters"]["max_dis"])


class HBondElementAnalysis(properties.HasProperties):
    hbond_database: HBondDatabase = Instance(
        "hbond parameter database", HBondDatabase, default=tmol.database.default.scoring.hbond)

    atom_types = Array("atomic types", dtype=object)[:]
    bonds = Array("inter-atomic bond graph", dtype=int)[:, 2]

    donor_dtype = numpy.dtype([("d", int), ("h", int)])
    donors = Array("Identified donor atom indices.", dtype=donor_dtype)[:]

    acceptor_dtype = numpy.dtype([("a", int), ("b", int), ("b0", int)])
    acceptors = Array("Identified acceptor atom indices.", dtype=acceptor_dtype)[:]

    @classmethod
    def setup(cls, atom_types, bonds):
        self = cls(atom_types = atom_types, bonds = bonds)
        bond_types = self.atom_types[self.bonds] 

        bond_table = pandas.DataFrame.from_dict({
            "i_i" : self.bonds[:,0],
            "i_t" : bond_types[:,0],
            "j_i" : self.bonds[:,1],
            "j_t" : bond_types[:,1],
        })

        inc_cols = {
            "i_i" : "j_i",
            "i_t" : "j_t",
            "j_i" : "k_i",
            "j_t" : "k_t",
        }

        donor_types = pandas.DataFrame.from_records(cattr.unstructure(self.hbond_database.donors))
        donor_table = pandas.merge(donor_types, bond_table, how="inner", left_on=["d", "h"], right_on=["i_t", "j_t"])
        self.donors = (
                donor_table[["i_i","j_i"]].values.copy()
                .view(self.donor_dtype).squeeze(axis=-1)
        )

        acceptor_types = pandas.DataFrame.from_records(cattr.unstructure(self.hbond_database.acceptors))
        ab_table = pandas.merge(
            acceptor_types, bond_table,
            how="inner", left_on=["a", "b"], right_on=["i_t", "j_t"]
        )
        bb0_table = pandas.merge(
            acceptor_types, bond_table.rename(inc_cols, axis="columns"),
            how="inner", left_on=["b", "b0"], right_on=["j_t", "k_t"]
        )
        acceptor_table = pandas.merge(ab_table, bb0_table)
        self.acceptors = (
            acceptor_table[["i_i", "j_i", "k_i"]].values.copy()
            .view(self.acceptor_dtype).squeeze(axis=-1)
        )

        return self
