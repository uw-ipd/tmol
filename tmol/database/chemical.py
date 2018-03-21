import io
import re
from toolz import curry
from toolz.curried import first, filter, map, compose

import pandas
import numpy

import properties
from properties import HasProperties, String, Instance

from tmol.properties.reactive import cached, derived_from
from tmol.properties.array import Array

lj_lk_param_dtype = numpy.dtype([
    ("lj_radius", numpy.float),
    ("lj_wdepth", numpy.float),
    ("lk_dgfree", numpy.float),
    ("lk_lambda", numpy.float),
    ("lk_volume", numpy.float),
    ("is_donor", numpy.bool),
    ("is_acceptor", numpy.bool),
    ("is_hydroxyl", numpy.bool),
    ("is_polarh", numpy.bool)
])

lj_lk_pair_param_dtype = numpy.dtype([
    ("lj_rad1", numpy.float),
    ("lj_rad2", numpy.float),
    ("lj_r6_coeff", numpy.float),
    ("lj_r12_coeff", numpy.float),
    ("lj_sigma", numpy.float),
    ("lk_coeff1", numpy.float),
    ("lk_coeff2", numpy.float),
    ("lk_inv_lambda1", numpy.float),
    ("lk_inv_lambda2", numpy.float)
])

class AtomProperties(properties.HasProperties):
    source = properties.String("atom properties table")

    @properties.File("source file", mode="r")
    def source_file(self):
        return self.source

    @cached(properties.List("source property lines"))
    def lines(self):
        sub = curry(re.sub)

        filters = compose(
            str.strip,
            sub("#.*", "")
        )

        return list(filter(None, map(filters, self.source_file.readlines())))

    @cached(Instance("atom properties, index by atom type name", pandas.DataFrame))
    def table(self):
        table_width = len(self.lines[0])
        table_lines = [l[:table_width] for l in self.lines]

        header_widths = [l + 1 for l in map(len, table_lines[0].split())]
        header_widths = [5, 6, header_widths[2] - 1] + header_widths[3:] 
        table = pandas.read_fwf(io.StringIO("\n".join(table_lines)), colspecs=None, widths=header_widths)

        for col in table_lines[0].split()[2:]:
            assert table.dtypes[col] == float

        cols = {c : c.lower() for c in table.columns}
        cols["ATOM"] = "elem"

        table = table.rename(columns=cols)

        tags = [l[table_width:].split() for l in self.lines[1:]]

        tag_mapping = {
            "is_donor" : "DONOR",
            "is_acceptor" : "ACCEPTOR",
            "is_polarh" : "POLAR_H",
            "is_hydroxyl" : "HYDROXYL",
        }

        for t, at in tag_mapping.items():
            table[t] = [at in ts for ts in tags]

        return table.set_index("name")

    @derived_from("table",
        properties.Instance("atom type name to atom type index mapping", pandas.Series))
    def name_to_idx(self):
        return pandas.Series(
            data = numpy.arange(len(self.table.index)),
            index = self.table.index
        )

    @derived_from(("table"),
        Array("per-atom lk/lj score parameters", dtype=lj_lk_param_dtype)[:])
    def atom_lj_lk_params(self):
        result = numpy.empty(len(self.table), dtype=lj_lk_param_dtype)

        for n in result.dtype.names:
            result[n] = self.table[n]

        return result

    @derived_from("atom_lj_lk_params",
        Array("atom pair lk/lj score parameters", dtype=lj_lk_pair_param_dtype)[:,:])
    def pairwise_lj_lk_params(self):

        a = self.atom_lj_lk_params.reshape((-1, 1))
        b = self.atom_lj_lk_params.reshape((1, -1))

        # update derived parameters
        # could 1/2 this calculation
        lj_lk_pair_data = numpy.empty(numpy.broadcast(a, b).shape, dtype=lj_lk_pair_param_dtype )

        # lj
        # these are only dependent on atom1/atom2 ... can this be more efficient?
        lj_lk_pair_data["lj_rad1"] = a["lj_radius"]
        lj_lk_pair_data["lj_rad2"] = b["lj_radius"]

        sigma = a["lj_radius"] + b["lj_radius"]
        don_acc_pair_mask = ( \
            ( a["is_donor"] & b["is_acceptor"] ) |
            ( b["is_donor"] & a["is_acceptor"] ))
        sigma[ don_acc_pair_mask ] = 1.75 #lj_hbond_hdis
        don_acc_pair_mask = ( \
            ( a["is_donor"] & a["is_hydroxyl"] & b["is_acceptor"] ) |
            ( b["is_donor"] & b["is_hydroxyl"] & a["is_acceptor"] ))
        sigma[ don_acc_pair_mask ] = 2.6 #lj_hbond_OH_donor_dis

        # lj 
        sigma  = sigma * sigma * sigma;
        sigma6  = sigma * sigma;
        sigma12 = sigma6 * sigma6;
        wdepth = numpy.sqrt( a["lj_wdepth"] + b["lj_wdepth"] );

        lj_lk_pair_data["lj_sigma"] = sigma
        lj_lk_pair_data["lj_r6_coeff"] = -2 * wdepth * sigma6
        lj_lk_pair_data["lj_r12_coeff"] = wdepth * sigma12

        # lk
        inv_neg2_tms_pi_sqrt_pi = -0.089793561062583294
        inv_lambda1 = 1.0 / a["lk_lambda"]
        lj_lk_pair_data["lk_inv_lambda1"] = inv_lambda1
        lj_lk_pair_data["lk_coeff1"] = (
            inv_neg2_tms_pi_sqrt_pi * a["lk_lambda"] * inv_lambda1 * inv_lambda1 * b["lk_volume"]
        )

        inv_lambda2 = 1.0 / b["lk_lambda"]
        lj_lk_pair_data["lk_inv_lambda2"] = inv_lambda2
        lj_lk_pair_data["lk_coeff2"] = (
            inv_neg2_tms_pi_sqrt_pi * b["lk_lambda"] * inv_lambda2 * inv_lambda2 * a["lk_volume"] )

        return lj_lk_pair_data

class ChemicalDatabase(HasProperties):
    source = String("Database source file.")

    @cached(Instance("atom properties table", AtomProperties))
    def atom_properties(self):
        return AtomProperties(source = f"{self.source}/atom_properties.txt")

