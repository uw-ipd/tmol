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

class ChemicalDatabase(HasProperties):
    source = String("Database source file.")

    @cached(Instance("atom properties table", AtomProperties))
    def atom_properties(self):
        return AtomProperties(source = f"{self.source}/atom_properties.txt")

