import os
import io
import re
import toolz
from toolz import curry, groupby, compose, filter
from collections import defaultdict

import pandas

class FlatFileDB():
    """Generic multi-table flat-file databases.

    Stores a set of pandas tables as a simple flat-file db. Each table keyed by
    a line prefix with whitespace-delimited records. Lines are grouped by
    prefix and must a single header row.

    Blank lines and comments, delimited by "#" are ignored.

    Eg:

    table_a ca1    ca2    ca3
    table_a a      b      1
    table_a a      c      2


    table_b cb1    cb2
    table_b b      3
    table_b b      4

    # A hanging, out of order, record.
    table_a a      d      3

    is mapped to the record format:
    {
        "table_a" : [
            { "ca1" : "a", "ca2" : "b", "ca3" : 1},
            { "ca1" : "a", "ca2" : "c", "ca3" : 2},
            { "ca1" : "a", "ca2" : "d", "ca3" : 3},
        ],
        "table_b" : [
            { "cb1" : "b", "cb2" : 3},
            { "ca1" : "b", "cb2" : 4},
        ],
    }
    """

    filters = compose(
        str.strip,
        curry(re.sub)("#.*", "")
    )

    @classmethod
    def read(cls, f):
        if isinstance(f, str):
            if os.path.exists(f):
                with open(f, "r") as inf:
                    f = inf.read()
            f = io.StringIO(f)

        line_blocks = defaultdict(list)

        for l in filter(None, map(cls.filters, f.readlines())):
            tag, line = l.split(maxsplit=1)
            line_blocks[tag].append(line)

        return {
            n : pandas.read_table(io.StringIO("\n".join(l)), sep="\s+")
            for n, l in line_blocks.items()
        }

    @classmethod
    def write(cls, tables, buf=None, order=None):

        res = []

        if not order:
            order = sorted(tables.keys())

        for n in order:
            res.append("\n".join(
                f"{n} {l}"
                for l in tables[n].to_string(index=False).split("\n")
            ))

        if buf:
            buf.write("\n\n".join(res))
        else:
            return "\n\n".join(res)
