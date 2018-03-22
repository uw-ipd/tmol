import os
import io
import re
import toolz
from toolz import curry, groupby, compose, filter
from collections import defaultdict

import pandas

class FlatFileDB():
    """Generic multi-table flat-file databases.

    Stores a set of pandas tables as a simple flat-file db.
    """

    filters = compose(
        str.strip,
        curry(re.sub)("#.*", "")
    )

    @classmethod
    def read(cls, f):
        if isinstance(f, str):
            if os.path.exists(f):
                f = open(f, "r")
            else:
                f = io.StringIO(f)

        line_blocks = defaultdict(list)

        for l in filter(None, map(cls.filters, f.readlines())):
            t, l = l.split(maxsplit=1)
            line_blocks[t].append(l)

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
