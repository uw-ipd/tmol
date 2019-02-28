import pytest
import pandas
import cattr
import numpy

from tmol.database.scoring.rama import RamaDatabase
from tmol.database import ParameterDatabase

import re

import timeit


def test_rama(default_database):
    db = default_database.scoring.rama
    assert len(db.rama_tables) == 40
    assert len(db.rama_lookup) == 40

    alltables = [x.name for x in db.rama_tables]
    allrules = [x.name for x in db.rama_lookup]

    # ensure each table is defined
    for rrule in allrules:
        assert rrule in alltables

    # ensure there is a rule for each table
    for rtbl in alltables:
        assert rtbl in allrules


def test_rama_mapper(default_database):
    db = default_database.scoring.rama

    allaas = [
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    ]

    # test mapping
    rama_records = pandas.DataFrame.from_records(
        cattr.unstructure(db.rama_lookup), index="name"
    ).reindex([x.name for x in db.rama_tables])

    rama_index = pandas.Index(rama_records[["res_middle", "res_upper"]])

    allpairs = numpy.array([[i, j] for i in allaas for j in allaas])
    indices = rama_index.get_indexer([allpairs[:, 0], allpairs[:, 1]])
    indices[indices == -1] = rama_index.get_indexer(
        [allpairs[indices == -1, 0], numpy.full(numpy.sum(indices == -1), "")]
    )

    assert numpy.sum(indices == -1) == 0

    for row, i in enumerate(indices):
        name_resolved = db.rama_tables[i].name
        assert rama_records.loc[name_resolved].res_middle == allpairs[row, 0]
        assert (
            rama_records.loc[name_resolved].res_upper == allpairs[row, 1]
            or rama_records.loc[name_resolved].res_upper == ""
        )
