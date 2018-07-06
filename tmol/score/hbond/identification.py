import attr
import cattr
import toolz

from tmol.types.functional import convert_args
from tmol.types.attrs import ValidateAttrs
from tmol.types.array import NDArray

import numpy
import pandas

from tmol.database.scoring import HBondDatabase

acceptor_dtype = numpy.dtype([
    ("a", int),
    ("b", int),
    ("b0", int),
    ("acceptor_type", object),
])

donor_dtype = numpy.dtype([
    ("d", int),
    ("h", int),
    ("donor_type", object),
])


def merge_tables(*tables, **kwargs):
    assert len(tables) >= 2
    return toolz.reduce(toolz.curry(pandas.merge, **kwargs), tables)


def df_to_struct(df, column_mapping):
    """Convert a subset of columns into a structured array."""
    df_subset = df[list(column_mapping.keys())].rename(columns=column_mapping)

    numpy_record_table = df_subset.to_records(index=False)
    return numpy_record_table.view(numpy_record_table.dtype.fields)


@attr.s(frozen=True, slots=True, auto_attribs=True)
class HBondElementAnalysis(ValidateAttrs):
    """Construct a set of indices of atoms that are
    1) hbond donors,
    2) sp2-hybridized hbond acceptors,
    3) sp3-hybridized hbond acceptors, or
    4) ring-hybridized hbond acceptors
    from two input NDArrays -- one listing the atom types of all the atoms,
    and a second listing the set of chemical bonds.

    The set of atoms that are donors will be checked against the three sets of
    atoms that are marked as acceptors by the HBondPairs class.

    This code uses sql-like joins provided by the pandas package in order
    to do graph matching: e.g.,

    "I have atoms a, b, and c that are bonded in a pattern
    of a-b and b-c and I know their chemical types as ta, tb, and tc; do these three
    atoms match any of the known sets of sp2 hybridized acceptors?"

    This analysis is performed on the CPU

    There are a few tricky bits. First pandas data frames are constructed by
    "unstructuring" the lists of atom-types held in the HBondDatabase
    (read in from a yaml file, such as the one in
    tmol/database/default/scoring/hbond.yaml). Another tricky bit is the construction
    of the graphs of bonded atoms, e.g. i-j-k, or k-i-j, where the original list
    of bonded atoms is converted into a dataframe of i, itype, j, jtype,
    and then this dataframe is joined with a variant of itself, where the columns
    have been relabeled as j, jtype, k, ktype, e.g.. Joining on j gives you
    a new dataframe of atom-triples, i, itype, j, jtype, k, ktype. The
    "variant" dataframe is created with the "inc_cols" sub-function.
    """

    donors: NDArray(donor_dtype)[:]
    sp2_acceptors: NDArray(acceptor_dtype)[:]
    sp3_acceptors: NDArray(acceptor_dtype)[:]
    ring_acceptors: NDArray(acceptor_dtype)[:]

    @classmethod
    @convert_args
    def setup(
            cls,
            hbond_database: HBondDatabase,
            atom_types: NDArray(object)[:],
            bonds: NDArray(int)[:, 2],
    ):
        atom_types = pandas.Categorical(atom_types)

        # Generate bond arrangement tables containing:
        #   atom_index '*_a'
        #   atom_type  '*_t'
        #   # of bonds '*_nbond'
        #
        # with column names i, j [and later k]
        def inc_cols(table, *cols):
            order = {"i": "j", "j": "k"}
            res = []
            for n in cols:
                nn = order[n]
                res.append((n + "_a", nn + "_a"))
                res.append((n + "_t", nn + "_t"))
                res.append((n + "_nbond", nn + "_nbond"))
            return table.rename(columns=dict(res))

        bond_table = pandas.DataFrame.from_dict({
            "i_a": bonds[:, 0],
            "i_t": atom_types[bonds[:, 0]],
            "j_a": bonds[:, 1],
            "j_t": atom_types[bonds[:, 1]],
        })

        # Calculate bond counts for each atom then merge by the atom index
        bond_table = merge_tables(
            bond_table,
            bond_table["i_a"].value_counts().to_frame("i_nbond")
            .rename_axis("i_a").reset_index(),
            bond_table["j_a"].value_counts().to_frame("j_nbond")
            .rename_axis("j_a").reset_index(),
        )

        # Index of bond arragments of the form:
        #
        #    i---j
        #
        bond_pair_table = bond_table

        # Index of all bonded arrangments of the form:
        #
        #   i---j
        #    \
        #     k
        #
        # pruned by unique atom types on j & k
        bond_sibling_table = (
            merge_tables(
                bond_table.query("i_nbond == 2"),
                inc_cols(bond_table, "j"),
            )
            .query("j_a != k_a")
            .drop_duplicates(("i_a", "j_t", "k_t"))
        ) # yapf: disable

        # Index of all arrangments of the form:
        #
        #   i---j---?
        #        \
        #         k
        #
        # pruned by unique atom types on k
        bond_parent_table = (
            merge_tables(
                bond_table.query("i_nbond == 1"),
                inc_cols(bond_table, "i", "j"),
            )
            .query("i_a != k_a")
            .drop_duplicates(("i_a", "k_t"))
        ) # yapf: disable

        if hbond_database.atom_groups.donors:
            # Identify donors by donor-hydrogen bonds:
            #
            #   d---h
            #
            # by matching atom types.

            donor_types = pandas.DataFrame.from_records(
                cattr.unstructure(hbond_database.atom_groups.donors)
            )[["d", "h", "donor_type"]]
            donor_table = pandas.merge(
                donor_types,
                bond_pair_table,
                how="inner",
                left_on=["d", "h"],
                right_on=["i_t", "j_t"]
            )
            donors = df_to_struct(
                donor_table, {
                    "i_a": "d",
                    "j_a": "h",
                    "donor_type": "donor_type"
                }
            )
        else:
            donors = numpy.empty(0, donor_dtype)

        if hbond_database.atom_groups.sp2_acceptors:
            # Identify sp2 by acceptor-base-base0
            #
            #   a---b---?
            #        \
            #         b0
            #
            # by matching atom types.
            sp2_acceptor_types = pandas.DataFrame.from_records(
                cattr.unstructure(hbond_database.atom_groups.sp2_acceptors)
            )
            sp2_acceptor_table = pandas.merge(
                sp2_acceptor_types,
                bond_parent_table,
                how="inner",
                left_on=["a", "b", "b0"],
                right_on=["i_t", "j_t", "k_t"]
            )
            sp2_acceptors = df_to_struct(
                sp2_acceptor_table, {
                    "i_a": "a",
                    "j_a": "b",
                    "k_a": "b0",
                    "acceptor_type": "acceptor_type"
                }
            )
        else:
            sp2_acceptors = numpy.empty(0, acceptor_dtype)

        if hbond_database.atom_groups.sp3_acceptors:
            # Identify sp3 by acceptor-base-base0
            #
            #   a---b
            #    \
            #     b0
            #
            # by matching atom types.
            sp3_acceptor_types = pandas.DataFrame.from_records(
                cattr.unstructure(hbond_database.atom_groups.sp3_acceptors)
            )
            sp3_acceptor_table = pandas.merge(
                sp3_acceptor_types,
                bond_sibling_table,
                how="inner",
                left_on=["a", "b", "b0"],
                right_on=["i_t", "j_t", "k_t"]
            )
            sp3_acceptors = df_to_struct(
                sp3_acceptor_table, {
                    "i_a": "a",
                    "j_a": "b",
                    "k_a": "b0",
                    "acceptor_type": "acceptor_type"
                }
            )
        else:
            sp3_acceptors = numpy.empty(0, acceptor_dtype)

        if hbond_database.atom_groups.ring_acceptors:
            # Identify ring by acceptor-base-base0
            #
            #   a---b
            #    \
            #     b0
            #
            # by matching atom types.
            ring_acceptor_types = pandas.DataFrame.from_records(
                cattr.unstructure(hbond_database.atom_groups.ring_acceptors)
            )
            ring_acceptor_table = pandas.merge(
                ring_acceptor_types,
                bond_sibling_table,
                how="inner",
                left_on=["a", "b", "b0"],
                right_on=["i_t", "j_t", "k_t"]
            )
            ring_acceptors = df_to_struct(
                ring_acceptor_table, {
                    "i_a": "a",
                    "j_a": "b",
                    "k_a": "b0",
                    "acceptor_type": "acceptor_type"
                }
            )
        else:
            ring_acceptors = numpy.empty(0, acceptor_dtype)

        return cls(
            donors=donors,
            sp2_acceptors=sp2_acceptors,
            sp3_acceptors=sp3_acceptors,
            ring_acceptors=ring_acceptors,
        )
