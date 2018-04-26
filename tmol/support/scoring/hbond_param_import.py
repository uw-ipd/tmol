""" Parse and import rosetta hydrogen bond parameters.

Manages parsing and import a subset of rosetta hydrogen bond parameters into a hydrogen bond
parameter database file. Selects a minimal subset of polynomial parameters and type pair parameters
to cover a specificed set of donor/acceptor types.

Example:

with open("sp2_elec_hbond_params.yaml", "w") as outfile:
    params = RosettaHBParams(
        "~/workspace/rosetta/main/database/scoring/score_functions/hbonds/sp2_elec_params/")
    params.to_yaml(outfile)
"""

import numpy
import pandas
import yaml
import attr
import toolz
from toolz.curried import reduce

import os
import io
import IPython.lib.pretty

table_schema = yaml.load(
    """
    HBondWeightType:
      - id
      - name
      - comment

    HBAccChemType:
      - id
      - name
      - name_long
      - comment

    HBDonChemType:
      - id
      - name
      - name_lon
      - comment

    HBAccHybridization:
      - acc_chem_type
      - hybridization
      - comment

    HBSeqSep:
      - id
      - name
      - comment

    HybridizationType:
      - id
      - name
      - comment

    HBPoly1D:
      - id
      - name
      - classic_name
      - dimension
      - xmin
      - xmax
      - min_val
      - max_val
      - root1
      - root2
      - degree
      - c_a
      - c_b
      - c_c
      - c_d
      - c_e
      - c_f
      - c_g
      - c_h
      - c_i
      - c_j
      - c_k

    HBEval:
      - don_chem_type
      - acc_chem_type
      - separation
      - AHdist_short_fade
      - AHdist_long_fade
      - cosBAH_fade
      - cosBAH2_fade
      - cosAHD_fade
      - chi_fade
      - AHdist
      - cosBAH_short
      - cosBAH_long
      - cosBAH2_long
      - cosAHD_short
      - cosAHD_long
      - weight_type
      - comment

    HBFadeIntervals:
      - id
      - name
      - junction_type
      - min0
      - fmin
      - fmax
      - max0
      - comment
"""
)

RawParams = attr.make_class("RawParams", list(table_schema.keys()))


@attr.s
class RosettaHBParams:
    target_donors = (
        'hbdon_PBA',
        'hbdon_CXA',
        'hbdon_IMD',
        'hbdon_IME',
        'hbdon_IND',
        'hbdon_AMO',
        'hbdon_GDE',
        'hbdon_GDH',
        'hbdon_AHX',
        'hbdon_HXL',
        'hbdon_H2O',
    )

    target_acceptors = (
        'hbacc_PBA',
        'hbacc_CXA',
        'hbacc_CXL',
        'hbacc_IMD',
        'hbacc_IME',
        'hbacc_AHX',
        'hbacc_HXL',
        'hbacc_H2O',
    )

    path: str = attr.ib(
        converter=toolz.compose(
            os.path.abspath,
            os.path.expanduser,
            os.path.expandvars,
        )
    )
    tables: RawParams = attr.ib()

    @tables.default
    def _load_tables(self):
        return RawParams(**{
            t: pandas.read_csv(
                f"{self.path}/{t}.csv", header=None, names=table_schema[t]
            )
            for t in table_schema
        })  # yapf:disable

    donor_types: pandas.DataFrame = attr.ib()

    @donor_types.default
    def _load_donor_types(self):
        donor_table = (
            pandas.merge(
                self.tables.HBDonChemType,
                pandas.DataFrame({
                    "name": self.target_donors
                })
            ).drop(["id"], axis="columns")
        )

        assert len(donor_table) == len(self.target_donors)
        return donor_table

    acceptor_types: pandas.DataFrame = attr.ib()

    @acceptor_types.default
    def _load_acceptor_types(self):
        acceptor_table = reduce(pandas.merge)((
            self.tables.HBAccChemType.drop(["id"], axis="columns"),
            pandas.DataFrame({"name": self.target_acceptors}),
            self.tables.HBAccHybridization[["acc_chem_type", "hybridization"]]
                .rename(columns={"acc_chem_type": "name"}),
        ))  # yapf: disable

        assert len(acceptor_table) == len(self.target_acceptors)
        return acceptor_table

    pair_params: pandas.DataFrame = attr.ib()

    @pair_params.default
    def _load_pair_params(self):
        # Cart product trick...
        # target_pairs = pandas.merge(
        #     self.donor_types["name"].to_frame("don_chem_type").assign(idx=0),
        #     self.acceptor_types["name"].to_frame("acc_chem_type").assign(idx=0)
        # ).drop("idx", axis="columns")

        pair_params = reduce(pandas.merge)((
            self.tables.HBEval
                .drop(
                    filter(lambda c: c.endswith("fade"), self.tables.HBEval.columns),
                    axis="columns")
                .drop(["weight_type", "separation", "comment"], axis="columns")
                .drop_duplicates(),
            self.donor_types["name"].to_frame("don_chem_type"),
            self.acceptor_types["name"].to_frame("acc_chem_type"),
        ))  # yapf: disable

        assert len(pair_params.groupby(["don_chem_type", "acc_chem_type"])) == \
               len(self.target_acceptors) * len(self.target_donors)

        return pair_params

    polynomial_parameters: pandas.DataFrame = attr.ib()

    @polynomial_parameters.default
    def _load_polynomial_params(self):
        term_names = (
            "AHdist", "cosBAH_short", "cosBAH_long", "cosBAH2_long",
            "cosAHD_short", "cosAHD_long"
        )

        poly_names = reduce(set.union)(
            (set(self.pair_params[c].values) for c in term_names)
        )

        poly_param_table = pandas.merge(
            self.tables.HBPoly1D.drop(["id", "classic_name"], axis="columns"),
            pandas.DataFrame({
                "name": list(poly_names)
            })
        )

        for c in term_names:
            merged = pandas.merge(
                self.pair_params[c].to_frame("name"), poly_param_table
            )
            assert len(merged) == len(self.pair_params[c]), \
                f"Missing poly parameter set: {c}"

        return poly_param_table

    def to_yaml(self, outfile=None):
        if not outfile:
            obuf = io.StringIO()

            p = IPython.lib.pretty.PrettyPrinter(
                output=obuf, max_width=int(1e6), max_seq_length=int(1e6)
            )
        else:
            p = IPython.lib.pretty.PrettyPrinter(
                output=outfile, max_width=int(1e6), max_seq_length=int(1e6)
            )

        def _(t):
            p.break_()
            p.text(t)

        _("chemical_types:")
        with p.group(2, ):
            _("donors:")
            with p.group(2):
                for i, d in self.donor_types.iterrows():
                    _(f"- {d['name']} #{d['comment']}")

            _("sp2_acceptors:")
            with p.group(2):
                for i, a in self.acceptor_types.query(
                        "hybridization == 'SP2_HYBRID'").iterrows():
                    _(f"- {a['name']} #{a['comment']}")

            _("sp3_acceptors:")
            with p.group(2):
                for i, a in self.acceptor_types.query(
                        "hybridization == 'SP3_HYBRID'").iterrows():
                    _(f"- {a['name']} #{a['comment']}")

            _("ring_acceptors:")
            with p.group(2):
                for i, a in self.acceptor_types.query(
                        "hybridization == 'RING_HYBRID'").iterrows():
                    _(f"- {a['name']} #{a['comment']}")

        _("polynomial_parameters:")
        with p.group(2):
            for r in self.polynomial_parameters.to_dict(orient="records"):
                p.break_()
                p.text("- {")
                p.text(
                    ", ".join(
                        f"{c}: {r[c]}"
                        for c in self.polynomial_parameters.columns
                    )
                )
                p.text("}")

        _("pair_parameters:")
        with p.group(2):
            for r in self.pair_params.to_dict(orient="records"):
                p.break_()
                p.text("- {")
                p.text(
                    ", ".join(
                        f"{c}: {r[c]}" for c in self.pair_params.columns
                    )
                )
                p.text("}")

        if not outfile:
            return obuf.getvalue()
        else:
            return None


basetype_for_dtype = {
    numpy.dtype("O"): "str",
    numpy.dtype("int64"): "int",
    numpy.dtype("float64"): "float",
}


def attrs_for_dtypes(name, dtypes):
    obuf = io.StringIO()
    p = IPython.lib.pretty.PrettyPrinter(
        output=obuf, max_width=int(1e6), max_seq_length=int(1e6)
    )

    p.text('@attr.s(auto_attribs=True, slots=True, frozen=True)')
    p.break_()
    p.text(f"class {name}:")
    with p.group(4):
        for n, v in dtypes.items():
            p.break_()
            p.text(f"{n}: {basetype_for_dtype[v]}")

    return obuf.getvalue()
