"""Support functions to plot benchmark results."""

import argparse
import json

from toolz import merge, dissoc
import pandas
from matplotlib import pyplot


class BenchmarkPlot:
    @classmethod
    def can_plot(cls, benchmark_rows):
        entries = benchmark_rows.query(cls.query)
        return len(entries) > 0

    @classmethod
    def get_data_rows(cls, benchmark_data):
        if not isinstance(benchmark_data, pandas.DataFrame):
            benchmark_data = cls.load_benchmarks(benchmark_data)

        benchmark_data = benchmark_data.applymap(
            lambda x: tuple(x) if isinstance(x, list) else x
        )

        return benchmark_data.query(cls.query)

    @classmethod
    def load_benchmarks(cls, benchmark_datafiles):
        if isinstance(benchmark_datafiles, (str, bytes)):
            d = json.load(open(benchmark_datafiles))

            rows = cls.extract_benchmark_rows(d)
            rows["benchmark_run"] = benchmark_datafiles

            return rows

        else:
            return pandas.concat([cls.load_benchmarks(f) for f in benchmark_datafiles])

    @staticmethod
    def extract_benchmark_rows(benchmark_data):
        return pandas.DataFrame.from_dict(
            merge(
                dict(
                    group=r["group"],
                    name=r["name"],
                    basename=r["name"].split("[")[0],
                    walltime=t,
                ),
                r["params"] if r["params"] else {},
                dissoc(benchmark_data["commit_info"], "time"),
            )
            for r in benchmark_data["benchmarks"]
            for t in r["stats"]["data"]
        )

    @classmethod
    def __main__(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("benchmark_datafiles", nargs="+")
        args = parser.parse_args()

        cls.plot(args.benchmark_datafiles)
        pyplot.show()
