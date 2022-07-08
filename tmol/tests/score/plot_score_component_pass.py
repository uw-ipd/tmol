import pandas
import seaborn

from tmol.tests.benchmark_plot import BenchmarkPlot


class TotalScoreParts(BenchmarkPlot):
    query = "basename=='test_end_to_end_score_system'"

    @classmethod
    def plot(cls, benchmark_data):
        rows = cls.get_data_rows(benchmark_data)
        metadata = rows.query(TotalScoreParts.query).name.str.extract(
            r"(?x) \[ (?P<_device>\w*) - (?P<score_pass> \w*)-(?P<component> \w*)\]",
            expand=True,
        )
        data = pandas.concat([rows, metadata], axis=1)

        if len(data.benchmark_run.unique()) > 1:
            data["score_pass"] = data.score_pass.str.cat(data.benchmark_run, sep=":")

        g = seaborn.catplot(
            data=data[~data.component.str.startswith("total")],
            x="component",
            y="walltime",
            hue="score_pass",
            hue_order=sorted(data["score_pass"].unique(), reverse=True),
            row="torch_device",
            kind="box",
            sharey=False,
            aspect=2,
        )
        for a in g.axes.ravel():
            a.set_ylim(0, None)
        g.set_titles("end_to_end components: {row_var}={row_name}")


if __name__ == "__main__":
    TotalScoreParts.__main__()
