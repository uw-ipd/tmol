import seaborn

from tmol.tests.benchmark_plot import BenchmarkPlot


class TotalScoreOnepass(BenchmarkPlot):
    query = "basename=='test_full' and group=='total_score_onepass'"

    @classmethod
    def plot(cls, benchmark_data):
        g = seaborn.FacetGrid(
            cls.get_data_rows(benchmark_data),
            row="torch_device",
            hue="benchmark_run",
            sharey=False,
            aspect=2,
        )

        g.map(seaborn.lineplot, "system_size", "walltime", err_style="bars", ci=68)

        for a in g.axes.ravel():
            a.set_ylim(0, None)

        g.set_titles("total_score_onepass: {row_var}={row_name}")
        g.add_legend()

        return g


if __name__ == "__main__":
    TotalScoreOnepass.__main__()
