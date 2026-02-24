"""Support functions for pytest_benchmark.

Support functions for working with pytest_benchmark.

Example:

    Create a benchmark fixture for interactive use or as an alternative to
    timeit-style components::

        fixture = make_fixture()
        fixture(a_simple_test_function)

        stat_frame(fixture.stats).time.describe()


    Use subfixtures to benchmark two sub-functions in single case::

        def test_str_join(benchmark):
            @subfixture(benchmark)
            def mult():
                return "foo" * 100

            @subfixture(benchmark)
            def add():
                foo = ""
                for _ in range(100):
                    foo += "foo"
                return foo

            assert mult == add
"""

import types
import logging
import warnings

from functools import singledispatch

import pandas
import toolz

import pytest_benchmark
from pytest_benchmark.fixture import BenchmarkFixture


def make_fixture(
    name="",
    add_stats=None,
    min_rounds=3,
    min_time=0.000005,
    max_time=1.0,
    timer=pytest_benchmark.utils.NameWrapper(  # noqa
        pytest_benchmark.timers.default_timer
    ),
    calibration_precision=10,
    warmup=False,
    warmup_iterations=0,
    disable_gc=False,
    logger=logging,
    warner=warnings,
    disabled=False,
    cprofile=False,
    cprofile_loops=None,
    cprofile_dump=None,
    **extra_info,
):
    """Create a pytest_benchmark fixture.

    Args:
        name: Reported fixture name.
        add_stats: Callback function for post-run stats. Eg: ``results.append``
        **extra_info: Extra kwargs stored as extra_info entries.
    """

    node = types.SimpleNamespace(name=name, _nodeid=name)

    benchmark = pytest_benchmark.fixture.BenchmarkFixture(
        node=node,
        add_stats=add_stats if add_stats is not None else lambda s: None,
        min_rounds=min_rounds,
        min_time=min_time,
        max_time=max_time,
        timer=timer,
        calibration_precision=calibration_precision,
        warmup=warmup,
        warmup_iterations=warmup_iterations,
        disable_gc=disable_gc,
        logger=logger,
        warner=warner,
        disabled=disabled,
        cprofile=cprofile,
        cprofile_loops=cprofile_loops,
        cprofile_dump=cprofile_dump,
    )

    benchmark.extra_info.update(extra_info)

    return benchmark


def make_subfixture(
    parent: BenchmarkFixture, name_suffix: str, set_mode=True, **extra_info
) -> BenchmarkFixture:
    """Create subfixture, cloning parent configuration.

    Args:
        parent: Source pytest_benchmark fixture.
        name_suffix: String suffix appending to fixture name for reporting.
        set_mode: Set _mode, marking fixture as "used".
        **extra_info: Kwargs are added to as extra info.

    Returns:
        Initialized subfixture.
    """
    node = types.SimpleNamespace(
        name=parent.name + name_suffix, _nodeid=parent.fullname + name_suffix
    )

    benchmark = pytest_benchmark.fixture.BenchmarkFixture(
        node=node,
        disable_gc=parent._disable_gc,
        timer=pytest_benchmark.utils.NameWrapper(parent._timer),
        min_rounds=parent._min_rounds,
        min_time=parent._min_time,
        max_time=parent._max_time,
        warmup=parent._warmup,
        warmup_iterations=parent._warmup,
        calibration_precision=parent._calibration_precision,
        add_stats=parent._add_stats,
        logger=parent._logger,
        warner=parent._warner,
        disabled=parent.disabled,
        cprofile=parent.cprofile,
        cprofile_loops=parent.cprofile_loops,
        cprofile_dump=parent.cprofile_dump,
    )

    benchmark.extra_info.update(parent.extra_info)
    benchmark.extra_info.update(extra_info)

    if set_mode:
        parent._mode = "subfixture"

    return benchmark


def subfixture(parent, set_mode=True, **extra_info):
    """Decorator benchmarking function as named subfixture.

    Args:
        parent: Source pytest_benchmark fixture.
        set_mode: Set _mode, marking fixture as "used".
        **extra_info: Kwargs are added to as extra info.

    Returns:
        Subfixture decorator.

    Example:
    """

    def subfixture_decorator(fun):
        sf = make_subfixture(parent, name_suffix="." + fun.__name__, **extra_info)
        return sf(fun)

    return subfixture_decorator


@singledispatch
def stat_frame(benchmark_stats):
    """Convert benchmark stats object into DataFrame of run times."""

    return pandas.DataFrame.from_dict(
        toolz.merge(
            {"name": benchmark_stats["name"], "time": benchmark_stats["stats"]["data"]},
            benchmark_stats["extra_info"],
        )
    )


@stat_frame.register(pytest_benchmark.stats.Metadata)
def stat_frame_from_metadata(metadata):
    return stat_frame(metadata.as_dict())


@stat_frame.register(list)
def stat_frame_from_result_list(result_list):
    return pandas.concat(map(stat_frame, result_list))
