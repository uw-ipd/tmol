import pytest

from ._benchmark import make_fixture, subfixture


def test_make_fixture():
    benchmark = make_fixture(max_time=0.001)
    assert benchmark(lambda: "value") == "value"


def test_str_join_method(benchmark):
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


@pytest.mark.xfail
def test_str_join_invalid(benchmark):
    @subfixture(benchmark)
    def times10():
        return "foo" * 10

    @subfixture(benchmark)
    def times100():
        return "foo" * 100

    assert times10 == times100
