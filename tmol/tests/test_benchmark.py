import pytest

from .benchmark import subfixture


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
