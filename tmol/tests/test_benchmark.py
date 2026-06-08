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
