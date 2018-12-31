def test_flatten():
    from tmol.utility.dicttoolz import flat_items, unflatten

    examples = [
        ({"a": 1, "b": 2, "c": 3}, {("a",): 1, ("b",): 2, ("c",): 3}),
        (
            {"a": 1, "b": {"foo": 2, "bar": 3}},
            {("a",): 1, ("b", "foo"): 2, ("b", "bar"): 3},
        ),
        (
            {"a": 1, "b": {"foo": 2, "bar": {"bat": 3, "baz": 4}}},
            {
                ("a",): 1,
                ("b", "foo"): 2,
                ("b", "bar", "bat"): 3,
                ("b", "bar", "baz"): 4,
            },
        ),
        ({"a": 1, "b": 2, "c": [3, 4, 5]}, {("a",): 1, ("b",): 2, ("c",): [3, 4, 5]}),
    ]

    for ex, flat_ex in examples:
        assert dict(flat_items(ex)) == flat_ex
        assert unflatten(flat_ex) == ex
        assert unflatten(flat_ex.items()) == ex
