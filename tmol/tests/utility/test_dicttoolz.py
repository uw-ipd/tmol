def test_flatten():
    from tmol.utility._dicttoolz import _flat_items, _unflatten

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
        assert dict(_flat_items(ex)) == flat_ex
        assert _unflatten(flat_ex) == ex
        assert _unflatten(flat_ex.items()) == ex


def test_values_are_a_live_view():
    from tmol.utility._dicttoolz import _vals

    mapping = {"first": 1}
    view = _vals(mapping)
    mapping["second"] = 2
    assert list(view) == [1, 2]
