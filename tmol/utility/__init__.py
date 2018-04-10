from toolz import first


def unique_val(vals):
    """Extract a single, unique value from a collection of values."""
    return just_one(set(vals))


def just_one(vals):
    """Extract a single value from a length one collection of values."""
    assert len(vals) == 1
    return first(vals)
