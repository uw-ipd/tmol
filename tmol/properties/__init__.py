def eq_by_is(prop):
    "Override property equality function to check via 'is'"

    prop.equal = lambda a, b: a is b
    return prop
