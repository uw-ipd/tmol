"""Support for `enum`/`pandas.Categorical <pandas:api.categorical>` interconversion."""

import pandas
import enum


def enum_val_catdtype(enum_type: enum.Enum) -> pandas.api.types.CategoricalDtype:
    """Generate categorial dtype convering enumeratation values."""
    if issubclass(enum_type, enum.Flag):
        raise NotImplementedError(f"enum.Flag categorical not supported: {enum_type}")

    return pandas.api.types.CategoricalDtype(
        categories=list(enum_type.__members__.values())
    )


def enum_name_catdtype(enum_type: enum.Enum) -> pandas.api.types.CategoricalDtype:
    """Generate categorial dtype convering enumeratation member names."""
    if issubclass(enum_type, enum.Flag):
        raise NotImplementedError(f"enum.Flag categorical not supported: {enum_type}")

    return pandas.api.types.CategoricalDtype(
        categories=list(enum_type.__members__.keys())
    )


def vals_to_val_cat(enum_type: enum.Enum, values) -> pandas.Categorical:
    """Convert enum values to a categorial."""
    return pandas.Categorical(values, dtype=enum_val_catdtype(enum_type))


def vals_to_name_cat(enum_type: enum.Enum, values) -> pandas.Categorical:
    """Convert enum values to a categorial of member names."""
    return vals_to_val_cat(enum_type, values).rename_categories(
        enum_name_catdtype(enum_type).categories
    )


def names_to_name_cat(enum_type: enum.Enum, values) -> pandas.Categorical:
    """Convert enum names to a categorial."""
    return pandas.Categorical(values, dtype=enum_name_catdtype(enum_type))


def names_to_val_cat(enum_type: enum.Enum, values) -> pandas.Categorical:
    """Convert enum names to a categorial of enum values."""
    return names_to_name_cat(enum_type, values).rename_categories(
        enum_val_catdtype(enum_type).categories
    )
