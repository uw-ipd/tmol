#######################################################################
# Implements a topological sort algorithm.
#
# Copyright 2014 True Blade Systems, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Inline from:
# https://bitbucket.org/ppentchev/toposort/src/pp-updates/toposort.py
# @ cd1dd65
# with updates to error handling for un-sortable entries.
#
########################################################################
import itertools

from functools import reduce as _reduce

try:
    from typing import Any, Callable, Dict, Iterable, Iterator, List, Set, TypeVar, cast

    T = TypeVar("T")  # pylint: disable=invalid-name

    def _cast(func):
        # type: (Any) -> Callable[[Iterable[T]], List[T]]
        """Work around some type-checking things."""
        return cast(Callable[[Iterable[T]], List[T]], func)


except ImportError:

    def _cast(func):
        # type: (Any) -> Callable[[Iterable[T]], List[T]]
        """A placeholder for the type-checking cast() method."""
        return func  # type: ignore


__all__ = ["toposort", "toposort_flatten", "CircularDependencyError"]


class CircularDependencyError(ValueError):
    """A circular dependency was detected in the input data."""

    def __init__(self, data):
        # type: (CircularDependencyError, Dict[T, Set[T]]) -> None
        # Sort the data just to make the output consistent, for use in
        #  error messages.  That's convenient for doctests.
        msg = "Circular dependencies exist among these items: {{{}}}".format(
            ", ".join(
                "{}:{}".format(key, value)
                for key, value in sorted((repr(k), repr(v)) for k, v in data.items())
            )
        )
        super(CircularDependencyError, self).__init__(msg)
        self.data = data


def toposort(data):
    # type: (Dict[T, Set[T]]) -> Iterator[Set[T]]
    """Dependencies are expressed as a dictionary whose keys are items
    and whose values are a set of dependent items. Output is a list of
    sets in topological order. The first set consists of items with no
    dependences, each subsequent set consists of items that depend upon
    items in the preceeding sets.
    """

    # Special case empty input.
    if len(data) == 0:
        return

    # Copy the input so as to leave it unmodified.
    data = data.copy()

    # Ignore self dependencies.
    for key, value in data.items():
        value.discard(key)
    # Find all items that don't depend on anything.
    extra_items_in_deps = _reduce(set.union, data.values()) - set(data.keys())
    # Add empty dependences where needed.
    data.update({item: set() for item in extra_items_in_deps})
    while True:
        ordered = set(item for item, dep in data.items() if len(dep) == 0)
        if not ordered:
            break
        yield ordered
        data = {
            item: (dep - ordered) for item, dep in data.items() if item not in ordered
        }
    if len(data) != 0:
        raise CircularDependencyError(data)


def toposort_flatten(data, sort=True):
    # type: (Dict[T, Set[T]], bool) -> List[T]
    """Returns a single list of dependencies. For any set returned by
    toposort(), those items are sorted and appended to the result (just to
    make the results deterministic)."""

    handler = _cast(sorted if sort else list)
    return list(itertools.chain(*map(handler, toposort(data))))
