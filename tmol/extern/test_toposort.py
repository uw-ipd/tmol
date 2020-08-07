"""Tests for toposort module.

Copyright 2014 True Blade Systems, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Notes:
"""

import unittest

try:
    import typing

    HAVE_TYPING = True
except ImportError:
    HAVE_TYPING = False

import tmol.extern.toposort as toposort


class TestCase(unittest.TestCase):
    """Test the operation of the main toposort() function."""

    def test_simple(self):
        # type: (TestCase) -> None
        """Test a couple of simple cases."""
        self.assertEqual(
            list(
                toposort.toposort(
                    {2: {11}, 9: {11, 8}, 10: {11, 3}, 11: {7, 5}, 8: {7, 3}}
                )
            ),
            [{3, 5, 7}, {8, 11}, {2, 9, 10}],
        )

        # make sure self dependencies are ignored
        self.assertEqual(
            list(
                toposort.toposort(
                    {2: {2, 11}, 9: {11, 8}, 10: {10, 11, 3}, 11: {7, 5}, 8: {7, 3}}
                )
            ),
            [{3, 5, 7}, {8, 11}, {2, 9, 10}],
        )

        data_one = {1: set()}  # type: typing.Dict[int, typing.Set[int]]
        self.assertEqual(list(toposort.toposort(data_one)), [{1}])
        self.assertEqual(list(toposort.toposort({1: {1}})), [{1}])

    def test_no_dependencies(self):
        # type: (TestCase) -> None
        """Test that values with no dependencies are returned first."""
        self.assertEqual(
            list(toposort.toposort({1: {2}, 3: {4}, 5: {6}})), [{2, 4, 6}, {1, 3, 5}]
        )

        data_three = {
            1: set(),
            3: set(),
            5: set(),
        }  # type: typing.Dict[int, typing.Set[int]]
        self.assertEqual(list(toposort.toposort(data_three)), [{1, 3, 5}])

    def test_empty(self):
        # type: (TestCase) -> None
        """Test "sorting" an empty input set."""
        nothing = {}  # type: typing.Dict[object, typing.Set[object]]
        self.assertEqual(list(toposort.toposort(nothing)), [])

    def test_strings(self):
        # type: (TestCase) -> None
        """Test sorting strings."""
        self.assertEqual(
            list(
                toposort.toposort(
                    {
                        "2": {"11"},
                        "9": {"11", "8"},
                        "10": {"11", "3"},
                        "11": {"7", "5"},
                        "8": {"7", "3"},
                    }
                )
            ),
            [{"3", "5", "7"}, {"8", "11"}, {"2", "9", "10"}],
        )

    def test_objects(self):
        # type: (TestCase) -> None
        """Test sorting arbitrary objects."""
        obj2 = object()
        obj3 = object()
        obj5 = object()
        obj7 = object()
        obj8 = object()
        obj9 = object()
        obj10 = object()
        obj11 = object()
        self.assertEqual(
            list(
                toposort.toposort(
                    {
                        obj2: {obj11},
                        obj9: {obj11, obj8},
                        obj10: {obj11, obj3},
                        obj11: {obj7, obj5},
                        obj8: {obj7, obj3, obj8},
                    }
                )
            ),
            [{obj3, obj5, obj7}, {obj8, obj11}, {obj2, obj9, obj10}],
        )

    def test_cycle(self):
        # type: (TestCase) -> None
        """Make sure cycles are detected."""
        # a simple, 2 element cycle
        # make sure we can catch this both as ValueError and
        # toposort.CircularDependencyError
        self.assertRaises(ValueError, list, toposort.toposort({1: {2}, 2: {1}}))
        with self.assertRaises(toposort.CircularDependencyError) as ex:
            list(toposort.toposort({1: {2}, 2: {1}}))
        self.assertEqual(ex.exception.data, {1: {2}, 2: {1}})

        # an indirect cycle
        self.assertRaises(ValueError, list, toposort.toposort({1: {2}, 2: {3}, 3: {1}}))
        with self.assertRaises(toposort.CircularDependencyError) as ex:
            list(toposort.toposort({1: {2}, 2: {3}, 3: {1}}))
        self.assertEqual(ex.exception.data, {1: {2}, 2: {3}, 3: {1}})

        # not all elements involved in a cycle
        with self.assertRaises(toposort.CircularDependencyError) as ex:
            list(toposort.toposort({1: {2}, 2: {3}, 3: {1}, 5: {4}, 4: {6}}))
        self.assertEqual(ex.exception.data, {1: set([2]), 2: set([3]), 3: set([1])})

        # Cycles are properly raised even for non-sortable objects
        obj2 = object()
        obj3 = object()
        with self.assertRaises(toposort.CircularDependencyError) as ex:
            list(toposort.toposort({obj2: {obj3}, obj3: {obj2}}))

    def test_input_not_modified(self):
        # type: (TestCase) -> None
        """Make sure the input is not modified in a successful run."""
        data = {
            2: {11},
            9: {11, 8},
            10: {11, 3},
            11: {7, 5},
            8: {7, 3, 8},  # includes something self-referential
        }
        orig = data.copy()
        list(toposort.toposort(data))
        self.assertEqual(data, orig)

    def test_input_not_modified_when_cycle_error(self):
        # type: (TestCase) -> None
        """Make sure the input is unmodified even if an error is raised."""
        data = {1: {2}, 2: {1}, 3: {4}}
        orig = data.copy()
        self.assertRaises(ValueError, list, toposort.toposort(data))
        self.assertEqual(data, orig)


class TestCaseAll(unittest.TestCase):
    """Make sure toposort_flatten() works."""

    def test_sort_flatten(self):
        # type: (TestCaseAll) -> None
        """Make sure toposort_flatten() works."""
        data = {
            2: {11},
            9: {11, 8},
            10: {11, 3},
            11: {7, 5},
            8: {7, 3, 8},  # includes something self-referential
        }
        expected = [{3, 5, 7}, {8, 11}, {2, 9, 10}]
        self.assertEqual(list(toposort.toposort(data)), expected)

        # now check the sorted results
        results = []
        for item in expected:
            results.extend(sorted(item))
        self.assertEqual(toposort.toposort_flatten(data), results)

        # and the unsorted results. break the results up into groups to
        # compare them
        actual = toposort.toposort_flatten(data, False)
        nonflat_results = [set(actual[0:3]), set(actual[3:5]), set(actual[5:8])]
        self.assertEqual(nonflat_results, expected)


class TestAll(unittest.TestCase):
    """Check that the __all__ descriptor is correct."""

    def test_all(self):
        # type: (TestAll) -> None
        """Check that toposort.__all__ contains what it needs to.

        Check that __all__ in the module contains everything that should be
        public, and only those symbols.
        """
        t_all = set(toposort.__all__)

        # check that things in __all__ only appear once
        self.assertEqual(
            len(t_all),
            len(toposort.__all__),
            "some symbols appear more than once in __all__",
        )

        # get the list of public symbols
        found = set(name for name in dir(toposort) if not name.startswith("_"))
        found.discard("itertools")

        # remove any symbols imported from the typing module and the T TypeVar
        if HAVE_TYPING:
            typing_symbols = set(dir(typing))
            typing_symbols.add("T")
            found -= typing_symbols

        # make sure it matches __all__
        self.assertEqual(t_all, found)
