import unittest
import properties

from tmol.properties.reactive import derived_from, cached

import re
from collections import Counter
from toolz import get_in

class TestReactiveProperties(unittest.TestCase):
    def test_cached(self):
        class TProp(properties.HasProperties):
            raw_text = properties.String("Raw source text")

            @cached(properties.String("Source text, lowercase normalized"))
            def norm_text(self):
                return re.sub(r'([^\s\w]|_)+', '', self.raw_text).lower()

            @cached(properties.Dictionary("Normalized word counts"))
            def word_counts(self):
                return Counter(self.norm_text.split())
            
        tp = TProp(raw_text="foo bar. Bat bazz. Foo bar.")

        self.assertCountEqual(tp._backend.keys(), ["raw_text"])
        
        self.assertDictEqual(dict(tp.word_counts), {"foo" : 2, "bar" : 2, "bat" : 1, "bazz" : 1})
        self.assertCountEqual(tp._backend.keys(), ["raw_text", "norm_text", "word_counts"])
        
        tp.raw_text = "Lorum ipsum"
        self.assertDictEqual(dict(tp.word_counts), {"foo" : 2, "bar" : 2, "bat" : 1, "bazz" : 1})
        self.assertCountEqual(tp._backend.keys(), ["raw_text", "norm_text", "word_counts"])
        
        del tp.norm_text
        self.assertDictEqual(dict(tp.word_counts), {"foo" : 2, "bar" : 2, "bat" : 1, "bazz" : 1})
        self.assertCountEqual(tp._backend.keys(), ["raw_text", "word_counts"])
        
        
    def test_derived(self):
    
        class TProp(properties.HasProperties):
            raw_text = properties.String("Raw source text")

            @derived_from(
                "raw_text",
                properties.String("Source text, lowercase normalized"))
            def norm_text(self):
                return re.sub(r'([^\s\w]|_)+', '', self.raw_text).lower()

            @derived_from(
                "norm_text",
                properties.Dictionary("Normalized word counts"))
            def word_counts(self):
                return Counter(self.norm_text.split())

        tp = TProp(raw_text="foo bar. Bat bazz. Foo bar.")

        self.assertCountEqual(tp._backend.keys(), ["raw_text"])
        self.assertEqual(tp._listeners, {})

        self.assertDictEqual(dict(tp.word_counts), {"foo" : 2, "bar" : 2, "bat" : 1, "bazz" : 1})

        self.assertCountEqual(tp._backend.keys(), ["raw_text", "norm_text", "word_counts"])
        self.assertEqual(len(get_in(["raw_text", "observe_set"], tp._listeners)), 1)
        self.assertEqual(len(get_in(["norm_text", "observe_set"], tp._listeners)), 1)

        tp.raw_text = "Lorum ipsum"
        self.assertCountEqual(tp._backend.keys(), ["raw_text"])
        self.assertEqual(len(get_in(["raw_text", "observe_set"], tp._listeners)), 1)
        self.assertEqual(len(get_in(["norm_text", "observe_set"], tp._listeners)), 1)

        self.assertDictEqual(dict(tp.word_counts), {"lorum" : 1, "ipsum" : 1})

        self.assertCountEqual(tp._backend.keys(), ["raw_text", "norm_text", "word_counts"])
        self.assertEqual(len(get_in(["raw_text", "observe_set"], tp._listeners)), 1)
        self.assertEqual(len(get_in(["norm_text", "observe_set"], tp._listeners)), 1)

if __name__ == "__main__":
        unittest.main()
