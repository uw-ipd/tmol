import unittest

def run():
    return unittest.TextTestRunner().run(
        unittest.defaultTestLoader.discover(__name__)
    )
