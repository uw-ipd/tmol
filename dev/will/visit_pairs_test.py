"""
start with a simple cuda extension that visits atom pairs the brute force way
"""

import tmol.utility.cpp_extension

_visit_pairs = tmol.utility.cpp_extension.load(
    'visit_pairs', ['./visit_pairs.cuda.cpp', './visit_pairs.cuda.cu'])


def main():
    assert _visit_pairs.foo(3) == 6
    assert _visit_pairs.foocuda(3) == 6

    print("DONE")


if __name__ == '__main__':
    main()