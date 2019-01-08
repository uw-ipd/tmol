"""
start with a simple cuda extension that visits atom pairs the brute force way
"""

import torch
import tmol.utility.cpp_extension

_visit_pairs = tmol.utility.cpp_extension.load(
    'visit_pairs', ['./visit_pairs.cuda.cpp', './visit_pairs.cuda.cu'])
print('extension:', _visit_pairs)


def main():
    npts = 1000
    pts = torch.randn(npts, 3).cumsum(dim=0)
    result = _visit_pairs.visit_pairs_cpu_naive(pts, 10.0)
    print(result)

    print("DONE")


if __name__ == '__main__':
    main()