"""Vectorized varients of baseline potential implementations."""

import numba

from .lj import lj, d_lj_d_dist
from .lk_isotropic import lk_isotropic_mutual, d_lk_isotropic_mutual_d_dist

lj = numba.vectorize(lj.py_func)
d_lj_d_dist = numba.vectorize(d_lj_d_dist.py_func)

lk_isotropic_mutual = numba.vectorize(lk_isotropic_mutual.py_func)
d_lk_isotropic_mutual_d_dist = numba.vectorize(d_lk_isotropic_mutual_d_dist.py_func)
