"""Vectorized varients of baseline potential implementations."""

import numba

from .lj import lj, d_lj_d_dist
from .lk_isotropic import lk_isotropic, d_lk_isotropic_d_dist

lj = numba.vectorize(lj.py_func)
d_lj_d_dist = numba.vectorize(d_lj_d_dist.py_func)

lk_isotropic = numba.vectorize(lk_isotropic.py_func)
d_lk_isotropic_d_dist = numba.vectorize(d_lk_isotropic_d_dist.py_func)
