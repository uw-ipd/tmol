# TMol

[![docs status](https://img.shields.io/website-up-down-green-red/http/shields.io.svg?label=docs)](http://tmol.ipd.uw.edu)
[![build status](https://badge.buildkite.com/0608cfe87394e48f6ffd7008b0634cb5be1b807e4b25f0d3e1.svg?branch=master)
](https://buildkite.com/uw-ipd/tmol)
[![codecov](https://codecov.io/gh/uw-ipd/tmol/branch/master/graph/badge.svg?token=OoO0dtKDBK)
](https://codecov.io/gh/uw-ipd/tmol)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

`tmol`, short for TensorMol, is a faithful reimplementation of the [Rosetta all atom energy
function](https://doi.org/10.1021/acs.jctc.7b00125) ("beta_nov2016_cart") in [PyTorch](https://pytorch.org) with custom kernels written in C++ and CUDA. Given the coordinates of one or more proteins (TODO: should this say 'proteins' or 'structures'?), `tmol` can compute both energies and derivatives. `tmol` can also perform gradient-based minimization on those structures. Thus, ML models that produce cartesian coordinates for proteins can include biophysical features in their loss during training or refine their output structures using Rosetta's experimentally validated energy function.

See our [wiki](./wiki) for details on our development environment, system
architecture, and development workflow.
