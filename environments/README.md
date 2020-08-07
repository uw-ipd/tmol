`tmol` environment dependencies are organized into rough layers for `core`
run deps, `test` suite deps, `docs` build deps and `dev` interactive
development deps. These layers are maintained to enable specification of
a minimal set of runtime dependencies, while still allowing broad use of
external libraries for test and development.

## Setup

All package versioning is handled via
[`conda-lock`](https://github.com/mariusvniekerk/conda-lock).


The `render` script is used to generate lockfiles for each environment
layer under `locks`. These locked package lists can be converted into
environments via `conda create`. 
For example `conda create -h tmol --file environments/locks/dev-linux-64.lock`)

## Adding Dependencies

Follow these ~~simple~~ steps to add a new dependency:

1. Identify the lowest level / first dependency layer from the following:

  * `core` - Referenced by the `tmol` package.
  * `test` - Referenced by the `tmol.tests` package.
  * `docs` - Referenced under `docs`.
  * `dev`  - Referenced under `dev`.
  * `support` - Referenced by the `tmol.support` package.

2. Determine the dependency source. If the package is available under the
   main/conda-forge/pytorch conda channels then include as
   a `.dependencies` reference. Elif available on PyPI, render a conda
   package via `conda skeleton pypi <package_name>` and upload to the
   uw-ipd anaconda.org channel. See
   https://github.com/uw-ipd/conda-recipes for examples. Iff the package
   is only available via a custom https channel, (eg.
   http://conda.ipd.uw.edu) then include this channel via a `.channels`
   reference.

3. Verify that the dependency isn't specified on multiple levels. (Eg.
   Moved from `dev` to `core`.) `grep` is your friend.

4. Test the locks via `render`.

## Test Environment

The "standard" tmol test environment is specified via a docker image under
`test/Dockerfile`. This environment is used for *all* CI test steps, and
includes the `test` and `docs` environment layers.
