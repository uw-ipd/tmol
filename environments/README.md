# Development Environments

`tmol` environment dependencies are organized into rough layers for `core`
run deps, `test` suite deps, `docs` build deps and `dev` interactive
development deps. These layers are maintained to enable specification of
a minimal set of runtime dependencies, while still allowing broad use of
external libraries for test and development.

## Setup

The `render` script is used to generate an `environment.yml` file for use
via `conda env update -f environment.yml`. See `render --help` for
details.

All environments are managed via `conda`, however not all external
packages may be available as conda packages. Requirements lists are
therefor split into "dependencies.txt" and "requirements.txt", specifying
`conda`-managed conda dependencies and `pip`-managed pypi requirements
respectively.

The `core`-level requirement is augmented via the `linux.cpu`,
`linux.cuda` or `osx.cpu` platform-specific sub files, allowing
installation of a cpu-only or cpu-and-cuda environment.

## Adding Dependencies

Follow these simple steps to add a new dependency:

1. Identify the lowest level / first dependency layer from the following:

  * `core` - Referenced by the `tmol` package.
  * `{platform}.core` - Referenced by cuda-specific components of the
    `tmol` package.
  * `test` - Referenced by the `tmol.tests` package.
  * `docs` - Referenced under `docs`.
  * `dev`  - Referenced under `dev`.

2. Determine the dependency source. If the package is available under
   a standard conda channel (not `conda-forge`) then include as
   a `.dependencies` reference. Otherwise include include as
   a `.requirements` reference. 

3. Verify that the dependency isn't specified on multiple levels. (Eg.
   Moved from `dev` to `core`.) `grep` is your friend.

4. Test and verify the rendered `environment.yml` file.
