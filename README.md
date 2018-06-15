# ToyMol

## Development Environment

`tmol` depends several components, outlined in
[environments](./environments/), most easily installed into
a [`conda`](https://conda.io) environment via [`dev_setup`](./dev_setup).
This script _requires_ a functional `conda` installation and, by default,
initializes a conda environment named `tmol`. It is _recommended_ that you
use [`direnv`](https://direnv.net) to ensure that the `tmol` environment
is activated.

Tests are managed via [`pytest`](https://pytest.org) and are roughly
partitioned into `testing`, `linting` and `benchmark` phases. See the
[`buildkite build phases`](./.buildkite/bin/) for details on specific
invocations. Tests are executed within the development environment via
`pytest`. The docker-compose service `local` can be used to reproduce the
continuous integration environment locally via `docker-compose run
--user=${UID} [...pytest...]`.
