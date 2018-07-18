# Docs

This directory contains a `sphinx`-based build configuration combining
docstring-based api documentation and freeform notes. Package
documentation is generated via a templated `apidoc` pass and freeform
documentation is included from `*.rst` documents in this directory.

## Building 

The documentation build is handled by the `make` script in this directory,
which outputs an html documentation build to `_build/html`. Execute via
`docs/make` from the project root directory.

Document editing is supported via the `watch` script, which performs
a clean documentation build, launches a local browser over the build, and
then triggers rebuilds in response to source changes. Execute via
`docs/watch` from the project root directory.

The docs build from the `master` branch is auto-deployed to
http://tmol.ipd.uw.edu via the `gh-pages` branch during the `buildkite`
"Pages" build step. Documentation builds from non-`master` branches are
available as a build artifact of the "Docs" build step under
`docs/_build/html.tgz`.

## Editing

All docstring markup and freeform notes are written in
[reStructuredText](http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
and formatted by `sphinx`. `index.rst` is the documentation root. Any new
top-level modules in `tmol` should be added to `index.rst`, as should any
new freeform documentation added in this directory.

