# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINX_APIDOC_OPTS = -M -e 
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = TMol
MODULEDIR     = ../tmol
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

clean-build:
	rm -rf build

clean-apidoc:
	rm -rf source/apidoc

clean: clean-apidoc clean-build

apidoc: clean-apidoc
	sphinx-apidoc $(SPHINX_APIDOC_OPTS) -o "$(SOURCEDIR)"/apidoc "$(MODULEDIR)" "$(MODULEDIR)/tests"

html: apidoc 
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

singlehtml: apidoc 
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

text: apidoc 
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

coverage: apidoc 
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
