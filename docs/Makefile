# Minimal makefile for Sphinx documentation
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = ./
BUILDDIR      = build/
MODULEDIR     = ../tmol

# apidoc configuration from
# 
SPHINX_APIDOC_OPTS = -M -e 

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

clean: clean-apidoc clean-build

clean-build:
	rm -rf "$(SOURCEDIR)"/build

clean-apidoc:
	rm -rf "$(SOURCEDIR)"/apidoc

apidoc:
	better-apidoc -t _templates $(SPHINX_APIDOC_OPTS) -o "$(SOURCEDIR)"/apidoc "$(MODULEDIR)" "$(MODULEDIR)/tests"

html: apidoc 
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

singlehtml: apidoc 
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

text: apidoc 
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

coverage: apidoc 
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
