# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = ./
BUILDDIR      = ./

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

clean-build:
	rm -rf build

clean-apidoc:
	rm -rf source/apidoc

clean: clean-apidoc clean-build

html: clean-apidoc 
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

livehtml: clean-apidoc 
	sphinx-autobuild -b html -i apidoc -i html "$(SOURCEDIR)" $(SPHINXOPTS) $(BUILDDIR)/html

coverage: clean-apidoc 
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
