#!/bin/bash

set -e
set -x

PYTORCH_BUILD_VERSION="0.4.0"
#PYTORCH_BUILD_NUMBER=0
#PYTORCH_BUILD_REVISION=1936753

time conda build --python 3.5 pytorch-cpu-$PYTORCH_BUILD_VERSION $*
#time conda build -c $ANACONDA_USER --no-anaconda-upload --python 3.6 pytorch-cpu-$BUILD_VERSION
