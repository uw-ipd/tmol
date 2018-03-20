#!/bin/bash

set -e
set -x

TORCH_TARGET_VERSION="1936753"

# Requires an active root conda installation see:
# https://conda.io/miniconda.html
conda env update -f environment.yml
source activate pytorch_install

mkdir -p src
if [ ! -d "src/pytorch" ]; then
  git clone --recursive https://github.com/pytorch/pytorch src/pytorch
fi

pushd src/pytorch

export CMAKE_PREFIX_PATH="$(conda info | grep 'active env location' | sed 's/.*: //')" # [anaconda root directory]
export NO_CUDA=1

git clean -xd -i
git checkout $TORCH_TARGET_VERSION
python setup.py bdist_wheel

popd
