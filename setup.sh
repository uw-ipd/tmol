#!/bin/bash

set -e
set -x

# Install basic dependencies
TORCH_TARGET_VERSION="0ef1038"

conda env update -f environment.yml
source activate tmol

mkdir -p src
if [ ! -d "src/pytorch" ]; then
  git clone --recursive https://github.com/pytorch/pytorch src/pytorch
fi

pushd src/pytorch

export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" # [anaconda root directory]
export NO_CUDA=1

git clean -xd -i
git checkout $TORCH_TARGET_VERSION
python setup.py install
