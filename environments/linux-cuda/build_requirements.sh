#!/usr/bin/env bash

pip-compile --verbose --no-emit-index-url --resolver=backtracking -o requirements-linux-cuda.txt ../../requirements.in
pip-compile --verbose --no-emit-index-url --resolver=backtracking -c requirements-linux-cuda.txt -o requirements-dev-linux-cuda.txt ../../requirements-dev.in
