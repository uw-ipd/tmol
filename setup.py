#!/usr/bin/env python

import os
from setuptools import setup
import subprocess

version = subprocess.check_output(
        ["git", "describe", "--tags", "--match", "[0-9]*"]
    ).strip().decode()

setup(
    name='tmol',
    version=version,
    packages=['tmol'],
    install_requires=[
        l.strip()
        for l in open("requirements.txt").readlines()
    ],
    zip_safe=False
)
