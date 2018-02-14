#!/usr/bin/env python

import os
from setuptools import setup
import subprocess
import re

def git_version():
    git_describe = subprocess.check_output(
            ["git", "describe", "--long", "--tags", "--match", "[0-9]*"]
        ).strip().decode()

    describe_match = re.match(
        r"(?P<version>[0-9.]+)(-(?P<post_revision>\d+)-g(?P<commit>\w+))?",
        git_describe)
    if not describe_match:
        raise ValueError("Invalid version.", git_describe)
    else:
        desc = describe_match.groupdict()

    desc["post_revision"] = int(desc.get("post_revision", 0))
    if not desc["post_revision"]:
        version = '{desc[version]}+{desc[commit]}'.format(**vars())
    else:
        version = '{desc[version]}.post.dev+{desc[post_revision]}.{desc[commit]}'.format(**vars())

    return version

setup(
    name='tmol',
    version=git_version(),
    packages=['tmol'],
    install_requires=[
        l.strip()
        for l in open("requirements.txt").readlines()
    ],
    zip_safe=False
)
