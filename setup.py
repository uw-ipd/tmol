#!/usr/bin/env python

import sys
from setuptools import setup
import subprocess
import re


def git_version():
    try:
        git_describe = (
            subprocess.check_output(
                ["git", "describe", "--long", "--tags", "--match", "[0-9]*"]
            )
            .strip()
            .decode()
        )
    except subprocess.CalledProcessError:
        version = "0.0.0"
        return version

    describe_match = re.match(
        r"(?P<version>[0-9.]+)(-(?P<post_revision>\d+)-g(?P<commit>\w+))?", git_describe
    )
    if not describe_match:
        raise ValueError("Invalid version.", git_describe)
    else:
        desc = describe_match.groupdict()

    desc["post_revision"] = int(desc.get("post_revision", 0))
    if not desc["post_revision"]:
        version = f"{desc['version']}+{desc['commit']}"
    else:
        version = f"{desc['version']}.post.dev+{desc['post_revision']}.{desc['commit']}"

    return version


needs_pytest = {"pytest", "test"}.intersection(sys.argv)
pytest_runner = ["pytest-runner"] if needs_pytest else []

setup(
    name="tmol",
    version=git_version(),
    packages=["tmol"],
    setup_requires=pytest_runner,
    # tests_require=[l.strip() for l in open("requirements.tests.txt").readlines()],
    # install_requires=[l.strip() for l in open("requirements.txt").readlines()],
    zip_safe=False,
)
