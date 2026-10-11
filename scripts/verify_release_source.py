#!/usr/bin/env python3
"""Verify immutable release source and optional candidates for recovery.

Recovery revalidates artifacts from the original tag run with the current CI
tools. Manual runs without candidates build the verified tag with current CI
tools. Manual publication requires master; branch runs are validation only.
"""

import json
import os
from pathlib import Path
import re
import subprocess
import tomllib

from release_matrix import linux_wheel_rows, macos_wheel_rows


def validate_candidate_run(run, artifacts, jobs, *, tag, sha, expected_names):
    if not (
        run["status"] == "completed"
        and run["event"] == "push"
        and run["path"] == ".github/workflows/publish.yml"
        and run["head_branch"] == tag
        and run["head_sha"] == sha
    ):
        raise ValueError(
            "Candidate run must be the completed publish workflow for this tag"
        )
    candidates = [
        a for a in artifacts if a["name"] == "sdist" or a["name"].startswith("wheel-")
    ]
    if (
        len(candidates) != len(expected_names)
        or {a["name"] for a in candidates} != expected_names
    ):
        raise ValueError(
            "Candidate run does not contain exactly the complete release manifest"
        )
    for artifact in candidates:
        source = artifact["workflow_run"]
        if (
            artifact["expired"]
            or source["id"] != run["id"]
            or source["head_sha"] != sha
            or source["repository_id"] != source["head_repository_id"]
        ):
            raise ValueError(
                "Candidate artifact has expired or has incorrect provenance"
            )
    if not any(
        j["name"] == "Build and test source distribution / install"
        and j["conclusion"] == "success"
        for j in jobs
    ):
        raise ValueError(
            "Original source distribution build/install test must have passed"
        )


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def api(path, key=None):
    if key:
        pages = json.loads(command("gh", "api", "--paginate", "--slurp", path))
        return [item for page in pages for item in page[key]]
    return json.loads(command("gh", "api", path))


def main():
    source_run = os.environ.get("CANDIDATE_RUN_ID", "")
    requested_tag = os.environ.get("RELEASE_TAG", "")
    manual = bool(source_run or requested_tag)
    if manual:
        if os.environ["GITHUB_EVENT_NAME"] != "workflow_dispatch" or (
            source_run and not source_run.isdecimal()
        ):
            raise ValueError(
                "Manual release requires workflow_dispatch and an optional numeric candidate run ID"
            )
        if (
            os.environ.get("VALIDATE_ONLY") != "true"
            and os.environ["GITHUB_REF"] != "refs/heads/master"
        ):
            raise ValueError(
                "Manual publication must use the reviewed workflow on master"
            )
        tag = requested_tag
    else:
        if os.environ["GITHUB_REF_TYPE"] != "tag":
            raise ValueError("Normal publication must run on a version tag")
        tag = os.environ["GITHUB_REF_NAME"]
    if not re.fullmatch(r"v[0-9][A-Za-z0-9.+-]*", tag):
        raise ValueError("Invalid release tag")
    # Checkout may create a lightweight local ref for an annotated remote tag.
    # Fetch into FETCH_HEAD so neither local nor remote tags are replaced.
    subprocess.run(
        ["git", "fetch", "--no-tags", "origin", f"refs/tags/{tag}"],
        check=True,
    )
    sha = command("git", "rev-parse", "FETCH_HEAD^{commit}")
    project = tomllib.loads(command("git", "show", f"{sha}:pyproject.toml"))
    version = project["project"]["version"]
    if tag != f"v{version}":
        raise ValueError("Tag and tagged project version differ")
    if manual:
        for path in ("pyproject.toml", "scripts/release_matrix.py"):
            if (
                command("git", "show", f"{sha}:{path}")
                != Path(path).read_text().strip()
            ):
                raise ValueError(f"Manual release must preserve the tagged {path}")
    elif sha != command("git", "rev-parse", "HEAD"):
        raise ValueError("Checkout does not match release tag")
    if source_run:
        base = f"repos/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{source_run}"
        run = api(base)
        artifacts = api(f"{base}/artifacts?per_page=100", "artifacts")
        jobs = api(f"{base}/jobs?per_page=100", "jobs")
        expected_names = {
            f"wheel-cp{row['python-tag']}-{row['local-tag']}-{row['arch']}"
            for row in linux_wheel_rows() + macos_wheel_rows()
        } | {"sdist"}
        validate_candidate_run(
            run, artifacts, jobs, tag=tag, sha=sha, expected_names=expected_names
        )
    print(f"Verified {tag} at {sha}; candidate run: {source_run or 'current run'}")
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        for key, value in {
            "release_tag": tag,
            "version": version,
            "candidate_run_id": source_run,
            "source_sha": sha,
            "build_sha": "" if source_run else sha,
        }.items():
            stream.write(f"{key}={value}\n")


if __name__ == "__main__":
    main()
