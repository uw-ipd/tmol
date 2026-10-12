"""Authorize GPU scheduling on a hosted runner before inspecting the target code."""

import json
import os
from pathlib import Path
import re
from urllib.parse import quote
from urllib.request import Request, urlopen


def api(path):
    request = Request(
        f"{os.environ['GITHUB_API_URL']}/repos/{os.environ['GITHUB_REPOSITORY']}/{path}",
        headers={
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def authorize(env, event, request=api):
    """Fail closed on trigger, current permissions, target, or API failures."""
    default = event["repository"]["default_branch"]
    trigger, ref = env["GITHUB_EVENT_NAME"], env["GITHUB_REF"]
    release = env.get("GPU_ALLOW_RELEASE") == "true"
    if trigger == "workflow_dispatch":
        if ref != f"refs/heads/{default}":
            raise ValueError("Dispatch the trusted workflow from the default branch")
    elif not (release and trigger == "push" and ref.startswith("refs/tags/v")):
        raise ValueError("GPU jobs require a maintainer dispatch or release tag push")

    for actor in {env["GITHUB_ACTOR"], env["GITHUB_TRIGGERING_ACTOR"]}:
        if not actor:
            raise ValueError("Missing workflow actor")
        permission = request(f"collaborators/{quote(actor, safe='')}/permission")
        # GitHub maps maintain to write and triage to read in this field.
        if permission.get("permission") not in {"write", "maintain", "admin"}:
            raise PermissionError(f"{actor} requires write, maintain, or admin access")

    pr = env.get("GPU_PR", "").strip()
    target = env.get("GPU_REF", "").strip()
    if pr and target:
        raise ValueError("Choose a pull request OR a ref, not both")
    if pr:
        if not re.fullmatch(r"[1-9][0-9]*", pr):
            raise ValueError("Pull request must be a positive integer")
        pull = request(f"pulls/{pr}")
        if (
            pull["base"]["repo"]["full_name"].lower()
            != env["GITHUB_REPOSITORY"].lower()
        ):
            raise ValueError("Pull request must target this repository")
        sha = pull["head"]["sha"]
    else:
        # Pin default selection to this run, including release tags and reruns.
        sha = request(f"commits/{quote(target or env['GITHUB_SHA'], safe='')}")["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("GitHub did not resolve an immutable commit SHA")
    return {"sha": sha, "attempt": env["GITHUB_RUN_ATTEMPT"]}


def main():
    result = authorize(
        os.environ, json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    )
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        for name, value in result.items():
            stream.write(f"{name}={value}\n")
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as stream:
        stream.write(
            f"Authorized GPU validation of `{result['sha']}` (attempt {result['attempt']}).\n"
        )


if __name__ == "__main__":
    main()
