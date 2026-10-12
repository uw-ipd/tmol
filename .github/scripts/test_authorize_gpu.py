"""GPU authorization and exact-source selection must fail before runner scheduling."""

from pathlib import Path

import pytest
import yaml

from authorize_gpu import authorize

ROOT = Path(__file__).resolve().parents[2]
SHA = "a" * 40
HEAD = "b" * 40


@pytest.fixture
def env():
    return {
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_REPOSITORY": "RosettaCommons/tmol",
        "GITHUB_ACTOR": "maintainer",
        "GITHUB_TRIGGERING_ACTOR": "maintainer",
        "GITHUB_SHA": SHA,
        "GITHUB_RUN_ATTEMPT": "1",
    }


def run(env, permission="write", target=HEAD):
    calls = []

    def request(path):
        calls.append(path)
        if path.startswith("collaborators/"):
            return {"permission": permission}
        if path.startswith("pulls/"):
            return {
                "base": {"repo": {"full_name": "RosettaCommons/tmol"}},
                "head": {"sha": target},
            }
        return {"sha": target}

    return authorize(env, {"repository": {"default_branch": "main"}}, request), calls


@pytest.mark.parametrize("permission", ["write", "maintain", "admin"])
def test_maintainer_resolves_exact_run_commit(env, permission):
    result, calls = run(env, permission)
    assert result == {"sha": HEAD, "attempt": "1"}
    assert calls[-1] == f"commits/{SHA}"


@pytest.mark.parametrize("permission", ["read", "triage", "none", "", None])
def test_nonmaintainers_are_denied_before_target_lookup(env, permission):
    with pytest.raises(PermissionError):
        run(env, permission)


def test_unprivileged_rerun_cannot_reuse_original_actor(env):
    env.update(GITHUB_TRIGGERING_ACTOR="contributor", GITHUB_RUN_ATTEMPT="2")
    calls = []

    def request(path):
        calls.append(path)
        return {"permission": "read" if "contributor" in path else "admin"}

    with pytest.raises(PermissionError, match="contributor"):
        authorize(env, {"repository": {"default_branch": "main"}}, request)
    assert all(path.startswith("collaborators/") for path in calls)


@pytest.mark.parametrize("trigger", ["pull_request", "pull_request_target", "schedule"])
def test_automatic_triggers_cannot_authorize_gpu(env, trigger):
    env["GITHUB_EVENT_NAME"] = trigger
    with pytest.raises(ValueError, match="maintainer dispatch"):
        run(env)


def test_dispatch_must_use_trusted_default_branch(env):
    env["GITHUB_REF"] = "refs/heads/unreviewed"
    with pytest.raises(ValueError, match="default branch"):
        run(env)


def test_release_tags_need_explicit_release_lane(env):
    env.update(GITHUB_EVENT_NAME="push", GITHUB_REF="refs/tags/v0.1.64")
    with pytest.raises(ValueError):
        run(env)
    env["GPU_ALLOW_RELEASE"] = "true"
    result, calls = run(env)
    assert result["sha"] == HEAD
    assert calls[-1] == f"commits/{SHA}"
    env["GITHUB_REF"] = "refs/heads/main"
    with pytest.raises(ValueError):
        run(env)


def test_pr_selection_uses_head_not_merge_commit(env):
    env["GPU_PR"] = "12"
    result, calls = run(env)
    assert calls[-1] == "pulls/12"
    assert result["sha"] == HEAD


def test_refs_are_encoded_and_pinned(env):
    env["GPU_REF"] = "feature/scoring"
    result, calls = run(env)
    assert calls[-1] == "commits/feature%2Fscoring"
    assert result["sha"] == HEAD


@pytest.mark.parametrize("pr", ["-1", "0", "12/foo", "$(command)"])
def test_invalid_pr_rejected(env, pr):
    env["GPU_PR"] = pr
    with pytest.raises(ValueError, match="positive integer"):
        run(env)


def test_ambiguous_target_rejected(env):
    env.update(GPU_PR="12", GPU_REF="main")
    with pytest.raises(ValueError, match="OR a ref"):
        run(env)


def test_api_errors_fail_closed(env):
    def unavailable(path):
        raise OSError("API unavailable")

    with pytest.raises(OSError):
        authorize(env, {"repository": {"default_branch": "main"}}, unavailable)


def test_nonimmutable_target_rejected(env):
    with pytest.raises(ValueError, match="immutable"):
        run(env, target="main")


def workflow(name):
    # BaseLoader avoids YAML 1.1 treating the Actions 'on' key as a boolean.
    return yaml.load(
        (ROOT / f".github/workflows/{name}.yml").read_text(), Loader=yaml.BaseLoader
    )


def test_automatic_ci_is_hosted_and_explicitly_cpu_only():
    ci = workflow("ci")
    assert ci["on"]["push"]["branches"] == ["main"]
    assert ci["jobs"]["ci"]["name"] == "ci (CPU)"
    assert ci["jobs"]["ci"]["needs"] == [
        "lint",
        "packaging",
        "jax-artifacts",
        "jax-cpu",
        "cpu",
    ]
    assert "self-hosted" not in str(ci)
    for name in ("wheel-smoke", "gpu-ci", "cuda-arm64-dependencies"):
        assert set(workflow(name)["on"]) == {"workflow_dispatch"}


@pytest.mark.parametrize(
    "name,job",
    [
        ("gpu-ci", "gpu"),
        ("gpu-ci", "jax-gpu"),
        ("_test_gpu_wheel", "install"),
    ],
)
def test_selfhosted_jobs_require_fresh_hosted_authorization(name, job):
    jobs = workflow(name)["jobs"]
    assert jobs["authorize"]["uses"] == "./.github/workflows/_authorize_gpu.yml"
    gpu = jobs[job]
    needs = gpu["needs"]
    assert needs == "authorize" or "authorize" in needs
    assert gpu["if"] == "needs.authorize.outputs.attempt == github.run_attempt"
    checkout = gpu["steps"][0]["with"]
    assert checkout["ref"] == "${{ needs.authorize.outputs.sha }}"
    assert checkout["persist-credentials"] == "false"
    assert jobs["result"]["if"] == "always()"
    assert job in jobs["result"]["needs"]


def test_jax_release_gpu_artifacts_are_explicit_opt_in():
    reusable = workflow("_test_gpu_wheel")
    assert (
        reusable["on"]["workflow_call"]["inputs"]["test-jax"]["default"]
        == "false"
    )
    jax_artifacts = {
        "tmol-jax-pure",
        "jax-native-cu12-x86_64",
        "jax-native-cu13-x86_64",
    }
    downloads = [
        step
        for step in reusable["jobs"]["install"]["steps"]
        if step.get("with", {}).get("name") in jax_artifacts
    ]
    assert {step["with"]["name"] for step in downloads} == jax_artifacts
    assert all(step["if"] == "inputs.test-jax" for step in downloads)
    assert (
        workflow("wheel-smoke")["jobs"]["test-gpu"]["with"]["test-jax"]
        == "true"
    )
    publish_with = workflow("publish")["jobs"]["test_gpu_runtime"]["with"]
    assert "test-jax" not in publish_with
    release_script = (ROOT / ".github/ci/test_release_gpu.sh").read_text()
    assert 'if [ "${TEST_JAX:-false}" = true ]' in release_script


def test_authorization_reads_only_trusted_default_branch_code():
    job = workflow("_authorize_gpu")["jobs"]["authorize"]
    assert job["runs-on"] == "ubuntu-latest"
    assert (
        job["steps"][0]["with"]["ref"]
        == "${{ github.event.repository.default_branch }}"
    )


def test_gpu_workflow_is_manual_and_reports_exact_sha():
    gpu = workflow("gpu-ci")
    assert list(gpu["on"]) == ["workflow_dispatch"]
    assert (
        gpu["jobs"]["jax-artifacts"]["with"]["checkout_ref"]
        == "${{ needs.authorize.outputs.sha }}"
    )
    report = gpu["jobs"]["result"]["steps"][0]
    assert report["env"]["TESTED_SHA"] == "${{ needs.authorize.outputs.sha }}"
    assert (
        "process.env.AUTHORIZED_ATTEMPT !== process.env.RUN_ATTEMPT"
        in report["with"]["script"]
    )
    assert "sha: process.env.TESTED_SHA" in report["with"]["script"]
    assert "if (!passed) core.setFailed" in report["with"]["script"]
