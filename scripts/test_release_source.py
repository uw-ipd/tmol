"""Recovery must use complete, unexpired artifacts from the immutable tag run."""

from copy import deepcopy

import pytest

from verify_release_source import validate_candidate_run
import verify_release_source


@pytest.fixture
def candidate():
    run = {
        "id": 123,
        "status": "completed",
        "event": "push",
        "path": ".github/workflows/publish.yml",
        "head_branch": "v0.1.60",
        "head_sha": "abc",
    }
    artifacts = [
        {
            "name": name,
            "expired": False,
            "workflow_run": {
                "id": 123,
                "head_sha": "abc",
                "repository_id": 1,
                "head_repository_id": 1,
            },
        }
        for name in ("sdist", "wheel-cp312-cu132torch2.14-aarch64")
    ]
    jobs = [
        {
            "name": "Build and test source distribution / install",
            "conclusion": "success",
        }
    ]
    return run, artifacts, jobs


def validate(candidate):
    validate_candidate_run(
        *candidate,
        tag="v0.1.60",
        sha="abc",
        expected_names={"sdist", "wheel-cp312-cu132torch2.14-aarch64"},
    )


def test_complete_tag_candidates_accepted(candidate):
    validate(candidate)


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "in_progress"),
        ("event", "pull_request"),
        ("head_sha", "other"),
        ("head_branch", "master"),
        ("path", ".github/workflows/ci.yml"),
    ],
)
def test_wrong_source_run_rejected(candidate, field, value):
    candidate[0][field] = value
    with pytest.raises(ValueError, match="completed publish workflow"):
        validate(candidate)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "duplicate",
        "expired",
        "wrong_sha",
        "wrong_run",
        "fork",
        "source_failed",
    ],
)
def test_invalid_artifacts_or_source_test_rejected(candidate, mutation):
    _, artifacts, jobs = candidate
    if mutation == "missing":
        artifacts.pop()
    elif mutation in {"extra", "duplicate"}:
        artifacts.append(deepcopy(artifacts[-1]))
        if mutation == "extra":
            artifacts[-1]["name"] = "wheel-unexpected"
    elif mutation == "expired":
        artifacts[0]["expired"] = True
    elif mutation == "wrong_sha":
        artifacts[0]["workflow_run"]["head_sha"] = "other"
    elif mutation == "wrong_run":
        artifacts[0]["workflow_run"]["id"] = 456
    elif mutation == "fork":
        artifacts[0]["workflow_run"]["head_repository_id"] = 2
    else:
        jobs[0]["conclusion"] = "failure"
    with pytest.raises(ValueError):
        validate(candidate)


@pytest.mark.parametrize("ref", ["refs/heads/repair", "refs/tags/v0.1.60"])
def test_recovery_cannot_publish_from_an_unreviewed_branch_or_tag(monkeypatch, ref):
    monkeypatch.setenv("CANDIDATE_RUN_ID", "123")
    monkeypatch.setenv("RELEASE_TAG", "v0.1.60")
    monkeypatch.setenv("VALIDATE_ONLY", "false")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", ref)
    with pytest.raises(ValueError, match="reviewed workflow on master"):
        verify_release_source.main()
