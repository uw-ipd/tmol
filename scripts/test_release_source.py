"""Recovery must use complete, unexpired artifacts from the immutable tag run."""

from copy import deepcopy
from pathlib import Path
import subprocess

import pytest
import yaml

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
@pytest.mark.parametrize("source_run", ["", "123"])
def test_manual_release_cannot_publish_from_an_unreviewed_ref(
    monkeypatch, ref, source_run
):
    monkeypatch.setenv("CANDIDATE_RUN_ID", source_run)
    monkeypatch.setenv("RELEASE_TAG", "v0.1.60")
    monkeypatch.setenv("VALIDATE_ONLY", "false")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", ref)
    with pytest.raises(ValueError, match="reviewed workflow on master"):
        verify_release_source.main()


def git(cwd, *args):
    return subprocess.check_output(
        ["git", "-C", str(cwd), *args], text=True, stderr=subprocess.PIPE
    ).strip()


@pytest.fixture
def release_checkout(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    remote.mkdir()
    git(remote, "init")
    (remote / "pyproject.toml").write_text('[project]\nversion = "9.9.9"\n')
    (remote / "scripts").mkdir()
    (remote / "scripts/release_matrix.py").write_text("# immutable matrix\n")
    (remote / "runtime.py").write_text("# tagged runtime\n")
    git(remote, "add", ".")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Release test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "release@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Release test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "release@example.invalid")
    git(remote, "commit", "-m", "Release source")
    sha = git(remote, "rev-parse", "HEAD")
    git(remote, "tag", "-a", "v9.9.9", "-m", "Annotated release")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    git(checkout, "init")
    git(checkout, "remote", "add", "origin", str(remote))
    # Reproduce actions/checkout: fetch a commit into a lightweight tag ref.
    git(checkout, "fetch", "--no-tags", "origin", f"{sha}:refs/tags/v9.9.9")
    git(checkout, "checkout", "--detach", "v9.9.9")
    monkeypatch.chdir(checkout)
    for key, value in {
        "CANDIDATE_RUN_ID": "",
        "RELEASE_TAG": "",
        "VALIDATE_ONLY": "",
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REPOSITORY": "example/release-test",
        "GITHUB_REF_TYPE": "tag",
        "GITHUB_REF": "refs/tags/v9.9.9",
        "GITHUB_REF_NAME": "v9.9.9",
        "GITHUB_OUTPUT": str(tmp_path / "outputs"),
    }.items():
        monkeypatch.setenv(key, value)
    return remote, checkout, sha, tmp_path / "outputs"


@pytest.mark.parametrize("annotated", [True, False])
def test_tag_fetch_preserves_local_and_remote_refs(release_checkout, annotated):
    remote, checkout, sha, outputs = release_checkout
    if not annotated:
        git(remote, "tag", "-d", "v9.9.9")
        git(remote, "tag", "v9.9.9")
    tag_object = git(remote, "rev-parse", "v9.9.9")
    verify_release_source.main()
    values = dict(line.split("=", 1) for line in outputs.read_text().splitlines())
    assert values["source_sha"] == values["build_sha"] == sha
    assert values["candidate_run_id"] == ""
    assert git(remote, "rev-parse", "v9.9.9") == tag_object
    assert git(checkout, "rev-parse", "v9.9.9") == sha


@pytest.mark.parametrize("validate_only", [False, True])
def test_manual_fresh_build_pins_tag_not_workflow_head(
    release_checkout, monkeypatch, validate_only
):
    _, checkout, sha, outputs = release_checkout
    Path("runtime.py").write_text("# unrelated later runtime\n")
    git(checkout, "commit", "-am", "Later workflow source")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv(
        "GITHUB_REF", "refs/heads/repair" if validate_only else "refs/heads/master"
    )
    monkeypatch.setenv("RELEASE_TAG", "v9.9.9")
    monkeypatch.setenv("VALIDATE_ONLY", str(validate_only).lower())
    verify_release_source.main()
    values = dict(line.split("=", 1) for line in outputs.read_text().splitlines())
    assert values["build_sha"] == sha != git(checkout, "rev-parse", "HEAD")
    assert values["candidate_run_id"] == ""


@pytest.mark.parametrize("path", ["pyproject.toml", "scripts/release_matrix.py"])
def test_manual_build_rejects_changed_release_configuration(
    release_checkout, monkeypatch, path
):
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/master")
    monkeypatch.setenv("RELEASE_TAG", "v9.9.9")
    Path(path).write_text(Path(path).read_text() + "# changed configuration\n")
    with pytest.raises(ValueError, match="must preserve the tagged"):
        verify_release_source.main()


def test_tag_push_rejects_a_different_checkout(release_checkout):
    _, checkout, _, outputs = release_checkout
    Path("runtime.py").write_text("# wrong source\n")
    git(checkout, "commit", "-am", "Wrong source")
    with pytest.raises(ValueError, match="Checkout does not match release tag"):
        verify_release_source.main()
    assert not outputs.exists()


@pytest.mark.parametrize("missing_artifact", [False, True])
def test_candidate_recovery_still_validates_artifacts_without_rebuilding(
    release_checkout, candidate, monkeypatch, missing_artifact
):
    _, _, sha, outputs = release_checkout
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/master")
    monkeypatch.setenv("RELEASE_TAG", "v9.9.9")
    monkeypatch.setenv("CANDIDATE_RUN_ID", "123")
    run, artifacts, jobs = candidate
    run.update(head_branch="v9.9.9", head_sha=sha)
    original = artifacts[0]
    original["workflow_run"]["head_sha"] = sha
    names = {
        f"wheel-cp{row['python-tag']}-{row['local-tag']}-{row['arch']}"
        for row in verify_release_source.linux_wheel_rows()
        + verify_release_source.macos_wheel_rows()
    } | {"sdist"}
    artifacts = [dict(deepcopy(original), name=name) for name in names]
    if missing_artifact:
        artifacts.pop()
    monkeypatch.setattr(
        verify_release_source,
        "api",
        lambda path, key=None: {None: run, "artifacts": artifacts, "jobs": jobs}[key],
    )
    if missing_artifact:
        with pytest.raises(ValueError, match="complete release manifest"):
            verify_release_source.main()
        assert not outputs.exists()
    else:
        verify_release_source.main()
        values = dict(line.split("=", 1) for line in outputs.read_text().splitlines())
        assert values["candidate_run_id"] == "123"
        assert values["source_sha"] == sha
        assert values["build_sha"] == ""


@pytest.mark.parametrize(
    "event,source_run", [("push", ""), ("workflow_dispatch", "bad")]
)
def test_manual_release_rejects_wrong_event_or_candidate_id(
    monkeypatch, event, source_run
):
    monkeypatch.setenv("GITHUB_EVENT_NAME", event)
    monkeypatch.setenv("RELEASE_TAG", "v9.9.9")
    monkeypatch.setenv("CANDIDATE_RUN_ID", source_run)
    with pytest.raises(ValueError, match="Manual release requires"):
        verify_release_source.main()


def test_all_release_builds_checkout_verified_source():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / ".github/workflows/publish.yml").read_text())
    jobs = workflow["jobs"]
    assert (
        jobs["verify_release_version"]["outputs"]["build_sha"]
        == "${{ steps.source.outputs.build_sha }}"
    )
    for name in (
        "build_wheels",
        "build_cpu_wheel",
        "build_macos_cpu_wheel",
        "build_sdist",
    ):
        job = jobs[name]
        assert (
            job["with"]["source-ref"]
            == "${{ needs.verify_release_version.outputs.build_sha }}"
        )
        template = yaml.safe_load((root / job["uses"]).read_text())
        for child in template["jobs"].values():
            checkouts = [
                step
                for step in child["steps"]
                if step.get("uses", "").startswith("actions/checkout@")
            ]
            assert len(checkouts) == 1
            assert checkouts[0]["with"]["ref"] == "${{ inputs.source-ref }}"
