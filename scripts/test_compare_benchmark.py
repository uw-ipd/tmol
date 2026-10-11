"""Benchmark failures and revision isolation must preserve developers' work."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import compare_benchmark as compare

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repository with spaces"
    root.mkdir()
    compare.git(root, "init", "--quiet")
    compare.git(root, "config", "user.email", "test@example.com")
    compare.git(root, "config", "user.name", "Test")
    (root / "tracked.txt").write_text("original\n")
    (root / "tracked.bin").write_bytes(b"\0\x01")
    (root / "deleted.txt").write_text("remove me\n")
    (root / ".gitignore").write_text("ignored/\n")
    compare.git(root, "add", ".")
    compare.git(root, "commit", "--quiet", "-m", "initial")
    return root


def test_tree_snapshot_keeps_staged_unstaged_untracked_and_deletions(
    repository, tmp_path
):
    root = repository
    (root / "tracked.txt").write_text("staged\n")
    compare.git(root, "add", "tracked.txt")
    (root / "tracked.txt").write_text("unstaged\n")
    (root / "tracked.bin").write_bytes(b"\0\xff")
    (root / "deleted.txt").unlink()
    (root / "untracked.dat").write_bytes(b"\0\xff")
    (root / "ignored").mkdir()
    (root / "ignored" / "cache").write_text("do not copy")
    (root / "link").symlink_to("tracked.txt")
    before = compare.git(root, "status", "--porcelain=v1", "-z")
    index = compare.git(root, "diff", "--cached", "--binary")
    head = compare.git(root, "rev-parse", "HEAD")
    record = compare.snapshot(root, "TREE", tmp_path / "tree")
    compare.snapshot(root, "HEAD", tmp_path / "committed")
    assert (tmp_path / "tree" / "tracked.txt").read_text() == "unstaged\n"
    assert (tmp_path / "tree" / "tracked.bin").read_bytes() == b"\0\xff"
    assert not (tmp_path / "tree" / "deleted.txt").exists()
    assert (tmp_path / "tree" / "untracked.dat").read_bytes() == b"\0\xff"
    assert (tmp_path / "tree" / "link").is_symlink()
    assert not (tmp_path / "tree" / "ignored").exists()
    assert (tmp_path / "committed" / "tracked.txt").read_text() == "original\n"
    assert record["tree_sha256"]
    assert compare.git(root, "status", "--porcelain=v1", "-z") == before
    assert compare.git(root, "diff", "--cached", "--binary") == index
    assert compare.git(root, "rev-parse", "HEAD") == head
    assert compare.git(root, "stash", "list") == b""


def test_invalid_revision_does_not_change_checkout(repository, tmp_path):
    before = compare.git(repository, "status", "--porcelain=v1", "-z")
    with pytest.raises(subprocess.CalledProcessError):
        compare.snapshot(repository, "missing-revision", tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
    assert compare.git(repository, "status", "--porcelain=v1", "-z") == before


def test_tree_digest_includes_untracked_executable_mode(repository, tmp_path):
    script = repository / "untracked.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o644)
    first = compare.snapshot(repository, "TREE", tmp_path / "first")
    script.chmod(0o755)
    second = compare.snapshot(repository, "TREE", tmp_path / "second")
    assert first["tree_sha256"] != second["tree_sha256"]
    assert (tmp_path / "second/untracked.sh").stat().st_mode & 0o111


@pytest.mark.parametrize(
    "stage,failure",
    [
        ("building", subprocess.CalledProcessError(7, "build")),
        ("building", KeyboardInterrupt()),
        ("benchmarking", subprocess.CalledProcessError(3, "pytest")),
    ],
)
def test_failure_retains_logs_manifest_and_caller_tree(
    repository, tmp_path, monkeypatch, stage, failure
):
    out = tmp_path / "results"
    out.mkdir()
    (repository / "tracked.txt").write_text("local edit")
    before = compare.git(repository, "status", "--porcelain=v1", "-z")
    monkeypatch.setattr(compare, "environment", lambda path: (Path(sys.executable), {}))

    def fail(command, *, cwd, env, log):
        if stage == "building" or "--benchmark-only" in command:
            log.write_text("diagnostic output")
            raise failure

    monkeypatch.setattr(compare, "run", fail)
    with pytest.raises(type(failure)):
        compare.compare(repository, [], ["TREE", "HEAD"], out)
    assert (out / "0.log").read_text() == "diagnostic output"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["runs"][0]["status"] == stage
    assert not (out / "comparison.txt").exists()
    assert all(not Path(run["source"]).exists() for run in manifest["runs"])
    assert compare.git(repository, "status", "--porcelain=v1", "-z") == before
    assert (repository / "tracked.txt").read_text() == "local edit"


def test_comparison_uses_each_revisions_environment_and_build(
    repository, tmp_path, monkeypatch
):
    out = tmp_path / "results"
    out.mkdir()
    calls = []
    temporary_directory = compare.tempfile.TemporaryDirectory
    resolved_roots = []

    @contextmanager
    def aliased_temporary_directory(**kwargs):
        with temporary_directory(dir=tmp_path, **kwargs) as directory:
            resolved_roots.append(Path(directory).resolve())
            alias = tmp_path / "aliased-build-directory"
            alias.symlink_to(directory, target_is_directory=True)
            yield str(alias)

    monkeypatch.setattr(
        compare.tempfile, "TemporaryDirectory", aliased_temporary_directory
    )

    def environment(path):
        return path / "bin/python", {"TORCH_EXTENSIONS_DIR": str(path / "jit")}

    def run(command, *, cwd, env, log):
        calls.append((command, cwd, env))
        log.write_text("success")
        for arg in command:
            if arg.startswith("--benchmark-json="):
                Path(arg.split("=", 1)[1]).write_text(
                    '{"benchmarks": [{"name": "example"}]}'
                )

    monkeypatch.setattr(compare, "environment", environment)
    monkeypatch.setattr(compare, "run", run)
    compare.compare(
        repository, ["tmol/tests/score", "-k", "selection"], ["TREE", "HEAD"], out
    )
    builds = [(cmd, cwd, env) for cmd, cwd, env in calls if "install" in cmd]
    benchmarks = [
        (cmd, cwd, env) for cmd, cwd, env in calls if "--benchmark-only" in cmd
    ]
    assert len(builds) == len(benchmarks) == 2
    assert builds[0][0][0] != builds[1][0][0]
    assert builds[0][1] != builds[1][1]
    assert all(cwd.is_relative_to(resolved_roots[0]) for _, cwd, _ in builds)
    assert builds[0][2]["TORCH_EXTENSIONS_DIR"] != builds[1][2]["TORCH_EXTENSIONS_DIR"]
    for build, benchmark in zip(builds, benchmarks):
        assert build[0][0] == benchmark[0][0]
        assert build[1] == benchmark[1]
        assert "--no-deps" in build[0]
        assert "-Ccmake.define.TMOL_BUILD_TESTS=ON" in build[0]
    assert all(
        run["status"] == "complete"
        for run in json.loads((out / "manifest.json").read_text())["runs"]
    )


def test_run_preserves_failed_command_output(tmp_path):
    log = tmp_path / "failure.log"
    with pytest.raises(subprocess.CalledProcessError) as exc:
        compare.run(
            [sys.executable, "-c", "print('useful error'); raise SystemExit(7)"],
            cwd=tmp_path,
            env=os.environ.copy(),
            log=log,
        )
    assert exc.value.returncode == 7
    assert "useful error" in log.read_text()


def test_benchmark_wrapper_propagates_pytest_failure(repository, tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "pytest"
    fake.write_text("#!/bin/sh\necho failing-test\nexit 7\n")
    fake.chmod(0o755)
    env = dict(
        os.environ,
        PATH=str(bindir) + os.pathsep + os.environ["PATH"],
        HOSTNAME="host with spaces",
    )
    result = subprocess.run(
        ["bash", str(ROOT / "dev/bin/benchmark")],
        cwd=repository,
        env=env,
        capture_output=True,
    )
    assert result.returncode == 7
    assert b"failing-test" in result.stdout
    assert len(list((repository / "dev/benchmark").rglob("*.summary.txt"))) == 1


def test_argument_order_is_preserved():
    _, selectors, revisions = compare.parse_args(
        ["tmol/tests/score", "-k", "some benchmark", "--", "HEAD", "TREE", "HEAD"]
    )
    assert selectors == ["tmol/tests/score", "-k", "some benchmark"]
    assert revisions == ["HEAD", "TREE"]
    assert compare.parse_args([])[2] == ["TREE", "HEAD"]
    options, selectors, _ = compare.parse_args(
        ["--benchmark-outdir", "results", "tmol/tests", "-o", "addopts="]
    )
    assert options.benchmark_outdir == Path("results")
    assert selectors == ["tmol/tests", "-o", "addopts="]


def test_environment_shares_dependencies_without_parent_editable_hooks(
    tmp_path, monkeypatch
):
    dependencies = tmp_path / "dependencies"
    dependencies.mkdir()
    (dependencies / "benchmark_shared_dependency.py").write_text("VALUE = 17\n")
    (dependencies / "old_editable.pth").write_text(
        "import sys; sys.OLD_TMOL_HOOK = True\n"
    )
    monkeypatch.setattr(compare.site, "getsitepackages", lambda: [str(dependencies)])
    monkeypatch.setenv("PYTHONPATH", "an-unrelated-checkout")
    python, env = compare.environment(tmp_path / "isolated")
    assert "PYTHONPATH" not in env
    subprocess.run(
        [
            str(python),
            "-c",
            "import sys, benchmark_shared_dependency as dep; "
            "assert dep.VALUE == 17; assert not hasattr(sys, 'OLD_TMOL_HOOK')",
        ],
        env=env,
        check=True,
    )


@pytest.mark.parametrize("contents", [None, '{"benchmarks": []}'])
def test_empty_benchmark_results_are_not_success(
    repository, tmp_path, monkeypatch, contents
):
    output = tmp_path / "results"
    output.mkdir()
    monkeypatch.setattr(compare, "environment", lambda path: (Path(sys.executable), {}))

    def run(command, **kwargs):
        for arg in command:
            if arg.startswith("--benchmark-json=") and contents is not None:
                Path(arg.split("=", 1)[1]).write_text(contents)

    monkeypatch.setattr(compare, "run", run)
    with pytest.raises(ValueError, match="No benchmark samples"):
        compare.compare(repository, [], ["TREE", "HEAD"], output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert "No benchmark samples" in manifest["failure"]
    assert not (output / "comparison.txt").exists()
