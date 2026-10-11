#!/usr/bin/env python3
"""Compare freshly built revisions without changing the caller's checkout."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import site
import subprocess
import sys
import tempfile
import venv


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def parse_args(argv):
    separator = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--benchmark-outdir", type=Path)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    options, unknown = parser.parse_known_args(argv[:separator])
    pytest_args = unknown + options.pytest_args
    revisions = argv[separator + 1 :]
    if not revisions:
        revisions = ["TREE", "HEAD"]
    elif len(revisions) == 1:
        revisions = ["TREE", *revisions]
    return options, pytest_args, list(dict.fromkeys(revisions))


def snapshot(root, revision, destination):
    """Clone committed sources, adding tracked edits/untracked files for TREE."""
    commit = (
        git(
            root,
            "rev-parse",
            "--verify",
            "--end-of-options",
            f"{'HEAD' if revision == 'TREE' else revision}^{{commit}}",
        )
        .decode()
        .strip()
    )
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--shared",
            "--no-checkout",
            "--",
            str(root),
            str(destination),
        ],
        check=True,
    )
    git(destination, "checkout", "--quiet", "--detach", commit)
    tree_digest = None
    if revision == "TREE":
        patch = git(root, "diff", "HEAD", "--binary", "--no-ext-diff", "--no-textconv")
        if patch:
            subprocess.run(
                ["git", "-C", str(destination), "apply", "--binary", "-"],
                input=patch,
                check=True,
            )
        digest = hashlib.sha256(patch)
        untracked = git(root, "ls-files", "--others", "--exclude-standard", "-z")
        for raw_name in untracked.split(b"\0"):
            if not raw_name:
                continue
            name = os.fsdecode(raw_name)
            src, dst = root / name, destination / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst, follow_symlinks=False)
            contents = (
                os.fsencode(os.readlink(src)) if src.is_symlink() else src.read_bytes()
            )
            digest.update(len(raw_name).to_bytes(8, "big"))
            digest.update(raw_name)
            digest.update(src.lstat().st_mode.to_bytes(4, "big"))
            digest.update(hashlib.sha256(contents).digest())
        tree_digest = digest.hexdigest()
    return {"revision": revision, "commit": commit, "tree_sha256": tree_digest}


def run(command, *, cwd, env, log):
    print(f"{shlex.join(map(str, command))}\n  log: {log}", flush=True)
    with log.open("a") as output:
        subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )


def environment(directory):
    """Share installed dependencies, but not another checkout's editable hooks."""
    venv.EnvBuilder(with_pip=True).create(directory)
    python = directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    purelib = subprocess.check_output(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        text=True,
    ).strip()
    # Adding paths does not execute their .pth files, so another editable TMol
    # installation cannot register an import hook inside this environment.
    (Path(purelib) / "benchmark_dependencies.pth").write_text(
        "\n".join(site.getsitepackages()) + "\n"
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(
        PYTHONNOUSERSITE="1",
        TMOL_DISABLE_WHEEL_FETCH="1",
        TORCH_EXTENSIONS_DIR=str(directory / "torch-extensions"),
        PATH=str(python.parent) + os.pathsep + env.get("PATH", ""),
        VIRTUAL_ENV=str(directory),
    )
    return python, env


def compare(root, pytest_args, revisions, outdir):
    manifest = {"python": sys.version, "pytest_args": pytest_args, "runs": []}
    manifest_path = outdir / "manifest.json"
    try:
        with tempfile.TemporaryDirectory(prefix="tmol-benchmark-build-") as scratch:
            scratch = Path(scratch).resolve()
            # Snapshot every revision before builds/tests can create any files.
            for index, revision in enumerate(revisions):
                source = scratch / f"source-{index}"
                record = snapshot(root, revision, source)
                record["source"] = str(source)
                manifest["runs"].append(record)
            results = []
            for index, record in enumerate(manifest["runs"]):
                source = Path(record["source"])
                python, env = environment(scratch / f"env-{index}")
                log = outdir / f"{index}.log"
                result = outdir / f"{index}.json"
                record.update(result=result.name, status="building")
                record["environment"] = {
                    name: env[name]
                    for name in (
                        "CC",
                        "CXX",
                        "CMAKE_ARGS",
                        "CMAKE_CUDA_ARCHITECTURES",
                        "CUDA_VISIBLE_DEVICES",
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "TMOL_USE_JIT",
                        "TORCH_CUDA_ARCH_LIST",
                        "TORCH_EXTENSIONS_DIR",
                    )
                    if name in env
                }
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                run(
                    [
                        str(python),
                        "-m",
                        "pip",
                        "install",
                        "--ignore-installed",
                        "--no-deps",
                        "--no-build-isolation",
                        "-e",
                        str(source),
                        "-Ccmake.define.TMOL_BUILD_TESTS=ON",
                    ],
                    cwd=source,
                    env=env,
                    log=log,
                )
                run(
                    [
                        str(python),
                        "-c",
                        "import pathlib, torch, tmol; "
                        f"assert pathlib.Path(tmol.__file__).resolve().is_relative_to(pathlib.Path({str(source)!r})); "
                        "print('tmol:', tmol.__file__); "
                        "print('torch:', torch.__version__, 'CUDA:', torch.version.cuda); "
                        "print('threads:', torch.get_num_threads()); "
                        "print(torch.__config__.show())",
                    ],
                    cwd=source,
                    env=env,
                    log=log,
                )
                run([str(python), "-m", "pip", "freeze"], cwd=source, env=env, log=log)
                record["status"] = "benchmarking"
                run(
                    [
                        str(python),
                        "-m",
                        "pytest",
                        "--benchmark-enable",
                        "--benchmark-only",
                        f"--benchmark-json={result}",
                        *pytest_args,
                    ],
                    cwd=source,
                    env=env,
                    log=log,
                )
                if not result.is_file() or not json.loads(result.read_text()).get(
                    "benchmarks"
                ):
                    raise ValueError(f"No benchmark samples were recorded in {result}")
                record["status"] = "complete"
                results.append(str(result))
            run(
                [
                    str(python),
                    "-c",
                    "from pytest_benchmark.cli import main; main()",
                    "compare",
                    *results,
                    "--columns=ops,mean,iqr",
                    "--sort=name",
                ],
                cwd=outdir,
                env=env,
                log=outdir / "comparison.txt",
            )
        print((outdir / "comparison.txt").read_text())
    except (Exception, KeyboardInterrupt) as exc:
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main(argv=None):
    options, pytest_args, revisions = parse_args(sys.argv[1:] if argv is None else argv)
    root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").decode().strip())
    if options.benchmark_outdir:
        outdir = options.benchmark_outdir.resolve()
        outdir.mkdir(parents=True, exist_ok=False)
    else:
        outdir = Path(tempfile.mkdtemp(prefix="tmol-benchmark-results-")).resolve()
    print(f"Benchmark results (retained on failure): {outdir}", flush=True)
    try:
        compare(root, pytest_args, revisions, outdir)
    except (
        subprocess.CalledProcessError,
        OSError,
        ValueError,
        KeyboardInterrupt,
    ) as exc:
        print(f"Benchmark comparison failed: {exc}\nSee {outdir}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
