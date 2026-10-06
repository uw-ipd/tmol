"""Exercise release conditions locally and on GitHub's actual job scheduler.

The generated reusable workflow preserves the production dependency graph and
conditions, replacing build/test/upload steps with harmless shell commands.
Regenerate it with this file's --write-fixture option after changing the graph.
"""

import ast
from pathlib import Path
import re
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/publish.yml"
FIXTURE = ROOT / ".github/workflows/_test_publish_scheduling.yml"
PUBLISH = ("upload", "publish_indexes", "publish_pypi", "verify_public")
JOBS = yaml.safe_load(WORKFLOW.read_text())["jobs"]


def dependencies(job):
    needs = JOBS[job].get("needs", [])
    return [needs] if isinstance(needs, str) else needs


def allows(
    job, results, *, trigger="workflow_dispatch", dry_run=False, cancelled=False
):
    """Evaluate the boolean subset used by these conditions, including defaults.

    GitHub adds success() when no status function is present. The regression
    case supplies a skipped build ancestor even when every direct gate passed.
    The live fixture below independently checks that scheduler behavior.
    """
    expression = JOBS[job].get("if", "success()")
    expression = expression.removeprefix("${{").removesuffix("}}").strip()
    if not re.search(r"\b(?:success|failure|cancelled|always)\(", expression):
        expression = f"success() && ({expression})"
    expression = re.sub(
        r"needs\.([a-z_]+)\.result",
        lambda match: repr(results[match[1]]),
        expression,
    )
    expression = expression.replace("github.event_name", repr(trigger))
    expression = expression.replace("inputs.validate-only", repr(dry_run))
    expression = expression.replace("cancelled()", repr(cancelled))
    expression = expression.replace(
        "success()", repr(all(value == "success" for value in results.values()))
    )
    expression = expression.replace("always()", "True")
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", " not ", expression)
    tree = ast.parse(f"({expression.strip()})", mode="eval")
    allowed = (
        ast.Expression,
        ast.BoolOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Constant,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Eq,
        ast.NotEq,
    )
    assert all(isinstance(node, allowed) for node in ast.walk(tree))
    return eval(compile(tree, "<release condition>", "eval"), {"__builtins__": {}})


def successful_results():
    return dict.fromkeys(JOBS, "success")


@pytest.mark.parametrize("job", PUBLISH)
@pytest.mark.parametrize(
    "trigger,recovery", [("push", False), ("workflow_dispatch", True)]
)
def test_publication_runs_after_all_required_gates_pass(job, trigger, recovery):
    results = successful_results()
    if recovery:
        for build in ("build_wheels", "build_cpu_wheel", "build_macos_cpu_wheel"):
            results[build] = "skipped"
    assert allows(job, results, trigger=trigger)


@pytest.mark.parametrize(
    "job,gate", [(job, gate) for job in PUBLISH for gate in dependencies(job)]
)
@pytest.mark.parametrize("status", ["failure", "cancelled", "skipped"])
def test_missing_required_gate_prevents_publication(job, gate, status):
    results = successful_results()
    results[gate] = status
    assert not allows(job, results)


@pytest.mark.parametrize("job", PUBLISH)
def test_cancellation_prevents_publication(job):
    assert not allows(job, successful_results(), cancelled=True)


def test_validation_only_skips_the_entire_publication_chain():
    results = successful_results()
    for job in PUBLISH:
        assert not allows(job, results, dry_run=True)
        results[job] = "skipped"


def scheduling_fixture():
    jobs = {}
    for name, original in JOBS.items():
        job = {key: original[key] for key in ("needs", "if") if key in original}
        if "if" in job:
            job["if"] = job["if"].replace("github.event_name", "inputs.trigger")
        job.update({"runs-on": "ubuntu-latest", "steps": [{"run": "true"}]})
        if name == "verify_release_version":
            job["outputs"] = {
                "candidate_run_id": "${{ inputs.recovery && 'original' || '' }}"
            }
        if name == "test_gpu_runtime":
            job["if"] = job["if"].replace("}}", "&& !inputs.skip-gate }}")
        jobs[name] = job
    jobs["assert_scheduling"] = {
        "needs": list(jobs),
        "if": "${{ always() }}",
        "runs-on": "ubuntu-latest",
        "env": {
            "RESULTS": "${{ toJSON(needs) }}",
            "NO_PUBLISH": "${{ inputs.validate-only || inputs.skip-gate }}",
        },
        "steps": [
            {
                "name": "Assert every publication stage ran or was blocked",
                "run": (
                    "python - <<'PY'\n"
                    "import json, os\n"
                    "results = json.loads(os.environ['RESULTS'])\n"
                    "expected = 'skipped' if os.environ['NO_PUBLISH'] == 'true' else 'success'\n"
                    f"for job in {PUBLISH!r}:\n"
                    "    actual = results[job]['result']\n"
                    "    assert actual == expected, (job, actual, expected)\n"
                    "print(json.dumps(results, indent=2))\n"
                    "PY\n"
                ),
            }
        ],
    }
    inputs = {
        "trigger": {"type": "string", "required": True},
        **{
            name: {"type": "boolean", "required": True}
            for name in ("recovery", "validate-only", "skip-gate")
        },
    }
    return {
        "name": "Release scheduling fixture (no uploads)",
        "on": {"workflow_call": {"inputs": inputs}},
        "permissions": {"contents": "read"},
        "jobs": jobs,
    }


def test_live_fixture_matches_production_conditions_and_dependency_graph():
    assert yaml.safe_load(FIXTURE.read_text()) == scheduling_fixture()


def fixture_yaml():
    class Dumper(yaml.SafeDumper):
        pass

    def represent_string(dumper, value):
        return dumper.represent_scalar(
            "tag:yaml.org,2002:str", value, style="|" if "\n" in value else None
        )

    Dumper.add_representer(str, represent_string)
    return yaml.dump(scheduling_fixture(), Dumper=Dumper, sort_keys=False, width=88)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write-fixture"]:
        raise SystemExit("Use pytest, or --write-fixture to regenerate the live probe")
    FIXTURE.write_text(
        "# Generated by scripts/test_publish_scheduling.py --write-fixture.\n"
        "# No secrets or publication commands; only the real scheduling graph.\n"
        + fixture_yaml()
    )
