"""Paired cold HBond table assembly, exact tensor checks and setup allocations."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import attr
import torch

from tmol.database import ParameterDatabase
from tmol.score.hbond._params import HBondParamResolver, CompactedHBondDatabase
from tmol.utility.weak_identity_cache import WeakIdentityLRU
from profile_parameter_coverage import measure

BASELINE = "ee7d4512f5de3fd5da839d5ccd2b2bcdcc3c5bb8"


def tensors(value, prefix=""):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif attr.has(type(value)):
        for field in attr.fields(type(value)):
            yield from tensors(getattr(value, field.name), prefix + "." + field.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:tmol/score/hbond/_params.py"], text=True
    )
    namespace = {
        "__name__": "tmol.score.hbond._params",
        "__package__": "tmol.score.hbond",
    }
    exec(compile(source, "previous_hbond_params.py", "exec"), namespace)
    database = ParameterDatabase.get_default()
    classes = {
        "before": (
            namespace["HBondParamResolver"],
            namespace["CompactedHBondDatabase"],
        ),
        "after": (HBondParamResolver, CompactedHBondDatabase),
    }
    hb = database.scoring.hbond
    models = {
        "default": hb,
        "reversed": attr.evolve(
            hb,
            donor_type_params=tuple(reversed(hb.donor_type_params)),
            acceptor_type_params=tuple(reversed(hb.acceptor_type_params)),
            pair_parameters=tuple(reversed(hb.pair_parameters)),
            polynomial_parameters=tuple(reversed(hb.polynomial_parameters)),
        ),
    }
    checks = {}
    for model_name, model in models.items():
        results = {}
        for label, (resolver, compact) in classes.items():
            resolver._from_db_cache = WeakIdentityLRU()
            results[label] = (
                resolver.from_database(database.chemical, model, device),
                compact._from_database(database.chemical, model, device),
            )
        fields = []
        for i, kind in enumerate(("resolver", "compact")):
            before = dict(tensors(results["before"][i], kind))
            after = dict(tensors(results["after"][i], kind))
            assert before.keys() == after.keys()
            for field, a in before.items():
                b = after[field]
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                assert a.stride() == b.stride(), (field, a.stride(), b.stride())
                fields.append(field)
        for index in ("donor_type_index", "acceptor_type_index"):
            assert getattr(results["before"][0], index).equals(
                getattr(results["after"][0], index)
            )
        checks[model_name] = dict(
            exact_tensor_fields=fields,
            exact_strides=True,
            tensor_bytes=sum(
                v.numel() * v.element_size()
                for result in results["after"]
                for _, v in tensors(result)
            ),
        )
    measurements = {}
    for i, kind in enumerate(("resolver", "compact")):

        def build(label):
            resolver, compact = classes[label]
            # The compact builder must also build its resolver in cold rounds.
            resolver._from_db_cache = WeakIdentityLRU()
            return (resolver, compact)[i]._from_database(database.chemical, hb, device)

        measurements[kind] = measure(
            {label: lambda label=label: build(label) for label in classes},
            device,
            calls=3,
        )
    output = dict(
        baseline=BASELINE,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        device=str(device),
        torch=torch.__version__,
        checks=checks,
        measurements=measurements,
        scope="Nine alternating warm process rounds of three cold table constructions. Cache lookup hits are excluded; compact includes resolver construction. Allocation tracing measures Python heap only, excluding native/RSS/GPU memory. Tensor values, strides and logical bytes are checked separately. No fitting or scoring speedup claim.",
    )
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
