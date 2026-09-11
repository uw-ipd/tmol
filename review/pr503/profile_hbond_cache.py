"""Reachable tensor storage for many independently fitted HBond databases."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import weakref


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.baseline:
        from check_nonbonded_baseline import BaselineImports, MODULES

        MODULES.add("tmol.score.hbond._params")
        sys.meta_path.insert(0, BaselineImports())

    import attr
    import torch
    from tmol.database import ParameterDatabase
    from tmol.score.hbond._params import HBondParamResolver, CompactedHBondDatabase
    from tmol.utility.weak_identity_cache import WeakIdentityLRU

    classes = (HBondParamResolver, CompactedHBondDatabase)
    for cls in classes:
        if args.baseline:
            cls._from_db_cache.clear()
        else:
            cls._from_db_cache = WeakIdentityLRU()
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    default = ParameterDatabase.get_default()
    models = [
        attr.evolve(
            default.scoring.hbond,
            donor_type_params=tuple(
                attr.evolve(row, weight=row.weight * (0.8 + 0.001 * i))
                for row in default.scoring.hbond.donor_type_params
            ),
        )
        for i in range(args.count)
    ]
    refs = [weakref.ref(model) for model in models]

    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif attr.has(type(value)):
            for field in attr.fields(type(value)):
                yield from tensors(getattr(value, field.name))
        elif isinstance(value, dict):
            for item in value.values():
                yield from tensors(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from tensors(item)

    def cache_storage():
        result = {}
        for cls in classes:
            cache = cls._from_db_cache
            values = cache if args.baseline else cache._entries
            storage = {
                (
                    str(t.device),
                    t.untyped_storage().data_ptr(),
                ): t.untyped_storage().nbytes()
                for t in tensors(values)
            }
            result[cls.__name__] = dict(
                entries=len(cache), tensor_bytes=sum(storage.values())
            )
        return result

    fingerprints = []
    for model in models:
        values = [cls.from_database(default.chemical, model, device) for cls in classes]
        digest = hashlib.sha256()
        for tensor in tensors(values):
            digest.update(tensor.detach().cpu().numpy().tobytes())
        fingerprints.append(digest.hexdigest())
    before = cache_storage()
    del model, models
    gc.collect()
    after = cache_storage()
    assert not any(ref() is not None for ref in refs)
    args.output.write_text(
        json.dumps(
            dict(
                baseline=args.baseline,
                count=args.count,
                device=str(device),
                before_gc=before,
                after_gc=after,
                table_fingerprints=fingerprints,
                limits="Cache-reachable tensor storage only; excludes Python metadata, caller-held outputs and allocator reservations.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
