"""Move a PoseStack and its PackedBlockTypes to another device.

PoseStack has no `.to()`, and building one directly on MPS is not yet
possible: pose construction runs its own operators (HIS tautomer resolution,
leaf-atom generation) and those terms have no Metal backend. For benchmarking
the elec term, the pose is built on CPU and relocated here.

This walks attrs fields recursively and moves every tensor it finds, which is
enough for the structures elec reads. It is a scratch helper, not a
replacement for a real PoseStack.to().
"""

import attr
import torch


def to_device(obj, device, _memo=None):
    if _memo is None:
        _memo = {}
    key = id(obj)
    if key in _memo:
        return _memo[key]

    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device, _memo) for v in obj)
    if isinstance(obj, dict):
        return {k: to_device(v, device, _memo) for k, v in obj.items()}
    # Residue types are shared, not relocated. The energy terms annotate them
    # in place and memoize the annotation by content, so an evolved copy would
    # hit the cache from the original device's pass and silently skip the
    # setattr. Their tensors are parameter tables that do not need to move:
    # PackedBlockTypes holds the device-resident copies.
    if type(obj).__name__ in ("RefinedResidueType", "ResidueType"):
        return obj

    if attr.has(type(obj)):
        changes = {}
        for field in attr.fields(type(obj)):
            value = getattr(obj, field.name)
            if field.name == "device":
                changes[field.name] = device
            else:
                moved = to_device(value, device, _memo)
                if moved is not value:
                    changes[field.name] = moved
        moved_obj = attr.evolve(obj, **changes) if changes else obj
        _memo[key] = moved_obj
        return moved_obj
    return obj
