"""Compare whole/split ligand gradients with identical coordinates and parameters."""

import argparse
import json
from pathlib import Path

import torch

from tmol.score import beta2016_score_function
from tmol.tests.ligand.test_fragmented_ligand_scoring import (
    _load_fixture,
    _annotate_at_bridge,
    _build,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    structure, params_path, preparation = _load_fixture()
    annotated = _annotate_at_bridge(structure, preparation)
    poses, contexts, mappings = zip(
        *[
            _build(array, params_path, device, fragmented=fragmented)
            for array, fragmented in ((structure, False), (annotated, True))
        ]
    )
    scorers = [
        beta2016_score_function(device, param_db=context.parameter_database)
        for context in contexts
    ]
    modules = [
        score.render_whole_pose_scoring_module(pose)
        for score, pose in zip(scorers, poses)
    ]
    names = [st.name for st in scorers[0].all_score_types()]
    atom_indices = []
    for pose in poses:
        index = {}
        for bi, ti in enumerate(pose.block_type_ind[0].tolist()):
            bt = pose.packed_block_types.active_block_types[ti]
            if bt.base_name != "LG1":
                continue
            start = int(pose.block_coord_offset[0, bi])
            index.update((atom.name, start + i) for i, atom in enumerate(bt.atoms))
        atom_indices.append(index)
    assert atom_indices[0].keys() == atom_indices[1].keys()
    atom_names = sorted(atom_indices[0])
    aligned = [
        torch.tensor([indices[n] for n in atom_names], device=device)
        for indices in atom_indices
    ]
    torch.testing.assert_close(
        poses[0].coords[0, aligned[0]], poses[1].coords[0, aligned[1]], rtol=0, atol=0
    )
    rows, gradients = [], {}
    for dtype in (torch.float32, torch.float64):
        coords = [
            pose.coords.to(dtype).detach().clone().requires_grad_(True)
            for pose in poses
        ]
        energies = [
            module(xyz, sum_terms=False, apply_weights=False)
            for module, xyz in zip(modules, coords)
        ]
        for term, name in enumerate(names):
            grads = [
                torch.autograd.grad(e[term].sum(), xyz, retain_graph=True)[0][
                    0, indices
                ]
                for e, xyz, indices in zip(energies, coords, aligned)
            ]
            delta = (grads[0] - grads[1]).abs()
            flat = int(delta.argmax())
            rows.append(
                dict(
                    dtype=str(dtype),
                    term=name,
                    max_gradient_difference=float(delta.max()),
                    atom=atom_names[flat // 3],
                    component=flat % 3,
                    values=[float(g.flatten()[flat]) for g in grads],
                    max_gradient_magnitude=max(float(g.abs().max()) for g in grads),
                )
            )
            gradients[str(dtype), name] = [g.detach() for g in grads]
        print(
            str(dtype),
            sorted(
                [r for r in rows if r["dtype"] == str(dtype)],
                key=lambda r: r["max_gradient_difference"],
                reverse=True,
            )[:5],
            flush=True,
        )
    for row in rows:
        if row["dtype"] == "torch.float32":
            row["max_float32_reference_error"] = [
                float((a.double() - b).abs().max())
                for a, b in zip(
                    gradients["torch.float32", row["term"]],
                    gradients["torch.float64", row["term"]],
                )
            ]
    args.output.write_text(
        json.dumps(
            dict(device=str(device), torch=torch.__version__, rows=rows), indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
