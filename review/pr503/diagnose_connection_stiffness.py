"""Probe the cartbonded restoring force of declared conjugation bonds."""

import argparse
import json
import math
from pathlib import Path

import networkx as nx
import torch

from tmol.pose._conjugated_groups import find_conjugated_groups
from tmol.score.cartbonded import CartBondedEnergyTerm
from tmol.tests.pack.test_conjugated_group_packing import FIXTURES
from tmol.tests.data import data_path
from tmol.io import atom_array_from_cif, pose_stack_from_biotite
from tmol.tests.pack.rotamer.test_group_constraints import pose_bonds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--minimize-biotin", action="store_true")
    parser.add_argument(
        "--fixtures", nargs="+", choices=list(FIXTURES), default=list(FIXTURES)
    )
    args = parser.parse_args()
    rows = []
    for name in args.fixtures:
        stem = FIXTURES[name]
        array = atom_array_from_cif(data_path("covalent_fixtures", stem + ".cif"))
        pose, context = pose_stack_from_biotite(
            array,
            torch.device(args.device),
            prepare_ligands=True,
            no_optH=True,
            return_context=True,
            ligand_seed=20250828,
        )
        database = context.parameter_database
        records = database.scoring.cartbonded.connection_params
        term = CartBondedEnergyTerm(param_db=database, device=pose.device)
        for block_type in pose.packed_block_types.active_block_types:
            term.setup_block_type(block_type)
        term.setup_packed_block_types(pose.packed_block_types)
        term.setup_poses(pose)
        module = term.render_whole_pose_scoring_module(pose)
        graph = nx.Graph()
        graph.add_edges_from(pose_bonds(pose))
        starts = pose.block_coord_offset[0].tolist()
        types = [
            pose.packed_block_types.active_block_types[t]
            for t in pose.block_type_ind[0].tolist()
        ]
        candidates = [
            (group.blocks[a], ac, group.blocks[b], bc, "conjugation")
            for group in find_conjugated_groups(pose)
            for a, ac, b, bc in group.links
        ]
        # Calibrate the same rigid-component displacement against one ordinary
        # peptide bond. Its expected harmonic stiffness is in the database.
        for ba, bt in enumerate(types):
            for ac, conn in enumerate(bt.connections):
                if conn.name != "up":
                    continue
                bb, bc = pose.inter_residue_connections[0, ba, ac].tolist()
                if bb < 0:
                    continue
                first = starts[ba] + int(bt.ordered_connection_atoms[ac])
                second = starts[bb] + int(types[bb].ordered_connection_atoms[bc])
                cut = graph.copy()
                cut.remove_edge(first, second)
                if not nx.has_path(cut, first, second):
                    candidates.append((ba, ac, bb, bc, "peptide_control"))
                    break
            else:
                continue
            break
        for ba, ac, bb, bc, kind in candidates:
            ia, ib = int(types[ba].ordered_connection_atoms[ac]), int(
                types[bb].ordered_connection_atoms[bc]
            )
            first, second = starts[ba] + ia, starts[bb] + ib
            cut = graph.copy()
            cut.remove_edge(first, second)
            moving = nx.node_connected_component(cut, second)
            if first in moving:
                continue
            coords = pose.coords.double()
            axis = coords[0, second] - coords[0, first]
            distance = float(axis.norm())
            axis /= distance
            orthogonal = torch.zeros_like(axis)
            orthogonal[torch.argmin(axis.abs())] = 1
            orthogonal = torch.linalg.cross(axis, orthogonal)
            orthogonal /= orthogonal.norm()
            probes = {}
            for probe, direction, score_index in (
                ("stretch", axis, 0),
                ("bend", orthogonal, 1),
            ):
                shift = torch.zeros_like(coords)
                shift[0, sorted(moving)] = direction
                samples = []
                for delta in (-0.1, 0.0, 0.1):
                    xyz = (coords + delta * shift).requires_grad_(True)
                    values = module(xyz)
                    selected_score = values[score_index].sum()
                    grad = torch.autograd.grad(selected_score, xyz)[0]
                    samples.append(
                        dict(
                            delta=delta,
                            energy=float(selected_score.detach()),
                            dE_ddelta=float((grad * shift).sum()),
                        )
                    )
                probes[probe] = dict(
                    samples=samples,
                    stiffness=(samples[2]["dE_ddelta"] - samples[0]["dE_ddelta"]) / 0.2,
                )
            row = dict(
                fixture=name,
                parameter_model="default",
                kind=kind,
                blocks=[ba, bb],
                atoms=[types[ba].atoms[ia].name, types[bb].atoms[ib].name],
                distance=distance,
                probes=probes,
            )
            if kind == "conjugation":
                key = (
                    types[ba].name,
                    types[ba].connections[ac].name,
                    types[bb].name,
                    types[bb].connections[bc].name,
                )
                record = next(
                    r
                    for r in records
                    if (r.block_type1, r.connection1, r.block_type2, r.connection2)
                    in (key, (*key[2:], *key[:2]))
                )
                parameter = record.length_parameters[0]
                expected_stiffness = float(
                    torch.tensor(parameter.K, dtype=torch.float32)
                )
                assert math.isclose(
                    probes["stretch"]["stiffness"],
                    expected_stiffness,
                    rel_tol=1e-8,
                    abs_tol=1e-8,
                )
                row["generated_bond"] = dict(x0=parameter.x0, K=expected_stiffness)
            if args.minimize_biotin and name == "biotin" and kind == "conjugation":
                from tmol.optimization import run_cart_min
                from tmol.score import beta2016_score_function

                score = beta2016_score_function(pose.device, param_db=database)
                whole = score.render_whole_pose_scoring_module(pose)
                mask = torch.zeros_like(pose.real_atoms)
                mask[0, sorted(moving)] = True
                row["cartesian_minimization"] = []
                for displacement in (0.0, 0.5, 1.0):
                    initial = pose.clone()
                    initial.coords[0, sorted(moving)] += displacement * axis.float()
                    before = float(whole(initial.coords).detach())
                    final = run_cart_min(
                        initial,
                        score,
                        coord_mask=mask,
                        optimizer_kwargs={"max_iter": 100},
                    )
                    final_distance = float(
                        (final.coords[0, first] - final.coords[0, second]).norm()
                    )
                    assert abs(final_distance - row["generated_bond"]["x0"]) < 0.1
                    assert float(whole(final.coords).detach()) < before
                    torch.testing.assert_close(
                        final.coords[~mask], initial.coords[~mask], rtol=0, atol=0
                    )
                    row["cartesian_minimization"].append(
                        dict(
                            displacement=displacement,
                            initial_energy=before,
                            final_energy=float(whole(final.coords).detach()),
                            initial_distance=float(
                                (
                                    initial.coords[0, first] - initial.coords[0, second]
                                ).norm()
                            ),
                            final_distance=float(
                                (
                                    final.coords[0, first] - final.coords[0, second]
                                ).norm()
                            ),
                        )
                    )
            rows.append(row)
            print(row, flush=True)
        args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
