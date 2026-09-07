from __future__ import annotations

import heapq
import math
from pathlib import Path

from common import ROOT, read_manifest


def coordinates(path: str, residue_name: str) -> dict[str, tuple[float, float, float]]:
    result = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("HETATM") and line[17:20].strip() == residue_name:
            result[line[12:16].strip()] = tuple(
                float(line[start : start + 8]) for start in (30, 38, 46)
            )
    return result


def neighbor_radius(preparation, structure_path: str) -> float:
    restype = preparation.residue_type
    xyz = coordinates(structure_path, restype.name3)
    graph: dict[str, list[tuple[str, float]]] = {name: [] for name in xyz}
    for atom1, atom2, *_ in restype.bonds:
        distance = math.dist(xyz[atom1], xyz[atom2])
        graph[atom1].append((atom2, distance))
        graph[atom2].append((atom1, distance))

    root = restype.default_jump_connection_atom
    shortest = {root: 0.0}
    queue = [(0.0, root)]
    while queue:
        distance, atom = heapq.heappop(queue)
        if distance != shortest[atom]:
            continue
        for neighbor, bond_length in graph[atom]:
            candidate = distance + bond_length
            if candidate < shortest.get(neighbor, math.inf):
                shortest[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    if shortest.keys() != graph.keys():
        raise ValueError(f"disconnected ligand graph in {structure_path}")
    return max(shortest.values())


def replace_neighbor_radius(path: Path, radius: float) -> None:
    lines = path.read_text().splitlines()
    matches = [
        index for index, line in enumerate(lines) if line.startswith("NBR_RADIUS")
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one NBR_RADIUS record in {path}")
    lines[matches[0]] = f"NBR_RADIUS {radius:.5f}"
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    from tmol.ligand import load_params_file, write_params_file

    output_dir = ROOT / "data/ligand_params"
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in read_manifest():
        if row["modality"] != "protein_ligand" or row["status"] != "ok":
            continue
        preparations = load_params_file(row["ligand_tmol_params"])
        if len(preparations) != 1:
            raise ValueError(f"Expected one ligand in {row['ligand_tmol_params']}")
        output = Path(row["ligand_rosetta_params"])
        write_params_file(preparations[0], output, format="rosetta")
        replace_neighbor_radius(
            output, neighbor_radius(preparations[0], row["structure_path"])
        )
        print(output)


if __name__ == "__main__":
    main()
