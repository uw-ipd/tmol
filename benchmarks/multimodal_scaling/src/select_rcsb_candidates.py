"""Freeze a size-stratified RCSB candidate pool for the expanded benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import urllib.request
from pathlib import Path

from common import ROOT

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"


def terminal(attribute: str, operator: str, value):
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {"attribute": attribute, "operator": operator, "value": value},
    }


def group(operator: str, *nodes):
    return {"type": "group", "logical_operator": operator, "nodes": list(nodes)}


def modality_query(modality: str, rows: int) -> dict:
    common = [
        terminal("rcsb_entry_info.resolution_combined", "less_or_equal", 3.0),
        terminal("exptl.method", "exact_match", "X-RAY DIFFRACTION"),
        terminal("rcsb_entry_info.polymer_entity_count_protein", "greater_or_equal", 1),
    ]
    no_na = [
        terminal("rcsb_entry_info.polymer_entity_count_DNA", "equals", 0),
        terminal("rcsb_entry_info.polymer_entity_count_RNA", "equals", 0),
    ]
    if modality == "protein":
        nodes = [
            *common,
            *no_na,
            terminal("rcsb_entry_info.nonpolymer_entity_count", "equals", 0),
        ]
    elif modality == "protein_ligand":
        nodes = [
            *common,
            *no_na,
            terminal("rcsb_entry_info.nonpolymer_entity_count", "greater_or_equal", 1),
        ]
    elif modality == "protein_nucleic":
        nodes = [
            *common,
            group(
                "or",
                terminal(
                    "rcsb_entry_info.polymer_entity_count_DNA", "greater_or_equal", 1
                ),
                terminal(
                    "rcsb_entry_info.polymer_entity_count_RNA", "greater_or_equal", 1
                ),
            ),
        ]
    else:
        raise ValueError(modality)
    return {
        "query": group("and", *nodes),
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": rows},
            "results_verbosity": "compact",
        },
    }


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "tmol-multimodal-benchmark/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def search(modality: str, rows: int) -> list[str]:
    result = post_json(SEARCH_URL, modality_query(modality, rows))
    return [
        item if isinstance(item, str) else item["identifier"]
        for item in result["result_set"]
    ]


def metadata(ids: list[str]) -> list[dict]:
    query = """
    query($ids:[String!]!) {
      entries(entry_ids:$ids) {
        rcsb_id
        rcsb_entry_info {
          deposited_polymer_monomer_count
          polymer_entity_count_protein
          polymer_entity_count_DNA
          polymer_entity_count_RNA
          nonpolymer_entity_count
        }
      }
    }
    """
    rows = []
    for start in range(0, len(ids), 250):
        result = post_json(
            GRAPHQL_URL,
            {"query": query, "variables": {"ids": ids[start : start + 250]}},
        )
        for entry in result["data"]["entries"]:
            info = entry["rcsb_entry_info"]
            rows.append(
                {
                    "dataset_id": entry["rcsb_id"].lower(),
                    "reported_polymer_residues": info[
                        "deposited_polymer_monomer_count"
                    ],
                    "protein_entities": info["polymer_entity_count_protein"],
                    "dna_entities": info["polymer_entity_count_DNA"],
                    "rna_entities": info["polymer_entity_count_RNA"],
                    "nonpolymer_entities": info["nonpolymer_entity_count"],
                }
            )
    return rows


def stable_tiebreaker(modality: str, dataset_id: str) -> str:
    return hashlib.sha256(f"20260907:{modality}:{dataset_id}".encode()).hexdigest()


def size_stratified(rows: list[dict], modality: str, count: int) -> list[dict]:
    eligible = [
        row for row in rows if 30 <= (row["reported_polymer_residues"] or 0) <= 1500
    ]
    # Pick evenly across log(size), using a stable hash to avoid accession-order
    # bias among similarly sized structures.
    eligible.sort(
        key=lambda row: (
            math.log(row["reported_polymer_residues"]),
            stable_tiebreaker(modality, row["dataset_id"]),
        )
    )
    if len(eligible) <= count:
        return eligible
    selected = []
    for index in range(count):
        position = round(index * (len(eligible) - 1) / (count - 1))
        selected.append(eligible[position])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-rows", type=int, default=5000)
    parser.add_argument("--candidates-per-modality", type=int, default=384)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "metadata/rcsb_candidates.json"
    )
    args = parser.parse_args()
    frozen = {
        "schema_version": 1,
        "selection_date": "2026-09-07",
        "selection_seed": 20260907,
        "selection": {
            "experimental_method": "X-RAY DIFFRACTION",
            "maximum_resolution_angstrom": 3.0,
            "minimum_polymer_residues": 30,
            "maximum_polymer_residues": 1500,
            "candidates_per_modality": args.candidates_per_modality,
        },
        "modalities": {},
    }
    for modality in ("protein", "protein_ligand", "protein_nucleic"):
        ids = search(modality, args.search_rows)
        rows = size_stratified(metadata(ids), modality, args.candidates_per_modality)
        frozen["modalities"][modality] = rows
        print(modality, len(ids), "queried;", len(rows), "frozen")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
