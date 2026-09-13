import json
import time
from pathlib import Path
import torch
from tmol.io import atom_array_from_cif
from tmol.io._pose_stack_from_biotite import build_context_from_biotite
from tmol.pack.rotamer import NaChiRotamerSampler

root = Path("tmol/tests/data")
rows = []
for rel in (
    "cif/1UBQ.cif",
    "covalent_fixtures/nglycan_tree_1ax2.cif",
    "covalent_fixtures/oglycan_sia_1g1s.cif",
    "ncaa_fixtures/na_rna_psu_1bzt.cif",
):
    context = build_context_from_biotite(
        atom_array_from_cif(root / rel),
        torch.device("cpu"),
        prepare_ligands=True,
        ligand_seed=20260909,
    )
    pbt = context.packed_block_types
    sampler = NaChiRotamerSampler.from_database(
        context.parameter_database, torch.device("cpu")
    )
    start = time.perf_counter()
    sampler.annotate_packed_block_types(pbt)
    elapsed = time.perf_counter() - start
    cache = pbt.na_chi_sampler_cache
    row = dict(
        fixture=rel,
        seconds=elapsed,
        shape=list(cache["proton"].shape),
        bytes=cache["proton"].numel() * cache["proton"].element_size(),
    )
    rows.append(row)
    print(json.dumps(row), flush=True)
