"""Safely expose the positive-pi lookup boundary with a guarded extra row.

Run from the repository root with a positional cpu or cuda:0 device argument.
The extra row prevents an out-of-bounds read; it does not make bin 36 valid in
the real 36-bin database. Successful execution is not a passed periodicity gate.
"""

import torch
import math
import json
import sys
from tmol.database import ParameterDatabase
from tmol.score.dunbrack import DunbrackParamResolver
from tmol.tests.pack.rotamer.dunbrack.test_dunbrack_chi_sampler import (
    get_compiled,
    _table_indices,
)

device = torch.device(sys.argv[1])
db = ParameterDatabase.get_default()
resolver = DunbrackParamResolver.from_database(db.scoring.dun, device)
d = resolver.sampling_db
(phe,) = _table_indices(resolver, ("PHE",), device)
# Deliberately guard the forbidden row/column so the probe cannot read OOB.
# Valid bins all select row0; forbidden bin36 selects row1 (both valid PHErows).
sorted_lookup = torch.zeros(
    (37, 37, d.sorted_rotamer_2_rotamer.shape[2]), dtype=torch.int64, device=device
)
sorted_lookup[36, :, :] = 1
sorted_lookup[:, 36, :] = 1
results = []
for phi, psi in ((-math.pi, 0), (math.pi, 0), (0, -math.pi), (0, math.pi)):
    result = torch.zeros(1, dtype=torch.float32, device=device)
    get_compiled().interpolate_probabilities_for_possible_rotamers(
        d.rotameric_prob_tables,
        d.rotprob_table_sizes,
        d.rotprob_table_strides,
        d.rotameric_bb_start,
        d.rotameric_bb_step,
        d.rotameric_bb_periodicity,
        d.rotameric_bb_source_start,
        d.rotameric_bb_is_mirrored,
        d.n_rotamers_for_tableset_offsets,
        sorted_lookup,
        torch.tensor([[0, phe]], dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.tensor([phi, psi], dtype=torch.float32, device=device),
        result,
    )
    results.append(dict(phi=phi, psi=psi, probability=float(result[0])))
print(
    json.dumps(
        dict(
            device=str(device),
            results=results,
            periodic_lookup_agrees=(
                results[0]["probability"] == results[1]["probability"]
                and results[2]["probability"] == results[3]["probability"]
            ),
            limits="Guarded oversized 37x37 lookup with valid sentinel rotamer indices makes forbidden bin 36 observable without executing an out-of-bounds access. Real source table has 36x36 rows. Endpoint probabilities must agree on each axis.",
        ),
        indent=2,
    )
)
