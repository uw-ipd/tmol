"""Fresh FastRelax measurements using the verified current modeling revision."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from repro_paths import ROOT as source, pyrosetta_path
import benchmark_tmol as bt
import benchmark_pyrosetta as bp
from common import SPEC, machine_metadata, sha256

parser = argparse.ArgumentParser()
parser.add_argument('--engine', choices=['tmol', 'pyrosetta'], required=True)
parser.add_argument('--dataset', required=True)
parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
parser.add_argument('--batch', type=int, default=1)
parser.add_argument('--threads', type=int, default=1)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
row = bt.select(args.dataset)
result = dict(engine=args.engine, protocol='fastrelax', dataset_id=args.dataset,
              modality=row['modality'], device=args.device, batch_size=args.batch,
              atoms=int(row['atoms']), residues=int(row['residues']),
              input_path=row['structure_path'], input_sha256=row['structure_sha256'],
              cpu_threads_requested=args.threads, seed=SPEC['seed'],
              num_repeats=SPEC['fastrelax_repeats'], timed_samples=1,
              engine_version='0.1.58' if args.engine=='tmol' else '2024.39',
              engine_commit='ef1dc2ad4c03cbec508838a1f9e70ddb7726d690' if args.engine=='tmol' else '59628fbc5bc09f1221e1642f1f8d157ce49b1410',
              modeling_master_commit='f92d40d12a8670dd2654db32f37ca885714d2f4a' if args.engine=='tmol' else None,
              machine=machine_metadata(), cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              source_harness=str(source/'scripts'), status='running',
              started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.with_suffix('.running.json').write_text(json.dumps(result, indent=2)+'\n')
started = time.perf_counter()
try:
    assert sha256(Path(row['structure_path'])) == row['structure_sha256']
    if args.engine == 'tmol':
        torch, tmol, *_ = bt.imports()
        # The CUDA wheel carries its ABI as a PEP 440 local-version suffix.
        assert tmol.__version__.split('+', 1)[0] == '0.1.58', tmol.__version__
        if args.device == 'cuda':
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        result.update(bt.benchmark_fastrelax(row, args.device, args.batch, cpu_threads=args.threads))
    else:
        assert args.device == 'cpu' and args.batch == 1 and args.threads == 1
        pyrosetta = bp.initialize(row, pyrosetta_path())
        pose = pyrosetta.pose_from_file(row['structure_path'])
        sfxn_name = 'beta_genpot_cart' if row['modality']=='protein_ligand' else 'beta_nov16_cart'
        sfxn = pyrosetta.create_score_function(sfxn_name)
        result.update(bp.benchmark_fastrelax(pyrosetta, pose, sfxn))
        result['score_function'] = sfxn_name
        result['pyrosetta_version_string'] = pyrosetta.version().splitlines()[-1]
        result['actual_rosetta_threads'] = pyrosetta.rosetta.basic.thread_manager.RosettaThreadManager.get_instance().total_threads()
    assert len(result['seconds_per_structure_samples']) == 1
    assert result['seconds_per_structure_samples'][0] > 0
    assert math.isfinite(result['validation_score_mean'])
    assert math.isfinite(result['initial_score_mean'])
    result['status'] = 'ok'
except Exception as exc:
    result.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
result['elapsed_seconds'] = time.perf_counter()-started
result['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
temporary = args.output.with_suffix('.tmp')
temporary.write_text(json.dumps(result, indent=2)+'\n')
temporary.replace(args.output)
args.output.with_suffix('.running.json').unlink()
print(json.dumps({k:result.get(k) for k in ['engine','dataset_id','device','batch_size','status','error','elapsed_seconds']}), flush=True)
if result['status'] != 'ok':
    raise SystemExit(2)
