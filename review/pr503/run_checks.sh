#!/usr/bin/env bash
# Run the reviewed suite and example workflows from this branch.
# Use a Python environment with tmol's project/dev and ligand dependencies.
# CUDA cases are selected automatically when a GPU is visible to PyTorch.
set -uo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
results=${1:-review-results}
mkdir -p -- "$results"
export TMOL_USE_JIT=${TMOL_USE_JIT:-1}
export MAX_JOBS=${MAX_JOBS:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export SPARSE_AUTO_DENSIFY=1
export PYTHONPATH=".:${PYTHONPATH:-}"

python -m pytest -v --tb=short --durations=20 --junitxml="$results/tests.xml" \
    tmol/tests/database/test_d_amino_acids.py \
    tmol/tests/database/scoring/test_rama.py \
    tmol/tests/database/scoring/test_content_hash.py \
    tmol/tests/ligand \
    tmol/tests/test_covalent_components.py \
    tmol/tests/test_cyclic_peptide.py \
    tmol/tests/io/test_cif_completion.py \
    tmol/tests/io/test_atomworks_reader.py \
    tmol/tests/io/test_atomworks_corpus_regressions.py \
    tmol/tests/io/test_filtered_covalent_partners.py \
    tmol/tests/io/test_pose_stack_from_biotite.py \
    tmol/tests/io/details/test_cyclic_search.py \
    tmol/tests/io/details/test_build_missing_leaf_atoms.py \
    tmol/tests/io/details/test_find_disulfides.py \
    tmol/tests/io/details/test_his_taut_resolution.py \
    tmol/tests/io/details/test_atomworks_histidine_parity.py \
    tmol/tests/io/details/test_left_justify_canonical_form.py \
    tmol/tests/io/details/test_select_from_canonical.py \
    tmol/tests/kinematics/test_fold_forest.py \
    tmol/tests/pose/test_util.py \
    tmol/tests/pack/test_conjugated_group_packing.py \
    tmol/tests/pack/rotamer/test_noncanonical_rotamers.py \
    tmol/tests/pack/rotamer/test_group_sampling_regressions.py \
    tmol/tests/pack/rotamer/test_build_rotamers.py \
    tmol/tests/pack/rotamer/test_na_chi_sampler.py \
    tmol/tests/pack/rotamer/dunbrack/test_dunbrack_chi_sampler.py \
    tmol/tests/score/test_score_function.py \
    tmol/tests/score/test_noncanonical_scoring.py \
    tmol/tests/score/test_mirror_image_scoring.py \
    tmol/tests/score/test_torsion_ownership.py \
    tmol/tests/score/backbone_torsion/test_mirrored_tables.py \
    tmol/tests/score/backbone_torsion/test_symmetric_gly.py \
    tmol/tests/score/genbonded/test_genbonded_energy_term.py \
    tmol/tests/score/cartbonded \
    tmol/tests/score/disulfide
status=$?
python review/pr503/run_examples.py --device cpu --output "$results/examples-cpu.json"
if [ "$?" -ne 0 ]; then status=1; fi
if python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    python review/pr503/run_examples.py --device cuda --output "$results/examples-gpu.json"
    if [ "$?" -ne 0 ]; then status=1; fi
fi
exit "$status"
