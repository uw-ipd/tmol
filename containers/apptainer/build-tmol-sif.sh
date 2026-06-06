#!/usr/bin/env bash
# Build tmol.sif from containers/apptainer/tmol-dev.def on digs.
#
# Requires apptainer and permission to use --fakeroot (or run as root).
# Run from anywhere; defaults to writing tmol.sif in the repo root.
#
# Examples:
#   containers/apptainer/build-tmol-sif.sh
#   containers/apptainer/build-tmol-sif.sh /home/bench/git_ci_apptainer/tmol.sif
#   containers/apptainer/build-tmol-sif.sh --deploy-ci

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEF="${REPO_ROOT}/containers/apptainer/tmol-dev.def"
CI_SIF="/home/bench/git_ci_apptainer/tmol.sif"

DEPLOY_CI=0
OUTPUT=""

for arg in "$@"; do
  case "${arg}" in
    --deploy-ci)
      DEPLOY_CI=1
      OUTPUT="${CI_SIF}"
      ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      if [ -n "${OUTPUT}" ] && [ "${OUTPUT}" != "${CI_SIF}" ]; then
        echo "ERROR: unexpected extra argument: ${arg}" >&2
        exit 1
      fi
      OUTPUT="${arg}"
      ;;
  esac
done

if [ -z "${OUTPUT}" ]; then
  OUTPUT="${REPO_ROOT}/tmol.sif"
fi

if [ ! -f "${DEF}" ]; then
  echo "ERROR: definition file not found: ${DEF}" >&2
  exit 1
fi

BASE_SIF="/net/software/containers/versions/modelhub/pytorch_25.11-py3.sif"
if [ ! -f "${BASE_SIF}" ]; then
  echo "ERROR: NGC base image not found: ${BASE_SIF}" >&2
  echo "Update containers/apptainer/tmol-dev.def Bootstrap From: path for this host." >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT}")"
TMP_OUT="${OUTPUT}.tmp.$$"
cleanup() {
  rm -f "${TMP_OUT}"
}
trap cleanup EXIT

echo "=== Building tmol Apptainer image ==="
echo "  def:    ${DEF}"
echo "  base:   ${BASE_SIF}"
echo "  output: ${OUTPUT}"
echo

cd "${REPO_ROOT}"
# Host bind overrides (e.g. CI APPTAINER_BIND) must not leak into image builds.
unset APPTAINER_BIND SINGULARITY_BIND
apptainer build --fakeroot "${TMP_OUT}" "${DEF}"

echo
echo "=== Verifying X11 libs and Open Babel ==="
unset APPTAINER_BIND SINGULARITY_BIND
apptainer exec "${TMP_OUT}" bash -lc '
  set -e
  for lib in libXrender.so.1 libX11.so.6 libXext.so.6; do
    ldconfig -p | grep -q "${lib}" || { echo "missing ${lib}"; exit 1; }
  done
  python -c "from openbabel import pybel; assert len(pybel.informats) > 100"
  echo "Open Babel OK ($(python -c "from openbabel import pybel; print(len(pybel.informats))") input formats)"
'

mv "${TMP_OUT}" "${OUTPUT}"
trap - EXIT

echo
echo "=== Done ==="
echo "  ${OUTPUT}"
if [ "${DEPLOY_CI}" -eq 1 ] || [ "${OUTPUT}" = "${CI_SIF}" ]; then
  echo "  CI runner will use this image via SIF=${CI_SIF} in .github/workflows/ci.yml"
else
  echo "  To install for CI:"
  echo "    ${SCRIPT_DIR}/build-tmol-sif.sh --deploy-ci"
  echo "  or:"
  echo "    cp ${OUTPUT} ${CI_SIF}"
fi
