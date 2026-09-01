#!/usr/bin/env bash
#SBATCH --job-name=tiling_modernization_tests
#SBATCH --partition=grace
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=scripts/logs/tiling_modernization_tests_%j.out
#SBATCH --error=scripts/logs/tiling_modernization_tests_%j.err

set -euo pipefail

START_TIME="$(date +%s)"
START_READABLE="$(date)"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_REL="scripts/shell/all_tasks/sbatch_tiling_modernization_tests.sh"

if [[ -f "${SUBMIT_DIR}/${SCRIPT_REL}" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/sbatch_tiling_modernization_tests.sh" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/../../.." && pwd)"
else
  echo "Could not locate the LFM repository from: ${SUBMIT_DIR}" >&2
  echo "Submit this script from the repository root or its own directory." >&2
  exit 1
fi

REPO_PARENT="$(dirname "${REPO_DIR}")"
CONTAINER_PATH="${CONTAINER_PATH:-/explore/nobackup/projects/lfm/containers/lfm-container}"
APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
APPTAINER_BIND_PATHS="${APPTAINER_BIND_PATHS:-/panfs/ccds02/nobackup:/explore/nobackup}"

cd "${REPO_DIR}"
mkdir -p scripts/logs

echo "Job started at: ${START_READABLE}"
echo "Job ID: ${SLURM_JOB_ID:-not submitted through Slurm}"
echo "Node list: ${SLURM_NODELIST:-unknown}"
echo "Repository: ${REPO_DIR}"
echo "Container: ${CONTAINER_PATH}"
echo "Bind paths: ${APPTAINER_BIND_PATHS}"
echo

APPTAINER_ARGS=(
  exec
  --bind "${APPTAINER_BIND_PATHS}"
  --bind "${REPO_DIR}"
  --pwd "${REPO_PARENT}"
  "${CONTAINER_PATH}"
)

echo "Checking GDAL and the repository lunar WKT..."
"${APPTAINER_BIN}" "${APPTAINER_ARGS[@]}" python - <<'PY'
from osgeo import gdal, osr

from lfm.model.lunar_crs import (
    LUNAR_GEOGRAPHIC_WKT_PATH,
    load_lunar_geographic_wkt,
)

wkt = load_lunar_geographic_wkt()
srs = osr.SpatialReference()
assert srs.ImportFromWkt(wkt) == 0

print("GDAL:", gdal.VersionInfo("--version"))
print("WKT:", LUNAR_GEOGRAPHIC_WKT_PATH)
print("CRS:", srs.GetName())
PY

echo
echo "Running the modern tiling contract tests..."
"${APPTAINER_BIN}" "${APPTAINER_ARGS[@]}" python -m unittest \
  lfm.model.tests.test_tiling_config \
  lfm.model.tests.test_lunar_crs \
  lfm.model.tests.test_vector_index \
  lfm.model.tests.test_vector_index_builder \
  lfm.model.tests.test_tiling_policy \
  lfm.model.tests.test_tiling_results \
  lfm.model.tests.test_tiling_api \
  lfm.model.tests.test_configured_tiler

echo
echo "Running the safe legacy regression subset..."
"${APPTAINER_BIN}" "${APPTAINER_ARGS[@]}" python -m unittest \
  lfm.model.tests.test_TmsTileDef \
  lfm.model.tests.test_TmsZoneDef \
  lfm.model.tests.test_TmsIntersector \
  lfm.model.tests.test_Pipeline.PipelineTestCase.testInit \
  lfm.model.tests.test_Pipeline.PipelineTestCase.testClip \
  lfm.model.tests.test_Pipeline.PipelineTestCase.testQuery \
  lfm.model.tests.test_Pipeline.PipelineTestCase.testCornerAlignment \
  lfm.model.tests.test_Pipeline.PipelineTestCase.testInitWithTargetProductID \
  lfm.model.tests.test_Pipeline.PipelineTestCase.testInitWithoutTargetProductID \
  lfm.model.tests.test_Pipeline.PipelineTestCase.testProductIDExtractionFromFilename \
  lfm.model.tests.test_Pipeline.PipelineTestCase.testNonMatchingProductIDSkipsFiles

echo
echo "Running the filtered one-tile legacy integration test..."
"${APPTAINER_BIN}" "${APPTAINER_ARGS[@]}" python -m unittest \
  lfm.model.tests.test_Pipeline.PipelineTestCase.testRunTileIndexWithTargetProductID

END_TIME="$(date +%s)"
END_READABLE="$(date)"
ELAPSED_SECONDS="$((END_TIME - START_TIME))"

printf -v ELAPSED_HMS "%02d:%02d:%02d" \
  "$((ELAPSED_SECONDS / 3600))" \
  "$(((ELAPSED_SECONDS % 3600) / 60))" \
  "$((ELAPSED_SECONDS % 60))"

echo
echo "All tiling modernization checks passed."
echo "Job finished at: ${END_READABLE}"
echo "Elapsed time: ${ELAPSED_HMS} (${ELAPSED_SECONDS} seconds)"
