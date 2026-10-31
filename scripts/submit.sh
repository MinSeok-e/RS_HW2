#!/bin/bash
# Submit the committed source; uncommitted tracked edits must be committed first.
set -euo pipefail
cd /home/minseok/data/semester/1/RS/RS_HW2
git diff --exit-code --quiet
git diff --cached --exit-code --quiet
RS_SOURCE_COMMIT=$(git rev-parse HEAD)
export RS_SOURCE_COMMIT
sbatch --export=ALL,RS_SOURCE_COMMIT="$RS_SOURCE_COMMIT" slurm/experiment.sbatch "$@"
