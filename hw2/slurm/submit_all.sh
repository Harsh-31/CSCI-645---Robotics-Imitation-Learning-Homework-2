#!/bin/bash
# Submits the full pipeline with dependencies. Run from hw2/:
#   ACCOUNT=<your_slurm_account> bash slurm/submit_all.sh
# Skip stages with e.g. SKIP_TEACHER=1 (when a teacher checkpoint exists).
set -euo pipefail
: "${ACCOUNT:?set ACCOUNT=<slurm account> (see: myaccount)}"
mkdir -p logs
A="--account=$ACCOUNT --parsable"
DEP=""
if [ -z "${SKIP_TEACHER:-}" ]; then
  T=$(sbatch $A slurm/01_teacher.slurm); echo "teacher  $T"; DEP="--dependency=afterok:$T"
fi
if [ -z "${SKIP_COLLECT:-}" ]; then
  C=$(sbatch $A $DEP slurm/02_collect.slurm); echo "collect  $C"; DEP="--dependency=afterok:$C"
fi
R=$(sbatch $A $DEP slurm/03_act.slurm); echo "act      $R"
echo "After all jobs finish: source slurm/env.sh && python code/analyze.py"
