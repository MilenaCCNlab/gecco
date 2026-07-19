#!/bin/bash
# Resilient per-participant driver for individual gecco.
# Runs each pid in [START, END) one at a time, skips pids whose best_model
# already exists, and retries each pid up to MAXTRIES on non-zero exit (503s
# etc.). Isolates transient Gemini failures to a single pid instead of killing
# a whole range. Does NOT modify reviewed pipeline code.
#
# Usage: run_individual_resilient.sh START END KEYNAME
#   e.g. run_individual_resilient.sh 7 25 LAKELAB
set -u
REPO=/Users/akshay/projects/gecco
cd "$REPO"
START=$1; END=$2; KEYNAME=$3
CFG=two_step_psychiatry_individual_function_gemini-3-pro_ocd_maxsetting_150.yaml
IND=results/two_step_psychiatry_individual_function_ocibalanced150_maxsetting_individual
MAXTRIES=6
set -a; source .env; set +a
KEYVAL=$(eval echo "\$GEMINI_API_KEY_$KEYNAME")

for p in $(seq "$START" $((END-1))); do
  if [ -f "$IND/models/best_model_0_participant$p.txt" ]; then
    echo "[resilient] pid $p already done — skip"
    continue
  fi
  ok=0
  for t in $(seq 1 "$MAXTRIES"); do
    echo "[resilient] pid $p attempt $t/$MAXTRIES ($(date +%H:%M:%S))"
    GEMINI_API_KEY="$KEYVAL" gecco-env313/bin/python scripts/two_step_individual_function.py \
      --config "$CFG" --participants "$p:$((p+1))"
    if [ -f "$IND/models/best_model_0_participant$p.txt" ]; then
      echo "[resilient] pid $p DONE"
      ok=1; break
    fi
    echo "[resilient] pid $p attempt $t failed; sleeping 45s before retry"
    sleep 45
  done
  if [ "$ok" -ne 1 ]; then
    echo "[resilient] pid $p STILL FAILING after $MAXTRIES tries — moving on"
  fi
done
echo "[resilient] range $START:$END complete"
