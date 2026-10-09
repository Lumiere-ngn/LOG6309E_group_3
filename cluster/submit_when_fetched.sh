#!/bin/bash
# Wait on the login node until fetch_loghub.sh logs DONE, then submit steps_3_6.sbatch.
# Gives up after 90 minutes, or as soon as the fetch process is gone without DONE.
#   setsid nohup bash cluster/submit_when_fetched.sh > logs/submit_waiter.log 2>&1 < /dev/null &
ROOT=/scratch/marwan/log6309e
cd "$ROOT" || exit 1
for _ in $(seq 180); do
  if grep -q DONE fetch.log; then
    echo "$(date '+%F %T') fetch DONE; submitting"
    sbatch cluster/steps_3_6.sbatch
    exit $?
  fi
  if ! pgrep -f "^bash cluster/fetch_loghub.sh" > /dev/null; then
    echo "$(date '+%F %T') fetch process gone without DONE; not submitting"; tail -5 fetch.log; exit 1
  fi
  sleep 30
done
echo "$(date '+%F %T') gave up after 90 minutes"; exit 1
