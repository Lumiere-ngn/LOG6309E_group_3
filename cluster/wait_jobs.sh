#!/bin/bash
# Wait from WSL until the given Fir jobs reach a final state; print their states. A failed poll is reported, never
# read as an empty queue. Usage: bash cluster/wait_jobs.sh 64409181,64409182,64409183
JOBS=$1
for _ in $(seq 1 180); do
  out=$(ssh fir "bash -l -c 'sacct -j $JOBS -X -n -o JobID,State,Elapsed'" 2>&1)
  rc=$?
  if [ $rc -ne 0 ] || [ -z "$out" ]; then echo "poll failed rc=$rc: $out"; sleep 60; continue; fi
  if ! echo "$out" | grep -qE "PENDING|RUNNING|REQUEUED"; then echo "$out"; exit 0; fi
  sleep 60
done
echo "timeout; last: $out"
exit 1
