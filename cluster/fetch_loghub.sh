#!/bin/bash
# Download and unpack Loghub HDFS_v1 and BGL (Zenodo record 8196385) into $ROOT/data/raw.
# Run on a login node (compute nodes may have no internet):
#   setsid nohup bash cluster/fetch_loghub.sh /scratch/$USER/log6309e > fetch.log 2>&1 &
# Retries each file because Zenodo answers 504 intermittently.
set -u
ROOT=${1:?usage: fetch_loghub.sh <repo root>}
RAW=$ROOT/data/raw
mkdir -p "$RAW/HDFS_v1" "$RAW/BGL"
cd "$RAW" || exit 1

fetch() {
  local name=$1
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if curl -fsSL --retry 3 -o "$name.zip" "https://zenodo.org/records/8196385/files/$name.zip?download=1" \
       && unzip -tq "$name.zip" > /dev/null; then
      echo "$(date '+%F %T') $name.zip ok ($(du -h "$name.zip" | cut -f1)) on attempt $attempt"
      return 0
    fi
    echo "$(date '+%F %T') $name.zip attempt $attempt failed; retrying in 60 s"
    sleep 60
  done
  return 1
}

fetch HDFS_v1 && (cd HDFS_v1 && unzip -oq ../HDFS_v1.zip) || exit 1
fetch BGL && (cd BGL && unzip -oq ../BGL.zip) || exit 1
find "$RAW" -maxdepth 3 -type f -name '*.log' -o -maxdepth 3 -name '*.csv' | xargs ls -la
echo "$(date '+%F %T') DONE"
