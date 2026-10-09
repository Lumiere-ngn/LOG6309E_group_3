#!/bin/bash
# Compare the torch builds of venvs that ran on Fir GPUs before with this project's venv.
module load StdEnv/2023 python/3.12 2>/dev/null
for v in /scratch/marwan/*/.venv /scratch/marwan/*/*/.venv /scratch/marwan/log6309e/.venv; do
  [ -x "$v/bin/python" ] || continue
  echo "$v: $("$v/bin/python" -c 'import torch; print(torch.__version__, torch.version.cuda)' 2>&1 | tail -1)"
done
grep -h -E "module load|source .*activate" /scratch/marwan/tribe/remote/*.sbatch /scratch/marwan/*/scripts/*.sbatch 2>/dev/null | sort | uniq -c | head
ls /scratch/marwan/log6309e/.venv/lib/python3.12/site-packages | grep -i -E "^nvidia|^torch" | head
