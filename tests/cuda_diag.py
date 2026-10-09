"""Explain why torch sees no GPU: device files, visibility variables, and the CUDA driver's own cuInit code.

cuInit codes: 0 ok, 3 not initialized, 35 insufficient driver, 100 no device, 999 unknown.
Usage: python tests/cuda_diag.py
"""

import ctypes
import glob
import os
import sys


def main() -> int:
    print("devices:", sorted(glob.glob("/dev/nvidia*")))
    for var in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "SLURM_JOB_GPUS", "GPU_DEVICE_ORDINAL",
                "SLURM_STEP_GPUS", "LD_LIBRARY_PATH"):
        print(f"{var}={os.environ.get(var)}")
    for dev in sorted(glob.glob("/dev/nvidia*")):
        print(dev, "read", os.access(dev, os.R_OK), "write", os.access(dev, os.W_OK))
    for name in ("libcuda.so.1", "/usr/lib64/nvidia/libcuda.so.1", "/usr/lib64/libcuda.so.1"):
        try:
            lib = ctypes.CDLL(name)
        except OSError as err:
            print(f"{name}: cannot load ({err})")
            continue
        code = lib.cuInit(0)
        count = ctypes.c_int(-1)
        if code == 0:
            lib.cuDeviceGetCount(ctypes.byref(count))
        print(f"{name}: cuInit -> {code}, device count {count.value}")
    try:
        import torch
        print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
        torch.cuda.init()
    except Exception as err:  # report whatever torch raises; this is a diagnostic
        print("torch init error:", repr(err))
    return 0


if __name__ == "__main__":
    sys.exit(main())
