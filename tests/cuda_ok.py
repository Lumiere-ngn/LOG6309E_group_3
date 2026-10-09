"""Exit 0 when torch can use a CUDA GPU, 3 otherwise; the job script stops before any run on 3.

Usage: python tests/cuda_ok.py
"""

import sys

import torch


def main() -> int:
    if not torch.cuda.is_available():
        print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) sees no GPU")
        return 3
    x = torch.ones(1024, 1024, device="cuda")
    print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}, matmul sum {float((x @ x).sum()):.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
