# Validation record

Validation date: 2026-08-17 (Asia/Shanghai)

## Local restructuring environment

- Windows
- Python 3.11.7
- PyTorch 2.5.0+cu121
- NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB

This is a repository-structure validation environment, not the locked paper
environment. The paper environment is specified in `environment.yml`.

## Checks performed

| Check | Result |
|---|---|
| Python compilation (`src`, `scripts`, third-party dependencies) | Passed |
| Ruff on maintained source, entry point, and tests | Passed |
| Unit tests | 5 passed |
| Ten CuPL JSON resources parse | Passed |
| DTD config (`dtd.yaml`) parse | Passed |
| DTD split construction | 1,692 test samples, 47 classes |
| CLI help/import smoke test | Passed |
| Local absolute-path audit | No matches |
| Model/data/cache/weight exclusion audit | Passed |

## GPU smoke test

One DTD item was loaded with the paper backbone and passed through the cached
CLIP ViT-B/16 image encoder on GPU 0. The observed output was:

```text
target shape: [1]
feature shape: [1, 512]
all values finite: true
normalized L2 norm: 1.0000001
```

The slight deviation from one is expected from floating-point arithmetic. This test
validates the model, data loader, augmentation, third-party import, and image
encoder path. It does **not** constitute a full DTD run or reproduction of
paper accuracy. Full numerical reproduction remains to be run in the locked
environment with the documented ten-dataset command.
