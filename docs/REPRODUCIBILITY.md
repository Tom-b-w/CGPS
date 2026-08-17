# Reproducibility protocol

## Scope

The primary target is the ten-dataset CLIP ViT-B/16 table. Exploratory
ImageNet-shift variants, rebuttal diagnostics, and later stress tests are not
part of the main reproduction claim.

## Fixed setup

- Python 3.10
- PyTorch 1.12.1, torchvision 0.13.1, CUDA 11.3
- One GPU, batch size 1
- Seed 42
- Unlabeled online test stream
- No gradient update to CLIP
- One CGPS trigger at 25% of the stream
- Confidence threshold 0.30
- Prompt selection fraction 0.10
- Minimum per-class consensus count 5

Random seeds are applied to Python, NumPy, PyTorch, and all CUDA devices. cuDNN
benchmarking is disabled and deterministic mode is enabled. Exact
floating-point identity across different GPUs and CUDA versions is not
guaranteed.

## Commands

Single dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --data-root /path/to/data \
  --datasets dtd \
  --seed 42 \
  --output outputs/dtd_seed42.json
```

All datasets:

```bash
bash scripts/reproduce.sh /path/to/data 0
```

The shell runner skips an existing non-empty JSON, enabling safe resume. Move
or delete a specific output only when deliberately rerunning that dataset.

## Acceptance criteria

1. The process exits with status 0 and writes valid JSON.
2. The JSON `config` object records the data root, dataset list, seed, backbone,
   and CGPS hyperparameters.
3. Each completed dataset contains the four result keys documented in the root
   README.
4. Differences from `results/paper_results.csv` are reported as measured and
   are never replaced with manuscript values.

## Known limits

- Dataset downloads and licenses are outside this repository.
- CLIP weights are downloaded separately and are not checksummed here.
- The evaluator retains diagnostic branches from the validated camera-ready
  file; only four output keys are used for the main paper claim.
- Existing logs show runtime variance under different machine load. No
  unmeasured full-suite runtime is claimed.
- Repository cleanup can validate code, tests, data construction, and a GPU
  smoke path, but full numerical reproduction requires all datasets and the
  documented CUDA environment.
