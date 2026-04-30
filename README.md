
# Artifact for CCS 2026 Submission (Anonymous)

This repository contains the artifacts required to reproduce the core results of our paper on reconstructing DNN layer architectures using cache side-channel analysis on ExecuTorch.

---

## Overview

This artifact includes:

- ExecuTorch layer profiling reconstruction pipeline (ETDump → layer timeline)
- Cache side-channel data (Prime+Probe spike logs)
- ONNX reference profiling traces
- Native Prime+Probe code
- Training and evaluation scripts for:
  - Layer type prediction
  - Kernel shape prediction
  - Input/output channel (Cin/Cout) prediction

The artifact supports **offline reproducibility** of all core experimental results without requiring an Android device.

---

## Repository Structure

```text
scripts/
  train_predict_layers_with_size.py
  relabel_all50_etdump_with_onnx.py
  run_and_collect_executorch.sh
  compute_hot_functions_cache_sets.py
  export_mobilenetv2_executorch.py
  export_resnet18_executorch.py
  prime_probe.c
  cache_sets.h

data/
  mobilenetv2/
  resnet18v1/
  resnet18v1/

models/
  ExecuTorch (.pte, .etrecord)
  ONNX models (ResNet18V1, MobileNetV2, EfficientNet-Lite4)

outputs/
  expected_results_MobileNetV2.txt
  expected_results_ResNet18v1.txt

requirements.txt

```
## Hardware requirements
- Google Pixel 7a or comparable rooted Android device.
- Linux/macOS host with ADB.
- Python 3.10+.
- Android Studio/Gradle for rebuilding apps.
- Root access required only for collecting perf/pagemap information and Prime+Probe setup.


## Environment Setup

We recommend Python 3.10+.
```
Create environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Reproducing Results (Offline)

This is the primary evaluation path.

Step 1: Run training and evaluation
```
cd scripts
python train_predict_layers_with_size.py
```

Step 2: Expected behavior

The script will:

- Load ExecuTorch layer timelines (CSV)
- Load spike logs (Prime+Probe)
- Extract features
- Train classifiers for:
- Layer type (C, P, F, FC)
- Kernel shape (K7, K3, K1, etc.)
- Input/output channels (Cin/Cout)
- Print accuracy and reconstruction results

Step 3: Verify output

Compare your output with:
```
outputs/expected_results.txt
```
## ExecuTorch Profiling Pipeline

We include scripts to reconstruct layer timelines from ETDump:

```
bash scripts/run_and_collect_executorch.sh
```

This step:

- Parses ETDump files,
- Aligns with ONNX profiling,
- Generates layer timeline CSV.


**Note:** ETDump files are not fully included due to size constraints.

## Artifact Limitations (Space Constraints)

Due to GitHub repository size limits (2GB), we provide:

- Representative profiling files (CSV)
- Representative spike logs
- Representative ONNX traces

We do **NOT** include:

- Full ETDump traces for all 50 runs
- All collected spike logs
- Android application source code (~5GB per app)

However:

- All provided samples are sufficient to reproduce the evaluation pipeline
- All scripts are included for full reproduction on a local device
- The methodology is identical across all models (MobileNetV2, ResNet18V1, EfficientNet-Lite4)

Thus, reviewers can fully evaluate the correctness of the approach using the provided artifacts.

