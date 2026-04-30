
# Artifact for CCS 2026 Submission (Anonymous)

This repository contains the artifacts required to reproduce the core results of our paper on reconstructing DNN layer architectures using cache side-channel analysis on ExecuTorch.

---

## Overview

This artifact includes:

- ExecuTorch layer profiling reconstruction pipeline (ETDump → layer timeline) (scripts/run_and_collect_executorch.sh)
- Cache side-channel data (Prime+Probe spike logs) (e.g., data/mobilenetv2/spike_log1.txt)
- ONNX reference profiling traces (e.g., data/mobilenetv2/onnx.json)
- ExecuTorch profiling traces (data/executorch_layer_timeline1.csv)
- Native Prime+Probe code (scripts/resnet18v1/prime_probe.c)
- Training and evaluation scripts (scripts/train_predict_layers_with_size.py) for:
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
  efficientnet/

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


## Artifact Limitations (Data Size and Responsible Release)

To enable offline reproducibility, we provide representative samples of:

- ExecuTorch layer timelines (CSV)
- Prime+Probe cache traces (spike logs)
- ONNX reference profiling traces

However, we do not include the full dataset used in the paper, which consists of multiple spike logs and corresponding profiling files per model.

### Reason 1: Storage Constraints

Each Prime+Probe spike log is approximately 400–500 MB.  
The complete dataset (multiple runs across multiple models) exceeds GitHub’s repository size limits (2GB).

To remain within these constraints, we include a **single representative spike log and profiling file per model**, which is sufficient to:

- demonstrate the full pipeline
- validate alignment and feature extraction
- reproduce the training and evaluation process

### Reason 2: Responsible Disclosure

The provided cache traces capture detailed microarchitectural behavior of real devices.  
Releasing large-scale raw traces may facilitate unintended misuse of side-channel techniques.

To mitigate this risk, we release only a limited subset of traces necessary for scientific evaluation, while preserving the reproducibility of the methodology.

### Reproducibility Impact

- The provided data allows reviewers to reproduce the full pipeline and verify correctness.
- Using a single trace per model may result in slightly lower accuracy compared to the full dataset used in the paper.
- The methodology, feature extraction, and classification pipeline remain identical.

### Full Reproduction

Researchers who wish to fully reproduce the dataset can:

1. Run the provided Android pipeline (Section: Full Device Reproduction)
2. Collect multiple spike logs (Prime+Probe traces)
3. Re-run the training script with additional data

This process is fully supported by the provided scripts.
