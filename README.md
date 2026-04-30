# Artifact for CCS 2026 Submission

This repository contains the artifacts for evaluating our Android cache-side-channel analysis pipeline for reconstructing DNN layer architectures from ExecuTorch inference.

## Contents
- Android ExecuTorch apps for MobileNetV2 and ResNet18V1
- Native Prime+Probe code
- ExecuTorch ETDump profiling pipeline
- ETDump-to-layer relabeling scripts
- ONNX reference profiling traces
- Sample spike logs and profiling files
- Training/evaluation script for layer type, kernel shape, Cin, and Cout prediction

## Artifact claims
The artifact supports:
1. Reproducing ExecuTorch layer timeline extraction.
2. Reproducing ETDump-to-semantic-layer relabeling.
3. Reproducing cache trace alignment.
4. Training/testing classifiers for layer type, kernel shape, Cin, and Cout.
5. Reproducing sample results for MobileNetV2 and ResNet18V1.

## Hardware requirements
- Google Pixel 7a or comparable rooted Android device.
- Linux/macOS host with ADB.
- Python 3.10+.
- Android Studio/Gradle for rebuilding apps.
- Root access required only for collecting perf/pagemap information and Prime+Probe setup.

## Quick offline reproduction
Run:
```bash
python src/training/train_predict_layers_with_size.py
