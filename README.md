# Android DNN Side-Channel Artifact

This repository contains the source code and artifacts required to reproduce the cache side-channel analysis and DNN architecture reconstruction pipeline described in the accompanying paper.

The artifact is organized to separate source code, input data, and model-related files.

---

## Directory Structure
```text
src/
├── prime_probe.c
├── cache_sets.h
├── compute_hot_functions_cache_sets.py
└── train_predict_layers_with_size.py

data/
├── spike_files/
├── perf_report.txt
├── maps.txt
└── pagemap.bin

models/
└── onnx_files/
    ├── onnx_aligned1.json
    └── onnx_aligned2.json
```


---

## Requirements

- Ubuntu Linux (recommended)
- Python 3.9+
- GCC or Clang

Python dependencies (if required):
```bash
pip install numpy pandas scikit-learn
```

Build Instructions

Compile the Prime+Probe binary:

```
gcc -O2 -o prime_probe src/prime_probe.c
```

Usage
1. Cache Set Identification
   ```
   python3 src/compute_hot_functions_cache_sets.py
   ```
   This script identifies cache sets corresponding to target functions using memory mappings and pagemap information.
2. Spike Processing and Layer Reconstruction
   ```
   python3 src/train_predict_layers_with_size.py
   ```
   This script reconstructs the DNN layer sequence and dimensions using cache-access traces and aligned ONNX profiling data.


## Input data

Cache access traces and spike logs are located in data/

ONNX profiling/alignment JSON files are located in models/onnx_files/

Paths can be adjusted inside the scripts if needed.


## Notes

The artifact is provided for reproducibility and evaluation purposes.

Example files are included, large raw traces may be omitted or downsampled.

All identifiers in the repository are anonymized.
