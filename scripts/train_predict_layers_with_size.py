#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, math, statistics, random
from bisect import bisect_left, bisect_right
from collections import defaultdict, Counter

import numpy as np
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier

# NEW:

from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
# NEW:
import umap


# ======= MODE & CONFIG =======
# MODE can be:
#   "train"        -> train layer/type/shape/Cin/Cout models (needs ONNX + spikes)
#   "test"         -> test those layer models on one ONNX + spikes pair
#   "train_model"  -> train model-identity classifier on spike-only captures
#   "test_model"   -> test / use the model-identity classifier on spike-only captures
#   "spike_only"   -> test layer heads using spike-only segmentation (needs ONNX only for run start/end)

MODE = "spike_only"
MODEL_PATH = "resnet18v1_layer_models_rf_cin_cout.joblib"

# Where to store the model-identity classifier (RF on spike-only features)
MODEL_ID_PATH = "model_classifier_rf.joblib"


JSON_FILE  = "cache_2025-12-01_18-52-05_onnx_aligned.json"
SPIKE_FILE = "spike_log.txt"
HOT_SETS   = [11644, 8735, 6434, 8481, 12850, 6710, 6711, 6713, 6715, 6688]

TRAIN_PAIRS = [
    #("profiling_files/resnet18v1_executorch_layer_timeline1.csv",  "spike_files_execu/ResNet_spike_log1.txt"),
    ("profiling_files/resnet18v1_executorch_layer_timeline13.csv",  "spike_files_execu/ResNet_spike_log13.txt"),
    ("profiling_files/resnet18v1_executorch_layer_timeline14.csv",  "spike_files_execu/ResNet_spike_log14.txt"),
    ("profiling_files/resnet18v1_executorch_layer_timeline15.csv",  "spike_files_execu/ResNet_spike_log15.txt"),
    ("profiling_files/resnet18v1_executorch_layer_timeline16.csv",  "spike_files_execu/ResNet_spike_log16.txt"),
    ("profiling_files/resnet18v1_executorch_layer_timeline17.csv",  "spike_files_execu/ResNet_spike_log17.txt"),
    ("profiling_files/resnet18v1_executorch_layer_timeline18.csv",  "spike_files_execu/ResNet_spike_log18.txt"),
    ("profiling_files/resnet18v1_executorch_layer_timeline19.csv",  "spike_files_execu/ResNet_spike_log19.txt"),
    ("profiling_files/resnet18v1_executorch_layer_timeline20.csv",  "spike_files_execu/ResNet_spike_log20.txt"),
    ("profiling_files/resnet18v1_executorch_layer_timeline21.csv",  "spike_files_execu/ResNet_spike_log21.txt"),
    ("profiling_files/resnet18v1_executorch_layer_timeline22.csv",  "spike_files_execu/ResNet_spike_log22.txt"),
    # add the rest similarly

]

# In TRAIN mode:
#   - If TEST_PAIR is None, we use the last TRAIN_PAIRS element as test
#   - If TEST_PAIR is not None, we train on all TRAIN_PAIRS and test on TEST_PAIR
#
# In TEST mode:
#   - We IGNORE TRAIN_PAIRS content and ONLY test on TEST_PAIR (must be set)
#TEST_PAIR = None
TEST_PAIR = ("profiling_files/resnet18v1_executorch_layer_timeline23.csv", "spike_files_execu/ResNet_spike_log23.txt")


# ======= Model-level (spike-only) classifier config =======
# Each entry in MODEL_TRAIN_SETS / MODEL_TEST_SETS is:
#     (label_string, spike_log_path)
#
# 
#
# For MODEL_TEST_SETS we can use ONNX or TFLite traces.
# If label is None, we only print predictions (no accuracy).
MODEL_TRAIN_SETS = [
    ("MobileNetV2",       "spike_files/spike_log1.txt"),
    ("MobileNetV2",       "spike_files/spike_log2.txt"),
    ("MobileNetV2",       "spike_files/spike_log3.txt"),
    ("MobileNetV2",       "spike_files/spike_log4.txt"),
    ("MobileNetV2",       "spike_files/spike_log5.txt"),
    ("MobileNetV2",       "spike_files/spike_log6.txt"),
    ("MobileNetV2",       "spike_files/spike_log7.txt"),
    ("MobileNetV2",       "spike_files/spike_log8.txt"),
    ("MobileNetV2",       "spike_files/spike_log9.txt"),
    ("MobileNetV2",       "spike_files/spike_log10.txt"),
    ("MobileNetV2",       "spike_files/spike_log11.txt"),
    ("MobileNetV2",       "spike_files/spike_log12.txt"),
    
    ("MobileNetV2-TFLite",       "../TFLite_MobileNetV2/spike_files/spike_log1.txt"),
    ("MobileNetV2-TFLite",       "../TFLite_MobileNetV2/spike_files/spike_log2.txt"),
    ("MobileNetV2-TFLite",       "../TFLite_MobileNetV2/spike_files/spike_log3.txt"),
    ("MobileNetV2-TFLite",       "../TFLite_MobileNetV2/spike_files/spike_log4.txt"),
    ("MobileNetV2-TFLite",       "../TFLite_MobileNetV2/spike_files/spike_log5.txt"),
    ("MobileNetV2-TFLite",       "../TFLite_MobileNetV2/spike_files/spike_log6.txt"),
    ("MobileNetV2-TFLite",       "../TFLite_MobileNetV2/spike_files/spike_log7.txt"),
    ("MobileNetV2-TFLite",       "../TFLite_MobileNetV2/spike_files/spike_log8.txt"),
    ("MobileNetV2-TFLite",       "../TFLite_MobileNetV2/spike_files/spike_log9.txt"),
    ("MobileNetV2-TFLite",       "../TFLite_MobileNetV2/spike_files/spike_log10.txt"),
    ("MobileNetV2-TFLite",       "../TFLite_MobileNetV2/spike_files/spike_log11.txt"),
    ("MobileNetV2-TFLite",       "../TFLite_MobileNetV2/spike_files/spike_log12.txt"),

    ("EfficientNet-Lite4","../EfficientNet/spike_files/spike_log1.txt"),
    ("EfficientNet-Lite4","../EfficientNet/spike_files/spike_log2.txt"),
    ("EfficientNet-Lite4","../EfficientNet/spike_files/spike_log3.txt"),
    ("EfficientNet-Lite4","../EfficientNet/spike_files/spike_log4.txt"),
    ("EfficientNet-Lite4","../EfficientNet/spike_files/spike_log5.txt"),
    ("EfficientNet-Lite4","../EfficientNet/spike_files/spike_log6.txt"),
    ("EfficientNet-Lite4","../EfficientNet/spike_files/spike_log7.txt"),
    ("EfficientNet-Lite4","../EfficientNet/spike_files/spike_log8.txt"),
    ("EfficientNet-Lite4","../EfficientNet/spike_files/spike_log9.txt"),
    ("EfficientNet-Lite4","../EfficientNet/spike_files/spike_log10.txt"),
    ("EfficientNet-Lite4","../EfficientNet/spike_files/spike_log11.txt"),
    ("EfficientNet-Lite4","../EfficientNet/spike_files/spike_log12.txt"),

]

MODEL_TEST_SETS = [
    # ("MobileNetV2",       "spike_files/spike_log13.txt"),
    # ("MobileNetV2-TFLite",       "../TFLite_MobileNetV2/spike_files/spike_log13.txt"),
    # ("EfficientNet-Lite4","../EfficientNet/spike_files/spike_log13.txt"),
    (None,                "../TFLite_MobileNetV2/spike_files/spike_log13.txt"),
]



COVERAGE_THRESHOLD = 0.85
FIXED_TICKS = 20
ADAPT_K     = 4.0          
ONLY_KERNEL_ROWS  = True   
ONLY_CPU_PROVIDER = True   # True = CPUExecutionProvider only

RNG_SEED = 7

# ======= helpers =======

#Executorch MobileNetV2 layer dimention mapping
MOBILENETV2_LAYER_DIMS = {
    "features.0.0":      (3, 32),

    "features.1.conv.0": (32, 32),
    "features.1.conv.3": (32, 16),

    "features.2.conv.0": (16, 96),
    "features.2.conv.3": (96, 96),
    "features.2.conv.6": (96, 24),

    "features.3.conv.0": (24, 144),
    "features.3.conv.3": (144, 144),
    "features.3.conv.6": (144, 24),

    "features.4.conv.0": (24, 144),
    "features.4.conv.3": (144, 144),
    "features.4.conv.6": (144, 32),

    "features.5.conv.0": (32, 192),
    "features.5.conv.3": (192, 192),
    "features.5.conv.6": (192, 32),

    "features.6.conv.0": (32, 192),
    "features.6.conv.3": (192, 192),
    "features.6.conv.6": (192, 32),

    "features.7.conv.0": (32, 192),
    "features.7.conv.3": (192, 192),
    "features.7.conv.6": (192, 64),

    "features.8.conv.0": (64, 384),
    "features.8.conv.3": (384, 384),
    "features.8.conv.6": (384, 64),

    "features.9.conv.0": (64, 384),
    "features.9.conv.3": (384, 384),
    "features.9.conv.6": (384, 64),

    "features.10.conv.0": (64, 384),
    "features.10.conv.3": (384, 384),
    "features.10.conv.6": (384, 64),

    "features.11.conv.0": (64, 384),
    "features.11.conv.3": (384, 384),
    "features.11.conv.6": (384, 96),

    "features.12.conv.0": (96, 576),
    "features.12.conv.3": (576, 576),
    "features.12.conv.6": (576, 96),

    "features.13.conv.0": (96, 576),
    "features.13.conv.3": (576, 576),
    "features.13.conv.6": (576, 96),

    "features.14.conv.0": (96, 576),
    "features.14.conv.3": (576, 576),
    "features.14.conv.6": (576, 160),

    "features.15.conv.0": (160, 960),
    "features.15.conv.3": (960, 960),
    "features.15.conv.6": (960, 160),

    "features.16.conv.0": (160, 960),
    "features.16.conv.3": (960, 960),
    "features.16.conv.6": (960, 160),

    "features.17.conv.0": (160, 960),
    "features.17.conv.3": (960, 960),
    "features.17.conv.6": (960, 320),

    "features.18.0":      (320, 1280),
}



RESNET18V1_LAYER_DIMS = {
    "conv0": (3, 64),

    "stage1.conv0": (64, 64),
    "stage1.conv1": (64, 64),
    "stage1.conv2": (64, 64),
    "stage1.conv3": (64, 64),

    "stage2.conv0": (64, 128),
    "stage2.conv1": (128, 128),
    "stage2.downsample.0": (64, 128),
    "stage2.conv3": (128, 128),
    "stage2.conv4": (128, 128),

    "stage3.conv0": (128, 256),
    "stage3.conv1": (256, 256),
    "stage3.downsample.0": (128, 256),
    "stage3.conv3": (256, 256),
    "stage3.conv4": (256, 256),

    "stage4.conv0": (256, 512),
    "stage4.conv1": (512, 512),
    "stage4.downsample.0": (256, 512),
    "stage4.conv3": (512, 512),
    "stage4.conv4": (512, 512),
}


def prefer(keys, d, default=None):
    for k in keys:
        if k in d: return d[k]
    return default



def make_execu_op_type(kernel_tag, semantic_role, event_name):
    k = (kernel_tag or "").upper()
    e = (event_name or "").lower()

    if k in ("K7", "K3", "K1", "FC", "GAP", "MAXPOOL", "FLATTEN", "RELU", "ADD", "TRANSPOSE"):
        return k

    if "clamp" in e:
        return "RELU"
    if "max pool" in e or "maxpool" in e:
        return "MAXPOOL"
    if "mean" in e or "average" in e:
        return "GAP"
    if "flatten" in e:
        return "FLATTEN"
    if "gemm" in e or "fully connected" in e:
        return "FC"
    if "convolution" in e or "igemm" in e:
        return "K3"
    if "add" in e:
        return "ADD"
    if "transpose" in e:
        return "TRANSPOSE"

    return "OTHER"


def load_events_executorch_csv(csv_path):
    import pandas as pd

    df = pd.read_csv(csv_path)
    df = df.sort_values(["inference_index", "op_index"]).reset_index(drop=True)

    out = []

    # 1) synthetic run events so detect_runs() still works unchanged
    run_df = (
        df[["inference_index", "inference_ts_start_ns", "inference_ts_end_ns"]]
        .drop_duplicates()
        .sort_values("inference_index")
    )

    for _, r in run_df.iterrows():
        rs_us = int(r["inference_ts_start_ns"]) // 1000
        re_us = int(r["inference_ts_end_ns"]) // 1000
        out.append({
            "start": rs_us,
            "end": re_us,
            "dur": re_us - rs_us,
            "name": "model_run",
            "op_type": "RUN",
            "provider": "",
            "args": {},
            "inference_index": int(r["inference_index"]),
        })

    # 2) actual layer events
    for _, r in df.iterrows():
        sem_name = str(r.get("semantic_layer_name", ""))
        kernel_tag = str(r.get("kernel_tag", ""))
        semantic_role = str(r.get("semantic_role", ""))
        event_name = str(r.get("event_name", ""))

        start_us = int(r["absolute_monotonic_start_ns"]) // 1000
        end_us   = int(r["absolute_monotonic_end_ns"]) // 1000
        dur_us   = end_us - start_us

        op_type = make_execu_op_type(kernel_tag, semantic_role, event_name)

        args = {}
        if sem_name in RESNET18V1_LAYER_DIMS:
            cin, cout = RESNET18V1_LAYER_DIMS[sem_name]
            args = {
                "input_type_shape":  [{"float": [1, cin, 1, 1]}],
                "output_type_shape": [{"float": [1, cout, 1, 1]}],
            }

        out.append({
            "start": start_us,
            "end": end_us,
            "dur": dur_us,
            "name": sem_name + "_kernel_time",
            "op_type": op_type,
            "provider": "CPUExecutionProvider",
            "args": args,
            "inference_index": int(r["inference_index"]),
            "event_name": event_name,
            "kernel_tag": kernel_tag,
            "semantic_role": semantic_role,
        })

    out.sort(key=lambda x: x["start"])
    return out


def load_events(path):
    path_l = path.lower()
    if path_l.endswith(".csv"):
        return load_events_executorch_csv(path)

    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        events = data
        top = {}
    else:
        events = data.get("events") or data.get("traceEvents") or data.get("Timeline") or []
        top = data

    base_us = prefer([
        "profiling_start_ts",
        "profiling_start_ts_us",
        "profiling_start_boot_us",
        "profiling_start_mono_us"
    ], top, None)

    out = []
    for e in events:
        if not isinstance(e, dict):
            continue
        ph = e.get("ph")
        if ph and ph != "X":
            continue

        dur = prefer(["dur", "dur_us"], e, None)
        if dur is None:
            continue
        try:
            dur = int(dur)
        except:
            try:
                dur = int(float(dur))
            except:
                continue
        if dur <= 0:
            continue

        ts_abs = prefer(["ts_abs_us", "ts_abs", "ts_absolute_us", "ts_boot_us", "ts_mono_us"], e, None)
        if ts_abs is None:
            ts_rel = prefer(["ts", "ts_us"], e, None)
            if ts_rel is None:
                continue
            try:
                ts_rel = int(ts_rel)
            except:
                try:
                    ts_rel = int(float(ts_rel))
                except:
                    continue
            if base_us is None:
                ts_abs = ts_rel
            else:
                ts_abs = base_us + ts_rel

        try:
            ts_abs = int(ts_abs)
        except:
            try:
                ts_abs = int(float(ts_abs))
            except:
                continue

        name = prefer(["name", "op_name", "node", "node_name"], e, "")
        op_type = prefer(["op_type", "type", "op"], e, "")
        args = e.get("args") or {}
        provider = args.get("provider") or ""

        out.append({
            "start": int(ts_abs),
            "end": int(ts_abs) + dur,
            "dur": int(dur),
            "name": str(name),
            "op_type": str(op_type),
            "provider": str(provider),
            "args": args,
        })

    out.sort(key=lambda x: x["start"])
    return out

def detect_runs(events):
    """
    Returns a list of (start, end) for *logical* inferences.
    Prefer 'model_run'; otherwise 'SequentialExecutor::Execute'.
    """
    runs_model = []
    runs_exec  = []

    for e in events:
        n = (e.get("name") or "").lower()
        if "model_run" in n:
            runs_model.append((e["start"], e["end"]))
        elif "sequentialexecutor::execute" in n:
            runs_exec.append((e["start"], e["end"]))

    runs = runs_model if runs_model else runs_exec
    runs.sort(key=lambda x: x[0])
    return runs

def parse_spikes(path):
    rows=[]
    with open(path,"r") as f:
        for line in f:
            s=line.strip()
            if not s:
                continue
            parts=[p.strip() for p in s.split(",")]
            if len(parts)<2:
                continue
            try:
                ts=int(parts[0])
            except:
                try:
                    ts=int(float(parts[0]))
                except:
                    continue
            lats=[]
            for p in parts[1:1+len(HOT_SETS)]:
                try:
                    v=int(p)
                except:
                    try:
                        v=int(float(p))
                    except:
                        v=0
                lats.append(v)
            if len(lats)!=len(HOT_SETS):
                continue
            rows.append({"ts": ts, "sets": lats})
    rows.sort(key=lambda r: r["ts"])
    return rows

def median_cadence_us(spikes):
    if len(spikes)<2:
        return 0.0
    diffs=[spikes[i+1]["ts"]-spikes[i]["ts"] for i in range(len(spikes)-1)]
    diffs=[d for d in diffs if d>0]
    if not diffs:
        return 0.0
    return statistics.median(diffs)

def rows_in_window(events, start, end, only_kernel, only_cpu):
    out=[]
    for e in events:
        if e["end"]<=start:
            continue
        if e["start"]>=end:
            break
        if only_kernel and "kernel_time" not in (e["name"] or ""):
            continue
        if only_cpu and e.get("provider") and "cpu" not in e["provider"].lower():
            continue
        out.append(e)
    return out

def label_from_op(op_type, name):
    n = (name or "").lower()
    o = (op_type or "").lower()

    # ---- ExecuTorch CSV path for ResNet18 ----
    if o in ("k7", "k3", "k1"):
        return "C"
    if o == "fc":
        return "FC"
    if o in ("gap", "maxpool"):
        return "P"
    if o == "flatten":
        return "F"

    # We do not want to consider residual_add or relu layers.
    if o in ("add", "relu", "transpose", "other"):
        return "O"

    # ---- original ONNX path ----
    if "reducemean" in n or "reducemean" in o:
        return "P"
    if "sub" in n or "div" in n or "sub" in o or "div" in o:
        return "N"
    if "gemm" in n or "gemm" in o or "matmul" in n or "matmul" in o or "fc" in n:
        return "FC"
    if "conv" in n or "conv" in o:
        return "C"
    if "pool" in n or "pool" in o:
        return "P"
    if "flatten" in n or "reshape" in n or "flatten" in o or "reshape" in o:
        return "F"
    if "relu6" in n or "relu6" in o or "relu" in n or "relu" in o:
        return "A"
    if "batchnorm" in n or "batchnorm" in o or "bn" in n:
        return "BN"
    return "O"

def skip_type_layer(op_type, name):
    n = (name or "").lower()
    o = (op_type or "").lower()

    # Exclude residual/add, ReLU/clamp, layout transforms, and unknown ops.
    if o in ("add", "relu", "transpose", "other"):
        return True

    if "residual_add" in n or "relu" in n or "clamp" in n:
        return True

    return False


def kernel_shape_from_name(op_type: str, name: str) -> str:
    """
    K1   = 1x1 conv
    K3   = 3x3 conv (non-depthwise)
    K3DW = 3x3 depthwise conv
    GAP  = global average pooling (ReduceMean)
    FC   = Gemm / fully-connected
    K?   = unknown conv
    O    = other
    """
    n = (name or "").lower()
    o = (op_type or "").lower()
        
    # Special-case the final pointwise conv in MobileNetV2
    if "features.18.0_kernel_time" in n:
        return "K1PW"
    
    # ---- ExecuTorch CSV path for ResNet18 ----
    if o in ("k7", "k3", "k1", "fc", "gap", "maxpool", "flatten"):
        return o.upper()

    # Exclude these from shape head
    if o in ("add", "relu", "transpose", "other"):
        return "O"


    # ---- original ONNX path ----
    if "gemm" in n or "gemm" in o or "matmul" in n or "matmul" in o or "fc" in n:
        return "FC"
    if "reducemean" in n or "globalaveragepool" in n or "global_average_pool" in n:
        return "GAP"

    if "conv" in n or "conv" in o:
        if "/features/0/0/conv" in n:
            return "K3"
        if "/conv/3/conv" in n or "/conv/3/" in n:
            return "K3DW"
        if "/conv/0/conv" in n or "/conv/6/conv" in n or "/conv/0/" in n or "/conv/6/" in n:
            return "K1"
        if "depth" in n or "depth" in o or "dw" in n:
            return "K3DW"
        return "K?"

    return "O"

def slice_spikes(spikes, ts_list, s, e):
    i=bisect_left(ts_list, s)
    j=bisect_right(ts_list, e)
    return i,j

def adapt_thresholds_per_run(spikes_run, k=ADAPT_K):
    per_set_vals=[[] for _ in HOT_SETS]
    for r in spikes_run:
        for i,v in enumerate(r["sets"]):
            per_set_vals[i].append(v)
    thr=[]
    for vs in per_set_vals:
        if not vs:
            thr.append(FIXED_TICKS)
        else:
            med=statistics.median(vs)
            mad=statistics.median([abs(x-med) for x in vs]) if vs else 0.0
            thr.append(int(med + k*(1.4826*mad if mad>0 else 0.0)))
    return thr

def entropy(p):
    s=0.0
    for x in p:
        if x>0:
            s -= x*math.log(x,2)
    return s

def gini_concentration(values):
    """Gini-like concentration index over a list of non-negative values."""
    vals = [v for v in values if v > 0]
    n = len(vals)
    if n == 0:
        return 0.0
    vals.sort()
    total = sum(vals)
    if total <= 0:
        return 0.0
    cum = 0.0
    for i, v in enumerate(vals, start=1):
        cum += i * v
    return (2 * cum) / (n * total) - (n + 1) / n
    
    
    
def robust_median(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    vals = sorted(vals)
    return vals[len(vals)//2]

def robust_filter_mad(vals, k=3.5):
    
    vals = [v for v in vals if v is not None]
    if len(vals) < 5:
        return vals
    med = statistics.median(vals)
    mad = statistics.median([abs(x - med) for x in vals])
    if mad <= 0:
        return vals
    s = 1.4826 * mad
    lo = med - k * s
    hi = med + k * s
    return [x for x in vals if lo <= x <= hi]

def build_duration_profile_from_training(train_pairs):
    """
    Learn per-layer duration template from ONNX layer durations across FULL runs.
    Returns dict with:
      - dur_profile_us: list[int] length = #layers
      - cum_profile_frac: list[float] cumulative fractions (same length)
    """
    durs_by_pos = defaultdict(list)   
    max_pos = -1

    for json_path, spike_path in train_pairs:
        events = load_events(json_path)
        spikes = parse_spikes(spike_path)
        if not spikes:
            continue
        ts_all = [r["ts"] for r in spikes]
        med_dt = median_cadence_us(spikes)
        runs = detect_runs(events)
        if not runs:
            continue

        # identify FULL runs (same logic as extract_features_for_pair)
        full = []
        for idx, (rs, re) in enumerate(runs):
            if rs < ts_all[0] or re > ts_all[-1]:
                continue
            i, j = slice_spikes(spikes, ts_all, rs, re)
            samples = j - i
            dur_us = re - rs
            cov = min(1.0, (samples * med_dt) / dur_us if (dur_us > 0 and med_dt > 0) else 0.0)
            if cov >= COVERAGE_THRESHOLD:
                full.append((idx, (rs, re)))

        # collect layer durations by position within run
        for ridx, (rs, re) in full:
            layers = rows_in_window(events, rs, re, ONLY_KERNEL_ROWS, ONLY_CPU_PROVIDER)
            for pos, L in enumerate(layers):
                durs_by_pos[pos].append(int(L["dur"]))
                if pos > max_pos:
                    max_pos = pos

    if max_pos < 0:
        return None

    dur_profile = []
    for pos in range(max_pos + 1):
        vals = robust_filter_mad(durs_by_pos.get(pos, []), k=3.5)
        med = robust_median(vals)
        if med is None:
            # fallback: 1 tick window (won't be great, but avoids crash)
            med = 1
        dur_profile.append(int(med))

    total = sum(dur_profile)
    if total <= 0:
        total = 1
    cum = 0
    cum_frac = []
    for d in dur_profile:
        cum += d
        cum_frac.append(cum / total)

    return {
        "dur_profile_us": dur_profile,
        "cum_profile_frac": cum_frac,
    }
    


# ======= Gaussian Naive Bayes (for TYPE head) =======
class GNB:
    def __init__(self):
        self.labels=[]
        self.priors={}
        self.mu={}
        self.var={}

    def fit(self, X, Y):
        self.labels=sorted(set(Y))
        n=len(X)
        d=len(X[0]) if X else 0
        for c in self.labels:
            idx=[i for i,y in enumerate(Y) if y==c]
            self.priors[c]=len(idx)/n
            mu=[0.0]*d
            for j in range(d):
                col=[X[i][j] for i in idx]
                m=sum(col)/len(col)
                mu[j]=m
            self.mu[c]=mu
            var=[0.0]*d
            for j in range(d):
                col=[X[i][j] for i in idx]
                m=mu[j]
                v=sum((v-m)*(v-m) for v in col)/max(1,len(col)-1)
                var[j]=max(v,1e-6)
            self.var[c]=var

    def predict_one(self, x):
        best_c=None
        best_ll=None
        for c in self.labels:
            mu=self.mu[c]; var=self.var[c]
            ll=math.log(self.priors[c] + 1e-12)
            for j, xj in enumerate(x):
                m=mu[j]; v=var[j]
                ll += -0.5*math.log(2*math.pi*v) - 0.5*((xj-m)*(xj-m))/v
            if best_ll is None or ll>best_ll:
                best_ll=ll; best_c=c
        return best_c

    def predict(self, X):
        return [self.predict_one(x) for x in X]

def standardize_fit(X):
    d=len(X[0])
    mu=[0.0]*d; sd=[0.0]*d
    for j in range(d):
        col=[x[j] for x in X]
        m=sum(col)/len(col)
        var=sum((v-m)*(v-m) for v in col)/max(1,len(col)-1)
        s=math.sqrt(var) if var>1e-12 else 1.0
        mu[j]=m; sd[j]=s
    return mu, sd

def standardize_apply(X, mu, sd):
    return [[(x[j]-mu[j])/sd[j] for j in range(len(x))] for x in X]



def extract_model_features_from_spike(spike_path):
    """
    Build a single feature vector summarizing an entire spike_log.txt capture.

    This uses ONLY the spikes (no ONNX), so it works for ONNX, TFLite,
    or any other framework as long as HOT_SETS matches the monitored cache sets.
    """
    spikes = parse_spikes(spike_path)
    if not spikes:
        print(f"[MODEL] No spikes found in {spike_path}; skipping.")
        return None

    ts_all = [r["ts"] for r in spikes]
    rs, re = ts_all[0], ts_all[-1]
    dur_us = max(re - rs, 1)
    dur_ms = dur_us / 1000.0
    samples = len(spikes)
    med_dt = median_cadence_us(spikes)

    # thresholds over the whole capture
    thr_adapt = adapt_thresholds_per_run(spikes, ADAPT_K)
    thr_fixed = [FIXED_TICKS] * len(HOT_SETS)

    any_fixed = 0
    any_adapt = 0
    per_set_cnt = [0] * len(HOT_SETS)
    per_set_sum = [0.0] * len(HOT_SETS)
    max_ticks = []

    for r in spikes:
        max_v = max(r["sets"])
        max_ticks.append(max_v)
        af = False
        aa = False
        for k, v in enumerate(r["sets"]):
            per_set_sum[k] += v
            if v >= thr_fixed[k]:
                af = True
                per_set_cnt[k] += 1
            if v >= thr_adapt[k]:
                aa = True
        if af:
            any_fixed += 1
        if aa:
            any_adapt += 1

    density_fixed = (any_fixed / samples) if samples > 0 else 0.0
    density_adapt = (any_adapt / samples) if samples > 0 else 0.0
    rate_ms       = (any_fixed / dur_ms) if dur_ms > 0 else 0.0

    if max_ticks:
        max_ticks.sort()
        p95      = max_ticks[int(0.95 * (len(max_ticks) - 1))]
        mean_max = sum(max_ticks) / len(max_ticks)
        if len(max_ticks) > 1:
            m = mean_max
            var_max = sum((v - m) * (v - m) for v in max_ticks) / (len(max_ticks) - 1)
            std_max = math.sqrt(var_max)
        else:
            std_max = 0.0
    else:
        p95 = 0
        mean_max = 0.0
        std_max  = 0.0

    p = [c / samples for c in per_set_cnt]   if samples > 0 else [0.0] * len(HOT_SETS)
    mean_set = [s / samples for s in per_set_sum] if samples > 0 else [0.0] * len(HOT_SETS)
    ent  = entropy(p)
    conc = gini_concentration(mean_set)

    # We add a few extra global scalars on top of the layer features
    feats = [
        math.log10(dur_us),
        math.log10(max(samples, 1)),
        math.log10(max(med_dt, 1.0)),
        density_fixed,
        density_adapt,
        rate_ms,
        p95,
        ent,
        mean_max,
        std_max,
        conc,
    ] + p + mean_set

    return feats



def segment_run_by_duration_profile_start_only(rs, dur_profile_us):
    """
    Split run using ONLY start time and the saved duration profile (no scaling).
    Returns list of (seg_start, seg_end) timestamps in us.
    """
    segs = []
    cur = rs
    for d in dur_profile_us:
        d = max(1, int(d))
        segs.append((cur, cur + d))
        cur += d
    run_end = cur
    return segs, run_end


def extract_features_for_pair_spike_only(json_path, spike_path, dur_profile_us):
    """
    Like extract_features_for_pair(), but:
      - uses ONNX only to detect FULL run boundaries
      - uses learned dur_profile_us to create per-layer windows
      - DOES NOT use per-layer ONNX timestamps
    GT labels are still derived from ONNX layer list (by position), so we can measure accuracy.
    """
    events = load_events(json_path)
    spikes = parse_spikes(spike_path)
    if not spikes:
        return [], [], [], [], [], [], [], [], [], [], [], 0.0

    ts_all = [r["ts"] for r in spikes]
    med_dt = median_cadence_us(spikes)
    runs = detect_runs(events)
    if not runs:
        return [], [], [], [], [], [], [], [], [], [], [], med_dt

    # pick FULL runs
    template_len = int(sum(dur_profile_us))
    if template_len <= 0:
        template_len = 1

    # pick FULL runs (use ONLY rs from ONNX, infer re from template_len)
    full = []
    for idx, (rs, _re_onnx) in enumerate(runs):
        re = rs + template_len

        if rs < ts_all[0] or re > ts_all[-1]:
            continue

        i, j = slice_spikes(spikes, ts_all, rs, re)
        samples = j - i

        cov = min(
            1.0,
            (samples * med_dt) / template_len if (template_len > 0 and med_dt > 0) else 0.0
        )
        if cov >= COVERAGE_THRESHOLD:
            full.append((idx, (rs, re)))


    # outputs
    X_all=[]; Y_all=[]; meta_all=[]
    X_sh=[];  Y_sh=[];  meta_sh=[]
    X_cin=[]; Y_cin=[]; meta_cin=[]
    X_cout=[];Y_cout=[];meta_cout=[]

    for ridx, (rs, re) in full:
        # adaptive thresholds per run
        i_run, j_run = slice_spikes(spikes, ts_all, rs, re)
        thr_adapt = adapt_thresholds_per_run(spikes[i_run:j_run], ADAPT_K)
        thr_fixed = [FIXED_TICKS]*len(HOT_SETS)

        # GT layer list (only for labels + Cin/Cout extraction), not for timing
        layers_gt = rows_in_window(events, rs, re, ONLY_KERNEL_ROWS, ONLY_CPU_PROVIDER)

        #segs = segment_run_by_duration_profile(rs, re, dur_profile_us)
        segs, re_seg = segment_run_by_duration_profile_start_only(rs, dur_profile_us)
        re = re_seg
        K = min(len(segs), len(layers_gt))
        

        for layer_idx in range(K):
            s_us, e_us = segs[layer_idx]
            i, j = slice_spikes(spikes, ts_all, s_us, e_us)
            rows = spikes[i:j]
            samples = len(rows)
            dur_us = max(1, e_us - s_us)
            dur_ms = dur_us/1000.0

            any_fixed=0; any_adapt=0
            per_set_cnt=[0]*len(HOT_SETS)
            per_set_sum=[0.0]*len(HOT_SETS)
            max_ticks=[]

            for r in rows:
                max_v = max(r["sets"])
                max_ticks.append(max_v)
                af=False; aa=False
                for k,v in enumerate(r["sets"]):
                    per_set_sum[k] += v
                    if v >= thr_fixed[k]:
                        af=True; per_set_cnt[k]+=1
                    if v >= thr_adapt[k]:
                        aa=True
                if af: any_fixed+=1
                if aa: any_adapt+=1

            density_fixed = (any_fixed / samples) if samples>0 else 0.0
            density_adapt = (any_adapt / samples) if samples>0 else 0.0
            rate_ms       = (any_fixed / dur_ms) if dur_ms>0 else 0.0

            if max_ticks:
                max_ticks.sort()
                p95 = max_ticks[int(0.95*(len(max_ticks)-1))]
                mean_max = sum(max_ticks)/len(max_ticks)
                std_max = statistics.pstdev(max_ticks) if len(max_ticks) > 1 else 0.0
            else:
                p95 = 0; mean_max = 0.0; std_max = 0.0

            p = [c/samples for c in per_set_cnt] if samples>0 else [0.0]*len(HOT_SETS)
            mean_set = [s/samples for s in per_set_sum] if samples>0 else [0.0]*len(HOT_SETS)
            ent = entropy(p)
            conc = gini_concentration(mean_set)

            mid = (s_us + e_us)/2.0
            run_pos = (mid - rs)/max(1.0,(re-rs))

            feats = [
                math.log10(max(dur_us,1)),
                density_fixed,
                density_adapt,
                rate_ms,
                p95,
                ent,
                mean_max,
                std_max,
                conc,
            ] + p + mean_set + [run_pos]

            # --- GT label by position ---
           

            L = layers_gt[layer_idx]
            lbl = label_from_op(L["op_type"], L["name"])
            if not skip_type_layer(L["op_type"], L["name"]):
                X_all.append(feats)
                Y_all.append(lbl)
                meta_all.append((ridx, layer_idx, L["name"], L["op_type"], dur_us, s_us, e_us))

            shape_lbl = kernel_shape_from_name(L["op_type"], L["name"])
            # if shape_lbl in ("K1","K3","K3DW","GAP","FC","K?"):
            
            if shape_lbl in ("K7", "K3", "K1", "MAXPOOL", "GAP", "FLATTEN", "FC"):
                X_sh.append(feats)
                Y_sh.append(shape_lbl)
                meta_sh.append((ridx, layer_idx, L["name"], L["op_type"], dur_us, s_us, e_us))

            # Cin/Cout (still extracted from ONNX args, but window is spike-only)
           
            if shape_lbl in ("K7", "K3", "K1"):
                cin = cout = None
                args = L.get("args", {}) or {}
                in_shapes  = args.get("input_type_shape")  or []
                out_shapes = args.get("output_type_shape") or []

                for t in in_shapes:
                    sh = t.get("float")
                    if isinstance(sh, list) and len(sh) == 4:
                        _, C_in, _, _ = sh
                        cin = C_in
                        break
                for t in out_shapes:
                    sh = t.get("float")
                    if isinstance(sh, list) and len(sh) == 4:
                        _, C_out, _, _ = sh
                        cout = C_out
                        break

                if cin is not None and cout is not None:
                    X_cin.append(feats);  Y_cin.append(f"CIN_{cin}")
                    meta_cin.append((ridx, layer_idx, L["name"], cin, cout))
                    X_cout.append(feats); Y_cout.append(f"COUT_{cout}")
                    meta_cout.append((ridx, layer_idx, L["name"], cin, cout))

    return full, X_all, Y_all, meta_all, \
           X_sh, Y_sh, meta_sh, \
           X_cin, Y_cin, meta_cin, \
           X_cout, Y_cout, meta_cout, med_dt



# ======= Feature extraction for one pair (shared by train & test) =======
def extract_features_for_pair(json_path, spike_path):
    """
    Returns:
      full_runs : list of (run_idx, (start, end))
      X_all, Y_all, meta_all
      X_sh,  Y_sh,  meta_sh
      X_cin, Y_cin, meta_cin
      X_cout,Y_cout,meta_cout
      med_dt
    """
    events = load_events(json_path)
    spikes = parse_spikes(spike_path)
    if not spikes:
        print(f"[WARN] No spikes found in {spike_path}; skipping.")
        return [], [], [], [], [], [], [], [], [], [], [], 0.0

    ts_all=[r["ts"] for r in spikes]
    med_dt = median_cadence_us(spikes)

    runs = detect_runs(events)
    if not runs:
        print(f"[WARN] No runs detected in {json_path}; skipping.")
        return [], [], [], [], [], [], [], [], [], [], [], med_dt

    full=[]
    for idx,(rs,re) in enumerate(runs):
        if rs<ts_all[0] or re>ts_all[-1]:
            continue
        i,j = slice_spikes(spikes, ts_all, rs, re)
        samples = j-i
        dur_us  = re-rs
        cov = min(1.0, (samples*med_dt)/dur_us if (dur_us>0 and med_dt>0) else 0.0)
        if cov >= COVERAGE_THRESHOLD:
            full.append((idx,(rs,re)))

    X_all=[]; Y_all=[]; meta_all=[]
    X_sh=[];  Y_sh=[];  meta_sh=[]
    X_cin=[]; Y_cin=[]; meta_cin=[]
    X_cout=[];Y_cout=[];meta_cout=[]

    for ridx,(rs,re) in full:
        i_run, j_run = slice_spikes(spikes, ts_all, rs, re)
        thr_adapt = adapt_thresholds_per_run(spikes[i_run:j_run], ADAPT_K)
        thr_fixed = [FIXED_TICKS]*len(HOT_SETS)

        layers = rows_in_window(events, rs, re, ONLY_KERNEL_ROWS, ONLY_CPU_PROVIDER)
        for layer_idx, L in enumerate(layers):
            i,j = slice_spikes(spikes, ts_all, L["start"], L["end"])
            rows = spikes[i:j]
            samples = len(rows)
            dur_us = L["dur"]; dur_ms = dur_us/1000.0

            any_fixed=0; any_adapt=0
            per_set_cnt=[0]*len(HOT_SETS)
            per_set_sum=[0.0]*len(HOT_SETS)
            max_ticks=[]

            for r in rows:
                max_v = max(r["sets"])
                max_ticks.append(max_v)
                af=False; aa=False
                for k,v in enumerate(r["sets"]):
                    per_set_sum[k] += v
                    if v >= thr_fixed[k]:
                        af=True; per_set_cnt[k]+=1
                    if v >= thr_adapt[k]:
                        aa=True
                if af: any_fixed+=1
                if aa: any_adapt+=1

            density_fixed = (any_fixed / samples) if samples>0 else 0.0
            density_adapt = (any_adapt / samples) if samples>0 else 0.0
            rate_ms       = (any_fixed / dur_ms) if dur_ms>0 else 0.0

            if max_ticks:
                max_ticks.sort()
                p95 = max_ticks[int(0.95*(len(max_ticks)-1))]
                mean_max = sum(max_ticks)/len(max_ticks)
                if len(max_ticks) > 1:
                    m = mean_max
                    var_max = sum((v-m)*(v-m) for v in max_ticks)/ (len(max_ticks)-1)
                    std_max = math.sqrt(var_max)
                else:
                    std_max = 0.0
            else:
                p95 = 0
                mean_max = 0.0
                std_max  = 0.0

            p = [c/samples for c in per_set_cnt] if samples>0 else [0.0]*len(HOT_SETS)
            mean_set = [s/samples for s in per_set_sum] if samples>0 else [0.0]*len(HOT_SETS)
            ent = entropy(p)
            conc = gini_concentration(mean_set)
            mid = (L["start"]+L["end"])/2.0
            run_pos = (mid - rs)/max(1.0,(re-rs))

            feats = [
                math.log10(max(dur_us,1)),
                density_fixed,
                density_adapt,
                rate_ms,
                p95,
                ent,
                mean_max,
                std_max,
                conc,
            ] + p + mean_set + [run_pos]



            lbl = label_from_op(L["op_type"], L["name"])
            if not skip_type_layer(L["op_type"], L["name"]):
                X_all.append(feats)
                Y_all.append(lbl)
                meta_all.append((ridx, layer_idx, L["name"], L["op_type"], L["dur"]))

            shape_lbl = kernel_shape_from_name(L["op_type"], L["name"])
            # if shape_lbl in ("K1","K3","K3DW","GAP","FC","K?"):
            # if shape_lbl in ("K1","K1PW","K3","K3DW","GAP","FC","K?"):
            if shape_lbl in ("K7", "K3", "K1", "MAXPOOL", "GAP", "FLATTEN", "FC"):
                X_sh.append(feats)
                Y_sh.append(shape_lbl)
                #meta_sh.append((ridx, layer_idx, L["name"], L["op_type"], L["dur"]))
                meta_sh.append((ridx, layer_idx, L["name"], L["op_type"], L["dur"], L["start"], L["end"]))


            # --- Cin / Cout extraction: only for convs (K1/K3/K3DW) ---
           
            if shape_lbl in ("K7", "K3", "K1"):
                cin = cout = None
                args = L.get("args", {}) or {}
                in_shapes  = args.get("input_type_shape")  or []
                out_shapes = args.get("output_type_shape") or []

                # activation input: [N, Cin, H, W]
                for t in in_shapes:
                    sh = t.get("float")
                    if isinstance(sh, list) and len(sh) == 4:
                        _, C_in, _, _ = sh
                        cin = C_in
                        break

                # activation output: [N, Cout, H, W]
                for t in out_shapes:
                    sh = t.get("float")
                    if isinstance(sh, list) and len(sh) == 4:
                        _, C_out, _, _ = sh
                        cout = C_out
                        break

                if cin is not None and cout is not None:
                    X_cin.append(feats)
                    Y_cin.append(f"CIN_{cin}")
                    meta_cin.append((ridx, layer_idx, L["name"], cin, cout))

                    X_cout.append(feats)
                    Y_cout.append(f"COUT_{cout}")
                    meta_cout.append((ridx, layer_idx, L["name"], cin, cout))

    return full, X_all, Y_all, meta_all, \
           X_sh, Y_sh, meta_sh, \
           X_cin, Y_cin, meta_cin, \
           X_cout, Y_cout, meta_cout, med_dt






def summarize_shape_feature_ranges(X_sh, Y_sh):
    """
    Print per-class statistics for the main features used by the SHAPE head.

    We summarize the interpretable scalar features:
      0 : log10(dur_us)
      1 : density_fixed
      2 : density_adapt
      3 : rate_ms
      4 : p95
      5 : entropy
      6 : mean_max
      7 : std_max
      8 : concentration
      last : run_pos (position of the layer within the run)
    """
    if not X_sh:
        print("[SHAPE FEATS] No training samples.")
        return

    dim = len(X_sh[0])
    idx_run_pos = dim - 1  # we appended run_pos at the very end

    feature_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, idx_run_pos]
    feature_names   = [
        "log10_dur_us",
        "dens_fixed",
        "dens_adapt",
        "rate_ms",
        "p95_ticks",
        "entropy",
        "mean_max",
        "std_max",
        "concentration",
        "run_pos",
    ]

    by_class = defaultdict(list)
    for x, y in zip(X_sh, Y_sh):
        by_class[y].append(x)

    print("\n[SHAPE FEATS] Per-class feature stats (from TRAIN data):")
    for cls in sorted(by_class.keys()):
        rows = by_class[cls]
        print(f"\n  Class {cls}  (N={len(rows)})")
        for fi, fname in zip(feature_indices, feature_names):
            vals = [r[fi] for r in rows]
            if not vals:
                continue
            mean_val = sum(vals) / len(vals)
            if len(vals) > 1:
                std_val = statistics.stdev(vals)
            else:
                std_val = 0.0
           
     
            print(f"    {fname:14s} mean±std = {mean_val:7.4f} ± {std_val:7.4f}")


def plot_shape_pca(X_sh, Y_sh, mu_sh=None, sd_sh=None,
                   out_path="resnet18v1_shape_pca_3d.png"):
    """
    3D PCA visualization of kernel-shape training features,
    using ONLY the 10 interpretable scalar features:

      0 : log10(dur_us)
      1 : density_fixed
      2 : density_adapt
      3 : rate_ms
      4 : p95
      5 : entropy
      6 : mean_max
      7 : std_max
      8 : concentration
      last : run_pos

    This usually separates K1 / K3 / K3DW / GAP / FC much better than
    using the full high-dimensional vector (which is dominated by
    per-set means and counts).
    """
    if not X_sh:
        print("[PCA] No SHAPE samples; skipping PCA plot.")
        return

    # Build a matrix with only the 10 scalar features
    dim = len(X_sh[0])
    idx_run_pos = dim - 1  # run_pos appended at the end
    scalar_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, idx_run_pos]

    X_scalar = [[x[i] for i in scalar_indices] for x in X_sh]

    # Standardize these 10-D features
    mu, sd = standardize_fit(X_scalar)
    XZ = standardize_apply(X_scalar, mu, sd)

    # PCA to 3 components
    pca = PCA(n_components=3, random_state=RNG_SEED)
    X_emb = pca.fit_transform(XZ)
    
    # ---- Outlier removal for visualization only ----
    # Compute Mahalanobis distance in PCA space
    mean_emb = X_emb.mean(axis=0)
    cov_emb = np.cov(X_emb, rowvar=False)
    inv_cov_emb = np.linalg.inv(cov_emb)

    diff = X_emb - mean_emb
    mahal = np.einsum("ij,jk,ik->i", diff, inv_cov_emb, diff)

    # Threshold: keep points within percentile (e.g., 97%)
    thr = np.percentile(mahal, 100)
    keep_mask = mahal <= thr

    X_emb_plot = X_emb[keep_mask]
    Y_sh_plot = [y for i, y in enumerate(Y_sh) if keep_mask[i]]


    # map internal labels to human-readable labels
    label_mapping = {
        "K7": "7x7 Conv",
        "K3": "3x3 Conv",
        "K1": "1x1 Conv",
        "MAXPOOL": "MaxPool",
        "GAP": "Global AvgPool",
        "FLATTEN": "Flatten",
        "FC": "FC",
    }
    
    # Color / marker per kernel shape
    colors = {
        "K7": "tab:cyan",
        "K3": "tab:orange",
        "K1": "tab:blue",
        "MAXPOOL": "tab:green",
        "GAP": "tab:purple",
        "FLATTEN": "tab:gray",
        "FC": "tab:red",
    }
    markers = {
        "K7": "P",
        "K3": "^",
        "K1": "o",
        "MAXPOOL": "v",
        "GAP": "D",
        "FLATTEN": "x",
        "FC": "s",
    }

    labels = sorted(set(Y_sh_plot))

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    for lab in labels:
        idxs = [i for i, y in enumerate(Y_sh_plot) if y == lab]
        if not idxs:
            continue
        pts = X_emb_plot[idxs]
        display_label = label_mapping.get(lab, lab)  # map to human-readable
        
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            label=display_label,
            s=20,
            alpha=0.7,
            marker=markers.get(lab, "o"),
            c=colors.get(lab, None),
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Kernel-shape feature space (PCA 3D, scalar features only)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    var = pca.explained_variance_ratio_
    print(f"[PCA] Saved 3D kernel-shape PCA plot to {out_path}")
    print(f"[PCA] Explained variance: PC1={var[0]:.3f}, PC2={var[1]:.3f}, PC3={var[2]:.3f}")




def plot_shape_umap(
    X_sh,
    Y_sh,
    out_path="resnet18v1_shape_umap_2d.png",
    scalar_only=True,
    use_pca=True,
    pca_dim=10,
):
    """
    2D *supervised* UMAP visualization of kernel-shape training features.

    - By default uses only the 10 interpretable scalar features:
        0 : log10(dur_us)
        1 : density_fixed
        2 : density_adapt
        3 : rate_ms
        4 : p95
        5 : entropy
        6 : mean_max
        7 : std_max
        8 : concentration
        last : run_pos
    - First applies PCA to 'pca_dim' dimensions (if use_pca=True).
    - Then runs UMAP with labels (Y_sh) as supervision, so clusters
      are encouraged to be label-pure and visually separated.
    """
    if not X_sh:
        print("[UMAP] No SHAPE samples; skipping UMAP plot.")
        return

    # ---- 1) select features ----
    if scalar_only:
        dim = len(X_sh[0])
        idx_run_pos = dim - 1
        scalar_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, idx_run_pos]
        X = np.array([[x[i] for i in scalar_indices] for x in X_sh], dtype=float)
    else:
        X = np.array(X_sh, dtype=float)

    # ---- 2) standardize ----
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0.0] = 1.0
    Z = (X - mu) / sd

    # ---- 3)  PCA pre-step ----
    if use_pca and Z.shape[1] > pca_dim:
        k = min(pca_dim, Z.shape[1])
        pca = PCA(n_components=k, random_state=RNG_SEED)
        Z_red = pca.fit_transform(Z)
        print(f"[UMAP] PCA pre-step: reduced {Z.shape[1]}D → {k}D "
              f"(explained var sum={pca.explained_variance_ratio_.sum():.3f})")
    else:
        Z_red = Z

    # ---- 4) supervised UMAP ----
    # y_raw: string labels like "K1", "K3DW", ...
    y_raw = np.array(Y_sh)

    # Encode labels as integers for UMAP
    unique_labels = sorted(set(Y_sh))
    label2id = {lab: i for i, lab in enumerate(unique_labels)}
    y_enc = np.array([label2id[lab] for lab in Y_sh], dtype=np.int32)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=20,
        min_dist=0.05,
        metric="euclidean",
        target_metric="categorical",
        target_weight=0.8,
        random_state=RNG_SEED,
    )
    emb = reducer.fit_transform(Z_red, y=y_enc)
    # map internal labels to human-readable labels
    label_mapping = {
        "K7": "7x7 Conv",
        "K3": "3x3 Conv",
        "K1": "1x1 Conv",
        "MAXPOOL": "MaxPool",
        "GAP": "Global AvgPool",
        "FLATTEN": "Flatten",
        "FC": "FC",
    }

    # ---- 5) plot ----
    colors = {
        "K7": "tab:cyan",
        "K3": "tab:orange",
        "K1": "tab:blue",
        "MAXPOOL": "tab:green",
        "GAP": "tab:purple",
        "FLATTEN": "tab:gray",
        "FC": "tab:red",
    }
    markers = {
        "K7": "P",
        "K3": "^",
        "K1": "o",
        "MAXPOOL": "v",
        "GAP": "D",
        "FLATTEN": "x",
        "FC": "s",
    }

    labels = sorted(set(Y_sh))

    fig, ax = plt.subplots(figsize=(6, 5))
    for lab in labels:
        idxs = np.where(y_raw == lab)[0]
        if len(idxs) == 0:
            continue
        pts = emb[idxs]
        display_label = label_mapping.get(lab, lab)  # map to human-readable
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            label=display_label,
            s=12,
            alpha=0.8,
            marker=markers.get(lab, "o"),
            c=colors.get(lab, None),
        )


    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("Kernel-shape feature space (UMAP 2D)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"[UMAP] Saved 2D supervised UMAP plot to {out_path}")




import os

def plot_raw_latency_heatmaps_by_shape(
    test_json,
    test_spike,
    full_runs,
    meta_te_sh,
    y_labels_sh,
    out_dir="raw_shape_latency_plots",
    max_layers_per_shape=8,
    pick_from_first_k_runs=20,
    max_time_rows=250,
    time_downsample=1,
):
    """
    For each kernel-shape class, save a few example heatmaps of raw per-set latencies
    within the exact ONNX layer window.

    Heatmap:
      x-axis: HOT_SETS (cache sets)
      y-axis: time samples (prime+probe steps)
      value : latency ticks
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Human-readable kernel-shape labels
    SHAPE_LABELS = {
        "FC": "FC layer",
        "GAP": "global average pooling",
        "MAXPOOL": "max pooling",
        "FLATTEN": "flatten",
        "K1": "1×1 Conv",
        "K3": "3×3 Conv",
        "K7": "7×7 Conv",
    }

    # Load spikes once
    spikes = parse_spikes(test_spike)
    if not spikes:
        print("[RAW VIS] No spikes; skip raw visualization.")
        return
    ts_list = [r["ts"] for r in spikes]

    
    chosen_runs = set(r for r, _ in full_runs[:max(1, pick_from_first_k_runs)])

    # Group layer-instances by shape label
    by_shape = defaultdict(list)
    for m, y in zip(meta_te_sh, y_labels_sh):
        # meta_te_sh: (ridx, layer_idx, name, op_type, dur, start, end)
        ridx = m[0]
        if ridx not in chosen_runs:
            continue
        by_shape[y].append(m)

    # For each shape, pick a few layer windows and plot
    for shape in sorted(by_shape.keys()):
        candidates = by_shape[shape]
        if not candidates:
            continue

        # Pick up to max_layers_per_shape examples (prefer earlier layer_idx)
        candidates = sorted(candidates, key=lambda t: (t[0], t[1]))[:max_layers_per_shape]

        for ex_i, m in enumerate(candidates, start=1):
            ridx, layer_idx, name, op_type, dur, start_us, end_us = m

            i, j = slice_spikes(spikes, ts_list, start_us, end_us)
            rows = spikes[i:j]
            if not rows:
                continue

            # Build matrix [T x S]
            M = np.array([r["sets"] for r in rows], dtype=float)  # T x |HOT_SETS|

            
            if time_downsample > 1 and M.shape[0] > time_downsample:
                M = M[::time_downsample, :]

           
            if max_time_rows and M.shape[0] > max_time_rows:
                M = M[:max_time_rows, :]

            fig, ax = plt.subplots(figsize=(7, 4))
            im = ax.imshow(M, aspect="auto")  # default colormap is fine

            #ax.set_title(f"{shape} raw latencies (run={ridx}, layer={layer_idx})")
            shape_label = SHAPE_LABELS.get(shape, shape)
            
            ax.set_title(f"Raw cache contention latencies ({shape_label})")
            ax.set_xlabel("Selected cache set")
            ax.set_ylabel(f"Time samples (~{median_cadence_us(spikes):.1f} µs each)")

            # x ticks = actual cache set numbers
            ax.set_xticks(range(len(HOT_SETS)))
            ax.set_xticklabels([str(s) for s in HOT_SETS], rotation=45, ha="right", fontsize=8)

            # Add a colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Latency (ticks)")

            fig.tight_layout()

            safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)[:120]
            out_path = os.path.join(out_dir, f"{shape}_ex{ex_i}_run{ridx}_layer{layer_idx}_{safe_name}.png")
            fig.savefig(out_path, dpi=300)
            plt.close(fig)

    print(f"[RAW VIS] Saved raw kernel-shape heatmaps under: {out_dir}")



def resample_time_axis(M, L=200):
    """
    Resample a [T x S] matrix to [L x S] using linear interpolation along time axis.
    Keeps cache-set axis intact.
    """
    T, S = M.shape
    if T == 0:
        return None
    if T == L:
        return M
    if T == 1:
        return np.repeat(M, L, axis=0)

    x_old = np.linspace(0.0, 1.0, T)
    x_new = np.linspace(0.0, 1.0, L)

    out = np.empty((L, S), dtype=float)
    for s in range(S):
        out[:, s] = np.interp(x_new, x_old, M[:, s])
    return out


def plot_mean_signature_by_shape(
    test_spike,
    full_runs,
    meta_te_sh,
    y_labels_sh,
    out_dir="raw_shape_mean_signatures",
    first_k_runs=1,
    L=200,
    per_shape_max_windows=300,
    use_gt_labels=True,
    robust_clip=True,
):
    """
    Build a single compact 'mean signature' per kernel-shape.

    Steps:
      1) Collect layer windows for each shape from the first_k_runs FULL runs.
      2) Slice raw spikes inside each window -> matrix M [T x S]
      3) Resample M -> [L x S]
      4) Average across windows -> mean_signature [L x S]
      5) Save one heatmap per shape.

    This complements (does not replace) the per-example heatmaps.
    """
    os.makedirs(out_dir, exist_ok=True)

    spikes = parse_spikes(test_spike)
    if not spikes:
        print("[MEAN VIS] No spikes; skip mean-signature plots.")
        return
    ts_list = [r["ts"] for r in spikes]

    # choose the first K FULL runs
    chosen = set(r for r, _ in full_runs[:max(1, first_k_runs)])
    if not chosen:
        print("[MEAN VIS] No chosen FULL runs; skip.")
        return

    # group windows by shape
    by_shape = defaultdict(list)
    # meta_te_sh tuple: (ridx, layer_idx, name, op_type, dur, start, end)
    for m, y in zip(meta_te_sh, y_labels_sh):
        ridx = m[0]
        if ridx not in chosen:
            continue
        by_shape[y].append(m)

    # compute mean signature per shape
    for shape in sorted(by_shape.keys()):
        windows = by_shape[shape]
        if not windows:
            continue

        # limit windows to keep runtime bounded
        windows = sorted(windows, key=lambda t: (t[0], t[1]))[:per_shape_max_windows]

        acc = []
        for (ridx, layer_idx, name, op_type, dur, start_us, end_us) in windows:
            i, j = slice_spikes(spikes, ts_list, start_us, end_us)
            rows = spikes[i:j]
            if not rows:
                continue

            M = np.array([r["sets"] for r in rows], dtype=float)  # [T x S]
            M_rs = resample_time_axis(M, L=L)
            if M_rs is None:
                continue
            acc.append(M_rs)

        if len(acc) < 5:
            print(f"[MEAN VIS] shape={shape}: too few windows ({len(acc)}); skipping.")
            continue

        A = np.stack(acc, axis=0)     # [N x L x S]
        mean_sig = A.mean(axis=0)     # [L x S]

        
        if robust_clip:
            lo = np.percentile(mean_sig, 5)
            hi = np.percentile(mean_sig, 95)
            mean_sig = np.clip(mean_sig, lo, hi)

        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.imshow(mean_sig, aspect="auto")

        ax.set_title(f"Mean latency signature: {shape}  (runs={first_k_runs}, windows={len(acc)})")
        ax.set_xlabel("Cache set")
        ax.set_ylabel(f"Normalized time (L={L} samples)")

        ax.set_xticks(range(len(HOT_SETS)))
        ax.set_xticklabels([str(s) for s in HOT_SETS], rotation=45, ha="right", fontsize=8)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Latency (ticks)")
        fig.tight_layout()

        out_path = os.path.join(out_dir, f"mean_{shape}_runs{first_k_runs}_L{L}.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    print(f"[MEAN VIS] Saved mean-signature plots under: {out_dir}")





# ======= TRAIN mode =======
def run_train_mode():
    if not TRAIN_PAIRS:
        print("Please populate TRAIN_PAIRS for TRAIN mode.")
        return

    train_pairs = list(TRAIN_PAIRS)

    # Decide which pair to use for testing during TRAIN
    if TEST_PAIR is not None:
        test_json, test_spike = TEST_PAIR
    else:
        if len(train_pairs) < 2:
            print("Need at least 2 TRAIN_PAIRS or set TEST_PAIR explicitly.")
            return
        test_json, test_spike = train_pairs[-1]
        train_pairs = train_pairs[:-1]

    print("[TRAIN] Training on the following pairs:")
    for j,s in train_pairs:
        print("  ", j, " / ", s)
    print("[TRAIN] Validation/Test on:", test_json, "/", test_spike)

    random.seed(RNG_SEED)

    # ----- build training data -----
    Xtr=[]; Ytr=[]
    Xtr_sh=[]; Ytr_sh=[]
    Xtr_cin=[]; Ytr_cin=[]
    Xtr_cout=[];Ytr_cout=[]

    for json_path, spike_path in train_pairs:
        (full, X_all, Y_all, meta_all,
         X_sh, Y_sh, meta_sh,
         X_cin, Y_cin, meta_cin,
         X_cout, Y_cout, meta_cout,
         med_dt) = extract_features_for_pair(json_path, spike_path)

        if not full:
            print(f"[WARN] No FULL runs in {json_path}; skipping.")
            continue

        Xtr.extend(X_all);   Ytr.extend(Y_all)
        Xtr_sh.extend(X_sh); Ytr_sh.extend(Y_sh)
        Xtr_cin.extend(X_cin);   Ytr_cin.extend(Y_cin)
        Xtr_cout.extend(X_cout); Ytr_cout.extend(Y_cout)

    if not Xtr:
        print("No training samples; aborting.")
        return

    print(f"Total TRAIN layers (type head):  {len(Xtr)}")
    print(f"Total TRAIN layers (shape head): {len(Xtr_sh)}")
    print(f"Total TRAIN convs for Cin:       {len(Xtr_cin)}")
    print(f"Total TRAIN convs for Cout:      {len(Xtr_cout)}")
    
    # Dump stats for kernel-shape features
    summarize_shape_feature_ranges(Xtr_sh, Ytr_sh)

    # # ----- TYPE head: GNB -----
    # mu_type, sd_type = standardize_fit(Xtr)
    # XtrZ_type = standardize_apply(Xtr, mu_type, sd_type)
    # clf_type = GNB()
    # clf_type.fit(XtrZ_type, Ytr)

    # ----- TYPE head: RandomForest -----
    mu_type, sd_type = standardize_fit(Xtr)
    XtrZ_type = standardize_apply(Xtr, mu_type, sd_type)

    clf_type = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RNG_SEED,
    )
    clf_type.fit(XtrZ_type, Ytr)
    print("[RF TYPE ] Class counts:", dict(Counter(Ytr)))

    # ----- SHAPE head: RandomForest -----
    clf_shape = None
    mu_sh = sd_sh = None
    clf_cin = None
    clf_cout = None

    if Xtr_sh:
        mu_sh, sd_sh = standardize_fit(Xtr_sh)
        XtrZ_sh    = standardize_apply(Xtr_sh,    mu_sh, sd_sh)
        XtrZ_cin   = standardize_apply(Xtr_cin,   mu_sh, sd_sh) if Xtr_cin  else []
        XtrZ_cout  = standardize_apply(Xtr_cout,  mu_sh, sd_sh) if Xtr_cout else []

        clf_shape = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RNG_SEED,
        )
        clf_shape.fit(XtrZ_sh, Ytr_sh)
        print("[RF SHAPE] Class counts:", dict(Counter(Ytr_sh)))
        
        # NEW: PCA visualization of SHAPE features
        plot_shape_pca(Xtr_sh, Ytr_sh, mu_sh, sd_sh,
                       out_path="resnet18v1_shape_pca_3d.png")
        
        # NEW: UMAP 2D plot for kernel-shape features
        plot_shape_umap(Xtr_sh, Ytr_sh,
                        out_path="resnet18v1_shape_umap_2d.png",
                        scalar_only=True)

        if Xtr_cin:
            clf_cin = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RNG_SEED,
            )
            clf_cin.fit(XtrZ_cin, Ytr_cin)
            print("[RF CIN ] Class counts:", dict(Counter(Ytr_cin)))

        if Xtr_cout:
            clf_cout = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RNG_SEED,
            )
            clf_cout.fit(XtrZ_cout, Ytr_cout)
            print("[RF COUT] Class counts:", dict(Counter(Ytr_cout)))

    
    # ----- NEW: learn layer-duration profile for spike-only segmentation -----
    dur_profile = build_duration_profile_from_training(train_pairs)
    if dur_profile is None:
        print("[DUR PROF] Could not build duration profile (no FULL runs?).")
    else:
        print(f"[DUR PROF] Learned duration profile for {len(dur_profile['dur_profile_us'])} layers "
              f"(sum={sum(dur_profile['dur_profile_us'])/1000.0:.1f} ms)")

    # ----- Save models to disk -----
    dump({
        "clf_type":  clf_type,
        "mu_type":   mu_type,
        "sd_type":   sd_type,
        "clf_shape": clf_shape,
        "mu_sh":     mu_sh,
        "sd_sh":     sd_sh,
        "clf_cin":   clf_cin,
        "clf_cout":  clf_cout,
        "dur_profile": dur_profile,
    }, MODEL_PATH)
    print(f"\n[TRAIN] Saved models to: {MODEL_PATH}")

    # ----- Also evaluate on held-out pair -----
    print("\n[TRAIN] Now evaluating on held-out pair...")
    run_test_on_pair(test_json, test_spike,
                     clf_type, mu_type, sd_type,
                     clf_shape, mu_sh, sd_sh,
                     clf_cin, clf_cout)

def per_sample_acc_for_runs(meta, y_true, y_pred, allowed_runs):
    """Per-sample accuracy restricted to a subset of run indices."""
    idxs = [i for i, m in enumerate(meta) if m[0] in allowed_runs]
    if not idxs:
        return 0.0
    correct = sum(1 for i in idxs if y_true[i] == y_pred[i])
    return correct / len(idxs)


def majority_acc_for_runs(meta, y_true, y_pred, allowed_runs):
    """
    Per-layer majority-vote accuracy restricted to a subset of run indices.
    Layers are grouped by 'name' (3rd element in meta tuples).
    """
    from collections import defaultdict, Counter

    by_layer = defaultdict(list)
    for i, m in enumerate(meta):
        ridx = m[0]
        if ridx not in allowed_runs:
            continue
        name = m[2]   # (ridx, layer_idx, name, ...)
        by_layer[name].append(i)

    if not by_layer:
        return 0.0, 0, 0

    correct = 0
    total = 0
    for name, idxs in by_layer.items():
        gt = y_true[idxs[0]]
        preds = [y_pred[i] for i in idxs]
        maj = Counter(preds).most_common(1)[0][0]
        if maj == gt:
            correct += 1
        total += 1

    return correct / total, correct, total






def summarize_reconstructed_layers(
    meta_te,      Yte,      Yhat_type,
    meta_te_sh,   Yte_sh,   Yhat_sh,
    meta_te_cin,  Yte_cin,  Yhat_cin,
    meta_te_cout, Yte_cout, Yhat_cout,
):
    """
    Print a final per-layer reconstruction table:
      idx, GT type/shape/Cin/Cout, Pred type/shape/Cin/Cout, layer name.

    All predictions are majority-voted across all FULL runs.
    """

    if Yhat_type is None or Yhat_sh is None:
        print("\n[RECON] Missing type/shape predictions; skip reconstruction.")
        return

    # --- 1) basic maps from layer name -> position / op_type (for ordering) ---
    pos_by_name = {}
    optype_by_name = {}
    #for ridx, layer_idx, name, op_type, dur in meta_te:
    for m in meta_te:
        ridx, layer_idx, name, op_type, dur = m[:5]
        if name not in pos_by_name or layer_idx < pos_by_name[name]:
            pos_by_name[name] = layer_idx
            optype_by_name[name] = op_type

    # container for all per-layer info
    layers = {}

    def ensure_entry(name):
        if name not in layers:
            layers[name] = {
                "pos": pos_by_name.get(name, 9999),
                "op":  optype_by_name.get(name, "?"),
                "gt_type": "-",
                "pr_type": "-",
                "gt_shape": "-",
                "pr_shape": "-",
                "gt_cin": "-",
                "pr_cin": "-",
                "gt_cout": "-",
                "pr_cout": "-",
            }
        return layers[name]

    # --- 2) TYPE majority vote ---
    by_layer_type = defaultdict(list)
    #for idx, (ridx, layer_idx, name, op_type, dur) in enumerate(meta_te):
    for idx, m in enumerate(meta_te):
        ridx, layer_idx, name, op_type, dur = m[:5]

        by_layer_type[name].append(idx)

    for name, idxs in by_layer_type.items():
        gt = Yte[idxs[0]]
        preds = [Yhat_type[i] for i in idxs]
        maj = Counter(preds).most_common(1)[0][0]
        ent = ensure_entry(name)
        ent["gt_type"] = gt
        ent["pr_type"] = maj

    # --- 3) SHAPE majority vote ---
    by_layer_sh = defaultdict(list)
    for idx, m in enumerate(meta_te_sh):
        ridx, layer_idx, name, op_type, dur = m[:5]

        by_layer_sh[name].append(idx)

    for name, idxs in by_layer_sh.items():
        gt = Yte_sh[idxs[0]]
        preds = [Yhat_sh[i] for i in idxs]
        maj = Counter(preds).most_common(1)[0][0]
        ent = ensure_entry(name)
        ent["gt_shape"] = gt
        ent["pr_shape"] = maj

    # --- 4) CIN majority vote (conv-only) ---
    if Yhat_cin is not None and meta_te_cin:
        by_layer_cin = defaultdict(list)
        for idx, (ridx, layer_idx, name, cin_true, cout_true) in enumerate(meta_te_cin):
            by_layer_cin[name].append(idx)

        for name, idxs in by_layer_cin.items():
            gt = Yte_cin[idxs[0]]
            preds = [Yhat_cin[i] for i in idxs]
            maj = Counter(preds).most_common(1)[0][0]
            ent = ensure_entry(name)
            ent["gt_cin"] = gt
            ent["pr_cin"] = maj

    # --- 5) COUT majority vote (conv-only) ---
    if Yhat_cout is not None and meta_te_cout:
        by_layer_cout = defaultdict(list)
        for idx, (ridx, layer_idx, name, cin_true, cout_true) in enumerate(meta_te_cout):
            by_layer_cout[name].append(idx)

        for name, idxs in by_layer_cout.items():
            gt = Yte_cout[idxs[0]]
            preds = [Yhat_cout[i] for i in idxs]
            maj = Counter(preds).most_common(1)[0][0]
            ent = ensure_entry(name)
            ent["gt_cout"] = gt
            ent["pr_cout"] = maj

    # --- 6) nicely formatted table ---
    def strip_prefix(lbl, prefix):
        if lbl in ("-", None):
            return "-"
        if isinstance(lbl, str) and lbl.startswith(prefix):
            return lbl[len(prefix):]
        return lbl

    print("\n[RECON] Layer-by-layer ground truth vs predicted "
          "(majority over FULL runs):")
    header = ("idx", "T_gt", "T_pr", "K_gt", "K_pr",
              "Cin_gt", "Cin_pr", "Cout_gt", "Cout_pr", "name")
    print("{:>3} {:>5} {:>5} {:>5} {:>5} {:>7} {:>7} {:>8} {:>8}  {}".format(*header))

    for idx, (name, ent) in enumerate(
            sorted(layers.items(), key=lambda kv: kv[1]["pos"])):
        cin_gt  = strip_prefix(ent["gt_cin"],  "CIN_")
        cin_pr  = strip_prefix(ent["pr_cin"],  "CIN_")
        cout_gt = strip_prefix(ent["gt_cout"], "COUT_")
        cout_pr = strip_prefix(ent["pr_cout"], "COUT_")
        print("{:3d} {:>5} {:>5} {:>5} {:>5} {:>7} {:>7} {:>8} {:>8}  {}".format(
            idx,
            ent["gt_type"], ent["pr_type"],
            ent["gt_shape"], ent["pr_shape"],
            cin_gt, cin_pr,
            cout_gt, cout_pr,
            name,
        ))






def run_test_on_pair_spike_only(test_json, test_spike,
                clf_type, mu_type, sd_type,
                clf_shape, mu_sh, sd_sh,
                clf_cin, clf_cout,
                dur_profile_us):
    (full, Xte, Yte, meta_te,
     Xte_sh, Yte_sh, meta_te_sh,
     Xte_cin, Yte_cin, meta_te_cin,
     Xte_cout, Yte_cout, meta_te_cout,
     med_dt) = extract_features_for_pair_spike_only(test_json, test_spike, dur_profile_us)

    if not full:
        print("No FULL runs in TEST file; aborting evaluation.")
        return


    print(f"[TEST] Median cadence ≈ {med_dt:.1f} µs")
    print("[TEST] FULL runs:", [i for i,_ in full])
    
    # placeholders so we can summarize at the end
    Yhat_type = None
    Yhat_sh   = None
    Yhat_cin  = None
    Yhat_cout = None

    # ----- TYPE head -----
    XteZ_type = standardize_apply(Xte, mu_type, sd_type)
    Yhat_type = clf_type.predict(XteZ_type)

    cm = defaultdict(int)
    correct=0
    for g,p in zip(Yte, Yhat_type):
        cm[(g,p)] += 1
        if g==p: correct+=1
    acc = correct/len(Yte) if Yte else 0.0
    labels_type = sorted(set(Yte))
    print(f"\n[TYPE] Test set across {len(full)} FULL runs, layers={len(Yte)}, accuracy={acc*100:.1f}%")
    print("Confusion (GT → Pred):")
    for g in labels_type:
        row=[cm[(g,p)] for p in labels_type]
        print(f"  {g:>3}: " + " ".join(f"{v:3d}" for v in row) + f"   | sum={sum(row)}")

    def short(lbl: str) -> str:
        if lbl in ("C","DW","FC","P","BN","F","A","N","O"): return lbl
        return "?"
    first_run = min(r for r,_ in full)
    seq_gt=[]; seq_pred=[]
    #for (ridx, layer_idx, name, op_type, dur), g, p in zip(meta_te, Yte, Yhat_type):
    for m, g, p in zip(meta_te, Yte, Yhat_type):
        ridx, layer_idx, name, op_type, dur = m[:5]
        if ridx == first_run:
            seq_gt.append(short(g))
            seq_pred.append(short(p))
    if seq_gt:
        print("\n[TYPE] First FULL run layer-type sequence:")
        print("   GT:", "".join(seq_gt))
        print("  PRD:", "".join(seq_pred))

    # ----- SHAPE head -----
    if clf_shape is None or not Xte_sh or mu_sh is None:
        print("\n[SHAPE] Not enough shape-labeled samples or no trained shape model.")
        return

    XteZ_sh = standardize_apply(Xte_sh, mu_sh, sd_sh)
    Yhat_sh = clf_shape.predict(XteZ_sh)

    lab_sh = sorted(set(Yte_sh))
    cm_sh = defaultdict(int); correct_sh=0
    for g,p in zip(Yte_sh, Yhat_sh):
        cm_sh[(g,p)] += 1
        if g==p: correct_sh+=1
    acc_sh = correct_sh/len(Yte_sh) if Yte_sh else 0.0
    print(f"\n[SHAPE] Per-sample kernel-shape accuracy across {len(full)} FULL runs: {acc_sh*100:.1f}%")
    print("Confusion (GT → Pred):")
    for g in lab_sh:
        row=[cm_sh[(g,p)] for p in lab_sh]
        print(f"  {g:>4}: " + " ".join(f"{v:3d}" for v in row) + f"  | sum={sum(row)}")

    by_layer_sh = defaultdict(list)
    #for idx,(ridx, layer_idx, name, op_type, dur) in enumerate(meta_te_sh):
    for idx, m in enumerate(meta_te_sh):
        ridx, layer_idx, name, op_type, dur = m[:5]
        # start_us, end_us = m[5], m[6]   #  if we need them here later

        by_layer_sh[name].append(idx)

    correct_mv_sh=0; total_layers_sh=0
    for name, idxs in by_layer_sh.items():
        gt = Yte_sh[idxs[0]]
        preds = [Yhat_sh[i] for i in idxs]
        maj = Counter(preds).most_common(1)[0][0]
        if maj==gt:
            correct_mv_sh+=1
        total_layers_sh+=1
    acc_mv_sh = correct_mv_sh/total_layers_sh if total_layers_sh>0 else 0.0
    print(f"\n[SHAPE] Majority-vote kernel-shape accuracy over {total_layers_sh} unique layers "
          f"using {len(full)} FULL runs: {acc_mv_sh*100:.1f}%")
    
    
    # --- Raw visualization (GT windows from ONNX) ---
    plot_raw_latency_heatmaps_by_shape(
        test_json, test_spike,
        full_runs=full,
        meta_te_sh=meta_te_sh,
        y_labels_sh=Yte_sh,     # use GT labels to avoid “wrong bin” visuals
        out_dir="raw_shape_latency_plots",
        max_layers_per_shape=8,
        pick_from_first_k_runs=20,
        max_time_rows=250,
        time_downsample=1,
    )
    


    

    # ----- CIN head -----
    if clf_cin is not None and Xte_cin and mu_sh is not None:
        XteZ_cin = standardize_apply(Xte_cin, mu_sh, sd_sh)
        Yhat_cin = clf_cin.predict(XteZ_cin)

        lab_cin = sorted(set(Yte_cin))
        cm_cin = defaultdict(int); correct_cin=0
        for g,p in zip(Yte_cin, Yhat_cin):
            cm_cin[(g,p)] += 1
            if g==p: correct_cin+=1
        acc_cin = correct_cin/len(Yte_cin) if Yte_cin else 0.0
        print(f"\n[CIN ] Per-sample Cin accuracy across {len(full)} FULL runs: {acc_cin*100:.1f}%")
        print("Confusion (GT → Pred):")
        for g in lab_cin:
            row=[cm_cin[(g,p)] for p in lab_cin]
            print(f"  {g:>8}: " + " ".join(f"{v:3d}" for v in row) + f"  | sum={sum(row)}")

        by_layer_cin = defaultdict(list)
        for idx,(ridx, layer_idx, name, cin_true, cout_true) in enumerate(meta_te_cin):
            by_layer_cin[name].append(idx)
        correct_mv_cin=0; total_layers_cin=0
        for name, idxs in by_layer_cin.items():
            gt = Yte_cin[idxs[0]]
            preds = [Yhat_cin[i] for i in idxs]
            maj = Counter(preds).most_common(1)[0][0]
            if maj==gt:
                correct_mv_cin+=1
            total_layers_cin+=1
        acc_mv_cin = correct_mv_cin/total_layers_cin if total_layers_cin>0 else 0.0
        print(f"\n[CIN ] Majority-vote Cin accuracy over {total_layers_cin} conv layers "
              f"using {len(full)} FULL runs: {acc_mv_cin*100:.1f}%")
    else:
        print("\n[CIN ] No trained Cin model or no Cin samples in test.")

    # ----- COUT head -----
    if clf_cout is not None and Xte_cout and mu_sh is not None:
        XteZ_cout = standardize_apply(Xte_cout, mu_sh, sd_sh)
        Yhat_cout = clf_cout.predict(XteZ_cout)

        lab_cout = sorted(set(Yte_cout))
        cm_cout = defaultdict(int); correct_cout=0
        for g,p in zip(Yte_cout, Yhat_cout):
            cm_cout[(g,p)] += 1
            if g==p: correct_cout+=1
        acc_cout = correct_cout/len(Yte_cout) if Yte_cout else 0.0
        print(f"\n[COUT] Per-sample Cout accuracy across {len(full)} FULL runs: {acc_cout*100:.1f}%")
        print("Confusion (GT → Pred):")
        for g in lab_cout:
            row=[cm_cout[(g,p)] for p in lab_cout]
            print(f"  {g:>8}: " + " ".join(f"{v:3d}" for v in row) + f"  | sum={sum(row)}")

        by_layer_cout = defaultdict(list)
        for idx,(ridx, layer_idx, name, cin_true, cout_true) in enumerate(meta_te_cout):
            by_layer_cout[name].append(idx)
        correct_mv_cout=0; total_layers_cout=0
        for name, idxs in by_layer_cout.items():
            gt = Yte_cout[idxs[0]]
            preds = [Yhat_cout[i] for i in idxs]
            maj = Counter(preds).most_common(1)[0][0]
            if maj==gt:
                correct_mv_cout+=1
            total_layers_cout+=1
        acc_mv_cout = correct_mv_cout/total_layers_cout if total_layers_cout>0 else 0.0
        print(f"\n[COUT] Majority-vote Cout accuracy over {total_layers_cout} conv layers "
              f"using {len(full)} FULL runs: {acc_mv_cout*100:.1f}%")
        
        # ====== Ablation over number of FULL runs ======
        # For MobileNetV2 we want #runs in {1, 10, 20, 47}.
        # We automatically cap at len(full) (we can replace 30 by this), so we can reuse this for other models.
        ablation_runs = [1, 10, 20, 30]

        # Guard: if we didn't train shape/Cin/Cout heads, just skip.
        have_shape = (clf_shape is not None and Xte_sh and mu_sh is not None)
        have_cin   = (clf_cin   is not None and Xte_cin and mu_sh is not None)
        have_cout  = (clf_cout  is not None and Xte_cout and mu_sh is not None)

        print("\n[ABLATION] Impact of number of FULL runs on per-layer majority accuracy:")
        for k in ablation_runs:
            if k <= 0:
                continue
            if k > 30:
                continue

            # use the first k FULL runs
            first_k_runs = set(r for r, _ in full[:k])

            # TYPE head (per-layer majority)
            type_acc_k, _, _ = majority_acc_for_runs(meta_te, Yte, Yhat_type, first_k_runs)

            # SHAPE head (per-layer majority)
            if have_shape:
                shape_acc_k, _, _ = majority_acc_for_runs(meta_te_sh, Yte_sh, Yhat_sh, first_k_runs)
            else:
                shape_acc_k = float("nan")

            # CIN head (per-layer majority)
            if have_cin:
                cin_acc_k, _, _ = majority_acc_for_runs(meta_te_cin, Yte_cin, Yhat_cin, first_k_runs)
            else:
                cin_acc_k = float("nan")

            # COUT head (per-layer majority)
            if have_cout:
                cout_acc_k, _, _ = majority_acc_for_runs(meta_te_cout, Yte_cout, Yhat_cout, first_k_runs)
            else:
                cout_acc_k = float("nan")

            print(f"  [runs={k:2d}] "
                  f"Type={type_acc_k*100:5.1f}%  "
                  f"Shape={shape_acc_k*100:5.1f}%  "
                  f"Cin={cin_acc_k*100:5.1f}%  "
                  f"Cout={cout_acc_k*100:5.1f}%")

        # ---- final layer-by-layer reconstruction summary ----
        summarize_reconstructed_layers(
            meta_te,      Yte,      Yhat_type,
            meta_te_sh,   Yte_sh,   Yhat_sh,
            meta_te_cin,  Yte_cin,  Yhat_cin,
            meta_te_cout, Yte_cout, Yhat_cout,
        )
    else:
        print("\n[COUT] No trained Cout model or no Cout samples in test.")
        
    






# ======= TEST helper (used in both modes) =======
def run_test_on_pair(test_json, test_spike,
                     clf_type, mu_type, sd_type,
                     clf_shape, mu_sh, sd_sh,
                     clf_cin, clf_cout):
    (full, Xte, Yte, meta_te,
     Xte_sh, Yte_sh, meta_te_sh,
     Xte_cin, Yte_cin, meta_te_cin,
     Xte_cout, Yte_cout, meta_te_cout,
     med_dt) = extract_features_for_pair(test_json, test_spike)

    if not full:
        print("No FULL runs in TEST file; aborting evaluation.")
        return


    print(f"[TEST] Median cadence ≈ {med_dt:.1f} µs")
    print("[TEST] FULL runs:", [i for i,_ in full])
    
    # placeholders so we can summarize at the end
    Yhat_type = None
    Yhat_sh   = None
    Yhat_cin  = None
    Yhat_cout = None

    # ----- TYPE head -----
    XteZ_type = standardize_apply(Xte, mu_type, sd_type)
    Yhat_type = clf_type.predict(XteZ_type)

    cm = defaultdict(int)
    correct=0
    for g,p in zip(Yte, Yhat_type):
        cm[(g,p)] += 1
        if g==p: correct+=1
    acc = correct/len(Yte) if Yte else 0.0
    labels_type = sorted(set(Yte))
    print(f"\n[TYPE] Test set across {len(full)} FULL runs, layers={len(Yte)}, accuracy={acc*100:.1f}%")
    print("Confusion (GT → Pred):")
    for g in labels_type:
        row=[cm[(g,p)] for p in labels_type]
        print(f"  {g:>3}: " + " ".join(f"{v:3d}" for v in row) + f"   | sum={sum(row)}")

    def short(lbl: str) -> str:
        if lbl in ("C","DW","FC","P","BN","F","A","N","O"): return lbl
        return "?"
    first_run = min(r for r,_ in full)
    seq_gt=[]; seq_pred=[]
    #for (ridx, layer_idx, name, op_type, dur), g, p in zip(meta_te, Yte, Yhat_type):
    for m, g, p in zip(meta_te, Yte, Yhat_type):
        ridx, layer_idx, name, op_type, dur = m[:5]
        if ridx == first_run:
            seq_gt.append(short(g))
            seq_pred.append(short(p))
    if seq_gt:
        print("\n[TYPE] First FULL run layer-type sequence:")
        print("   GT:", "".join(seq_gt))
        print("  PRD:", "".join(seq_pred))

    # ----- SHAPE head -----
    if clf_shape is None or not Xte_sh or mu_sh is None:
        print("\n[SHAPE] Not enough shape-labeled samples or no trained shape model.")
        return

    XteZ_sh = standardize_apply(Xte_sh, mu_sh, sd_sh)
    Yhat_sh = clf_shape.predict(XteZ_sh)

    lab_sh = sorted(set(Yte_sh))
    cm_sh = defaultdict(int); correct_sh=0
    for g,p in zip(Yte_sh, Yhat_sh):
        cm_sh[(g,p)] += 1
        if g==p: correct_sh+=1
    acc_sh = correct_sh/len(Yte_sh) if Yte_sh else 0.0
    print(f"\n[SHAPE] Per-sample kernel-shape accuracy across {len(full)} FULL runs: {acc_sh*100:.1f}%")
    print("Confusion (GT → Pred):")
    for g in lab_sh:
        row=[cm_sh[(g,p)] for p in lab_sh]
        print(f"  {g:>4}: " + " ".join(f"{v:3d}" for v in row) + f"  | sum={sum(row)}")

    by_layer_sh = defaultdict(list)
    #for idx,(ridx, layer_idx, name, op_type, dur) in enumerate(meta_te_sh):
    for idx, m in enumerate(meta_te_sh):
        ridx, layer_idx, name, op_type, dur = m[:5]
        # start_us, end_us = m[5], m[6]   

        by_layer_sh[name].append(idx)

    correct_mv_sh=0; total_layers_sh=0
    for name, idxs in by_layer_sh.items():
        gt = Yte_sh[idxs[0]]
        preds = [Yhat_sh[i] for i in idxs]
        maj = Counter(preds).most_common(1)[0][0]
        if maj==gt:
            correct_mv_sh+=1
        total_layers_sh+=1
    acc_mv_sh = correct_mv_sh/total_layers_sh if total_layers_sh>0 else 0.0
    print(f"\n[SHAPE] Majority-vote kernel-shape accuracy over {total_layers_sh} unique layers "
          f"using {len(full)} FULL runs: {acc_mv_sh*100:.1f}%")
    
    
    # --- Raw visualization (GT windows from ONNX) ---
    plot_raw_latency_heatmaps_by_shape(
        test_json, test_spike,
        full_runs=full,
        meta_te_sh=meta_te_sh,
        y_labels_sh=Yte_sh,     # use GT labels to avoid “wrong bin” visuals
        out_dir="raw_shape_latency_plots",
        max_layers_per_shape=8,
        pick_from_first_k_runs=20,
        max_time_rows=250,
        time_downsample=1,
    )



    

    # ----- CIN head -----
    if clf_cin is not None and Xte_cin and mu_sh is not None:
        XteZ_cin = standardize_apply(Xte_cin, mu_sh, sd_sh)
        Yhat_cin = clf_cin.predict(XteZ_cin)

        lab_cin = sorted(set(Yte_cin))
        cm_cin = defaultdict(int); correct_cin=0
        for g,p in zip(Yte_cin, Yhat_cin):
            cm_cin[(g,p)] += 1
            if g==p: correct_cin+=1
        acc_cin = correct_cin/len(Yte_cin) if Yte_cin else 0.0
        print(f"\n[CIN ] Per-sample Cin accuracy across {len(full)} FULL runs: {acc_cin*100:.1f}%")
        print("Confusion (GT → Pred):")
        for g in lab_cin:
            row=[cm_cin[(g,p)] for p in lab_cin]
            print(f"  {g:>8}: " + " ".join(f"{v:3d}" for v in row) + f"  | sum={sum(row)}")

        by_layer_cin = defaultdict(list)
        for idx,(ridx, layer_idx, name, cin_true, cout_true) in enumerate(meta_te_cin):
            by_layer_cin[name].append(idx)
        correct_mv_cin=0; total_layers_cin=0
        for name, idxs in by_layer_cin.items():
            gt = Yte_cin[idxs[0]]
            preds = [Yhat_cin[i] for i in idxs]
            maj = Counter(preds).most_common(1)[0][0]
            if maj==gt:
                correct_mv_cin+=1
            total_layers_cin+=1
        acc_mv_cin = correct_mv_cin/total_layers_cin if total_layers_cin>0 else 0.0
        print(f"\n[CIN ] Majority-vote Cin accuracy over {total_layers_cin} conv layers "
              f"using {len(full)} FULL runs: {acc_mv_cin*100:.1f}%")
    else:
        print("\n[CIN ] No trained Cin model or no Cin samples in test.")

    # ----- COUT head -----
    if clf_cout is not None and Xte_cout and mu_sh is not None:
        XteZ_cout = standardize_apply(Xte_cout, mu_sh, sd_sh)
        Yhat_cout = clf_cout.predict(XteZ_cout)

        lab_cout = sorted(set(Yte_cout))
        cm_cout = defaultdict(int); correct_cout=0
        for g,p in zip(Yte_cout, Yhat_cout):
            cm_cout[(g,p)] += 1
            if g==p: correct_cout+=1
        acc_cout = correct_cout/len(Yte_cout) if Yte_cout else 0.0
        print(f"\n[COUT] Per-sample Cout accuracy across {len(full)} FULL runs: {acc_cout*100:.1f}%")
        print("Confusion (GT → Pred):")
        for g in lab_cout:
            row=[cm_cout[(g,p)] for p in lab_cout]
            print(f"  {g:>8}: " + " ".join(f"{v:3d}" for v in row) + f"  | sum={sum(row)}")

        by_layer_cout = defaultdict(list)
        for idx,(ridx, layer_idx, name, cin_true, cout_true) in enumerate(meta_te_cout):
            by_layer_cout[name].append(idx)
        correct_mv_cout=0; total_layers_cout=0
        for name, idxs in by_layer_cout.items():
            gt = Yte_cout[idxs[0]]
            preds = [Yhat_cout[i] for i in idxs]
            maj = Counter(preds).most_common(1)[0][0]
            if maj==gt:
                correct_mv_cout+=1
            total_layers_cout+=1
        acc_mv_cout = correct_mv_cout/total_layers_cout if total_layers_cout>0 else 0.0
        print(f"\n[COUT] Majority-vote Cout accuracy over {total_layers_cout} conv layers "
              f"using {len(full)} FULL runs: {acc_mv_cout*100:.1f}%")
        
        # ====== Ablation over number of FULL runs ======
        # For MobileNetV2 we want #runs in {1, 10, 20, 47}.
        # We automatically cap at len(full) (we can replace 30 by this), so we can reuse this for other models.
        ablation_runs = [1, 10, 20, 30]

        # Guard: if we didn't train shape/Cin/Cout heads, just skip.
        have_shape = (clf_shape is not None and Xte_sh and mu_sh is not None)
        have_cin   = (clf_cin   is not None and Xte_cin and mu_sh is not None)
        have_cout  = (clf_cout  is not None and Xte_cout and mu_sh is not None)

        print("\n[ABLATION] Impact of number of FULL runs on per-layer majority accuracy:")
        for k in ablation_runs:
            if k <= 0:
                continue
            if k > 30:
                continue

            # use the first k FULL runs
            first_k_runs = set(r for r, _ in full[:k])

            # TYPE head (per-layer majority)
            type_acc_k, _, _ = majority_acc_for_runs(meta_te, Yte, Yhat_type, first_k_runs)

            # SHAPE head (per-layer majority)
            if have_shape:
                shape_acc_k, _, _ = majority_acc_for_runs(meta_te_sh, Yte_sh, Yhat_sh, first_k_runs)
            else:
                shape_acc_k = float("nan")

            # CIN head (per-layer majority)
            if have_cin:
                cin_acc_k, _, _ = majority_acc_for_runs(meta_te_cin, Yte_cin, Yhat_cin, first_k_runs)
            else:
                cin_acc_k = float("nan")

            # COUT head (per-layer majority)
            if have_cout:
                cout_acc_k, _, _ = majority_acc_for_runs(meta_te_cout, Yte_cout, Yhat_cout, first_k_runs)
            else:
                cout_acc_k = float("nan")

            print(f"  [runs={k:2d}] "
                  f"Type={type_acc_k*100:5.1f}%  "
                  f"Shape={shape_acc_k*100:5.1f}%  "
                  f"Cin={cin_acc_k*100:5.1f}%  "
                  f"Cout={cout_acc_k*100:5.1f}%")

        # ---- final layer-by-layer reconstruction summary ----
        summarize_reconstructed_layers(
            meta_te,      Yte,      Yhat_type,
            meta_te_sh,   Yte_sh,   Yhat_sh,
            meta_te_cin,  Yte_cin,  Yhat_cin,
            meta_te_cout, Yte_cout, Yhat_cout,
        )
    else:
        print("\n[COUT] No trained Cout model or no Cout samples in test.")
        
    
        
        
# ======= Model-level (spike-only) classifier helpers =======

def run_train_model_classifier():
    """
    Train a model-identity classifier (e.g., MobileNetV2 vs EfficientNet-Lite4)
    using ONLY spike_log.txt captures listed in MODEL_TRAIN_SETS.
    """
    if not MODEL_TRAIN_SETS:
        print("[MODEL TRAIN] Please fill MODEL_TRAIN_SETS with (label, spike_path) entries.")
        return

    X = []
    Y = []
    for label, spike_path in MODEL_TRAIN_SETS:
        feats = extract_model_features_from_spike(spike_path)
        if feats is None:
            continue
        X.append(feats)
        Y.append(label)

    if not X:
        print("[MODEL TRAIN] No usable samples (all empty or failed).")
        return

    mu, sd = standardize_fit(X)
    XZ = standardize_apply(X, mu, sd)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RNG_SEED,
    )
    clf.fit(XZ, Y)

    dump({"clf_model": clf, "mu": mu, "sd": sd}, MODEL_ID_PATH)
    print(f"[MODEL TRAIN] Trained on {len(X)} samples; saved to {MODEL_ID_PATH}")

    if MODEL_TEST_SETS:
        print("\n[MODEL TRAIN] Quick evaluation on MODEL_TEST_SETS using the freshly trained model...")
        run_test_model_classifier(clf, mu, sd)


def run_test_model_classifier(clf=None, mu=None, sd=None):
    """
    Test or use the model-identity classifier on spike-only captures.

    If clf/mu/sd are not provided, they are loaded from MODEL_ID_PATH.
    """
    if not MODEL_TEST_SETS:
        print("[MODEL TEST] Please fill MODEL_TEST_SETS with (label, spike_path) entries.")
        return

    if clf is None or mu is None or sd is None:
        print(f"[MODEL TEST] Loading model from {MODEL_ID_PATH} ...")
        data = load(MODEL_ID_PATH)
        clf = data["clf_model"]
        mu  = data["mu"]
        sd  = data["sd"]

    total_labeled = 0
    correct = 0
    cm = Counter()
    label_set = set()

    for true_label, spike_path in MODEL_TEST_SETS:
        feats = extract_model_features_from_spike(spike_path)
        if feats is None:
            continue
        z = standardize_apply([feats], mu, sd)[0]
        pred = clf.predict([z])[0]

        if true_label is None:
            print(f"[MODEL TEST] {spike_path}: predicted = {pred}")
        else:
            print(f"[MODEL TEST] {spike_path}: predicted = {pred}, true = {true_label}")
            total_labeled += 1
            label_set.add(true_label)
            cm[(true_label, pred)] += 1
            if pred == true_label:
                correct += 1

    if total_labeled > 0:
        acc = 100.0 * correct / total_labeled
        print(f"\n[MODEL TEST] Accuracy over {total_labeled} labeled samples: {acc:.1f}%")
        labels = sorted(label_set)
        print("Confusion (GT → Pred):")
        for g in labels:
            row = [cm[(g, p)] for p in labels]
            print(f"  {g:>12}: " + " ".join(f"{v:3d}" for v in row) + f"  | sum={sum(row)}")



# ======= TEST mode =======
def run_test_mode():
    if TEST_PAIR is None:
        print("In TEST mode, please set TEST_PAIR = ('path_to_json', 'path_to_spike').")
        return
    test_json, test_spike = TEST_PAIR
    
    print(f"[TEST MODE] Loading models from {MODEL_PATH} ...")
    data = load(MODEL_PATH)
    clf_type  = data["clf_type"]
    mu_type   = data["mu_type"]
    sd_type   = data["sd_type"]
    clf_shape = data["clf_shape"]
    mu_sh     = data["mu_sh"]
    sd_sh     = data["sd_sh"]
    clf_cin   = data.get("clf_cin")
    clf_cout  = data.get("clf_cout")
    
    dur_profile = data.get("dur_profile")
    
    if MODE.lower() == "spike_only":
        if dur_profile is None:
            print("[SPIKE_ONLY] Missing dur_profile in model file. Re-train with MODE='train' first.")
            return
        dur_profile_us = dur_profile["dur_profile_us"]

        # call a spike-only version of test
        run_test_on_pair_spike_only(test_json, test_spike,
                                    clf_type, mu_type, sd_type,
                                    clf_shape, mu_sh, sd_sh,
                                    clf_cin, clf_cout,
                                    dur_profile_us)
        return


    
    print("[TEST MODE] Evaluating on:", test_json, "/", test_spike)
    run_test_on_pair(test_json, test_spike,
                     clf_type, mu_type, sd_type,
                     clf_shape, mu_sh, sd_sh,
                     clf_cin, clf_cout)

# ======= main =======
def main():
    mode = MODE.lower()
    if mode == "train":
        run_train_mode()
    elif mode == "test":
        run_test_mode()
    elif mode == "train_model":
        run_train_model_classifier()
    elif mode == "test_model":
        run_test_model_classifier()
    elif mode == "spike_only":
        run_test_mode()

    else:
        print(f"Unknown MODE='{MODE}'. Use 'train', 'test', 'train_model', or 'test_model', or 'spike_only'.")


if __name__=="__main__":
    main()

