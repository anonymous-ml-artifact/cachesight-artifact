import os
import sys
import ast
import json
import re
import argparse
from pathlib import Path

# Force a real working directory before importing executorch/torch
DEFAULT_BASE = Path("/Volumes/T7_Shield/Android/Projects/ResNet18V1ExecuTorchApp/profiling")
os.chdir(DEFAULT_BASE)
sys.path[0] = os.getcwd()

import pandas as pd
from executorch.devtools import Inspector


def parse_args():
    parser = argparse.ArgumentParser(
        description="Relabel 50 ExecuTorch ETDump traces using ONNX first-run semantic reference."
    )
    parser.add_argument(
        "--base",
        type=str,
        default=str(DEFAULT_BASE),
        help="Profiling base directory",
    )
    parser.add_argument(
        "--infer-jsonl",
        type=str,
        required=True,
        help="JSONL filename or full path for same-run native inference timing",
    )
    parser.add_argument(
        "--etdump-dir",
        type=str,
        default=None,
        help="Directory containing ETDump files (default: <base>/etdump_50)",
    )
    parser.add_argument(
        "--onnx-json",
        type=str,
        default=None,
        help="ONNX profiling JSON path (default: <base>/onnx/resnet18v1_onnx.json)",
    )
    parser.add_argument(
        "--etrecord",
        type=str,
        default="/Volumes/T7_Shield/Android/Projects/ResNet18V1ExecuTorchApp/model_export/output/resnet18_v1_xnnpack.etrecord",
        help="ETRecord path",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="executorch_layer_timeline_absolute_all50_relabeled",
        help="Prefix for output CSV/JSON files",
    )
    return parser.parse_args()


ARGS = parse_args()

BASE = Path(ARGS.base).resolve()
ETDUMP_DIR = Path(ARGS.etdump_dir).resolve() if ARGS.etdump_dir else (BASE / "etdump_50")
ONNX_JSON = Path(ARGS.onnx_json).resolve() if ARGS.onnx_json else (BASE / "onnx" / "resnet18v1_onnx.json")
ETRECORD_PATH = Path(ARGS.etrecord).resolve()

INFER_JSONL = Path(ARGS.infer_jsonl)
if not INFER_JSONL.is_absolute():
    INFER_JSONL = (BASE / INFER_JSONL).resolve()

OUT_CSV = BASE / f"{ARGS.output_prefix}.csv"
OUT_JSON = BASE / f"{ARGS.output_prefix}.json"
OUT_VERIFY_CSV = BASE / f"{ARGS.output_prefix}_verify.csv"


def parse_single_list(x):
    if isinstance(x, list):
        return x[0] if x else None
    if pd.isna(x):
        return None
    if isinstance(x, str):
        vals = ast.literal_eval(x)
        if isinstance(vals, list):
            return vals[0] if vals else None
        return vals
    return x


def load_infer_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No rows found in {path}")
    return df.sort_values("iter").reset_index(drop=True)


def load_first_onnx_run(json_path: Path):
    data = json.loads(json_path.read_text())

    model_runs = [x for x in data if x.get("name") == "model_run" and x.get("cat") == "Session"]
    if not model_runs:
        raise RuntimeError("No model_run session found in ONNX JSON")

    first_run = model_runs[0]
    run_start = first_run["ts_abs"]
    run_end = first_run["ts_abs"] + first_run["dur"]

    nodes = []
    for x in data:
        if x.get("cat") != "Node":
            continue
        if not str(x.get("name", "")).endswith("_kernel_time"):
            continue
        ts_abs = x.get("ts_abs")
        if ts_abs is None or not (run_start <= ts_abs <= run_end):
            continue

        name = x["name"]
        args = x.get("args", {})
        nodes.append({
            "name": name,
            "op_name": args.get("op_name", ""),
            "ts_abs": ts_abs,
            "dur": x.get("dur", 0),
            "input_type_shape": args.get("input_type_shape", []),
        })

    cleaned = []
    for n in nodes:
        name = n["name"]
        if "/Sub_" in name or "/Div_" in name:
            continue
        cleaned.append(n)

    return cleaned


def semantic_from_onnx_node(node):
    name = node["name"]
    op = node["op_name"]
    inputs = node.get("input_type_shape", [])

    # Stem
    if name == "resnetv15_conv0_fwd_kernel_time":
        return ("conv0", "K7", "stem_conv")

    if name == "resnetv15_pool0_fwd_kernel_time":
        return ("pool0", "MAXPOOL", "stem_maxpool")

    # Main ResNet conv layers
    m = re.search(r"resnetv15_stage(\d+)_conv(\d+)_fwd_kernel_time", name)
    if m:
        stage = int(m.group(1))
        conv = int(m.group(2))

        # Downsample/projection convs are 1x1. In your ONNX file these are conv2
        # for stage2/stage3/stage4.
        if stage in (2, 3, 4) and conv == 2:
            return (f"stage{stage}.downsample.0", "K1", "downsample_projection")

        # Other stage convs are 3x3 convolutions.
        return (f"stage{stage}.conv{conv}", "K3", "residual_conv")

    # Final pooling / flatten / FC
    if name == "resnetv15_pool1_fwd_kernel_time":
        return ("pool1", "GAP", "global_avg_pool")

    if name == "flatten_170_kernel_time":
        return ("flatten", "FLATTEN", "flatten")

    if name == "resnetv15_dense0_fwd_kernel_time":
        return ("dense0", "FC", "classifier")

    return (name, "OTHER", op)


def build_reference_from_onnx(onnx_nodes):
    ref = []
    for n in onnx_nodes:
        semantic_name, kernel_tag, role = semantic_from_onnx_node(n)
        ref.append({
            "onnx_name": n["name"],
            "semantic_layer_name": semantic_name,
            "kernel_tag": kernel_tag,
            "semantic_role": role,
        })
    return ref


def etdump_category(event_name: str):
    n = event_name.lower()

    if "transpose" in n:
        return "TRANSPOSE"

    if "clamp" in n:
        return "RELU"

    if "max pooling" in n or "maxpool" in n or "max pool" in n:
        return "MAXPOOL"

    if "mean" in n or "average" in n or "global average" in n:
        return "GAP"

    if "add" in n:
        return "ADD"

    # For ResNet18, all convs may look like generic IGEMM.
    if "convolution" in n or "igemm" in n:
        return "CONV"

    if "gemm" in n or "fully connected" in n:
        return "GEMM"

    return "OTHER"


def load_etdump_one(path: Path) -> pd.DataFrame:
    insp = Inspector(etdump_path=str(path), etrecord=str(ETRECORD_PATH))
    df = insp.to_dataframe()
    if df is None or df.empty:
        raise RuntimeError(f"Inspector dataframe is empty for {path}")

    exclude_prefixes = ("Method::", "Program::", "DELEGATE_CALL")
    df = df[~df["event_name"].fillna("").str.startswith(exclude_prefixes)].copy()

    df["duration_ms"] = df["raw"].apply(parse_single_list).astype(float)
    df["duration_ns"] = (df["duration_ms"] * 1_000_000.0).round().astype(int)
    # raw start time from Inspector (NOT relative yet)
    df["raw_start_ns"] = df["start_time"].apply(parse_single_list).astype(int)

    # normalize per inference (this ETDump = one inference)
    min_start = df["raw_start_ns"].min()

    df["relative_start_ns"] = df["raw_start_ns"] - min_start
    df["relative_end_ns"] = df["relative_start_ns"] + df["duration_ns"]

    df = df.sort_values("relative_start_ns").reset_index(drop=True)
    return df


def relabel_etdump_with_reference(etdump_df, ref):
    # Keep convolution references in ONNX order:
    # K7 stem, K3 residual convs, K1 downsample projections.
    conv_refs = [
        r for r in ref
        if r["kernel_tag"] in ("K7", "K3", "K1")
    ]

    pool_refs = [r for r in ref if r["kernel_tag"] == "MAXPOOL"]
    gap_refs = [r for r in ref if r["kernel_tag"] == "GAP"]
    fc_refs = [r for r in ref if r["kernel_tag"] == "FC"]
    flatten_refs = [r for r in ref if r["kernel_tag"] == "FLATTEN"]

    rows = []

    for _, row in etdump_df.iterrows():
        event_name = row["event_name"]
        cat = etdump_category(event_name)

        semantic_layer_name = ""
        kernel_tag = ""
        semantic_role = ""

        if cat == "TRANSPOSE":
            semantic_layer_name = "layout_transform"
            kernel_tag = "TRANSPOSE"
            semantic_role = "layout_transform"

        elif cat == "CONV":
            if conv_refs:
                r = conv_refs.pop(0)
                semantic_layer_name = r["semantic_layer_name"]
                kernel_tag = r["kernel_tag"]
                semantic_role = r["semantic_role"]
            else:
                semantic_layer_name = "conv_unknown"
                kernel_tag = "CONV_UNKNOWN"
                semantic_role = "unknown_conv"

        elif cat == "MAXPOOL":
            if pool_refs:
                r = pool_refs.pop(0)
                semantic_layer_name = r["semantic_layer_name"]
                kernel_tag = "MAXPOOL"
                semantic_role = r["semantic_role"]
            else:
                semantic_layer_name = "maxpool"
                kernel_tag = "MAXPOOL"
                semantic_role = "maxpool"

        elif cat == "RELU":
            semantic_layer_name = "relu"
            kernel_tag = "RELU"
            semantic_role = "activation"

        elif cat == "ADD":
            semantic_layer_name = "residual_add"
            kernel_tag = "ADD"
            semantic_role = "residual_add"

        elif cat == "GAP":
            if gap_refs:
                r = gap_refs.pop(0)
                semantic_layer_name = r["semantic_layer_name"]
                kernel_tag = "GAP"
                semantic_role = r["semantic_role"]
            else:
                semantic_layer_name = "global_avg_pool"
                kernel_tag = "GAP"
                semantic_role = "global_avg_pool"

        elif cat == "GEMM":
            if fc_refs:
                r = fc_refs.pop(0)
                semantic_layer_name = r["semantic_layer_name"]
                kernel_tag = "FC"
                semantic_role = r["semantic_role"]
            else:
                semantic_layer_name = "fc_unknown"
                kernel_tag = "FC"
                semantic_role = "classifier"

        elif cat == "OTHER":
            n = str(event_name).lower()
            if "flatten" in n and flatten_refs:
                r = flatten_refs.pop(0)
                semantic_layer_name = r["semantic_layer_name"]
                kernel_tag = "FLATTEN"
                semantic_role = r["semantic_role"]
            else:
                semantic_layer_name = "unknown"
                kernel_tag = "OTHER"
                semantic_role = "unknown"

        out = dict(row)
        out["semantic_layer_name"] = semantic_layer_name
        out["kernel_tag"] = kernel_tag
        out["semantic_role"] = semantic_role
        rows.append(out)

    return pd.DataFrame(rows)


def verify_labeled(df: pd.DataFrame):
    monotonic = df["relative_start_ns"].is_monotonic_increasing

    overlap_count = 0
    for i in range(len(df) - 1):
        if int(df.loc[i, "relative_end_ns"]) > int(df.loc[i + 1, "relative_start_ns"]):
            overlap_count += 1

    first_start = int(df["relative_start_ns"].min())
    last_end = int(df["relative_end_ns"].max())
    span_ns = last_end - first_start
    sum_ns = int(df["duration_ns"].sum())
    ratio = (sum_ns / span_ns) if span_ns > 0 else None

    counts = df["kernel_tag"].value_counts().to_dict()

    return {
        "num_ops": len(df),
        "monotonic": monotonic,
        "overlap_count": overlap_count,
        "span_ns": span_ns,
        "sum_ns": sum_ns,
        "ratio": ratio,
        "count_K7": counts.get("K7", 0),
        "count_K3": counts.get("K3", 0),
        "count_K1": counts.get("K1", 0),
        "count_GAP": counts.get("GAP", 0),
        "count_MAXPOOL": counts.get("MAXPOOL", 0),
        "count_RELU": counts.get("RELU", 0),
        "count_FLATTEN": counts.get("FLATTEN", 0),
        "count_FC": counts.get("FC", 0),
        "count_ADD": counts.get("ADD", 0),
        "count_TRANSPOSE": counts.get("TRANSPOSE", 0),
    }


def main():
    print(f"BASE:         {BASE}")
    print(f"ETDUMP_DIR:   {ETDUMP_DIR}")
    print(f"ONNX_JSON:    {ONNX_JSON}")
    print(f"INFER_JSONL:  {INFER_JSONL}")
    print(f"ETRECORD:     {ETRECORD_PATH}")
    print()

    infer_df = load_infer_jsonl(INFER_JSONL)
    onnx_nodes = load_first_onnx_run(ONNX_JSON)
    ref = build_reference_from_onnx(onnx_nodes)

    etdump_files = sorted(ETDUMP_DIR.glob("*.etdump"))
    if not etdump_files:
        raise RuntimeError(f"No ETDump files found in {ETDUMP_DIR}")

    usable_n = min(len(etdump_files), len(infer_df))
    all_rows = []
    verify_rows = []

    for idx in range(usable_n):
        etdump_path = etdump_files[idx]
        infer_row = infer_df.iloc[idx]

        df = load_etdump_one(etdump_path)
        df = relabel_etdump_with_reference(df, ref)

        infer_index = int(infer_row["iter"])
        infer_start_ns = int(infer_row["ts_start_ns"])
        infer_end_ns = int(infer_row["ts_end_ns"])
        infer_dur_us = int(infer_row["dur_us"])

        # ADD THIS BLOCK HERE
        print(f"[debug] inference {infer_index}: "
            f"rel_start min={df['relative_start_ns'].min()} "
            f"rel_end max={df['relative_end_ns'].max()} "
            f"infer_ns={(infer_end_ns - infer_start_ns)}")

        for op_idx, row in df.iterrows():
            rel_start_ns = int(row["relative_start_ns"])
            duration_ns = int(row["duration_ns"])
            abs_start_ns = infer_start_ns + rel_start_ns
            abs_end_ns = abs_start_ns + duration_ns

            all_rows.append({
                "inference_index": infer_index,
                "op_index": int(op_idx),
                "event_name": row["event_name"],
                "semantic_layer_name": row["semantic_layer_name"],
                "kernel_tag": row["kernel_tag"],
                "semantic_role": row["semantic_role"],
                "relative_start_ns": rel_start_ns,
                "duration_ns": duration_ns,
                "absolute_monotonic_start_ns": abs_start_ns,
                "absolute_monotonic_end_ns": abs_end_ns,
                "inference_ts_start_ns": infer_start_ns,
                "inference_ts_end_ns": infer_end_ns,
                "inference_dur_us": infer_dur_us,
                "source_etdump_file": etdump_path.name,
            })

        v = verify_labeled(df)
        v["inference_index"] = infer_index
        v["source_etdump_file"] = etdump_path.name
        verify_rows.append(v)

        print(
            f"[ok] inference {infer_index:02d}: "
            f"ops={v['num_ops']} K7={v['count_K7']} K3={v['count_K3']} "
            f"K1={v['count_K1']} MAXPOOL={v['count_MAXPOOL']} "
            f"RELU={v['count_RELU']} GAP={v['count_GAP']} "
            f"FC={v['count_FC']} ADD={v['count_ADD']} "
            f"ratio={v['ratio']:.6f}"
        )

    out_df = pd.DataFrame(all_rows)
    verify_df = pd.DataFrame(verify_rows)

    out_df.to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, "w") as f:
        json.dump(all_rows, f, indent=2)
    verify_df.to_csv(OUT_VERIFY_CSV, index=False)

    print()
    print(f"Saved final CSV:   {OUT_CSV}")
    print(f"Saved final JSON:  {OUT_JSON}")
    print(f"Saved verify CSV:  {OUT_VERIFY_CSV}")
    print()
    print(out_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()