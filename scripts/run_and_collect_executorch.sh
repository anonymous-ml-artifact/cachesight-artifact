#!/usr/bin/env bash
set -euo pipefail

APP_ID="com.example.resnet18v1executorch"
BASE="/Volumes/T7_Shield/Android/Projects/ResNet18V1ExecuTorchApp/profiling"
ETDUMP_DIR="$BASE/etdump_50"
PY_SCRIPT="$BASE/relabel_all50_etdump_with_onnx.py"
ONNX_JSON="$BASE/onnx/resnet18v1_onnx.json"

mkdir -p "$BASE"
mkdir -p "$ETDUMP_DIR"

echo "== Step 1: Find latest ExecuTorch manifest and JSONL on device =="

CACHE_LIST=$(
  adb shell run-as "$APP_ID" ls cache 2>/dev/null | tr -d '\r'
)

LATEST_MANIFEST_NAME=$(printf "%s\n" "$CACHE_LIST" | grep '_executorch_manifest\.json$' | sort | tail -n 1 || true)
LATEST_JSONL_NAME=$(printf "%s\n" "$CACHE_LIST" | grep '_executorch_native_infer\.jsonl$' | sort | tail -n 1 || true)

if [[ -z "${LATEST_MANIFEST_NAME}" ]]; then
  echo "ERROR: No *_executorch_manifest.json found in app cache."
  exit 1
fi

if [[ -z "${LATEST_JSONL_NAME}" ]]; then
  echo "ERROR: No *_executorch_native_infer.jsonl found in app cache."
  exit 1
fi

LATEST_MANIFEST="cache/${LATEST_MANIFEST_NAME}"
LATEST_JSONL="cache/${LATEST_JSONL_NAME}"

echo "Latest manifest: $LATEST_MANIFEST"
echo "Latest jsonl:    $LATEST_JSONL"

MANIFEST_BASENAME="$LATEST_MANIFEST_NAME"
JSONL_BASENAME="$LATEST_JSONL_NAME"

echo
echo "== Step 2: Pull manifest and JSONL =="

adb exec-out run-as "$APP_ID" cat "$LATEST_MANIFEST" > "$BASE/$MANIFEST_BASENAME"
adb exec-out run-as "$APP_ID" cat "$LATEST_JSONL" > "$BASE/$JSONL_BASENAME"

echo "Saved: $BASE/$MANIFEST_BASENAME"
echo "Saved: $BASE/$JSONL_BASENAME"

echo
echo "== Step 3: Parse manifest and pull all ETDump files =="

python3 - "$BASE/$MANIFEST_BASENAME" "$APP_ID" "$ETDUMP_DIR" <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
app_id = sys.argv[2]
out_dir = Path(sys.argv[3])

text = manifest_path.read_text().strip()
if not text:
    print(f"ERROR: Manifest file is empty: {manifest_path}")
    sys.exit(1)

data = json.loads(text)
files = data.get("files", [])

if not files:
    print("ERROR: No ETDump files listed in manifest.")
    sys.exit(1)

# Clean old ETDump files
for old in out_dir.glob("*.etdump"):
    old.unlink()

for idx, remote_path in enumerate(files, start=1):
    basename = os.path.basename(remote_path)
    local_path = out_dir / basename
    print(f"[{idx:02d}/{len(files):02d}] pulling {basename}")
    with open(local_path, "wb") as f:
        subprocess.run(
            ["adb", "exec-out", "run-as", app_id, "cat", remote_path],
            check=True,
            stdout=f
        )

print(f"Pulled {len(files)} ETDump files into {out_dir}")
PY

COUNT=$(find "$ETDUMP_DIR" -maxdepth 1 -type f -name "*.etdump" | wc -l | tr -d ' ')
echo "ETDump count: $COUNT"

if [[ "$COUNT" != "50" ]]; then
  echo "WARNING: Expected 50 ETDump files, found $COUNT"
fi

echo
echo "== Step 4: Run relabel pipeline =="

cd "$BASE"
python relabel_all50_etdump_with_onnx.py \
  --infer-jsonl "$JSONL_BASENAME" \
  --onnx-json "$ONNX_JSON" \
  --output-prefix "resnet18v1_executorch_layer_timeline_absolute_all50_relabeled"

echo
echo "== Done =="
echo "Final CSV:"
echo "$BASE/resnet18v1_executorch_layer_timeline_absolute_all50_relabeled.csv"
echo
echo "Final JSON:"
echo "$BASE/resnet18v1_executorch_layer_timeline_absolute_all50_relabeled.json"
echo
echo "Verify CSV:"
echo "$BASE/resnet18v1_executorch_layer_timeline_absolute_all50_relabeled_verify.csv"