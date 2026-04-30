import os
import copy
import torch
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.devtools import generate_etrecord

OUTPUT_DIR = "/Volumes/T7_Shield/Android/Projects/MobileNetV2TFPyTorch/model_export/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
sample_inputs = (torch.randn(1, 3, 224, 224),)

exported = torch.export.export(model, sample_inputs)

# Keep a copy for ETRecord before lowering mutates it
edge_manager = to_edge_transform_and_lower(
    exported,
    partitioner=[XnnpackPartitioner()],
)

et_program = edge_manager.to_executorch()

pte_path = os.path.join(OUTPUT_DIR, "mobilenet_v2_xnnpack.pte")
with open(pte_path, "wb") as f:
    et_program.write_to_file(f)

etrecord_path = os.path.join(OUTPUT_DIR, "mobilenet_v2_xnnpack.etrecord")
generate_etrecord(
    etrecord_path,
    edge_dialect_program=edge_manager,
    executorch_program=et_program,
)

print(f"Saved PTE: {pte_path}")
print(f"Saved ETRecord: {etrecord_path}")