import os
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower, ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from executorch.extension.export_util.utils import save_pte_program
from executorch.devtools import generate_etrecord

OUT_DIR = "/Volumes/T7_Shield/Android/Projects/ResNet18V1ExecuTorchApp/model_export/output"
PTE_PATH = os.path.join(OUT_DIR, "resnet18_v1_xnnpack.pte")
ETRECORD_PATH = os.path.join(OUT_DIR, "resnet18_v1_xnnpack.etrecord")

os.makedirs(OUT_DIR, exist_ok=True)

# 1) Load pretrained ResNet18 from torchvision
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval()

# 2) Example input
example_inputs = (torch.randn(1, 3, 224, 224),)

# 3) Export with torch.export
exported = torch.export.export(model, example_inputs)

# 4) Lower to ExecuTorch / XNNPACK
edge_program = to_edge_transform_and_lower(
    exported,
    partitioner=[XnnpackPartitioner()],
)

# 5) Convert to ExecuTorch program
exec_program = edge_program.to_executorch(
    config=ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass()
    )
)

generate_etrecord(
    ETRECORD_PATH,
    edge_program,
    exec_program,
)

# 6) Save .pte
save_pte_program(exec_program, PTE_PATH)

print(f"Saved PTE:      {PTE_PATH}")
print(f"Saved ETRecord: {ETRECORD_PATH}")