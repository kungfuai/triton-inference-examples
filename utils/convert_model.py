# This is intended to run inside the nvidia container.
# e.g. nvcr.io/nvidia/pytorch:22.09-py3
#
# Under construction.  Not ready for use yet.

import torch
import torch_tensorrt

print(torch_tensorrt.__version__)

model = torch.load("vector_similarity.pth")
print("loaded")
trt_model = torch_tensorrt.compile(
    model,
    inputs=[
        torch_tensorrt.Input(
            shape=(1, 3, 32, 32),
            # min_shape=(1, 3, 32, 32),
            # opt_shape=(1, 3, 512, 512),
            # max_shape=(16, 3, 1024, 1024),
            # min_shape=(1, 224, 224, 3),
            # opt_shape=(1, 512, 512, 3),
            # max_shape=(1, 1024, 1024, 3),
            dtype=torch.float32,
        )
    ],
    enabled_precisions={torch.half},  # Run with FP32
)
torch.jit.save(trt_model, "model_repo/ris/1/model.pt")
