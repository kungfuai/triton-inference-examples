from os import makedirs, path
import torch
import torchvision

TRITON_CONFIG = """
name: "image_encoder"
backend: "pytorch"
max_batch_size : 0

input [
  {
    name: "INPUT__0"
    data_type: TYPE_FP32
    dims: [ 3, -1, -1 ]
  }
]
output [
  {
    name: "OUTPUT__0"
    data_type: TYPE_FP32
    dims: [ 512 ]
  }
]
"""


class ImageEncoder(torch.nn.Module):
    def __init__(self, model_name="resnet50"):
        super().__init__()
        self.model_name = model_name
        self.model = self._get_model()
        self.model.fc = torch.nn.Identity()

    def _get_model(self):
        model = getattr(torchvision.models, self.model_name)(pretrained=True)
        model.eval()
        return model

    def forward(self, image):
        return self.model(image)

    def save(self, output_file: str, use_trace: bool = True, save_triton_config=True):

        makedirs(path.dirname(output_file), exist_ok=True)
        if use_trace:
            image = torch.randn(1, 3, 244, 244)
            m = torch.jit.trace(self.model, image)
            m.save(output_file)
        else:
            m = torch.jit.script(self.model)
            torch.jit.save(m, output_file)

        if save_triton_config:
            with open(
                path.join(path.dirname(path.dirname(output_file)), "config.pbtxt"), "w"
            ) as f:
                f.write(TRITON_CONFIG)
