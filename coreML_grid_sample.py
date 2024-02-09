import coremltools as ct
import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_grid_sample import grid_sample as custom_grid_sample


class GridSampleModel(nn.Module):
    def forward(self, x, grid):
        x1 = F.grid_sample(x, grid)
        return x1

    def __repr__(self):
        return "GridSampleModel"


class CustomGridSampleModel(nn.Module):
    def forward(self, x, grid):
        x1 = custom_grid_sample(x, grid)
        return x1

    def __repr__(self):
        return "CustomGridSampleModel"


if __name__ == "__main__":

    models = [GridSampleModel(), CustomGridSampleModel()]

    # Generate a random input image
    batch_size = 1
    input_image = torch.randn((batch_size, 512, 256, 256), dtype=torch.float32)

    # Generate a random grid with the same spatial dimensions as the input image
    grid = torch.randn((1, 256, 256, 2), dtype=torch.float32)

    for model in models:
        # Perform forward pass through the model
        output = model(input_image, grid)
        # trace the model
        traced_model = torch.jit.trace(
            model, example_inputs=[input_image, grid], strict=False)
        # Convert the traced model to CoreML
        coreml_model = ct.convert(traced_model,
                                  inputs=[ct.TensorType(shape=input_image.shape), ct.TensorType(shape=grid.shape)],
                                  minimum_deployment_target=ct.target.iOS17, compute_precision=ct.precision.FLOAT16)
        print(f"converting {repr(model)}")
        coreml_model.save(f"{repr(model)}.mlpackage")
        print("model converted")
