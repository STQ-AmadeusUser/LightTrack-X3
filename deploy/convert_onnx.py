import _init_paths
import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import lib.models.onnx_model as model
from lib.utils.utils import load_pretrain

batch_size = 1

# reference: https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
if __name__ == '__main__':

    # build siamese network neck_and_head for tracking
    siam_onnx = model.__dict__['ONNXModel']()
    siam_onnx = load_pretrain(siam_onnx, '../snapshot/LightTrackM.pth')
    siam_onnx.eval()
    # Input to the model
    torch_z = torch.randn(batch_size, 3, 128, 128, requires_grad=True)
    torch_x = torch.randn(batch_size, 3, 256, 256, requires_grad=True)
    torch_cls, torch_reg = siam_onnx(torch_z, torch_x)

    # Export the model
    torch.onnx.export(siam_onnx,  # model being run
                      (torch_z, torch_x),  # model input (or a tuple for multiple inputs)
                      "LightTrack.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input1', 'input2'],  # the model's input names
                      output_names=['output1', 'output2'],  # the model's output names
                      )
