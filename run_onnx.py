import onnxruntime as ort
import numpy as np
import yaml
import torch
from utils.loader import dataLoader_test as dataLoader, get_module
from utils.print_tool import datasize
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the ONNX model
session_int8 = ort.InferenceSession("onnx_models/superpoint_quantized_w8a8_act_float.onnx")
session_float = ort.InferenceSession("onnx_models/superpoint.onnx")

# Prepare input (make sure it matches the model input shape and dtype)
input_name_int8 = session_int8.get_inputs()[0].name
input_name_float = session_float.get_inputs()[0].name
# input_data = np.random.randn(1, 1, 1000, 1000).astype(np.float32)

config = 'configs/liteml_magicpoint_repeatability_heatmap_W8A8_PTQ.yaml'  # W8A8 PTQ model wrapped with LiteML

with open(config, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# data loading
task = config["data"]["dataset"]
data = dataLoader(config, dataset=task)
test_set, test_loader = data["test_set"], data["test_loader"]
datasize(test_loader, config, tag="test")

# model loading
Val_model_heatmap = get_module("", config["front_end_model"])
## load pretrained
val_agent = Val_model_heatmap(config["model"], device=device)
val_agent.calibration_data = config.get("calibration_data")  # for calibration in PTQ

val_agent.loadModel()
val_agent.net.to(device)

# get image
data_iter = iter(test_loader)
input_tensor = next(data_iter)['image'].to(device)
input_tensor = F.interpolate(input_tensor, size=(1000, 1000), mode='bilinear', align_corners=False)

input_data = input_tensor.detach().cpu().numpy().astype(np.float32)

# Run inference
# outputs_int8 = session_int8.run(['173', '175', '176', '181'], {input_name_int8: input_data}) # for inner buffers
outputs_int8 = session_int8.run(None, {input_name_int8: input_data})
# print('int8:', outputs_int8[0])

outputs_float = session_float.run(None, {input_name_float: input_data})
# print('float:', outputs_float[0])

output_torch_w8a8 = val_agent.net(input_tensor)
# print('torch:', output_torch_w8a8[0])
print('Done')
mse_semi_torch = np.sqrt(np.mean((output_torch_w8a8['semi'].detach().cpu().numpy() - outputs_int8[0])**2))
mse_desc_torch = np.sqrt(np.mean((output_torch_w8a8['desc'].detach().cpu().numpy() - outputs_int8[1])**2))

mse_semi_float = np.sqrt(np.mean((outputs_float[0] - outputs_int8[0])**2))
mse_desc_float = np.sqrt(np.mean((outputs_float[1] - outputs_int8[1])**2))

print('MSE semi torch: ', mse_semi_torch)
print('MSE desc torch: ', mse_desc_torch)
print('MSE semi float: ', mse_semi_float)
print('MSE desc float: ', mse_desc_float)