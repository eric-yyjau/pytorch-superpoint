from utils.loader import get_save_path
from utils.var_dim import squeezeToNumpy
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import requests
import matplotlib.pyplot as plt
import warnings
import json
from utils.loader import modelLoader
import yaml

use_liteml = True
onnx_path = 'onnx_models/superpoint_quantized_w8a8_act_float_240_320.onnx'
# onnx_path = 'onnx_models/superpoint_quantized_w4a8_new_format.onnx'
quant_params_name = 'onnx_models/superpoint_quantized_w4a8_new_format_quant_params.pickle'
shallow_onnx_name = 'onnx_models/superpoint_quantized_w4a8_new_format_shallow.onnx'

if use_liteml:
    # config = 'configs/liteml_magicpoint_repeatability_heatmap_W4A8_QAT.yaml'  # W4A8 QAT model wrapped with LiteML
    # config = 'configs/liteml_magicpoint_repeatability_heatmap_W8A8_PTQ.yaml'  # W8A8 PTQ model wrapped with LiteML
    config = 'configs/liteml_magicpoint_repeatability_heatmap_W8A8_act_float_PTQ.yaml'  # W8A8 PTQ model wrapped with LiteML
else:
    weights_path = 'logs/superpoint_coco_heat2_0/checkpoints/superPointNet_170000_checkpoint.pth.tar'  # for PTQ
    config = 'configs/magicpoint_repeatability_heatmap.yaml'  # Float model without LiteML
with open(config, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# basic settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## parameters
outputMatches = True
subpixel = config["model"]["subpixel"]["enable"]
patch_size = config["model"]["subpixel"]["patch_size"]

# data loading
from utils.loader import dataLoader_test as dataLoader
# task = config["data"]["dataset"]
# data = dataLoader(config, dataset=task)
# test_set, test_loader = data["test_set"], data["test_loader"]
# from utils.print_tool import datasize
# datasize(test_loader, config, tag="test")

# model loading
from utils.loader import get_module
Val_model_heatmap = get_module("", config["front_end_model"])
## load pretrained
val_agent = Val_model_heatmap(config["model"], device=device)
val_agent.calibration_data = config.get("calibration_data")  # for calibration in PTQ

val_agent.loadModel()
val_agent.net.to(device)
# inp = torch.randn((1, 1, 1000, 1000)).to(device)
inp = torch.randn((1, 1, 240, 320)).to(device)

val_agent.net.export_to_onnx(inp, onnx_path, inplace=True)
# val_agent.net.export_to_onnx(inp, name=onnx_path, quant_params_name=quant_params_name, shallow_onnx_name=shallow_onnx_name, inplace=True)
