import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import requests
import matplotlib.pyplot as plt
import warnings
import onnx
import json
from utils.loader import modelLoader
import yaml
from tqdm import tqdm
from export_to_onnx import convert_pth_to_onnx
from liteml.ailabs_quant.quant_modules import Quantizer
warnings.filterwarnings('ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')


def run_inference(config, use_liteml):
    from utils.loader import get_save_path
    from utils.var_dim import squeezeToNumpy

    # basic settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## parameters
    outputMatches = True
    subpixel = config["model"]["subpixel"]["enable"]
    patch_size = config["model"]["subpixel"]["patch_size"]

    # data loading
    from utils.loader import dataLoader_test as dataLoader
    task = config["data"]["dataset"]
    data = dataLoader(config, dataset=task)
    test_set, test_loader = data["test_set"], data["test_loader"]
    from utils.print_tool import datasize
    datasize(test_loader, config, tag="test")

    # model loading
    from utils.loader import get_module
    Val_model_heatmap = get_module("", config["front_end_model"])
    ## load pretrained
    val_agent = Val_model_heatmap(config["model"], device=device)
    val_agent.calibration_data = config.get("calibration_data")  # for calibration in PTQ

    val_agent.loadModel()
    val_agent.net.to(device)

    # Add hooks
    zero_percentages = {}
    if use_liteml:
        layer_names = register_hooks_quantized_model(val_agent.net, zero_percentages)  # for quantized model
    else:
        layer_names = register_hooks(val_agent.net, zero_percentages)


    ###### check!!!
    count = 0
    for i, sample in tqdm(enumerate(test_loader)):
        img_0, img_1 = sample["image"], sample["warped_image"]

        # first image, no matches
        # img = img_0
        def get_pts_desc_from_agent(val_agent, img, device="cpu"):
            """
            pts: list [numpy (3, N)]
            desc: list [numpy (256, N)]
            """
            heatmap_batch = val_agent.run(
                img.to(device)
            )  # heatmap: numpy [batch, 1, H, W]
            # heatmap to pts
            pts = val_agent.heatmap_to_pts()
            # print("pts: ", pts)
            if subpixel:
                pts = val_agent.soft_argmax_points(pts, patch_size=patch_size)
            # heatmap, pts to desc
            desc_sparse = val_agent.desc_to_sparseDesc()
            # print("pts[0]: ", pts[0].shape, ", desc_sparse[0]: ", desc_sparse[0].shape)
            # print("pts[0]: ", pts[0].shape)
            outs = {"pts": pts[0], "desc": desc_sparse[0]}
            return outs

        def transpose_np_dict(outs):
            for entry in list(outs):
                outs[entry] = outs[entry].transpose()

        outs = get_pts_desc_from_agent(val_agent, img_0, device=device)
        pts, desc = outs["pts"], outs["desc"]  # pts: np [3, N]
        if i == 200:
            break

    return layer_names, zero_percentages



def count_zeros_hook(module, input, output, layer_name, zero_percentages):
    input_tensor = input[0]
    zero_count = (input_tensor == 0).sum().item()
    total_elements = input_tensor.numel()
    zero_percentage = (zero_count / total_elements)
    # zero_percentages[layer_name] = zero_percentage
    if zero_percentages.get(layer_name) is None:
        zero_percentages[layer_name] = []
    zero_percentages[layer_name].append(zero_percentage)


def register_hooks(model, zero_percentages):
    layer_names = []
    for layer_name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Linear):
            layer_names.append(layer_name)
            layer.register_forward_hook(
                lambda module, input, output, name=layer_name: count_zeros_hook(module, input, output, name,
                                                                          zero_percentages))
    return layer_names


def register_hooks_quantized_model(model, zero_percentages):
    layer_names = []
    for layer_name, layer in model.named_modules():
        if isinstance(layer, Quantizer):
            layer_names.append(layer_name)
            layer.register_forward_hook(
                lambda module, input, output, name=layer_name: count_zeros_hook(module, input, output, name,
                                                                          zero_percentages))
    return layer_names

def export_to_onnx(onnx_model_path, weights_path):
    # torch.onnx.export(model, batch[0:1, :, :, :], path, verbose=True)
    # print(f"Model has been successfully exported to {path}")
    model = modelLoader(model='SuperPointNet_gauss2')

    checkpoint = torch.load(weights_path,
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])

    input_size = (1, 1, 1000, 1000)

    convert_pth_to_onnx(model, onnx_model_path, input_size)


def load_onnx_model(path):
    return onnx.load(path)


def create_layer_mapping(layer_names, onnx_model):
    onnx_conv_nodes = [node for node in onnx_model.graph.node if node.op_type in ['Conv', 'ConvTranspose', 'Gemm']]
    onnx_layer_names = [node.name for node in onnx_conv_nodes]
    return dict(zip(layer_names, onnx_layer_names))


def save_sparsity_config(zero_percentages, layer_mapping, path):
    quantization_list = []
    for layer, percentage in zero_percentages.items():
        instances = {
            "node_name": layer_mapping[layer],
            "data_sparsity": percentage
        }
        quantization_list.append({"instances": instances})

    with open(path, 'w') as json_file:
        json.dump({"quantization": quantization_list}, json_file, indent=4)


def main():
    # set use_liteml=True to wrap model with LiteML and apply quantization.
    # set use_liteml=False to run the float model.
    use_liteml = True
    onnx_path = 'onnx/superpoint.onnx'
    if use_liteml:
        config = 'configs/liteml_magicpoint_repeatability_heatmap_W4A8_QAT.yaml'  # W4A8 QAT model wrapped with LiteML
        # config = 'configs/liteml_magicpoint_repeatability_heatmap_W8A8_PTQ.yaml'  # W8A8 PTQ model wrapped with LiteML
    else:
        weights_path = 'logs/superpoint_coco_heat2_0/checkpoints/superPointNet_170000_checkpoint.pth.tar'  # for PTQ
        config = 'configs/magicpoint_repeatability_heatmap.yaml'  # Float model without LiteML
    with open(config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    layer_names, zero_percentages = run_inference(config, use_liteml)

    zero_percentages_avg = {key: np.average(zero_percentages[key]) for key in zero_percentages}

    for layer, percentage in zero_percentages_avg.items():
        print(f"Layer: {layer}, Zero Percentage: {percentage * 100:.2f}%")

    if not use_liteml:
        # The code below works only for models that are not wrapped with LiteML
        export_to_onnx(onnx_path, weights_path)
        onnx_model = load_onnx_model(onnx_path)
        layer_mapping = create_layer_mapping(layer_names, onnx_model)
        save_sparsity_config(zero_percentages, layer_mapping, './onnx/superpoint_sparsity_cfg.json')


if __name__ == "__main__":
    main()