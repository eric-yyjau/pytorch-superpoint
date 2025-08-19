from utils.loader import modelLoader
import torch
import onnx


def convert_pth_to_onnx(model, onnx_model_path, input_size=(1, 1, 1000, 1000)):

    dummy_input = torch.randn(*input_size)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['prob', 'desc'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'prob': {0: 'batch_size', 2: 'height', 3: 'width'},
            'desc': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )


if __name__ == "__main__":
    onnx_model_path = 'onnx/superPointNet_17000.onnx'
    weights_path = 'logs/superpoint_coco_heat2_0/checkpoints/superPointNet_170000_checkpoint.pth.tar'
    model = modelLoader(model='SuperPointNet_gauss2')

    checkpoint = torch.load(weights_path,
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])

    input_size = (1, 1, 1000, 1000)

    convert_pth_to_onnx(model, onnx_model_path, input_size)


