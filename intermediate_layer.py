import onnx
from onnx import helper

model_path = "onnx_models/superpoint_quantized_w8a8_zp_int_2.onnx"
model = onnx.load(model_path)
intermediate_layer_value_info = helper.ValueInfoProto()
intermediate_layer_value_info.name = "173"
model.graph.output.append(intermediate_layer_value_info)

intermediate_layer_value_info = helper.ValueInfoProto()
intermediate_layer_value_info.name = "175"
model.graph.output.append(intermediate_layer_value_info)

intermediate_layer_value_info = helper.ValueInfoProto()
intermediate_layer_value_info.name = "176"
model.graph.output.append(intermediate_layer_value_info)

intermediate_layer_value_info = helper.ValueInfoProto()
intermediate_layer_value_info.name = "181"
model.graph.output.append(intermediate_layer_value_info)

onnx.save(model, "onnx_models/superpoint_quantized_w8a8_zp_int_2_intermediate.onnx")