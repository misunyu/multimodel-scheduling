import multiprocessing as mp
import time
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch

from NeublaDriver import NeublaDriver

REPEAT = 3

def initialize_driver(npu_num, o_path):
    driver = NeublaDriver()
    assert driver.Init(npu_num) == 0
    assert driver.LoadModel(o_path) == 0
    print(f"Driver for NPU{npu_num} initialized and model loaded successfully.")
    return driver

def close_driver(driver):
    if driver is not None:
        assert driver.Close() == 0
        print("Driver closed successfully.")

def send_receive_data_npu(driver, input_data, input_size):
    assert driver is not None, "Driver must be initialized"
    assert driver.SendInput(input_data, input_size) == 0
    assert driver.Launch() == 0
    return driver.ReceiveOutputs()

def yolo_prepare_onnx_model(
    target_model,
    front_input_names=["input"],
    front_output_names=["input_quantized"],
    target_dequant_layer_name=[
        "onnx::Transpose_684_DequantizeLinear",
        "onnx::Transpose_688_DequantizeLinear",
        "onnx::Transpose_692_DequantizeLinear",
    ],
    back_input_names=[
        "onnx::Transpose_684",
        "onnx::Transpose_688",
        "onnx::Transpose_692",
    ],
    back_output_names=["dets", "labels"],
):
    ## Prepare quantize layer(front model), postprocess layers(back model) and scale factors of dequant layers

    onnx_model = onnx.load(target_model)
    onnx.checker.check_model(onnx_model)

    onnx_graph = onnx_model.graph
    onnx_graph_nodes = onnx_graph.node

    front_output_path = target_model + ".front"
    back_output_path = target_model + ".back"

    # Get front model
    onnx.utils.extract_model(
        target_model, front_output_path, front_input_names, front_output_names
    )
    front_sess = ort.InferenceSession(front_output_path)

    # Get scale factor
    scale = {}
    zero_point = {}
    for idx, node in enumerate(onnx_graph_nodes):
        if node.name in target_dequant_layer_name:
            factors = onnx_model.graph.initializer
            for idx, init in enumerate(factors):
                if factors[idx].name == node.input[1]:
                    scale[node.name] = onnx.numpy_helper.to_array(factors[idx])
                elif factors[idx].name == node.input[2]:
                    zero_point[node.name] = onnx.numpy_helper.to_array(factors[idx])

    # Get back model
    onnx.utils.extract_model(
        target_model, back_output_path, back_input_names, back_output_names
    )
    back_sess = ort.InferenceSession(back_output_path)

    return front_sess, back_sess, (scale, zero_point)

def resnet50_prepare_onnx_model(target_model):
    front_input_names = ["input"]
    front_output_names = ["input_quantized"]
    target_gemm_layer_name = ["/0/fc/Gemm_quant"]

    onnx_model = onnx.load(target_model)
    onnx.checker.check_model(onnx_model)
    front_output_path = target_model + ".front"

    onnx.utils.extract_model(target_model, front_output_path, front_input_names, front_output_names)
    front_sess = ort.InferenceSession(front_output_path)

    params = {}
    for node in onnx_model.graph.node:
        if node.name in target_gemm_layer_name:
            for init in onnx_model.graph.initializer:
                if init.name in node.input:
                    params[init.name] = onnx.numpy_helper.to_array(init)

    return front_sess, None, params

def yolo_preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (608, 608))
    image_data = np.expand_dims(np.transpose(img / 255.0, (2, 0, 1)), axis=0).astype(np.float32)
    return image_data

def resnet50_preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image_data = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0).astype(np.float32)
    return image_data

def yolo_process_frame_driver():
    driver = initialize_driver(0, "models/yolov3_big/npu_code/yolov3_half.o")
    img = cv2.imread("../yolov3/dog.jpg")
    front_sess, back_sess, (scale, zero_point) = yolo_prepare_onnx_model(
        "../yolov3/yolov3_d53_mstrain-608_273e_coco_optim_opset12.neubla_u8_lwq_movingaverage.onnx"
    )

    try:
        for _ in range(REPEAT):
            input_data = front_sess.run(None, {"input": yolo_preprocess(img)})[0].tobytes()
            raw_outputs = send_receive_data_npu(driver, input_data, 3 * 608 * 608)

            output_data = [np.frombuffer(out, dtype=np.uint8) for out in raw_outputs]
            output_dequant_data = [
                (data.astype(np.float32) - zero_point[name]) * scale[name]
                for name, data in zip(scale.keys(), output_data)
            ]

            back_feeds = {
                "onnx::Transpose_684": (
                    output_dequant_data[0][: 255 * 19 * 19]
                ).reshape(1, 255, 19, 19),
                "onnx::Transpose_688": (
                    output_dequant_data[1][: 255 * 38 * 38]
                ).reshape(1, 255, 38, 38),
                "onnx::Transpose_692": (
                    output_dequant_data[2][: 255 * 76 * 76]
                ).reshape(1, 255, 76, 76),
            }

            back_output = back_sess.run(None, back_feeds)
            print("Yolo done.")
    finally:
        close_driver(driver)

def resnet50_process_frame_driver():
    driver = initialize_driver(1, "./resnet50_neubla_ori_best.o")
    img = cv2.imread("../resnet/cat_285.jpg")
    front_sess, _, params = resnet50_prepare_onnx_model(
        "../resnet/resnet50-0676ba61_opset12.neubla_u8_lwq_percentile.onnx"
    )

    try:
        scale = params['/0/avgpool/GlobalAveragePool_output_0_scale'] * params['0.fc.weight_scale']
        zp_act = params['/0/avgpool/GlobalAveragePool_output_0_zero_point']
        zp_w = params['0.fc.weight_zero_point']
        scale_out = params['/0/fc/Gemm_output_0_scale']
        zp_out = params['/0/fc/Gemm_output_0_zero_point']
        weight_q = params['0.fc.weight_quantized'].T.astype(np.int32)

        for _ in range(REPEAT):
            input_data = front_sess.run(None, {"input": resnet50_preprocess(img)})[0].tobytes()
            raw_outputs = send_receive_data_npu(driver, input_data, 3 * 224 * 224)
            output_data = np.frombuffer(raw_outputs[0], dtype=np.uint8)

            output = np.matmul(output_data.astype(np.int32), weight_q)
            output -= zp_act * np.sum(weight_q, axis=0)
            output -= zp_w * np.sum(output_data, axis=0)
            output += zp_act * zp_w
            output = np.round(output * scale / scale_out) + zp_out
            output = output.astype(np.uint8)
            max_index = np.argmax(output)

            print(f"ResNet50 max index: {max_index}")
    finally:
        close_driver(driver)

def main():
    yolo_proc = mp.Process(target=yolo_process_frame_driver)
    resnet_proc = mp.Process(target=resnet50_process_frame_driver)

    yolo_proc.start()
    resnet_proc.start()
    yolo_proc.join()
    resnet_proc.join()

if __name__ == "__main__":
    main()
