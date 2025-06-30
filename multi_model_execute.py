import multiprocessing as mp
import threading
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort

from NeublaDriver import NeublaDriver

#import yaml
#parser = argparse.ArgumentParser(prog="Yolo Demo")

REPEAT = 3

def yolo_prepare_onnx_model(
    yolo_target_model,
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

    onnx_model = onnx.load(yolo_target_model)
    onnx.checker.check_model(onnx_model)

    onnx_graph = onnx_model.graph
    onnx_graph_nodes = onnx_graph.node

    front_output_path = yolo_target_model + ".front"
    back_output_path = yolo_target_model + ".back"

    # Get front model
    onnx.utils.extract_model(
        yolo_target_model, front_output_path, front_input_names, front_output_names
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
        yolo_target_model, back_output_path, back_input_names, back_output_names
    )
    back_sess = ort.InferenceSession(back_output_path)

    return front_sess, back_sess, (scale, zero_point)


def resnet50_prepare_onnx_model(
    target_model,
    front_input_names=["input"],
    front_output_names=["input_quantized"],
    #front_output_names=["input_quantized"],
    target_gemm_layer_name=[
        "/0/fc/Gemm_quant"
    ]
):

    onnx_model = onnx.load(target_model)
    onnx.checker.check_model(onnx_model)

    onnx_graph = onnx_model.graph
    onnx_graph_nodes = onnx_graph.node

    front_output_path = target_model + ".front"

    # Get front model
    onnx.utils.extract_model(
        target_model, front_output_path, front_input_names, front_output_names
    )
    front_sess = ort.InferenceSession(front_output_path)

    params = {}
    for idx, node in enumerate(onnx_graph_nodes):
        if node.name in target_gemm_layer_name:
            factors = onnx_model.graph.initializer
            for idx, init in enumerate(factors):
                if factors[idx].name in node.input:
                    param_name = factors[idx].name
                    params[param_name] = onnx.numpy_helper.to_array(factors[idx])

    return front_sess, None, params



def yolo_preprocess(raw_input_img):
    input_width = input_height = 608

    img_height, img_width = raw_input_img.shape[:2]
    img = cv2.cvtColor(raw_input_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_width, input_height))
    image_data = np.array(img) / 255.0  # normalize
    image_data = np.transpose(image_data, (2, 0, 1))  # Channel first
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

    return image_data, (img_width, img_height)


def resnet50_preprocess(raw_input_img):
    input_width = input_height = 224
    raw_input_img = cv2.imread("../resnet/cat_285.jpg")

    img_height, img_width = raw_input_img.shape[:2]
    img = cv2.cvtColor(raw_input_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_width, input_height))
    image_data = np.array(img) / 255.0  # normalize

    image_data = image_data - np.array([0.485, 0.456, 0.406])
    image_data = image_data / np.array([0.229, 0.224, 0.225])

    image_data = np.transpose(image_data, (2, 0, 1))  # Channel first
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

    return image_data, (img_width, img_height)


def yolo_process_frame_driver(npu_num):

    print(f"==== Start Driver #{npu_num} ====")

    raw_input_img = cv2.imread("../yolov3/dog.jpg")
    yolo_target_model = "../yolov3/yolov3_d53_mstrain-608_273e_coco_optim_opset12.neubla_u8_lwq_movingaverage.onnx"

    front_sess, back_sess, (scale, zero_point) = yolo_prepare_onnx_model(yolo_target_model)

    total_time = 0.0

    driver = NeublaDriver()

    assert driver.Init(npu_num) == 0 
    assert driver.LoadModel("./yolov3_half.o") == 0
    
    try:
        for i in range(REPEAT):
            #print(f"Repeat count: {i}")
            #print("." , end='', flush=True)

            start_time = time.time()
            input_img, (img_width, img_height) = yolo_preprocess(raw_input_img)
        
            input_data = front_sess.run(None, {"input": input_img})[
                0
            ].tobytes()  # to binary string

            assert driver.SendInput(input_data, 3 * 608 * 608) == 0
            assert driver.Launch() == 0
            raw_outputs = driver.ReceiveOutputs()

            output_data = [
                np.frombuffer(output, dtype=np.uint8) for output in raw_outputs
            ]

            output_dequant_data = [
                (data.astype(np.float32) - zero_point[name]) * scale[name]
                for name, data in zip(
                    [
                        "onnx::Transpose_684_DequantizeLinear",
                        "onnx::Transpose_688_DequantizeLinear",
                        "onnx::Transpose_692_DequantizeLinear",
                    ],
                    output_data,
                )
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

            output = back_sess.run(None, back_feeds)
            end_time = time.time()
            elapsed_time = (end_time - start_time)  * 1000
            #print(f"\nDriver0 time: {elapsed_time:.2f} ms")
            total_time = total_time + elapsed_time
            #time.sleep(0.5)

        average_time = total_time/REPEAT
        print(f"\n(Driver0) Average time over {REPEAT} runs is {average_time:.2f} ms")

        assert driver.Close() == 0

    except Exception as e:
        assert driver.Close() == 0
        print(e)
        print("Error occured. Closed successfully.")
        exit()


def resnet50_process_frame_driver(npu_num):

    print(f"==== Start Driver # {npu_num}  ====")

    driver = NeublaDriver()

    total_time = 0.0

    raw_input_img = cv2.imread("../resnet/cat_285.jpg")
    resnet50_target_model = "../resnet/resnet50-0676ba61_opset12.neubla_u8_lwq_percentile.onnx"

    front_sess, back_sess, params = resnet50_prepare_onnx_model(resnet50_target_model)
    dequant_scale = params['/0/avgpool/GlobalAveragePool_output_0_scale'] * params['0.fc.weight_scale']
    dequant_act_zero_point = params['/0/avgpool/GlobalAveragePool_output_0_zero_point']
    dequant_weight_zero_point = params['0.fc.weight_zero_point']
    requant_scale = params['/0/fc/Gemm_output_0_scale']
    requant_zero_point = params['/0/fc/Gemm_output_0_zero_point']


    assert driver.Init(npu_num) == 0
    #assert driver.LoadModel("../resnet/resnet50_neubla_ori_lw.o") == 0
    assert driver.LoadModel("./resnet50_neubla_ori_best.o") == 0

    try:
        for i in range(REPEAT):
            #print("*" , end='', flush=True)

            input_img, (img_width, img_height) = resnet50_preprocess(raw_input_img)

            #start_time = time.time()
            input_data = front_sess.run(None, {"input": input_img})[
                0
            ].tobytes()  # to binary string


            start_time = time.time()
            assert driver.SendInput(input_data, 3 * 224 * 224) == 0
            assert driver.Launch() == 0

            raw_outputs = driver.ReceiveOutputs()

            output_data = [
                np.frombuffer(output, dtype=np.uint8) for output in raw_outputs
            ]

            output_gemm = np.matmul(output_data[0].astype(np.int32), params['0.fc.weight_quantized'].T.astype(np.int32))
            output_a_sum = np.sum(output_data[0].astype(np.int32), axis=0)
            output_w_sum = np.sum(params['0.fc.weight_quantized'].T.astype(np.int32), axis=0)

            output_gemm = (output_gemm - (dequant_act_zero_point * output_w_sum) - (dequant_weight_zero_point * output_a_sum) + (dequant_act_zero_point * dequant_weight_zero_point)) * dequant_scale  # gemm + dequantize
            output = (np.round(output_gemm / requant_scale) + requant_zero_point).astype(np.uint8)  # requantize

            max_index = np.argmax(output)

            end_time = time.time()
            elapsed_time = (end_time - start_time)  * 1000
            total_time = total_time + elapsed_time

        print(f"\nmax_idx = {max_index}")
        average_time = total_time/REPEAT
        print(f"\n(Driver1) Average time over {REPEAT} runs is {average_time:.2f} ms")


        assert driver.Close() == 0

    except Exception as e:
        assert driver.Close() == 0
        print(e)
        print("Error occured. Closed successfully.")
        exit()

def run_driver_yolo(npu_num):
    yolo_process_frame_driver(npu_num)
    print(f"Multi-Antara Driver #{npu_num} ended")

def run_driver_resnet50(npu_num):
    resnet50_process_frame_driver(npu_num)
    print(f"Multi-Antara Driver #{npu_num} ended")

def main():

    #yolo_process_frame_driver0()
    #assert driver0.Close() == 0
    #print("Multi-Antara Driver 0 ended")
    
    #resnet50_process_frame_driver1()
    #assert driver1.Close() == 0
    #print("Multi-Antara Driver 1 ended")
   
    process_yolo = mp.Process(target=run_driver_yolo, args=(0,))
    process_resnet50 = mp.Process(target=run_driver_resnet50, args=(1,))

    process_yolo.start()
    process_resnet50.start()

    process_yolo.join()
    process_resnet50.join()

    return 0


if __name__ == "__main__":
    main()
    


