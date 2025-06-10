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
#from NeublaDriver0 import NeublaDriver0
#from NeublaDriver1 import NeublaDriver1

target_model = "../yolov3/yolov3_d53_mstrain-608_273e_coco_optim_opset12.neubla_u8_lwq_movingaverage.onnx"
raw_input_img = cv2.imread("../yolov3/dog.jpg")
input_width = input_height = 608

REPEAT = 1

def prepare_onnx_model(
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


def preprocess(raw_input_img):
    img_height, img_width = raw_input_img.shape[:2]
    img = cv2.cvtColor(raw_input_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_width, input_height))
    image_data = np.array(img) / 255.0  # normalize
    image_data = np.transpose(image_data, (2, 0, 1))  # Channel first
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

    return image_data, (img_width, img_height)


#input_image_name="yolov3/dog.jpg"
#def get_next_img():
#    raw_input_img = cv2.imread(input_image_name)


#    if not success:
#        return None
#    return raw_input_img


#target_model = "../yolov3/yolov3_d53_mstrain-608_273e_coco_optim_opset12.neubla_u8_lwq_movingaverage.onnx"
start_time = time.time()
front_sess, back_sess, (scale, zero_point) = prepare_onnx_model(target_model)
end_time = time.time()
elapsed_time = (end_time - start_time)  * 1000
print(f"\nFrontend/backend sesstion creation time: {elapsed_time:.2f} ms\n")

driver = NeublaDriver()
def process_frame_driver(npu_num):


    prep_time = 0.0
    quant_time = 0.0
    npu_time = 0.0
    cpu_time = 0.0
    dequant_time = 0.0

    print(f"==== process_frame_driver #{npu_num} ====")

    start_time = time.time()
    assert driver.Init(npu_num) == 0 
    end_time = time.time()
    elapsed_time = (end_time - start_time)  * 1000
    print(f"Driver init: {elapsed_time:.2f} ms")

    start_time = time.time()
    assert driver.LoadModel("./yolov3_half.o") == 0
    end_time = time.time()
    elapsed_time = (end_time - start_time)  * 1000
    print(f"Model load: {elapsed_time:.2f} ms")
    
    try:
        for i in range(REPEAT):

            start_time = time.time()
            input_img, (img_width, img_height) = preprocess(raw_input_img)
        
            end_time = time.time()
            elapsed_time = (end_time - start_time)  * 1000
            print(f"preprocessing time: {elapsed_time:.2f} ms")
            #prep_time = prep_time + elapsed_time

            start_time = time.time()
            input_data = front_sess.run(None, {"input": input_img})[
                0
            ].tobytes()  # to binary string

            end_time = time.time()
            elapsed_time = (end_time - start_time)  * 1000
            print(f"front time: {elapsed_time:.2f} ms")
            #quant_time = quant_time + elapsed_time

            start_time = time.time()
            assert driver.SendInput(input_data, 3 * 608 * 608) == 0
            end_time = time.time()
            elapsed_time = (end_time - start_time)  * 1000
            print(f"NPU input send time: {elapsed_time:.2f} ms")

            start_time = time.time()
            assert driver.Launch() == 0
            end_time = time.time()
            elapsed_time = (end_time - start_time)  * 1000
            print(f"NPU Launch time: {elapsed_time:.2f} ms")

            start_time = time.time()
            raw_outputs = driver.ReceiveOutputs()
            end_time = time.time()
            elapsed_time = (end_time - start_time)  * 1000
            print(f"NPU receive time: {elapsed_time:.2f} ms")

            start_time = time.time()
            output_data = [
                np.frombuffer(output, dtype=np.uint8) for output in raw_outputs
            ]

            #end_time = time.time()
            #elapsed_time = (end_time - start_time)  * 1000
            #print(f"NPU receive time: {elapsed_time} ms")
            #npu_time = npu_time + elapsed_time


            #start_time = time.time()
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

            end_time = time.time()
            elapsed_time = (end_time - start_time)  * 1000
            print(f"Dequant time: {elapsed_time:.2f} ms")
            #dequant_time = dequant_time + elapsed_time

            #start_time = time.time()
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
            #end_time = time.time()
            #elapsed_time = (end_time - start_time)  * 1000
            #print(f"Back time: {elapsed_time} ms")
            #cpu_time = cpu_time + elapsed_time

            #start_time = time.time()
            #output_image = postprocess(
            #raw_input_img, output, img_width, img_height
            #)
            #end_time = time.time()
            #elapsed_time = (end_time - start_time)  * 1000
            #print(f"Postprocessing time: {elapsed_time} ms")

    except Exception as e:
        assert driver.Close() == 0
        print(e)
        print("Error occured. Closed successfully.")
        exit()

    assert driver.Close() == 0
    print(f"Multi-Antara Driver #{npu_num} ended")
    #print(f"prep time: {prep_time/REPEAT} ms")
    #print(f"quant time: {quant_time/REPEAT} ms")
    #print(f"npu time: {npu_time/REPEAT} ms")
    #print(f"cpu time: {cpu_time/REPEAT} ms")
    #print(f"dequant time: {dequant_time/REPEAT} ms")



def process_onnxruntime_cpu():
    print("==== process_onnxruntime_cpu() ====")

    start_time = time.time()
    cpu_total_session = ort.InferenceSession(target_model)
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000
    print(f"session creation time: {elapsed_time:.2f} ms")

    for i in range(REPEAT):
        start_time = time.time()
        input_img, (img_width, img_height) = preprocess(raw_input_img)

        end_time = time.time()
        prep_time = (end_time - start_time) * 1000
        print(f"preprocessing time: {prep_time:.2f} ms")

    #input_names = [inp.name for inp in session.get_inputs()]
    #output_names = [inp.name for inp in session.get_outputs()]
   # print(input_names)
   # print(output_names)

        cpu_start_time = time.time()
        output = cpu_total_session.run(None, {"input": input_img})
        end_time = time.time()
        elapsed_cpu_time = (end_time - cpu_start_time) * 1000
        print(f"inference time: {elapsed_cpu_time:.2f} ms\n")

        #postp_start_time = time.time()
        #output_image = postprocess(
        #    raw_input_img, output, img_width, img_height
        #    )
        #end_time = time.time()
        #postp_elapsed_time = ((end_time - postp_start_time) * 1000)/REPEAT
        #print(f"Yolov3 Postprocessing time: {postp_elapsed_time} ms")

    # 출력 결과 확인
    #for i, output in enumerate(outputs):
    #    print(f"Output {i}: {output.shape}")




def main():

    #start_time = time.time()

    process_onnxruntime_cpu()


    process_frame_driver(0)

    #process_frame_driver1("./yolov3/yolov3_half.o")
    #process1 = mp.Process(target=process_frame_driver0, args=("./yolov3/yolov3_half.o",))
    #process1.start()

#    process2 = mp.Process(target=process_frame_driver1, args=("./yolov3/yolov3_half.o",))
#    process2.start()

    #process1.join()
#    process2.join()

   # end_time = time.time()
    #elapsed_time = ((end_time - start_time) * 1000)/REPEAT
   # elapsed_time = (end_time - start_time)*1000
    #print(f"Total time: {elapsed_time} ms")
   # print(f"Total time: {elapsed_time:.2f} ms")

    #assert driver0.Close() == 0
    #print("Multi-Antara Driver 0 ended")

    return 0


if __name__ == "__main__":
    main()
    


